"""End-to-end tests for worker death handling in the async servers.

Servers are built through their real ``__init__`` with ``main_loop``
monkeypatched to a lightweight fake worker, so the spawn / queue-bridge /
watchdog wiring under test is the production wiring. No GPU or model needed.

The fake workers below are executed in the spawned child, which re-imports
this module without the GPU shims installed by ``conftest``. Keep the
module-level imports light and free of ``nanovllm_voxcpm``.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import signal
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Fake workers (module level so ``spawn`` can pickle them by reference)
# ---------------------------------------------------------------------------


def _worker_silent(queue_in: Any, queue_out: Any, args: Any, kwargs: Any) -> None:
    """Acknowledges init, then never answers anything."""
    queue_out.put({"type": "init_ok"})
    while True:
        time.sleep(3600)


def _worker_one_chunk(queue_in: Any, queue_out: Any, args: Any, kwargs: Any) -> None:
    """Emits a single stream chunk for a request, then goes silent."""
    queue_out.put({"type": "init_ok"})
    while True:
        msg = queue_in.get()
        queue_out.put({"id": msg["id"], "type": "response", "data": None})
        if msg["type"] != "add_request":
            continue
        queue_out.put({"id": msg["args"][0], "type": "stream", "data": np.ones(2, dtype=np.float32)})
        while True:
            time.sleep(3600)


def _worker_healthy(queue_in: Any, queue_out: Any, args: Any, kwargs: Any, completion: Any = None) -> None:
    """Answers every command and completes streams normally."""
    queue_out.put({"type": "init_ok"})
    while True:
        msg = queue_in.get()
        op_id = msg["id"]
        if msg["type"] == "stop":
            queue_out.put({"id": op_id, "type": "response", "data": None})
            return
        if msg["type"] == "add_request":
            seq_id = msg["args"][0]
            queue_out.put({"id": op_id, "type": "response", "data": None})
            queue_out.put({"id": seq_id, "type": "stream", "data": np.ones(2, dtype=np.float32)})
            queue_out.put({"id": seq_id, "type": "stream", "data": completion})
            continue
        queue_out.put({"id": op_id, "type": "response", "data": {"status": "ok"}})


def _worker_healthy_voxcpm2(queue_in: Any, queue_out: Any, args: Any, kwargs: Any) -> None:
    from tests._shims import install_gpu_shims

    install_gpu_shims()
    from nanovllm_voxcpm.models.voxcpm2.server import _GenerationAttemptResult

    _worker_healthy(queue_in, queue_out, args, kwargs, _GenerationAttemptResult(None, 1, False))


_INIT_CRASH_EXITCODE = 42


def _worker_dies_during_init(queue_in: Any, queue_out: Any, args: Any, kwargs: Any) -> None:
    """Exits before the handshake, as a CUDA OOM during model load would."""
    os._exit(_INIT_CRASH_EXITCODE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SERVERS = [
    pytest.param("nanovllm_voxcpm.models.voxcpm.server", "AsyncVoxCPMServer", id="voxcpm"),
    pytest.param("nanovllm_voxcpm.models.voxcpm2.server", "AsyncVoxCPM2Server", id="voxcpm2"),
]

_READY_TIMEOUT = 60.0
_DEATH_TIMEOUT = 15.0


def _start(monkeypatch: pytest.MonkeyPatch, modname: str, clsname: str, worker: Any) -> Any:
    # ``spawn`` pickles the worker by reference and re-imports this module in
    # the child. pytest's importlib import mode does not put the repo root on
    # sys.path, so make it importable for the child process.
    monkeypatch.syspath_prepend(str(_REPO_ROOT))
    mod = importlib.import_module(modname)
    monkeypatch.setattr(mod, "main_loop", worker)
    return getattr(mod, clsname)("/fake/model")


async def _shutdown(server: Any) -> None:
    with contextlib.suppress(Exception):
        await asyncio.wait_for(server.stop(), timeout=30.0)


def _worker_died_error() -> type[BaseException]:
    from nanovllm_voxcpm.utils.worker_link import WorkerDiedError

    return WorkerDiedError


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def _wait_for_ready_fails_when_worker_dies_before_init(monkeypatch, modname, clsname):
    server = _start(monkeypatch, modname, clsname, _worker_dies_during_init)
    try:
        # Matching the exit code proves the fake worker actually ran, rather
        # than the child having failed to start for an unrelated reason.
        with pytest.raises(_worker_died_error(), match=f"exitcode={_INIT_CRASH_EXITCODE}"):
            await asyncio.wait_for(server.wait_for_ready(), timeout=_READY_TIMEOUT)
    finally:
        await _shutdown(server)


@pytest.mark.parametrize("modname,clsname", _SERVERS)
def test_wait_for_ready_fails_when_worker_dies_before_init(monkeypatch, modname, clsname):
    asyncio.run(_wait_for_ready_fails_when_worker_dies_before_init(monkeypatch, modname, clsname))


async def _pending_request_fails_and_new_ones_are_rejected(monkeypatch, modname, clsname):
    server = _start(monkeypatch, modname, clsname, _worker_silent)
    try:
        await asyncio.wait_for(server.wait_for_ready(), timeout=_READY_TIMEOUT)

        pending = asyncio.ensure_future(server.health())
        await asyncio.sleep(0)  # let submit() register the request
        assert len(server.op_table) == 1

        os.kill(server.process.pid, signal.SIGKILL)

        with pytest.raises(_worker_died_error()):
            await asyncio.wait_for(pending, timeout=_DEATH_TIMEOUT)
        assert server.op_table == {}

        # Subsequent requests fail immediately rather than queueing up behind
        # a worker that will never answer.
        with pytest.raises(_worker_died_error()):
            await asyncio.wait_for(server.health(), timeout=_DEATH_TIMEOUT)
        assert server.op_table == {}
    finally:
        await _shutdown(server)


@pytest.mark.parametrize("modname,clsname", _SERVERS)
def test_pending_request_fails_and_new_ones_are_rejected(monkeypatch, modname, clsname):
    asyncio.run(_pending_request_fails_and_new_ones_are_rejected(monkeypatch, modname, clsname))


async def _generate_fails_mid_stream_and_releases_slot(monkeypatch, modname, clsname):
    server = _start(monkeypatch, modname, clsname, _worker_one_chunk)
    try:
        await asyncio.wait_for(server.wait_for_ready(), timeout=_READY_TIMEOUT)

        stream = server.generate("hello")
        first = await asyncio.wait_for(stream.__anext__(), timeout=_READY_TIMEOUT)
        assert first.tolist() == [1.0, 1.0]
        assert len(server.stream_table) == 1

        # add_request has already been acknowledged, so the consumer is parked
        # on the stream queue: this is the path that used to hang forever.
        os.kill(server.process.pid, signal.SIGKILL)

        with pytest.raises(_worker_died_error()):
            await asyncio.wait_for(stream.__anext__(), timeout=_DEATH_TIMEOUT)

        assert server.stream_table == {}
        assert server.op_table == {}
    finally:
        await _shutdown(server)


@pytest.mark.parametrize("modname,clsname", _SERVERS)
def test_generate_fails_mid_stream_and_releases_slot(monkeypatch, modname, clsname):
    asyncio.run(_generate_fails_mid_stream_and_releases_slot(monkeypatch, modname, clsname))


async def _buffered_generation_discards_partial_audio_when_worker_dies(monkeypatch):
    from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2ServerPool

    server = _start(monkeypatch, "nanovllm_voxcpm.models.voxcpm2.server", "AsyncVoxCPM2Server", _worker_one_chunk)
    pool = object.__new__(AsyncVoxCPM2ServerPool)
    pool.servers = [server]
    pool.servers_load = np.zeros(1, dtype=np.int32)
    pool._prompt_pool = {}
    pool._registered_loras = set()
    pool._draining_loras = set()
    first_chunk = asyncio.Event()
    attempted_seeds = []
    generate_attempt = server._generate_attempt

    async def observe_attempt(**kwargs):
        attempted_seeds.append(kwargs["seed"])
        async for item in generate_attempt(**kwargs):
            if isinstance(item, np.ndarray):
                first_chunk.set()
            yield item

    server._generate_attempt = observe_attempt
    try:
        await asyncio.wait_for(server.wait_for_ready(), timeout=_READY_TIMEOUT)
        pending = asyncio.create_task(pool.generate_buffered("hello", seed=7, retry_badcase_max_times=3))
        await asyncio.wait_for(first_chunk.wait(), timeout=_READY_TIMEOUT)
        assert pool.servers_load.tolist() == [1]
        os.kill(server.process.pid, signal.SIGKILL)
        with pytest.raises(_worker_died_error()):
            await asyncio.wait_for(pending, timeout=_DEATH_TIMEOUT)
        assert attempted_seeds == [7]
        assert pool.servers_load.tolist() == [0]
        assert server.stream_table == {}
        assert server.op_table == {}
    finally:
        await _shutdown(server)


def test_buffered_generation_discards_partial_audio_when_worker_dies(monkeypatch):
    asyncio.run(_buffered_generation_discards_partial_audio_when_worker_dies(monkeypatch))


async def _healthy_worker_is_unaffected(monkeypatch, modname, clsname):
    worker = _worker_healthy_voxcpm2 if "voxcpm2" in modname else _worker_healthy
    server = _start(monkeypatch, modname, clsname, worker)
    try:
        await asyncio.wait_for(server.wait_for_ready(), timeout=_READY_TIMEOUT)

        assert await asyncio.wait_for(server.health(), timeout=_READY_TIMEOUT) == {"status": "ok"}

        chunks = [chunk async for chunk in server.generate("hello")]
        assert [chunk.tolist() for chunk in chunks] == [[1.0, 1.0]]

        assert server.stream_table == {}
        assert server.op_table == {}
    finally:
        await _shutdown(server)

    # A graceful shutdown must not be reported as a crash.
    assert not server._link.dead


@pytest.mark.parametrize("modname,clsname", _SERVERS)
def test_healthy_worker_is_unaffected(monkeypatch, modname, clsname):
    asyncio.run(_healthy_worker_is_unaffected(monkeypatch, modname, clsname))
