import asyncio

import numpy as np
import pytest

from nanovllm_voxcpm.models.voxcpm2.server import AsyncVoxCPM2Server
from nanovllm_voxcpm.utils.worker_link import WorkerLink


@pytest.mark.parametrize("cancel_fails", [False, True])
def test_cancelled_buffered_attempt_releases_stream_even_if_cancel_fails(cancel_fails):
    async def exercise():
        server = object.__new__(AsyncVoxCPM2Server)
        server.stream_table = {}
        server._link = WorkerLink(asyncio.get_running_loop())
        started = asyncio.Event()
        commands = []

        async def submit(command, *args):
            commands.append(command)
            if command == "add_request":
                await server.stream_table[args[0]].put(np.ones(2, dtype=np.float32))
                started.set()
            elif cancel_fails:
                raise RuntimeError("cancel failed")

        server.submit = submit
        pending = asyncio.create_task(server.generate_buffered("hello", seed=7))
        await asyncio.wait_for(started.wait(), timeout=1)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert commands == ["add_request", "cancel"]
        assert server.stream_table == {}

    asyncio.run(exercise())
