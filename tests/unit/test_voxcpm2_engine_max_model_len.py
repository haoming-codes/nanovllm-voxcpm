import pytest

torch = pytest.importorskip("torch")


def _make_engine(max_model_len: int, token_count: int):
    """Create a VoxCPM2Engine instance without heavy init."""

    from nanovllm_voxcpm.models.voxcpm2.engine import VoxCPM2Engine

    e = VoxCPM2Engine.__new__(VoxCPM2Engine)
    e.n_decode_pad_frames = 4
    e.feat_dim = 8
    e.patch_size = 1
    e.audio_start_token = 101
    e.block_size = 256
    e.max_model_len = max_model_len

    e.tokenizer = lambda _s: list(range(token_count))

    e._captured_seq = None
    e.add_sequence = lambda seq: setattr(e, "_captured_seq", seq)
    e.resolve_lora = lambda name: None if name is None else 9
    return e


def test_add_request_rejects_too_long_prompt():
    e = _make_engine(max_model_len=4, token_count=4)
    with pytest.raises(ValueError, match=r"Prompt is too long"):
        e.add_request(seq_id="s", target_text="x", max_generate_length=1)


def test_add_request_rejects_when_total_can_exceed_max_model_len():
    e = _make_engine(max_model_len=10, token_count=4)
    with pytest.raises(ValueError, match=r"may exceed max_model_len"):
        e.add_request(seq_id="s", target_text="x", max_generate_length=6)


def test_add_request_allows_on_boundary_and_enqueues_sequence():
    e = _make_engine(max_model_len=11, token_count=4)
    e.add_request(seq_id="s", target_text="x", max_generate_length=6)
    assert e._captured_seq is not None
    assert len(e._captured_seq) == 5


def test_add_request_requires_positive_max_generate_length():
    e = _make_engine(max_model_len=10, token_count=1)
    with pytest.raises(ValueError, match=r"max_generate_length must be >= 1"):
        e.add_request(seq_id="s", target_text="x", max_generate_length=0)


def test_add_request_applies_short_text_floor_to_text_relative_limit():
    e = _make_engine(max_model_len=64, token_count=2)
    e.add_request(seq_id="s", target_text="x", max_generate_length=2000)

    payload = e._captured_seq.custom_payload
    assert payload.target_text_length == 2
    assert payload.retry_badcase_ratio_threshold == 6.0
    assert payload.max_generate_length == 26


def test_add_request_excludes_prompt_text_from_text_relative_limit():
    e = _make_engine(max_model_len=64, token_count=2)
    e.tokenizer = lambda text: [1, 2] if text == "target" else list(range(20))
    e.add_request(
        seq_id="s",
        target_text="target",
        prompt_text="prompt",
        max_generate_length=2000,
    )

    payload = e._captured_seq.custom_payload
    assert payload.target_text_length == 2
    assert payload.max_generate_length == 26


def test_add_request_keeps_text_relative_limit_for_longer_targets():
    e = _make_engine(max_model_len=128, token_count=10)
    e.add_request(seq_id="s", target_text="long target", max_generate_length=2000)
    assert e._captured_seq.custom_payload.max_generate_length == 70


def test_add_request_preserves_lower_requested_limit():
    e = _make_engine(max_model_len=16, token_count=2)
    e.add_request(seq_id="s", target_text="x", max_generate_length=7)

    assert e._captured_seq.custom_payload.max_generate_length == 7


@pytest.mark.parametrize("ratio", [0, -1, float('nan'), float('inf')])
def test_add_request_requires_positive_finite_badcase_ratio(ratio):
    e = _make_engine(max_model_len=16, token_count=2)
    with pytest.raises(ValueError, match=r"retry_badcase_ratio_threshold must be > 0"):
        e.add_request(
            seq_id="s",
            target_text="x",
            max_generate_length=7,
            retry_badcase_ratio_threshold=ratio,
        )


def test_add_request_resolves_lora_name_into_adapter_id():
    e = _make_engine(max_model_len=11, token_count=4)
    e.add_request(seq_id="s", target_text="x", max_generate_length=6, lora_name="demo")
    assert e._captured_seq.lora_name == "demo"
    assert e._captured_seq.adapter_id == 9


def test_postprocess_advances_seed_step_after_generated_latent():
    import numpy as np

    from nanovllm_voxcpm.models.voxcpm2.engine import VoxCPM2Engine, VoxCPM2SeqPayload

    engine = VoxCPM2Engine.__new__(VoxCPM2Engine)
    engine.feat_dim = 2
    engine.patch_size = 4
    engine.n_decode_pad_frames = 4

    class _Seq:
        stoped = False

        def __init__(self):
            self.tokens = []
            self.custom_payload = VoxCPM2SeqPayload(
                feats=[],
                text_tokens=[],
                feat_masks=[],
                generated_waveforms=[],
                temperature=1.0,
                cfg_value=1.0,
                target_text_length=10,
                retry_badcase_ratio_threshold=6.0,
                max_generate_length=10,
                seed=123,
                seed_step=2,
            )

        def append_token(self, token):
            self.tokens.append(token)

    seq = _Seq()
    engine.postprocess_seq(
        seq,
        {
            "latents": np.zeros((engine.patch_size, engine.feat_dim), dtype=np.float32),
            "waveforms": np.zeros(8, dtype=np.float32),
            "stop_flag": 0,
        },
        is_prefill=False,
    )

    assert seq.custom_payload.seed_step == 3


@pytest.mark.parametrize(
    ("prior_steps", "expected_badcase"),
    [
        (4, False),
        (11, False),
        (14, False),
        (15, True),
    ],
)
def test_postprocess_uses_short_text_floor_in_ratio_detector(prior_steps, expected_badcase):
    import numpy as np

    from nanovllm_voxcpm.models.voxcpm2.engine import VoxCPM2Engine, VoxCPM2SeqPayload

    engine = VoxCPM2Engine.__new__(VoxCPM2Engine)
    engine.feat_dim = 2
    engine.patch_size = 4
    engine.n_decode_pad_frames = 4

    class _Seq:
        stoped = False

        def __init__(self):
            self.tokens = []
            self.custom_payload = VoxCPM2SeqPayload(
                feats=[],
                text_tokens=[],
                feat_masks=[],
                generated_waveforms=[np.zeros(8, dtype=np.float32) for _ in range(prior_steps)],
                temperature=1.0,
                cfg_value=1.0,
                target_text_length=2,
                retry_badcase_ratio_threshold=6.0,
                max_generate_length=22,
            )

        def append_token(self, token):
            self.tokens.append(token)

    seq = _Seq()
    engine.postprocess_seq(
        seq,
        {
            "latents": np.zeros((engine.patch_size, engine.feat_dim), dtype=np.float32),
            "waveforms": np.zeros(8, dtype=np.float32),
            "stop_flag": 1,
        },
        is_prefill=False,
    )

    assert seq.stoped is True
    assert seq.custom_payload.is_badcase is expected_badcase
