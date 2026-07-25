from nanovllm_voxcpm.utils.seed import derive_step_seed, materialize_generation_seed


def test_derive_step_seed_is_stable_for_same_request_step():
    assert derive_step_seed(123, 4) == derive_step_seed(123, 4)


def test_derive_step_seed_advances_for_different_steps():
    seeds = {derive_step_seed(123, step) for step in range(8)}
    assert len(seeds) == 8


def test_materialize_generation_seed_preserves_explicit_seed():
    assert materialize_generation_seed(123) == 123


def test_materialize_generation_seed_uses_openbmb_32_bit_range(monkeypatch):
    monkeypatch.setattr("nanovllm_voxcpm.utils.seed.torch.seed", lambda: (1 << 40) + 17)
    assert materialize_generation_seed(None) == 17
