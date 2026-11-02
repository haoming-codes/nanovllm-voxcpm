import importlib.util
from pathlib import Path

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).parents[2] / "benchmark" / "calibrate_waveform.py"
_SPEC = importlib.util.spec_from_file_location("calibrate_waveform", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

compare_waveforms = _MODULE.compare_waveforms
comparison_failures = _MODULE.comparison_failures


def test_compare_waveforms_rejects_empty_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        compare_waveforms(np.empty(0, dtype=np.float32), np.ones(1, dtype=np.float32))


def test_compare_waveforms_does_not_treat_different_constants_as_correlated():
    comparison = compare_waveforms(
        np.ones(4, dtype=np.float32),
        np.zeros(4, dtype=np.float32),
    )

    assert comparison["correlation"] == 0.0
    assert comparison["rmse"] == 1.0
    assert comparison["length_match"] is True


def test_comparison_failures_enforces_lengths_and_thresholds():
    comparison = compare_waveforms(
        np.array([0.0, 1.0, 4.0, 9.0], dtype=np.float32),
        np.array([0.0, 1.0, 2.0], dtype=np.float32),
    )

    failures = comparison_failures(
        comparison,
        require_same_length=True,
        max_rmse=0.01,
        max_abs_error=0.05,
        min_correlation=0.9999,
    )

    assert any("sample count mismatch" in failure for failure in failures)
    assert any("rmse" in failure for failure in failures)
    assert any("max_abs_error" in failure for failure in failures)
    assert any("correlation" in failure for failure in failures)
