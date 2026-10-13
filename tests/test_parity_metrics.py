import numpy as np

from pymaldi.parity.metrics import compute_raw_metrics


def test_compute_raw_metrics_identity() -> None:
    mz = np.array([1000.0, 1001.0, 1002.0])
    intensity = np.array([10.0, 20.0, 30.0])

    m = compute_raw_metrics(mz, intensity, mz, intensity)

    assert m["mz_median_abs_err_da"] == 0.0
    assert abs(m["intensity_pearson_r"] - 1.0) < 1e-12
    assert m["tic_rel_error"] == 0.0
