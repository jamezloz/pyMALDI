import numpy as np

from pymaldi.processing import peak_f1


def test_peak_f1_perfect_match() -> None:
    ref = np.array([1000.0, 2000.0, 3000.0])
    pred = np.array([1000.1, 1999.9, 3000.0])
    m = peak_f1(pred, ref, tolerance_da=0.2)
    assert m["f1"] == 1.0


def test_peak_f1_partial_match() -> None:
    ref = np.array([1000.0, 2000.0, 3000.0])
    pred = np.array([1000.1, 9000.0])
    m = peak_f1(pred, ref, tolerance_da=0.2)
    assert 0.0 < m["f1"] < 1.0
