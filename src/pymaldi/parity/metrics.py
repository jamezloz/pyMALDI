from __future__ import annotations

import math
from typing import Dict

import numpy as np


def mz_median_abs_error_da(a_mz: np.ndarray, b_mz: np.ndarray) -> float:
    n = min(a_mz.size, b_mz.size)
    if n == 0:
        return float("nan")
    return float(np.median(np.abs(a_mz[:n] - b_mz[:n])))


def pearson_intensity(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.size, b.size)
    if n < 2:
        return float("nan")
    a1 = a[:n]
    b1 = b[:n]
    if float(np.std(a1)) == 0.0 or float(np.std(b1)) == 0.0:
        return float("nan")
    return float(np.corrcoef(a1, b1)[0, 1])


def tic_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    ta = float(np.sum(a))
    tb = float(np.sum(b))
    if tb == 0.0:
        return 0.0 if ta == 0.0 else float("inf")
    return abs(ta - tb) / abs(tb)


def compute_raw_metrics(py_mz: np.ndarray, py_i: np.ndarray, ref_mz: np.ndarray, ref_i: np.ndarray) -> Dict[str, float]:
    return {
        "mz_median_abs_err_da": mz_median_abs_error_da(py_mz, ref_mz),
        "intensity_pearson_r": pearson_intensity(py_i, ref_i),
        "tic_rel_error": tic_relative_error(py_i, ref_i),
    }


def is_close_or_nan(value: float, threshold: float, mode: str) -> bool:
    if math.isnan(value):
        return False
    if mode == "le":
        return value <= threshold
    if mode == "ge":
        return value >= threshold
    raise ValueError(f"Unsupported mode: {mode}")
