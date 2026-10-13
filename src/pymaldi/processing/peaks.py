from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.ndimage import median_filter, maximum_filter1d, uniform_filter1d


@dataclass
class PeakList:
    mz: np.ndarray
    intensity: np.ndarray
    snr: np.ndarray
    index: np.ndarray


def _estimate_noise_mad(intensity: np.ndarray, half_window_size: int = 20) -> np.ndarray:
    med = float(np.median(intensity)) if intensity.size else 0.0
    mad = float(np.median(np.abs(intensity - med))) if intensity.size else 0.0
    noise = max(1.4826 * mad, 1e-12)
    return np.full_like(intensity, noise, dtype=float)


def _estimate_noise_supersmoother(intensity: np.ndarray, half_window_size: int = 20) -> np.ndarray:
    # Approximation of MALDIquant "SuperSmoother" noise mode.
    size = int(max(1, half_window_size) * 2 + 1)
    trend = uniform_filter1d(intensity, size=size, mode="nearest")
    return np.maximum(trend, 1e-12)


def estimate_noise(intensity: np.ndarray, method: str = "MAD", half_window_size: int = 20) -> np.ndarray:
    if method.lower() == "mad":
        method = "MAD"
    elif method.lower() == "supersmoother":
        method = "SuperSmoother"
    if method == "MAD":
        return _estimate_noise_mad(intensity, half_window_size=half_window_size)
    if method == "SuperSmoother":
        return _estimate_noise_supersmoother(intensity, half_window_size=half_window_size)
    raise ValueError(f"Unsupported noise method: {method}")


def find_local_maxima(intensity: np.ndarray, half_window_size: int = 20) -> np.ndarray:
    size = int(max(1, half_window_size) * 2 + 1)
    y = intensity.astype(float)
    mx = maximum_filter1d(y, size=size, mode="nearest")
    is_max = y == mx
    # Break ties on plateaus by keeping left-most index in each run.
    idx = np.where(is_max)[0]
    if idx.size == 0:
        return is_max
    keep = np.ones(idx.size, dtype=bool)
    keep[1:] = (idx[1:] - idx[:-1]) > 1
    out = np.zeros_like(is_max, dtype=bool)
    out[idx[keep]] = True
    return out


def detect_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    half_window_size: int = 20,
    method: str = "MAD",
    snr: float = 2.0,
) -> PeakList:
    y = intensity.astype(float)
    if y.size == 0:
        return PeakList(mz=np.array([]), intensity=np.array([]), snr=np.array([]), index=np.array([], dtype=int))

    noise = estimate_noise(y, method=method, half_window_size=half_window_size)
    is_local = find_local_maxima(y, half_window_size=half_window_size)
    is_above = y > (float(snr) * noise)
    idx = np.where(is_local & is_above)[0]

    peak_mz = mz[idx]
    peak_i = y[idx]
    peak_snr = peak_i / noise[idx]
    return PeakList(mz=peak_mz, intensity=peak_i, snr=peak_snr, index=idx.astype(int))


def peak_f1(
    mz_pred: np.ndarray,
    mz_ref: np.ndarray,
    tolerance_da: float = 0.5,
) -> dict[str, float]:
    if mz_pred.size == 0 and mz_ref.size == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched": 0.0}
    if mz_pred.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0.0}
    if mz_ref.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "matched": 0.0}

    p = np.sort(np.asarray(mz_pred, dtype=float))
    r = np.sort(np.asarray(mz_ref, dtype=float))
    i = 0
    j = 0
    matched = 0
    tol = float(tolerance_da)
    while i < p.size and j < r.size:
        d = p[i] - r[j]
        if abs(d) <= tol:
            matched += 1
            i += 1
            j += 1
        elif d < 0:
            i += 1
        else:
            j += 1

    prec = matched / p.size if p.size else 0.0
    rec = matched / r.size if r.size else 0.0
    f1 = 0.0 if (prec + rec) == 0 else (2.0 * prec * rec / (prec + rec))
    return {"precision": float(prec), "recall": float(rec), "f1": float(f1), "matched": float(matched)}
