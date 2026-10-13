from __future__ import annotations

import numpy as np


def calibrate_intensity_single(intensity: np.ndarray, method: str = "TIC") -> np.ndarray:
    method = method.strip()
    if method.lower() == "tic":
        method = "TIC"
    elif method.lower() == "pqn":
        method = "PQN"
    elif method.lower() == "median":
        method = "median"
    elif method.lower() == "none":
        method = "none"
    y = intensity.astype(float)
    if method == "none":
        return y
    if method == "TIC":
        denom = float(np.sum(y))
        return y / denom if denom > 0 else y
    if method == "median":
        med = float(np.median(y)) if y.size else 0.0
        return y / med if med > 0 else y
    if method == "PQN":
        raise ValueError("PQN requires a list/matrix of spectra; use calibrate_intensity_batch")
    raise ValueError(f"Unsupported calibrate method: {method}")


def calibrate_intensity_batch(intensity_list: list[np.ndarray], method: str = "TIC") -> list[np.ndarray]:
    method = method.strip()
    if method.lower() == "tic":
        method = "TIC"
    elif method.lower() == "pqn":
        method = "PQN"
    elif method.lower() == "median":
        method = "median"
    elif method.lower() == "none":
        method = "none"
    if method != "PQN":
        return [calibrate_intensity_single(y, method=method) for y in intensity_list]

    if not intensity_list:
        return []

    # Probabilistic Quotient Normalization (MALDIquant-style batch-level concept).
    # 1) TIC-normalize each spectrum.
    # 2) Build reference as median spectrum.
    # 3) Scale each spectrum by median quotient.
    tic_norm = [calibrate_intensity_single(y, method="TIC") for y in intensity_list]
    n = min(len(y) for y in tic_norm)
    mat = np.vstack([y[:n] for y in tic_norm])
    ref = np.median(mat, axis=0)
    ref = np.where(ref <= 0, np.nan, ref)

    out: list[np.ndarray] = []
    for y in tic_norm:
        q = y[:n] / ref
        factor = float(np.nanmedian(q))
        if not np.isfinite(factor) or factor <= 0:
            factor = 1.0
        out.append(y / factor)
    return out
