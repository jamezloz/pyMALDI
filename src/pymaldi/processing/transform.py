from __future__ import annotations

import numpy as np


def transform_intensity(intensity: np.ndarray, method: str = "none") -> np.ndarray:
    m = method.lower()
    y = np.maximum(intensity.astype(float), 0.0)
    if m == "none":
        return y
    if m == "sqrt":
        return np.sqrt(y)

    eps = 1e-12
    if m == "log":
        return np.log(y + eps)
    if m == "log2":
        return np.log2(y + eps)
    if m == "log10":
        return np.log10(y + eps)
    raise ValueError(f"Unsupported transform method: {method}")
