from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d


def smooth_intensity(
    intensity: np.ndarray,
    method: str = "SavitzkyGolay",
    half_window_size: int = 10,
    polyorder: int = 3,
) -> np.ndarray:
    ml = method.lower()
    if ml == "none":
        m = "none"
    elif ml == "movingaverage":
        m = "MovingAverage"
    elif ml == "savitzkygolay":
        m = "SavitzkyGolay"
    else:
        m = method
    y = intensity.astype(float)
    if m == "none":
        return y

    window = int(max(1, half_window_size) * 2 + 1)
    if y.size < window:
        return y

    if m == "MovingAverage":
        return uniform_filter1d(y, size=window, mode="nearest")

    if m == "SavitzkyGolay":
        p = int(max(1, polyorder))
        if p >= window:
            p = window - 1
        if p < 1:
            return y
        return savgol_filter(y, window_length=window, polyorder=p)

    raise ValueError(f"Unsupported smooth method: {method}")
