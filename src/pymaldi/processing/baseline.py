from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter, grey_opening
from scipy.spatial import ConvexHull


def _baseline_snip(y: np.ndarray, iterations: int = 40, decreasing: bool = True) -> np.ndarray:
    # SNIP baseline (Ryan et al., 1988; Morhac et al., 1997) applied directly to the intensities, as in MALDIquant.
    b = np.maximum(y.astype(float), 0.0).copy()
    n = b.size
    if n == 0:
        return b

    it = int(max(1, iterations))
    k_values = range(it, 0, -1) if decreasing else range(1, it + 1)
    for k in k_values:
        if 2 * k >= n:
            continue
        left = b[:-2 * k]
        right = b[2 * k :]
        avg = 0.5 * (left + right)
        core = b[k : n - k]
        b[k : n - k] = np.minimum(core, avg)
    return b


def _baseline_tophat(y: np.ndarray, half_window_size: int = 20) -> np.ndarray:
    size = int(max(1, half_window_size) * 2 + 1)
    opened = grey_opening(y, size=size)
    return np.minimum(opened, y)


def _baseline_convex_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if x.size < 3:
        return np.minimum(y, np.median(y) if y.size else 0.0)
    pts = np.column_stack([x, y])
    hull = ConvexHull(pts)
    hpts = pts[hull.vertices]
    hpts = hpts[np.argsort(hpts[:, 0])]
    lower = [hpts[0]]
    for p in hpts[1:]:
        while len(lower) >= 2:
            x1, y1 = lower[-2]
            x2, y2 = lower[-1]
            x3, y3 = p
            cross = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)
    low = np.array(lower)
    return np.interp(x, low[:, 0], low[:, 1])


def estimate_baseline(
    mz: np.ndarray,
    intensity: np.ndarray,
    method: str = "SNIP",
    half_window_size: int = 20,
    iterations: int = 40,
) -> np.ndarray:
    ml = method.lower()
    if ml == "none":
        method = "none"
    elif ml == "median":
        method = "median"
    elif ml == "snip":
        method = "SNIP"
    elif ml == "tophat":
        method = "TopHat"
    elif ml == "convexhull":
        method = "ConvexHull"
    y = intensity.astype(float)
    if method == "none":
        return np.zeros_like(y)
    if method == "median":
        size = int(max(1, half_window_size) * 2 + 1)
        return median_filter(y, size=size, mode="nearest")
    if method == "SNIP":
        return _baseline_snip(y, iterations=iterations, decreasing=True)
    if method == "TopHat":
        return _baseline_tophat(y, half_window_size=half_window_size)
    if method == "ConvexHull":
        return _baseline_convex_hull(mz, y)
    raise ValueError(f"Unsupported baseline method: {method}")


def remove_baseline(
    mz: np.ndarray,
    intensity: np.ndarray,
    method: str = "SNIP",
    half_window_size: int = 20,
    iterations: int = 40,
) -> np.ndarray:
    b = estimate_baseline(
        mz,
        intensity,
        method=method,
        half_window_size=half_window_size,
        iterations=iterations,
    )
    return np.maximum(intensity.astype(float) - b, 0.0)
