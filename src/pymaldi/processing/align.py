from __future__ import annotations

import numpy as np


def _within_tolerance(mass: float, ref_mass: float, tolerance: float) -> bool:
    if tolerance < 1.0:
        return abs(mass - ref_mass) / max(ref_mass, 1e-12) <= tolerance
    return abs(mass - ref_mass) <= tolerance


def _match_peaks_monotonic(ref_mz: np.ndarray, mz: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    if ref_mz.size == 0 or mz.size == 0:
        return np.array([]), np.array([])
    src: list[float] = []
    dst: list[float] = []
    j = 0
    for m in mz:
        best_j = -1
        best_d = np.inf
        k = j
        while k < ref_mz.size:
            r = float(ref_mz[k])
            d = abs(float(m) - r)
            if _within_tolerance(float(m), r, tolerance):
                if d < best_d:
                    best_d = d
                    best_j = k
            elif r > m and best_j >= 0:
                break
            k += 1
        if best_j >= 0:
            src.append(float(m))
            dst.append(float(ref_mz[best_j]))
            j = best_j + 1
            if j >= ref_mz.size:
                break
    return np.array(src, dtype=float), np.array(dst, dtype=float)


def build_reference_peaks(
    peak_mz_list: list[np.ndarray],
    tolerance: float = 0.002,
    min_frequency: float = 0.9,
) -> np.ndarray:
    n_spec = len(peak_mz_list)
    if n_spec == 0:
        return np.array([])
    needed = max(1, int(np.ceil(float(min_frequency) * n_spec)))

    all_peaks: list[tuple[float, int]] = []
    for sid, mz in enumerate(peak_mz_list):
        for m in np.asarray(mz, dtype=float):
            all_peaks.append((float(m), sid))
    if not all_peaks:
        return np.array([])
    all_peaks.sort(key=lambda x: x[0])

    clusters: list[dict[str, object]] = []
    for m, sid in all_peaks:
        placed = False
        for c in clusters:
            center = float(c["center"])
            if _within_tolerance(m, center, tolerance):
                members = c["members"]
                assert isinstance(members, list)
                members.append((m, sid))
                c["center"] = float(np.mean([mm for mm, _ in members]))
                placed = True
                break
        if not placed:
            clusters.append({"center": m, "members": [(m, sid)]})

    ref: list[float] = []
    for c in clusters:
        members = c["members"]
        assert isinstance(members, list)
        unique_samples = {sid for _, sid in members}
        if len(unique_samples) >= needed:
            # Reference peak center from one highest-frequency cluster.
            ref.append(float(np.median([m for m, _ in members])))
    if not ref:
        # fallback to median of all peaks by position-less approach
        merged = np.sort(np.concatenate([np.asarray(x, dtype=float) for x in peak_mz_list if len(x)]))
        return merged
    return np.array(sorted(ref), dtype=float)


def align_mz_axis(
    mz: np.ndarray,
    peak_mz: np.ndarray,
    ref_peak_mz: np.ndarray,
    tolerance: float = 0.002,
    method: str = "lowess",
    intensity: np.ndarray | None = None,
    ref_intensity: np.ndarray | None = None,
) -> np.ndarray:
    if method == "none":
        return mz.copy()

    peak_sorted = np.sort(np.asarray(peak_mz, dtype=float))
    src, dst = _match_peaks_monotonic(ref_peak_mz, peak_sorted, tolerance=tolerance)
    if src.size < 3:
        # Retry with relaxed tolerance when strict matching fails.
        relaxed_tol = max(float(tolerance), 0.002)
        src, dst = _match_peaks_monotonic(ref_peak_mz, peak_sorted, tolerance=relaxed_tol)
    if src.size < 3:
        # Fallback 0: quantile-based monotonic mapping between sample/reference peaks.
        if ref_peak_mz.size >= 10 and peak_sorted.size >= 10:
            k = min(ref_peak_mz.size, peak_sorted.size)
            n_anchor = min(25, k)
            q = np.linspace(0.0, 1.0, n_anchor)
            ref_anchor = np.quantile(np.sort(ref_peak_mz), q)
            src_anchor = np.quantile(peak_sorted, q)
            # Build delta over anchors and warp all mass values.
            delta_anchor = ref_anchor - src_anchor
            delta = np.interp(mz, src_anchor, delta_anchor, left=delta_anchor[0], right=delta_anchor[-1])
            return mz + delta

        # Fallback: robust global shift from nearest reference peaks.
        if ref_peak_mz.size and peak_mz.size:
            deltas = []
            rel_tol = max(float(tolerance) * 5.0, 0.005)
            for m in np.asarray(peak_mz, dtype=float):
                j = int(np.argmin(np.abs(ref_peak_mz - m)))
                r = float(ref_peak_mz[j])
                rel = abs(m - r) / max(r, 1e-12)
                if rel <= rel_tol:
                    deltas.append(r - float(m))
            if len(deltas) >= 3:
                shift = float(np.median(np.asarray(deltas, dtype=float)))
                return mz + shift
        # Secondary fallback: cross-correlation shift on intensities.
        if intensity is not None and ref_intensity is not None:
            y = np.asarray(intensity, dtype=float)
            r = np.asarray(ref_intensity, dtype=float)
            n = min(y.size, r.size)
            if n >= 16:
                y1 = y[:n] - np.mean(y[:n])
                r1 = r[:n] - np.mean(r[:n])
                corr = np.correlate(y1, r1, mode="full")
                shift_idx = int(np.argmax(corr) - (n - 1))
                if mz.size >= 2:
                    step = float(np.median(np.diff(mz)))
                    return mz + shift_idx * step
        return mz.copy()

    m = method.lower()
    if m == "linear":
        coef = np.polyfit(src, dst, 1)
        return np.polyval(coef, mz)
    if m == "quadratic":
        coef = np.polyfit(src, dst, 2)
        return np.polyval(coef, mz)

    # lowess-like fallback: smooth delta (d) then interpolate.
    order = np.argsort(src)
    src = src[order]
    d = (dst - src)[order]
    src_u, idx = np.unique(src, return_index=True)
    d_u = d[idx]
    if src_u.size < 2:
        return mz.copy()
    # Robust fallback for unstable warping pairs: apply only global shift.
    d_med = float(np.median(d_u))
    d_mad = float(np.median(np.abs(d_u - d_med)))
    if src_u.size < 6 or d_mad > 0.75:
        return mz + d_med
    # Moving-average smoothing over matched deltas.
    win = max(3, int(src_u.size // 10))
    if win % 2 == 0:
        win += 1
    pad = win // 2
    d_pad = np.pad(d_u, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=float) / win
    d_smooth = np.convolve(d_pad, kernel, mode="valid")
    delta = np.interp(mz, src_u, d_smooth, left=d_smooth[0], right=d_smooth[-1])
    return mz + delta
