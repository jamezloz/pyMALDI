from __future__ import annotations

import numpy as np

from .bruker_fid import SpectrumData


def open_qc_summary(spec: SpectrumData) -> dict[str, object]:
    mz = np.asarray(spec.mz, dtype=float)
    y = np.asarray(spec.intensity, dtype=float)

    issues: list[str] = []
    if mz.size == 0 or y.size == 0:
        issues.append("empty_spectrum")
    if mz.size != y.size:
        issues.append("shape_mismatch")
    if mz.size and not np.all(np.isfinite(mz)):
        issues.append("mz_non_finite")
    if y.size and not np.all(np.isfinite(y)):
        issues.append("intensity_non_finite")
    if y.size and np.min(y) < 0:
        issues.append("negative_intensity")
    if mz.size >= 2 and np.any(np.diff(mz) < 0):
        issues.append("non_monotonic_mz")

    return {
        "n_points": int(min(mz.size, y.size)),
        "mz_min": float(np.min(mz)) if mz.size else None,
        "mz_max": float(np.max(mz)) if mz.size else None,
        "intensity_min": float(np.min(y)) if y.size else None,
        "intensity_max": float(np.max(y)) if y.size else None,
        "is_profile": bool(spec.is_profile),
        "issues": issues,
        "ok": len(issues) == 0,
    }

