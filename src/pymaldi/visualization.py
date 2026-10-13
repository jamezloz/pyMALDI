from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pymaldi.io.options import ImportOptions
from pymaldi.io.universal import load_from_path
from pymaldi.io.universal import LoadedSpectrum


@dataclass
class PlotPayload:
    mz: np.ndarray
    intensity: np.ndarray
    is_profile: bool
    file_type: str
    source_id: str
    source_path: str
    metadata: dict[str, object]


def _normalize(y: np.ndarray, method: str) -> np.ndarray:
    m = method.lower()
    arr = np.asarray(y, dtype=float).copy()
    if m == "none":
        return arr
    if m == "tic":
        d = float(np.sum(arr))
    elif m == "max":
        d = float(np.max(arr)) if arr.size else 0.0
    else:
        raise ValueError(f"Unknown normalize method: {method}")
    if d > 0:
        arr /= d
    return arr


def downsample_peak_preserving(mz: np.ndarray, intensity: np.ndarray, target_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(mz, dtype=float)
    y = np.asarray(intensity, dtype=float)
    n = x.size
    if n <= max(10, int(target_points)):
        return x, y

    bins = max(2, int(target_points // 2))
    edges = np.linspace(0, n, bins + 1, dtype=int)
    idx: list[int] = []
    for i in range(bins):
        lo = int(edges[i])
        hi = int(edges[i + 1])
        if hi <= lo:
            continue
        yy = y[lo:hi]
        i_min = int(np.argmin(yy)) + lo
        i_max = int(np.argmax(yy)) + lo
        if i_min == i_max:
            idx.append(i_min)
        else:
            idx.extend(sorted((i_min, i_max)))
    idx = sorted(set(idx))
    return x[idx], y[idx]


def build_plot_payload(
    loaded: LoadedSpectrum,
    normalize: str = "none",
    clip_mz_min: float | None = None,
    clip_mz_max: float | None = None,
    downsample_to: int | None = None,
) -> PlotPayload:
    x = np.asarray(loaded.spectrum.mz, dtype=float)
    y = np.asarray(loaded.spectrum.intensity, dtype=float)

    if clip_mz_min is not None or clip_mz_max is not None:
        mask = np.ones(x.size, dtype=bool)
        if clip_mz_min is not None:
            mask &= x >= float(clip_mz_min)
        if clip_mz_max is not None:
            mask &= x <= float(clip_mz_max)
        x = x[mask]
        y = y[mask]

    y = _normalize(y, normalize)

    if downsample_to is not None and int(downsample_to) > 0:
        x, y = downsample_peak_preserving(x, y, target_points=int(downsample_to))

    return PlotPayload(
        mz=x,
        intensity=y,
        is_profile=bool(loaded.spectrum.is_profile),
        file_type=loaded.file_type,
        source_id=loaded.source_id,
        source_path=str(loaded.source_path),
        metadata=dict(loaded.metadata),
    )


def build_multiresolution_payloads(
    loaded: LoadedSpectrum, levels: tuple[int, ...] = (20000, 10000, 5000, 2500), normalize: str = "none"
) -> dict[int, PlotPayload]:
    out: dict[int, PlotPayload] = {}
    for level in levels:
        out[int(level)] = build_plot_payload(loaded, normalize=normalize, downsample_to=int(level))
    return out


def open_for_visualization(path: str | Path, recursive: bool = False, scan_index: int = 0) -> list[PlotPayload]:
    """
    Open spectra in strict visual-fidelity mode.
    - No import filtering
    - No normalization
    - No clipping
    - No downsampling
    """
    loaded = load_from_path(
        Path(path),
        recursive=recursive,
        scan_index=scan_index,
        options=ImportOptions(
            mass_min=None,
            mass_max=None,
            min_intensity=None,
            centroided_only=False,
            prefer_indexed=True,
            include_patterns=None,
            exclude_patterns=None,
        ),
    )
    return [build_plot_payload(item, normalize="none", downsample_to=None) for item in loaded]
