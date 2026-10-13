from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from pymaldi.io.bruker_fid import SpectrumData
from pymaldi.processing.pipeline import ProcessedSpectrum


def _within_tolerance(mass: float, ref_mass: float, tolerance: float) -> bool:
    if tolerance < 1.0:
        return abs(mass - ref_mass) / max(ref_mass, 1e-12) <= tolerance
    return abs(mass - ref_mass) <= tolerance


def _merge_reference_mz(all_peak_mz: list[np.ndarray], tolerance: float, min_frequency: float) -> np.ndarray:
    n_spec = len(all_peak_mz)
    if n_spec == 0:
        return np.array([], dtype=float)

    needed = max(1, int(np.ceil(float(min_frequency) * n_spec)))
    merged: list[dict[str, object]] = []

    peaks = []
    for sid, arr in enumerate(all_peak_mz):
        for m in np.asarray(arr, dtype=float):
            peaks.append((float(m), sid))
    peaks.sort(key=lambda x: x[0])

    for mass, sid in peaks:
        placed = False
        for cl in merged:
            center = float(cl["center"])
            if _within_tolerance(mass, center, tolerance):
                vals = cl["values"]
                assert isinstance(vals, list)
                vals.append((mass, sid))
                cl["center"] = float(np.median([v for v, _ in vals]))
                placed = True
                break
        if not placed:
            merged.append({"center": mass, "values": [(mass, sid)]})

    ref: list[float] = []
    for cl in merged:
        vals = cl["values"]
        assert isinstance(vals, list)
        if len({sid for _, sid in vals}) >= needed:
            ref.append(float(np.median([v for v, _ in vals])))

    if not ref:
        all_vals = np.concatenate([np.asarray(p, dtype=float) for p in all_peak_mz if np.asarray(p).size > 0])
        return np.sort(all_vals.astype(float)) if all_vals.size else np.array([], dtype=float)

    return np.array(sorted(ref), dtype=float)


def _simple_peak_pick(spec: SpectrumData, max_peaks: int = 300) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(spec.mz, dtype=float)
    y = np.asarray(spec.intensity, dtype=float)
    if x.size < 3 or y.size < 3:
        return x.copy(), y.copy()

    mask = (y[1:-1] >= y[:-2]) & (y[1:-1] > y[2:])
    idx = np.where(mask)[0] + 1
    if idx.size == 0:
        idx = np.argsort(y)[-min(max_peaks, y.size) :]

    if idx.size > max_peaks:
        keep = np.argsort(y[idx])[-max_peaks:]
        idx = idx[keep]

    order = np.argsort(x[idx])
    idx = idx[order]
    return x[idx], y[idx]


def _extract_peaklist(item: ProcessedSpectrum | SpectrumData) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(item, ProcessedSpectrum):
        if item.peaks is not None and item.peaks.mz.size > 0:
            return np.asarray(item.peaks.mz, dtype=float), np.asarray(item.peaks.intensity, dtype=float)
        return _simple_peak_pick(item.spectrum)
    return _simple_peak_pick(item)


def _normalize_rows(matrix: np.ndarray, method: str) -> np.ndarray:
    m = np.asarray(matrix, dtype=float).copy()
    norm = str(method).lower()
    if norm == "none":
        return m

    for i in range(m.shape[0]):
        row = m[i]
        if norm == "tic":
            d = float(np.sum(row))
        elif norm == "max":
            d = float(np.max(row))
        elif norm == "median":
            nz = row[row > 0.0]
            d = float(np.median(nz)) if nz.size else 0.0
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        if d > 0.0:
            m[i] = row / d
    return m


def intensity_to_binary(matrix: np.ndarray, threshold: float = 0.0, relative: bool = False) -> np.ndarray:
    x = np.asarray(matrix, dtype=float)
    if not relative:
        return (x > float(threshold)).astype(np.uint8)

    out = np.zeros_like(x, dtype=np.uint8)
    for i in range(x.shape[0]):
        row = x[i]
        rmax = float(np.max(row))
        th = float(threshold) * rmax
        out[i] = (row > th).astype(np.uint8)
    return out


@dataclass
class PeakMatrix:
    matrix: np.ndarray
    feature_mz: np.ndarray


def build_peak_matrix(
    spectra: Iterable[ProcessedSpectrum | SpectrumData],
    tolerance: float = 0.002,
    min_frequency: float = 0.5,
    normalize: str = "none",
) -> PeakMatrix:
    items = list(spectra)
    if not items:
        return PeakMatrix(matrix=np.zeros((0, 0), dtype=float), feature_mz=np.array([], dtype=float))

    peak_lists = [_extract_peaklist(it) for it in items]
    ref = _merge_reference_mz([mz for mz, _ in peak_lists], tolerance=tolerance, min_frequency=min_frequency)
    mat = np.zeros((len(items), ref.size), dtype=float)

    for i, (mz_vals, int_vals) in enumerate(peak_lists):
        for m, inten in zip(np.asarray(mz_vals, dtype=float), np.asarray(int_vals, dtype=float)):
            if ref.size == 0:
                continue
            j = int(np.argmin(np.abs(ref - m)))
            if _within_tolerance(float(m), float(ref[j]), tolerance):
                if float(inten) > mat[i, j]:
                    mat[i, j] = float(inten)

    mat = _normalize_rows(mat, normalize)
    return PeakMatrix(matrix=mat, feature_mz=ref)


def build_profile_matrix(
    spectra: Iterable[SpectrumData | ProcessedSpectrum],
    grid_size: int | None = None,
    normalize: str = "none",
) -> PeakMatrix:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for item in spectra:
        spec = item.spectrum if isinstance(item, ProcessedSpectrum) else item
        x = np.asarray(spec.mz, dtype=float)
        y = np.asarray(spec.intensity, dtype=float)
        if x.size == 0 or y.size == 0:
            continue
        xs.append(x)
        ys.append(y)

    if not xs:
        return PeakMatrix(matrix=np.zeros((0, 0), dtype=float), feature_mz=np.array([], dtype=float))

    low = max(float(np.min(x)) for x in xs)
    high = min(float(np.max(x)) for x in xs)
    if high <= low:
        low = min(float(np.min(x)) for x in xs)
        high = max(float(np.max(x)) for x in xs)

    if grid_size is None:
        grid_size = int(np.median([x.size for x in xs]))
    grid_size = max(64, int(grid_size))

    grid = np.linspace(low, high, grid_size, dtype=float)
    mat = np.zeros((len(xs), grid.size), dtype=float)

    for i, (x, y) in enumerate(zip(xs, ys)):
        mat[i] = np.interp(grid, x, y, left=0.0, right=0.0)

    mat = _normalize_rows(mat, normalize)
    return PeakMatrix(matrix=mat, feature_mz=grid)
