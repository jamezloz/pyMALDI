from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pymaldi.io.bruker_fid import SpectrumData
from pymaldi.io.universal import LoadedSpectrum, load_from_path
from pymaldi.matrix import PeakMatrix, build_peak_matrix, build_profile_matrix, intensity_to_binary
from pymaldi.processing import ProcessedSpectrum, ProcessingOptions, process_batch, process_spectrum


@dataclass
class DatasetResult:
    loaded: list[LoadedSpectrum]
    processed: list[ProcessedSpectrum]
    intensity_matrix: np.ndarray
    feature_mz: np.ndarray
    binary_matrix: np.ndarray


def preprocess_loaded(loaded: list[LoadedSpectrum], options: ProcessingOptions) -> list[ProcessedSpectrum]:
    specs: list[SpectrumData] = [x.spectrum for x in loaded]
    if not specs:
        return []

    # Batch mode preserves MALDIquant-like behavior for calibration and optional alignment.
    if len(specs) > 1:
        return process_batch(specs, options)
    return [process_spectrum(specs[0], options)]


def run_dataset_pipeline(
    input_path: str | Path,
    options: ProcessingOptions | None = None,
    recursive: bool = True,
    scan_index: int = 0,
    matrix_mode: str = "peaks",
    normalize: str = "none",
    tolerance: float = 0.002,
    min_frequency: float = 0.5,
    binary_threshold: float = 0.0,
    binary_relative: bool = False,
) -> DatasetResult:
    loaded = load_from_path(Path(input_path), recursive=recursive, scan_index=scan_index)
    opts = options or ProcessingOptions()
    processed = preprocess_loaded(loaded, opts)

    pm: PeakMatrix
    mode = matrix_mode.lower()
    if mode == "profile":
        pm = build_profile_matrix(processed, normalize=normalize)
    else:
        pm = build_peak_matrix(processed, tolerance=tolerance, min_frequency=min_frequency, normalize=normalize)

    binary = intensity_to_binary(pm.matrix, threshold=binary_threshold, relative=binary_relative)

    return DatasetResult(
        loaded=loaded,
        processed=processed,
        intensity_matrix=pm.matrix,
        feature_mz=pm.feature_mz,
        binary_matrix=binary,
    )
