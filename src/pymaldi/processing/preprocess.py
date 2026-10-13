from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymaldi.io.bruker_fid import SpectrumData
from .smooth import smooth_intensity
from .baseline import remove_baseline
from .calibrate import calibrate_intensity_single


@dataclass
class PreprocessConfig:
    smooth_window: int = 11
    smooth_polyorder: int = 3
    baseline_window: int = 101
    normalize: str = "tic"  # tic|max|none


def smooth(intensity: np.ndarray, window: int = 11, polyorder: int = 3) -> np.ndarray:
    half = max(1, int((window - 1) // 2))
    return smooth_intensity(intensity, method="SavitzkyGolay", half_window_size=half, polyorder=polyorder)


def baseline_median(intensity: np.ndarray, window: int = 101) -> np.ndarray:
    half = max(1, int((window - 1) // 2))
    x = np.arange(intensity.size, dtype=float)
    return remove_baseline(x, intensity, method="median", half_window_size=half)


def normalize(intensity: np.ndarray, method: str = "tic") -> np.ndarray:
    m = method.lower()
    if m == "max":
        vmax = float(np.max(intensity)) if intensity.size else 0.0
        return intensity / vmax if vmax > 0 else intensity.copy()
    if m == "none":
        return intensity.copy()
    return calibrate_intensity_single(intensity, method="TIC")


def preprocess_spectrum(spec: SpectrumData, cfg: PreprocessConfig) -> SpectrumData:
    y = smooth(spec.intensity, window=cfg.smooth_window, polyorder=cfg.smooth_polyorder)
    y = baseline_median(y, window=cfg.baseline_window)
    y = normalize(y, method=cfg.normalize)
    return SpectrumData(mz=spec.mz.copy(), intensity=y, is_profile=spec.is_profile)
