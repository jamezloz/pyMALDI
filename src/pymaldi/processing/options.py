from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TransformMethod = Literal["none", "sqrt", "log", "log2", "log10"]
SmoothMethod = Literal["none", "SavitzkyGolay", "MovingAverage"]
BaselineMethod = Literal["none", "SNIP", "TopHat", "ConvexHull", "median"]
CalibrateMethod = Literal["none", "TIC", "median", "PQN"]
NoiseMethod = Literal["MAD", "SuperSmoother"]
AlignMethod = Literal["none", "lowess", "linear", "quadratic"]


@dataclass
class ProcessingOptions:
    transform_method: TransformMethod = "none"
    smooth_method: SmoothMethod = "SavitzkyGolay"
    smooth_half_window_size: int = 10
    smooth_polyorder: int = 3

    baseline_method: BaselineMethod = "SNIP"
    baseline_half_window_size: int = 20
    baseline_iterations: int = 40

    calibrate_method: CalibrateMethod = "TIC"

    detect_half_window_size: int = 20
    detect_noise_method: NoiseMethod = "MAD"
    detect_snr: float = 2.0

    align_method: AlignMethod = "none"
    align_tolerance: float = 0.002
    align_warping_method: str = "lowess"
    align_min_frequency: float = 0.9
