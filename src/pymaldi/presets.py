from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pymaldi.processing import ProcessingOptions

RawPresetKey = Literal["raw_bruker", "raw_vitek", "mzxml"]


@dataclass(frozen=True)
class RawPreprocessPreset:
    do_smooth: bool
    window: int
    poly: int
    baseline: str


RAW_PREPROCESS_PRESETS: dict[str, RawPreprocessPreset] = {
    "raw_bruker": RawPreprocessPreset(do_smooth=True, window=9, poly=2, baseline="SNIP"),
    "raw_vitek": RawPreprocessPreset(do_smooth=True, window=11, poly=3, baseline="TopHat"),
    "mzxml": RawPreprocessPreset(do_smooth=True, window=11, poly=3, baseline="TopHat"),
}


def get_preprocess_ui_preset(key: RawPresetKey) -> dict[str, object]:
    p = RAW_PREPROCESS_PRESETS[key]
    return {
        "do_smooth": bool(p.do_smooth),
        "window": int(p.window),
        "poly": int(p.poly),
        "baseline": str(p.baseline),
    }


def _baseline_name(name: str) -> str:
    n = (name or "none").strip().lower()
    if n == "snip":
        return "SNIP"
    if n == "tophat":
        return "TopHat"
    if n == "convexhull":
        return "ConvexHull"
    if n == "median":
        return "median"
    return "none"


def build_processing_options_for_app_raw(
    key: RawPresetKey,
    do_smooth: bool | None = None,
    window: int | None = None,
    poly: int | None = None,
    baseline: str | None = None,
) -> ProcessingOptions:
    base = RAW_PREPROCESS_PRESETS[key]

    do_s = base.do_smooth if do_smooth is None else bool(do_smooth)
    win = int(base.window if window is None else window)
    pol = int(base.poly if poly is None else poly)
    bl = str(base.baseline if baseline is None else baseline)

    half_window = max(1, int((max(3, win) - 1) // 2))

    return ProcessingOptions(
        transform_method="none",
        smooth_method="SavitzkyGolay" if do_s else "none",
        smooth_half_window_size=half_window,
        smooth_polyorder=max(1, pol),
        baseline_method=_baseline_name(bl),
        baseline_half_window_size=20,
        baseline_iterations=40,
        calibrate_method="none",
        detect_noise_method="MAD",
        detect_half_window_size=20,
        detect_snr=2.0,
        align_method="none",
    )
