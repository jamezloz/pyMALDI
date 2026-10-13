"""Processing algorithms and MALDIquant-like configurable pipeline."""

from .options import ProcessingOptions
from .pipeline import ProcessedSpectrum, process_spectrum, process_batch
from .peaks import PeakList, detect_peaks, peak_f1
from .align import align_mz_axis

__all__ = [
    "ProcessingOptions",
    "ProcessedSpectrum",
    "process_spectrum",
    "process_batch",
    "PeakList",
    "detect_peaks",
    "peak_f1",
    "align_mz_axis",
]
