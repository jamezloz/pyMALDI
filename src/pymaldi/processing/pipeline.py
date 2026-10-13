from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pymaldi.io.bruker_fid import SpectrumData
from .options import ProcessingOptions
from .transform import transform_intensity
from .smooth import smooth_intensity
from .baseline import remove_baseline
from .calibrate import calibrate_intensity_single, calibrate_intensity_batch
from .peaks import PeakList, detect_peaks
from .align import align_mz_axis, build_reference_peaks


@dataclass
class ProcessedSpectrum:
    spectrum: SpectrumData
    peaks: PeakList | None = None


def process_spectrum(spec: SpectrumData, options: ProcessingOptions) -> ProcessedSpectrum:
    y = spec.intensity.astype(float)
    y = transform_intensity(y, method=options.transform_method)
    y = smooth_intensity(
        y,
        method=options.smooth_method,
        half_window_size=options.smooth_half_window_size,
        polyorder=options.smooth_polyorder,
    )
    y = remove_baseline(
        spec.mz,
        y,
        method=options.baseline_method,
        half_window_size=options.baseline_half_window_size,
        iterations=options.baseline_iterations,
    )
    y = calibrate_intensity_single(y, method=options.calibrate_method)

    out_spec = SpectrumData(mz=spec.mz.copy(), intensity=y, is_profile=spec.is_profile)
    p = detect_peaks(
        out_spec.mz,
        out_spec.intensity,
        half_window_size=options.detect_half_window_size,
        method=options.detect_noise_method,
        snr=options.detect_snr,
    )
    return ProcessedSpectrum(spectrum=out_spec, peaks=p)


def process_batch(specs: list[SpectrumData], options: ProcessingOptions) -> list[ProcessedSpectrum]:
    transformed: list[SpectrumData] = []
    for spec in specs:
        y = transform_intensity(spec.intensity, method=options.transform_method)
        y = smooth_intensity(
            y,
            method=options.smooth_method,
            half_window_size=options.smooth_half_window_size,
            polyorder=options.smooth_polyorder,
        )
        y = remove_baseline(
            spec.mz,
            y,
            method=options.baseline_method,
            half_window_size=options.baseline_half_window_size,
            iterations=options.baseline_iterations,
        )
        transformed.append(SpectrumData(mz=spec.mz.copy(), intensity=y, is_profile=spec.is_profile))

    calibrated = calibrate_intensity_batch([s.intensity for s in transformed], method=options.calibrate_method)

    out: list[ProcessedSpectrum] = []
    prealigned: list[ProcessedSpectrum] = []
    for spec, y in zip(transformed, calibrated):
        out_spec = SpectrumData(mz=spec.mz, intensity=np.array(y, dtype=float), is_profile=spec.is_profile)
        p = detect_peaks(
            out_spec.mz,
            out_spec.intensity,
            half_window_size=options.detect_half_window_size,
            method=options.detect_noise_method,
            snr=options.detect_snr,
        )
        prealigned.append(ProcessedSpectrum(spectrum=out_spec, peaks=p))

    if options.align_method == "none" or len(prealigned) < 2:
        return prealigned

    peak_list = [it.peaks.mz if it.peaks is not None else np.array([]) for it in prealigned]
    ref = build_reference_peaks(
        peak_list,
        tolerance=options.align_tolerance,
        min_frequency=options.align_min_frequency,
    )
    ref_intensity = prealigned[0].spectrum.intensity
    for item in prealigned:
        peak_mz = item.peaks.mz if item.peaks is not None else np.array([])
        mz_new = align_mz_axis(
            item.spectrum.mz,
            peak_mz=peak_mz,
            ref_peak_mz=ref,
            tolerance=options.align_tolerance,
            method=options.align_method,
            intensity=item.spectrum.intensity,
            ref_intensity=ref_intensity,
        )
        spec_aligned = SpectrumData(mz=mz_new, intensity=item.spectrum.intensity.copy(), is_profile=item.spectrum.is_profile)
        p_aligned = detect_peaks(
            spec_aligned.mz,
            spec_aligned.intensity,
            half_window_size=options.detect_half_window_size,
            method=options.detect_noise_method,
            snr=options.detect_snr,
        )
        out.append(ProcessedSpectrum(spectrum=spec_aligned, peaks=p_aligned))
    return out
