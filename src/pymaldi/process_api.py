from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pymaldi.io import ImportOptions
from pymaldi.io.universal import LoadedSpectrum, load_from_path
from pymaldi.presets import build_processing_options_for_app_raw
from pymaldi.processing import ProcessingOptions, process_batch, process_spectrum


PROCESSING_API_VERSION = "0.3.0"


@dataclass(frozen=True)
class ProcessingConfig:
    recursive: bool = False
    scan_index: int = 0
    process_peaklists: bool = False
    batch_align_raw: bool = False


@dataclass
class ProcessedRecord:
    source_path: str
    source_id: str
    file_type: str
    metadata: dict[str, object]
    mz: np.ndarray
    intensity: np.ndarray
    is_profile: bool
    peak_mz: np.ndarray
    peak_intensity: np.ndarray


@dataclass
class ProcessingResult:
    api_version: str
    input_path: str
    records: list[ProcessedRecord]


def default_options_for_file_type(file_type: str) -> ProcessingOptions:
    ft = file_type.lower()
    if ft == "bruker_fid":
        opts = build_processing_options_for_app_raw("raw_bruker")
        opts.baseline_iterations = 80
        return opts
    if ft == "mzxml":
        opts = build_processing_options_for_app_raw("mzxml")
        opts.baseline_iterations = 60
        return opts
    if ft == "mzml":
        opts = build_processing_options_for_app_raw("raw_vitek")
        opts.baseline_iterations = 60
        return opts

    # Peaklist-safe default: no smoothing/baseline, only peak detection pass-through.
    return ProcessingOptions(
        transform_method="none",
        smooth_method="none",
        baseline_method="none",
        calibrate_method="none",
        detect_noise_method="MAD",
        detect_half_window_size=20,
        detect_snr=2.0,
        align_method="none",
    )


def _is_raw_like(item: LoadedSpectrum) -> bool:
    return item.file_type in {"bruker_fid", "mzxml"} or (item.file_type == "mzml" and bool(item.spectrum.is_profile))


def _to_record(item: LoadedSpectrum, out) -> ProcessedRecord:
    peaks = out.peaks
    peak_mz = np.asarray(peaks.mz, dtype=float) if peaks is not None else np.array([], dtype=float)
    peak_int = np.asarray(peaks.intensity, dtype=float) if peaks is not None else np.array([], dtype=float)
    return ProcessedRecord(
        source_path=str(item.source_path),
        source_id=item.source_id,
        file_type=item.file_type,
        metadata=dict(item.metadata),
        mz=np.asarray(out.spectrum.mz, dtype=float),
        intensity=np.asarray(out.spectrum.intensity, dtype=float),
        is_profile=bool(getattr(out.spectrum, "is_profile", item.spectrum.is_profile)),
        peak_mz=peak_mz,
        peak_intensity=peak_int,
    )


def run_processing_records(
    path: str | Path,
    config: ProcessingConfig | None = None,
    import_options: ImportOptions | None = None,
    options_override: ProcessingOptions | None = None,
) -> ProcessingResult:
    cfg = config or ProcessingConfig()
    imp = import_options or ImportOptions(prefer_indexed=True)

    loaded = load_from_path(Path(path), recursive=cfg.recursive, scan_index=cfg.scan_index, options=imp)
    records: list[ProcessedRecord] = []

    if cfg.batch_align_raw:
        raw_items = [x for x in loaded if _is_raw_like(x)]
        raw_ids = {id(x) for x in raw_items}
        non_raw = [x for x in loaded if id(x) not in raw_ids]

        if raw_items:
            opts = options_override or default_options_for_file_type(raw_items[0].file_type)
            opts.align_method = "linear"
            opts.align_tolerance = 0.002
            opts.align_min_frequency = 0.7
            batch = process_batch([x.spectrum for x in raw_items], opts)
            for item, out in zip(raw_items, batch):
                records.append(_to_record(item, out))

        for item in non_raw:
            if not cfg.process_peaklists and not _is_raw_like(item):
                out = process_spectrum(item.spectrum, default_options_for_file_type("peaklist"))
            else:
                opts = options_override or default_options_for_file_type(item.file_type)
                out = process_spectrum(item.spectrum, opts)
            records.append(_to_record(item, out))
    else:
        for item in loaded:
            if not cfg.process_peaklists and not _is_raw_like(item):
                opts = default_options_for_file_type("peaklist")
            else:
                opts = options_override or default_options_for_file_type(item.file_type)
            out = process_spectrum(item.spectrum, opts)
            records.append(_to_record(item, out))

    return ProcessingResult(
        api_version=PROCESSING_API_VERSION,
        input_path=str(Path(path)),
        records=records,
    )
