from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import fnmatch
import numpy as np

from .bruker_fid import SpectrumData, read_bruker_fid
from .bruker_xml import read_bruker_xml_entries
from .mass_formats import read_mzml, read_mzmlb, read_mzxml
from .options import ImportOptions
from .peaklists import read_csv_peaklist, read_txt_peaklist
from .qc import open_qc_summary


@dataclass
class LoadedSpectrum:
    source_path: Path
    source_id: str
    file_type: str
    spectrum: SpectrumData
    metadata: dict[str, object]


def _apply_generic_filters(spec: SpectrumData, options: ImportOptions | None) -> SpectrumData:
    if options is None:
        return spec

    mz = spec.mz.astype(float, copy=True)
    intensity = spec.intensity.astype(float, copy=True)
    mask = mz == mz  # all True mask with correct shape

    lo, hi = options.mass_range()
    if lo is not None:
        mask &= mz >= float(lo)
    if hi is not None:
        mask &= mz <= float(hi)
    if options.min_intensity is not None:
        mask &= intensity >= float(options.min_intensity)

    if not np.all(mask):
        mz = mz[mask]
        intensity = intensity[mask]
    return SpectrumData(mz=mz, intensity=intensity, is_profile=spec.is_profile)


def detect_file_type(path: Path) -> str:
    p = Path(path)
    if p.is_dir():
        if (p / "fid").is_file() or p.name.lower().endswith(".fid"):
            return "bruker_fid"
        return "directory"

    suffix = p.suffix.lower()
    if suffix == ".mzml":
        return "mzml"
    if suffix == ".mzmlb":
        return "mzmlb"
    if suffix == ".mzxml":
        return "mzxml"
    if suffix == ".xml":
        return "bruker_xml"
    if suffix == ".txt":
        return "txt_peaklist"
    if suffix == ".csv":
        return "csv_peaklist"
    return "unknown"


def _find_fid_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for fid_file in root.rglob("fid"):
        if fid_file.is_file():
            out.append(fid_file.parent)
    # Preserve order but remove duplicates.
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in sorted(out):
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _allowed_by_patterns(path: Path, options: ImportOptions | None) -> bool:
    if options is None:
        return True
    rel = str(path)
    if options.include_patterns:
        if not any(fnmatch.fnmatch(rel, pat) for pat in options.include_patterns):
            return False
    if options.exclude_patterns:
        if any(fnmatch.fnmatch(rel, pat) for pat in options.exclude_patterns):
            return False
    return True


def list_supported_sources(path: Path, recursive: bool = True, options: ImportOptions | None = None) -> list[Path]:
    p = Path(path)
    if p.is_file():
        ftype = detect_file_type(p)
        return [p] if ftype != "unknown" and _allowed_by_patterns(p, options) else []

    if not p.is_dir():
        return []

    out: list[Path] = []
    if recursive:
        out.extend(_find_fid_dirs(p))
        for f in sorted(p.rglob("*")):
            if not f.is_file():
                continue
            if detect_file_type(f) in {"mzml", "mzmlb", "mzxml", "bruker_xml", "txt_peaklist", "csv_peaklist"} and _allowed_by_patterns(
                f, options
            ):
                out.append(f)
    else:
        if (p / "fid").is_file():
            out.append(p)
        for f in sorted(p.iterdir()):
            if not f.is_file():
                continue
            if detect_file_type(f) in {"mzml", "mzmlb", "mzxml", "bruker_xml", "txt_peaklist", "csv_peaklist"} and _allowed_by_patterns(
                f, options
            ):
                out.append(f)

    # De-duplicate while preserving order.
    seen: set[str] = set()
    uniq: list[Path] = []
    for item in out:
        key = str(item.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(item)
    return uniq


def read_spectra(path: Path, scan_index: int = 0, options: ImportOptions | None = None) -> list[LoadedSpectrum]:
    p = Path(path)
    ftype = detect_file_type(p)

    if ftype == "bruker_fid":
        spec = _apply_generic_filters(read_bruker_fid(p), options)
        return [
            LoadedSpectrum(
                source_path=p,
                source_id="scan:0",
                file_type="bruker_fid",
                spectrum=spec,
                metadata={"scan_index": 0, "total_scans": 1, "open_qc": open_qc_summary(spec)},
            )
        ]

    if ftype == "mzml":
        spec, selected, total = read_mzml(p, scan_index=scan_index, options=options)
        return [
            LoadedSpectrum(
                source_path=p,
                source_id=f"scan:{selected}",
                file_type="mzml",
                spectrum=spec,
                metadata={"scan_index": selected, "total_scans": total, "open_qc": open_qc_summary(spec)},
            )
        ]

    if ftype == "mzmlb":
        spec, selected, total = read_mzmlb(p, scan_index=scan_index, options=options)
        return [
            LoadedSpectrum(
                source_path=p,
                source_id=f"scan:{selected}",
                file_type="mzmlb",
                spectrum=spec,
                metadata={"scan_index": selected, "total_scans": total, "open_qc": open_qc_summary(spec)},
            )
        ]

    if ftype == "mzxml":
        spec, selected, total = read_mzxml(p, scan_index=scan_index, options=options)
        return [
            LoadedSpectrum(
                source_path=p,
                source_id=f"scan:{selected}",
                file_type="mzxml",
                spectrum=spec,
                metadata={"scan_index": selected, "total_scans": total, "open_qc": open_qc_summary(spec)},
            )
        ]

    if ftype == "bruker_xml":
        entries = read_bruker_xml_entries(p)
        out: list[LoadedSpectrum] = []
        for entry in entries:
            spec = _apply_generic_filters(entry.spectrum, options)
            out.append(
                LoadedSpectrum(
                    source_path=p,
                    source_id=entry.source_id,
                    file_type="bruker_xml",
                    spectrum=spec,
                    metadata={
                        "analyte_name": entry.analyte_name,
                        "sample_name": entry.sample_name,
                        "type_name": entry.type_name,
                        "position": entry.position,
                        "open_qc": open_qc_summary(spec),
                    },
                )
            )
        return out

    if ftype == "txt_peaklist":
        spec = _apply_generic_filters(read_txt_peaklist(p), options)
        return [
            LoadedSpectrum(
                source_path=p,
                source_id="scan:0",
                file_type="txt_peaklist",
                spectrum=spec,
                metadata={"scan_index": 0, "total_scans": 1, "open_qc": open_qc_summary(spec)},
            )
        ]

    if ftype == "csv_peaklist":
        spec = _apply_generic_filters(read_csv_peaklist(p), options)
        return [
            LoadedSpectrum(
                source_path=p,
                source_id="scan:0",
                file_type="csv_peaklist",
                spectrum=spec,
                metadata={"scan_index": 0, "total_scans": 1, "open_qc": open_qc_summary(spec)},
            )
        ]

    raise RuntimeError(f"Unsupported source: {p}")


def read_many(paths: Iterable[Path], scan_index: int = 0, options: ImportOptions | None = None) -> list[LoadedSpectrum]:
    out: list[LoadedSpectrum] = []
    for p in paths:
        out.extend(read_spectra(Path(p), scan_index=scan_index, options=options))
    return out


def load_from_path(path: Path, recursive: bool = True, scan_index: int = 0, options: ImportOptions | None = None) -> list[LoadedSpectrum]:
    p = Path(path)
    if p.is_file() or detect_file_type(p) == "bruker_fid":
        return read_spectra(p, scan_index=scan_index, options=options)

    sources = list_supported_sources(p, recursive=recursive, options=options)
    return read_many(sources, scan_index=scan_index, options=options)
