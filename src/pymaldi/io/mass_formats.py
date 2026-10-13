from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np

from .bruker_fid import SpectrumData
from .options import ImportOptions

try:  # pragma: no cover
    from pyteomics import mzml, mzxml
except Exception:  # pragma: no cover
    mzml = None
    mzxml = None
try:  # pragma: no cover
    from pyteomics import mzmlb
except Exception:  # pragma: no cover
    mzmlb = None
try:  # pragma: no cover
    from pyteomics.auxiliary.file_helpers import HierarchicalOffsetIndex
except Exception:  # pragma: no cover
    HierarchicalOffsetIndex = None


def _source_signature(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _default_mzml_cache_path(path: Path, cache_dir: Path | None = None) -> Path:
    if cache_dir is None:
        return path.with_suffix(path.suffix + ".pymaldi_idx.json")
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{path.stem}.{digest}.pymaldi_idx.json"


def _default_mzml_cache_dir() -> Path:
    override = os.environ.get("PYMALDI_MZML_INDEX_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "pymaldi" / "mzml_index"
    return Path.home() / ".cache" / "pymaldi" / "mzml_index"


def _effective_mzml_cache_dir(options: ImportOptions) -> Path:
    return options.mzml_cache_dir_path() or _default_mzml_cache_dir()


def _load_cached_index(path: Path, cache_path: Path) -> object | None:
    if HierarchicalOffsetIndex is None or not cache_path.is_file():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("format") != "pymaldi_mzml_offset_index_v1":
            return None
        if payload.get("source_signature") != _source_signature(path):
            return None
        raw = payload.get("index")
        if not isinstance(raw, dict):
            return None
        return HierarchicalOffsetIndex.load(io.StringIO(json.dumps(raw)))
    except Exception:
        return None


def _save_cached_index(path: Path, cache_path: Path, index_obj: object) -> None:
    if HierarchicalOffsetIndex is None or index_obj is None:
        return
    if not hasattr(index_obj, "save"):
        return
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        index_obj.save(buf)
        payload = {
            "format": "pymaldi_mzml_offset_index_v1",
            "source_path": str(path.resolve()),
            "source_signature": _source_signature(path),
            "index": json.loads(buf.getvalue()),
        }
        cache_path.write_text(json.dumps(payload), encoding="utf-8")
    except Exception:
        return


class _CachedPreIndexedMzML(mzml.PreIndexedMzML if mzml is not None else object):  # type: ignore[misc]
    """PreIndexed reader with pyMALDI sidecar-cache fallback for mzML offset indexes."""

    def __init__(
        self,
        source: str,
        *,
        use_cache: bool = True,
        force_reindex: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        self._pymaldi_source_path = Path(source)
        self._pymaldi_use_cache = bool(use_cache)
        self._pymaldi_force_reindex = bool(force_reindex)
        self._pymaldi_cache_path = _default_mzml_cache_path(self._pymaldi_source_path, cache_dir=cache_dir)
        super().__init__(source)

    def build_byte_index(self):  # type: ignore[override]
        if self._pymaldi_use_cache and not self._pymaldi_force_reindex:
            cached = _load_cached_index(self._pymaldi_source_path, self._pymaldi_cache_path)
            if cached is not None:
                return cached

        # Avoid PreIndexed warning path when embedded index is missing.
        built = mzml.MzML.build_byte_index(self)  # type: ignore[attr-defined]
        if self._pymaldi_use_cache and built is not None:
            _save_cached_index(self._pymaldi_source_path, self._pymaldi_cache_path, built)
        return built


def _infer_profile_flag(spectrum: dict) -> bool:
    if "profile spectrum" in spectrum:
        return True
    cv_params = spectrum.get("cvParams") or spectrum.get("cvparams") or []
    for item in cv_params:
        try:
            if str(item.get("name", "")).lower() == "profile spectrum":
                return True
        except Exception:
            continue
    return False


def _apply_import_filters(
    mz_arr: np.ndarray, int_arr: np.ndarray, is_profile: bool, options: ImportOptions | None
) -> tuple[np.ndarray, np.ndarray, bool]:
    if options is None:
        return mz_arr, int_arr, is_profile

    mz = np.asarray(mz_arr, dtype=float)
    intensity = np.asarray(int_arr, dtype=float)

    if options.centroided_only and is_profile:
        raise RuntimeError("centroided_only=True but spectrum is profile")

    lo, hi = options.mass_range()
    mask = np.ones(mz.size, dtype=bool)
    if lo is not None:
        mask &= mz >= float(lo)
    if hi is not None:
        mask &= mz <= float(hi)
    if options.min_intensity is not None:
        mask &= intensity >= float(options.min_intensity)

    if mask.size and not np.all(mask):
        mz = mz[mask]
        intensity = intensity[mask]
    return mz, intensity, is_profile


def _read_mzml_with_reader(reader: object, scan_index: int = 0, options: ImportOptions | None = None) -> tuple[SpectrumData, int, int]:
    target: dict | None = None
    last: dict | None = None
    total = 0
    wanted = max(0, int(scan_index))

    for spec in reader:  # type: ignore[operator]
        mz_arr = spec.get("m/z array")
        int_arr = spec.get("intensity array")
        if mz_arr is None or int_arr is None:
            continue
        if total == wanted:
            target = spec
        last = spec
        total += 1

    if total == 0 or last is None:
        raise RuntimeError("No valid spectra found in mzML")

    if target is None:
        target = last
        selected = total - 1
    else:
        selected = wanted

    mz_arr = np.asarray(target.get("m/z array"), dtype=float)
    int_arr = np.asarray(target.get("intensity array"), dtype=float)
    is_profile = _infer_profile_flag(target)
    mz_arr, int_arr, is_profile = _apply_import_filters(mz_arr, int_arr, is_profile, options)
    return SpectrumData(mz=mz_arr, intensity=int_arr, is_profile=is_profile), selected, total


def read_mzml(path: Path, scan_index: int = 0, options: ImportOptions | None = None) -> tuple[SpectrumData, int, int]:
    """Read one scan from mzML and return (spectrum, selected_index, total_scans)."""
    if mzml is None:
        raise RuntimeError("pyteomics.mzml is not available")

    opt = options or ImportOptions()
    cache_dir = _effective_mzml_cache_dir(opt)
    # Prefer indexed access and sidecar-cached offsets for consistent/random access.
    if opt.prefer_indexed and hasattr(mzml, "PreIndexedMzML"):
        try:
            with _CachedPreIndexedMzML(
                str(path),
                use_cache=opt.mzml_cache_index,
                force_reindex=opt.mzml_force_reindex,
                cache_dir=cache_dir,
            ) as reader:
                return _read_mzml_with_reader(reader, scan_index=scan_index, options=opt)
        except Exception:
            pass

    with mzml.MzML(str(path)) as reader:
        return _read_mzml_with_reader(reader, scan_index=scan_index, options=opt)


def build_mzml_index_cache(path: Path, options: ImportOptions | None = None) -> Path:
    """Force generation/update of the pyMALDI sidecar index cache for an mzML file."""
    if mzml is None:
        raise RuntimeError("pyteomics.mzml is not available")
    opt = options or ImportOptions()
    cache_dir = _effective_mzml_cache_dir(opt)
    with _CachedPreIndexedMzML(
        str(path),
        use_cache=opt.mzml_cache_index,
        force_reindex=True,
        cache_dir=cache_dir,
    ):
        pass
    return _default_mzml_cache_path(Path(path), cache_dir=cache_dir)


def read_mzmlb(path: Path, scan_index: int = 0, options: ImportOptions | None = None) -> tuple[SpectrumData, int, int]:
    """Optional mzMLb reader; available only when pyteomics.mzmlb is installed."""
    if mzmlb is None:
        raise RuntimeError("pyteomics.mzmlb is not available")
    opt = options or ImportOptions()
    with mzmlb.MzMLb(str(path)) as reader:
        return _read_mzml_with_reader(reader, scan_index=scan_index, options=opt)


def read_mzxml(path: Path, scan_index: int = 0, options: ImportOptions | None = None) -> tuple[SpectrumData, int, int]:
    """Read one scan from mzXML and return (spectrum, selected_index, total_scans)."""
    if mzxml is None:
        raise RuntimeError("pyteomics.mzxml is not available")

    target: dict | None = None
    last: dict | None = None
    total = 0
    wanted = max(0, int(scan_index))

    with mzxml.MzXML(str(path)) as reader:
        for spec in reader:
            mz_arr = spec.get("m/z array")
            int_arr = spec.get("intensity array")
            if mz_arr is None or int_arr is None:
                continue
            if total == wanted:
                target = spec
            last = spec
            total += 1

    if total == 0 or last is None:
        raise RuntimeError(f"No valid spectra found in mzXML: {path}")

    if target is None:
        target = last
        selected = total - 1
    else:
        selected = wanted

    mz_arr = np.asarray(target.get("m/z array"), dtype=float)
    int_arr = np.asarray(target.get("intensity array"), dtype=float)
    is_profile = _infer_profile_flag(target)
    mz_arr, int_arr, is_profile = _apply_import_filters(mz_arr, int_arr, is_profile, options)

    return SpectrumData(mz=mz_arr, intensity=int_arr, is_profile=is_profile), selected, total
