from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from pymaldi.io.universal import LoadedSpectrum, detect_file_type, read_spectra
from pymaldi.matrix import build_peak_matrix, intensity_to_binary
from pymaldi.presets import build_processing_options_for_app_raw
from pymaldi.processing import ProcessingOptions, process_spectrum


@dataclass
class SourceResult:
    path: str
    file_type: str
    loaded_count: int
    load_ok: bool
    plot_ok: bool
    preprocess_applicable: bool
    preprocess_ok: bool
    matrix_ok: bool
    error: str = ""


def _find_fid_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    out: list[Path] = []
    for fid in root.rglob("fid"):
        if fid.is_file():
            out.append(fid.parent)
    seen: set[str] = set()
    uniq: list[Path] = []
    for p in sorted(out):
        k = str(p.resolve())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    return uniq


def _discover_sources(espectros_root: Path) -> dict[str, list[Path]]:
    discovered: dict[str, list[Path]] = {
        "bruker_fid": [],
        "mzml_raw": [],
        "mzml_peaklist": [],
        "mzxml": [],
        "bruker_xml": [],
        "txt": [],
        "csv": [],
    }

    fid_low = espectros_root / "fid"
    fid_high = espectros_root / "fid alto peso"
    discovered["bruker_fid"].extend(_find_fid_dirs(fid_low))
    discovered["bruker_fid"].extend(_find_fid_dirs(fid_high))

    for p in sorted((espectros_root / "mzML RAW").glob("*.mzML")):
        discovered["mzml_raw"].append(p)

    for p in sorted((espectros_root / "mzML RUO IVD").glob("*.mzML")):
        discovered["mzml_peaklist"].append(p)

    for p in sorted((espectros_root / "mzXML").glob("*.mzXML")):
        discovered["mzxml"].append(p)

    for p in sorted((espectros_root / "XML").glob("*.xml")):
        discovered["bruker_xml"].append(p)

    for p in sorted((espectros_root / "TXT").glob("*.txt")):
        discovered["txt"].append(p)

    csv_dir = espectros_root / "CSV"
    if csv_dir.exists():
        for p in sorted(csv_dir.glob("*.csv")):
            discovered["csv"].append(p)

    return discovered


def _is_plot_ready(loaded: LoadedSpectrum) -> bool:
    mz = np.asarray(loaded.spectrum.mz, dtype=float)
    y = np.asarray(loaded.spectrum.intensity, dtype=float)
    return bool(mz.size > 0 and y.size == mz.size and np.any(np.isfinite(mz)) and np.any(np.isfinite(y)))


def _preprocess_options(source: LoadedSpectrum) -> ProcessingOptions:
    ftype = source.file_type
    if ftype == "bruker_fid":
        return build_processing_options_for_app_raw("raw_bruker")
    # mzml/mzxml raw defaults aligned with current app behavior.
    if ftype == "mzxml":
        return build_processing_options_for_app_raw("mzxml")
    return build_processing_options_for_app_raw("raw_vitek")


def _run_single(path: Path) -> SourceResult:
    guessed = detect_file_type(path)
    try:
        loaded = read_spectra(path)
    except Exception as exc:
        return SourceResult(
            path=str(path),
            file_type=guessed,
            loaded_count=0,
            load_ok=False,
            plot_ok=False,
            preprocess_applicable=False,
            preprocess_ok=False,
            matrix_ok=False,
            error=f"load_error: {exc}",
        )

    if not loaded:
        return SourceResult(
            path=str(path),
            file_type=guessed,
            loaded_count=0,
            load_ok=False,
            plot_ok=False,
            preprocess_applicable=False,
            preprocess_ok=False,
            matrix_ok=False,
            error="load_error: empty_result",
        )

    plot_ok = all(_is_plot_ready(x) for x in loaded)

    preprocess_applicable = any(
        (x.file_type in {"bruker_fid", "mzxml"}) or (x.file_type == "mzml" and bool(x.spectrum.is_profile))
        for x in loaded
    )
    preprocess_ok = True
    processed_items = []
    for x in loaded:
        # In app, preprocessing is currently focused on RAW-like types.
        is_raw_mzml = x.file_type == "mzml" and bool(x.spectrum.is_profile)
        if x.file_type not in {"bruker_fid", "mzxml"} and not is_raw_mzml:
            continue
        try:
            out = process_spectrum(x.spectrum, _preprocess_options(x))
            processed_items.append(out)
        except Exception:
            preprocess_ok = False

    # Matrix stage for all loaded spectra in the source.
    matrix_ok = True
    try:
        if processed_items:
            pm = build_peak_matrix(processed_items, tolerance=0.002, min_frequency=0.5, normalize="none")
        else:
            pm = build_peak_matrix([x.spectrum for x in loaded], tolerance=0.002, min_frequency=0.5, normalize="none")
        _ = intensity_to_binary(pm.matrix, threshold=0.0, relative=False)
    except Exception:
        matrix_ok = False

    return SourceResult(
        path=str(path),
        file_type=loaded[0].file_type,
        loaded_count=len(loaded),
        load_ok=True,
        plot_ok=plot_ok,
        preprocess_applicable=preprocess_applicable,
        preprocess_ok=preprocess_ok,
        matrix_ok=matrix_ok,
        error="",
    )


def _sample(items: Iterable[Path], max_items: int) -> list[Path]:
    paths = list(items)
    if max_items <= 0 or len(paths) <= max_items:
        return paths
    return paths[:max_items]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pyMALDI capability benchmark on app formats + CSV.")
    parser.add_argument(
        "--espectros-root",
        type=Path,
        required=True,
        help="Root folder containing Espectros datasets.",
    )
    parser.add_argument(
        "--max-per-group",
        type=int,
        default=40,
        help="Maximum sources tested per format group (0 = all).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports"),
        help="Output directory for JSON report.",
    )
    args = parser.parse_args()

    started = time.time()
    groups = _discover_sources(args.espectros_root)

    by_group: dict[str, dict[str, object]] = {}
    all_results: list[SourceResult] = []

    for group_name, paths in groups.items():
        tested = _sample(paths, args.max_per_group)
        results = [_run_single(p) for p in tested]
        all_results.extend(results)

        total = len(results)
        load_ok = sum(1 for r in results if r.load_ok)
        plot_ok = sum(1 for r in results if r.plot_ok)
        preprocess_applicable = sum(1 for r in results if r.preprocess_applicable)
        preprocess_ok = sum(1 for r in results if (r.preprocess_applicable and r.preprocess_ok))
        matrix_ok = sum(1 for r in results if r.matrix_ok)

        by_group[group_name] = {
            "discovered": len(paths),
            "tested": total,
            "load_ok": load_ok,
            "plot_ok": plot_ok,
            "preprocess_applicable": preprocess_applicable,
            "preprocess_ok": preprocess_ok,
            "matrix_ok": matrix_ok,
            "failed_examples": [asdict(r) for r in results if not (r.load_ok and r.plot_ok and r.preprocess_ok and r.matrix_ok)][:10],
        }

    elapsed = time.time() - started

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "espectros_root": str(args.espectros_root),
        "max_per_group": args.max_per_group,
        "elapsed_seconds": round(elapsed, 3),
        "summary": by_group,
        "results": [asdict(r) for r in all_results],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"capability_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Report: {out_json}")
    for g, info in by_group.items():
        print(
            f"{g}: tested={info['tested']} "
            f"load={info['load_ok']} plot={info['plot_ok']} "
            f"preprocess={info['preprocess_ok']}/{info['preprocess_applicable']} "
            f"matrix={info['matrix_ok']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
