from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pymaldi.io import ImportOptions
from pymaldi.io.universal import list_supported_sources, load_from_path
from pymaldi.visualization import build_plot_payload


def _group_name(path: Path) -> str:
    p = str(path)
    if "/fid alto peso/" in p or "/fid/" in p or path.name == "1SLin":
        return "fid"
    s = path.suffix.lower()
    if s == ".mzml":
        return "mzml"
    if s == ".mzxml":
        return "mzxml"
    if s == ".xml":
        return "xml"
    if s == ".txt":
        return "txt"
    if s == ".csv":
        return "csv"
    if s == ".mzmlb":
        return "mzmlb"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description="GO/NO-GO gate for visual-fidelity open mode")
    ap.add_argument("--espectros-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    options = ImportOptions(prefer_indexed=True)
    sources = list_supported_sources(args.espectros_root, recursive=True, options=options)

    by_group: dict[str, dict[str, float]] = {}
    failures: list[dict[str, object]] = []

    total_sources = len(sources)
    loaded_spectra = 0
    load_errors = 0
    fidelity_failures = 0
    qc_issues = 0
    max_abs_diff = 0.0
    load_times_ms: list[float] = []

    for src in sources:
        grp = _group_name(src)
        if grp not in by_group:
            by_group[grp] = {
                "sources": 0,
                "loaded_spectra": 0,
                "load_errors": 0,
                "fidelity_failures": 0,
                "qc_issues": 0,
            }
        by_group[grp]["sources"] += 1

        t0 = time.perf_counter()
        try:
            loaded = load_from_path(src, recursive=False, scan_index=0, options=options)
            load_times_ms.append((time.perf_counter() - t0) * 1000.0)
        except Exception as exc:
            load_errors += 1
            by_group[grp]["load_errors"] += 1
            failures.append({"source": str(src), "kind": "load_error", "error": str(exc)})
            continue

        for item in loaded:
            loaded_spectra += 1
            by_group[grp]["loaded_spectra"] += 1

            payload = build_plot_payload(item, normalize="none", downsample_to=None)
            mz_eq = np.array_equal(payload.mz, item.spectrum.mz)
            y_eq = np.array_equal(payload.intensity, item.spectrum.intensity)

            if not mz_eq or not y_eq:
                fidelity_failures += 1
                by_group[grp]["fidelity_failures"] += 1
                mz_d = float(np.max(np.abs(payload.mz - item.spectrum.mz))) if payload.mz.size == item.spectrum.mz.size else float("inf")
                y_d = (
                    float(np.max(np.abs(payload.intensity - item.spectrum.intensity)))
                    if payload.intensity.size == item.spectrum.intensity.size
                    else float("inf")
                )
                max_abs_diff = max(max_abs_diff, 0.0 if not np.isfinite(y_d) else y_d)
                failures.append(
                    {
                        "source": str(src),
                        "kind": "fidelity_mismatch",
                        "source_id": item.source_id,
                        "mz_max_abs_diff": mz_d,
                        "intensity_max_abs_diff": y_d,
                    }
                )

            qc = item.metadata.get("open_qc")
            if isinstance(qc, dict) and not bool(qc.get("ok", True)):
                qc_issues += 1
                by_group[grp]["qc_issues"] += 1
                failures.append({"source": str(src), "kind": "open_qc_issue", "source_id": item.source_id, "qc": qc})

    go = load_errors == 0 and fidelity_failures == 0

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "espectros_root": str(args.espectros_root),
        "go": go,
        "criteria": {
            "load_errors": 0,
            "fidelity_failures": 0,
            "note": "open_qc issues are reported but not hard-fail unless desired",
        },
        "summary": {
            "total_sources": total_sources,
            "loaded_spectra": loaded_spectra,
            "load_errors": load_errors,
            "fidelity_failures": fidelity_failures,
            "open_qc_issues": qc_issues,
            "max_abs_diff": max_abs_diff,
            "mean_load_ms": float(np.mean(load_times_ms)) if load_times_ms else None,
        },
        "by_group": by_group,
        "failures": failures[:200],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"visual_fidelity_gate_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"report={out}")
    print(f"GO={go} sources={total_sources} loaded={loaded_spectra} load_errors={load_errors} fidelity_failures={fidelity_failures} qc_issues={qc_issues}")
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
