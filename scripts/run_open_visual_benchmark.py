from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pymaldi.io import ImportOptions, load_from_path
from pymaldi.visualization import build_plot_payload


def _discover(root: Path) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {
        "fid": [],
        "mzml_raw": [],
        "mzml_peak": [],
        "mzxml": [],
        "xml": [],
        "txt": [],
        "csv": [],
    }

    for f in sorted((root / "mzML RAW").glob("*.mzML")):
        out["mzml_raw"].append(f)
    for f in sorted((root / "mzML RUO IVD").glob("*.mzML")):
        out["mzml_peak"].append(f)
    for f in sorted((root / "mzXML").glob("*.mzXML")):
        out["mzxml"].append(f)
    for f in sorted((root / "XML").glob("*.xml")):
        out["xml"].append(f)
    for f in sorted((root / "TXT").glob("*.txt")):
        out["txt"].append(f)
    for f in sorted((root / "CSV").glob("*.csv")):
        out["csv"].append(f)

    for d in [root / "fid", root / "fid alto peso"]:
        if not d.exists():
            continue
        for x in d.rglob("fid"):
            if x.is_file():
                out["fid"].append(x.parent)

    for k in out:
        out[k] = sorted(out[k])
    return out


def _sample(xs: list[Path], n: int) -> list[Path]:
    if n <= 0 or len(xs) <= n:
        return xs
    return xs[:n]


def _run_group(paths: list[Path], options: ImportOptions) -> dict[str, object]:
    load_ms: list[float] = []
    viz_ms: list[float] = []
    ok = 0
    n_points: list[int] = []
    issues = 0

    for p in paths:
        t0 = time.perf_counter()
        try:
            loaded = load_from_path(p, recursive=False, scan_index=0, options=options)
            load_ms.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            continue

        for item in loaded:
            t1 = time.perf_counter()
            payload = build_plot_payload(item, normalize="none", downsample_to=5000)
            viz_ms.append((time.perf_counter() - t1) * 1000.0)
            ok += 1
            n_points.append(int(payload.mz.size))
            qc = item.metadata.get("open_qc")
            if isinstance(qc, dict) and not bool(qc.get("ok", True)):
                issues += 1

    return {
        "tested_sources": len(paths),
        "loaded_spectra": ok,
        "mean_load_ms": float(np.mean(load_ms)) if load_ms else None,
        "mean_plot_payload_ms": float(np.mean(viz_ms)) if viz_ms else None,
        "mean_plot_points": float(np.mean(n_points)) if n_points else None,
        "open_qc_issues": issues,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark opening + plot payload generation in pyMALDI")
    ap.add_argument("--espectros-root", type=Path, required=True)
    ap.add_argument("--max-per-group", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    groups = _discover(args.espectros_root)
    options = ImportOptions(prefer_indexed=True)

    summary: dict[str, object] = {}
    for name, paths in groups.items():
        summary[name] = _run_group(_sample(paths, args.max_per_group), options)

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "espectros_root": str(args.espectros_root),
        "max_per_group": args.max_per_group,
        "summary": summary,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"open_visual_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"report={out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
