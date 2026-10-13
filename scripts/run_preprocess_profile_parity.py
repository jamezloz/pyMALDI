from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from pymaldi.io.bruker_fid import read_bruker_fid
from pymaldi.processing import ProcessingOptions, process_spectrum


def sample_id_from_folder(folder: Path) -> str:
    parts = folder.parts
    if len(parts) >= 4:
        return "__".join(parts[-4:])
    return folder.name


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    n = min(a.size, b.size)
    if n < 4:
        return float("nan")
    a = np.asarray(a[:n], dtype=float)
    b = np.asarray(b[:n], dtype=float)
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa == 0.0 or sb == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare pyMALDI preprocessed profiles against R MALDIquant references.")
    ap.add_argument("fid_root", type=Path)
    ap.add_argument("reference_dir", type=Path)
    ap.add_argument("--corr-min", type=float, default=0.98)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    opts = ProcessingOptions(
        transform_method="none",
        smooth_method="SavitzkyGolay",
        smooth_half_window_size=4,
        smooth_polyorder=2,
        baseline_method="SNIP",
        baseline_half_window_size=50,
        baseline_iterations=80,
        calibrate_method="none",
        align_method="none",
    )

    targets = sorted({p.parent for p in args.fid_root.rglob("fid") if p.is_file()})
    items: list[dict[str, object]] = []

    for folder in targets:
        sid = sample_id_from_folder(folder)
        ref_file = args.reference_dir / f"{sid}.json"
        if not ref_file.exists():
            continue

        spec = read_bruker_fid(folder)
        proc = process_spectrum(spec, opts)
        py_mz = np.asarray(proc.spectrum.mz, dtype=float)
        py_y = np.asarray(proc.spectrum.intensity, dtype=float)

        payload = json.loads(ref_file.read_text(encoding="utf-8"))
        r_mz = np.asarray(payload.get("mz", []), dtype=float)
        r_y = np.asarray(payload.get("intensity", []), dtype=float)

        if r_mz.size == 0 or r_y.size == 0 or py_mz.size == 0 or py_y.size == 0:
            corr = float("nan")
            n_common = 0
        else:
            # Compare on Python grid for stable summary.
            r_interp = np.interp(py_mz, r_mz, r_y, left=0.0, right=0.0)
            corr = pearson(py_y, r_interp)
            n_common = int(min(py_y.size, r_interp.size))

        items.append(
            {
                "sample_id": sid,
                "fid_folder": str(folder),
                "reference": str(ref_file),
                "corr": corr,
                "n_points": n_common,
                "pass": bool(np.isfinite(corr) and corr >= args.corr_min),
            }
        )

    corrs = np.asarray([it["corr"] for it in items if np.isfinite(it.get("corr", np.nan))], dtype=float)
    passed = sum(1 for it in items if it["pass"])
    failed = len(items) - passed

    report = {
        "total": len(items),
        "passed": passed,
        "failed": failed,
        "corr_min": args.corr_min,
        "corr_mean": float(np.mean(corrs)) if corrs.size else float("nan"),
        "corr_median": float(np.median(corrs)) if corrs.size else float("nan"),
        "corr_p05": float(np.percentile(corrs, 5)) if corrs.size else float("nan"),
        "items": items,
    }

    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("reports") / f"parity_preprocess_profile_{stamp}.json"
    else:
        out = args.out

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"report={out}")
    print(
        f"total={report['total']} passed={report['passed']} failed={report['failed']} "
        f"corr_mean={report['corr_mean']:.6f} corr_median={report['corr_median']:.6f}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
