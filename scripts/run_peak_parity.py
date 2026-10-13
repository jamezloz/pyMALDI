from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import json
import numpy as np

from pymaldi.io.bruker_fid import read_bruker_fid
from pymaldi.processing import ProcessingOptions, process_spectrum, peak_f1


def sample_id_from_folder(folder: Path) -> str:
    parts = folder.parts
    if len(parts) >= 4:
        return "__".join(parts[-4:])
    return folder.name


def main() -> int:
    ap = argparse.ArgumentParser(description="Run peak parity checks against R references")
    ap.add_argument("fid_root", type=Path)
    ap.add_argument("reference_dir", type=Path)
    ap.add_argument("--tolerance-da", type=float, default=0.5)

    ap.add_argument("--transform", default="none")
    ap.add_argument("--smooth", default="SavitzkyGolay")
    ap.add_argument("--smooth-half", type=int, default=10)
    ap.add_argument("--baseline", default="SNIP")
    ap.add_argument("--baseline-half", type=int, default=20)
    ap.add_argument("--baseline-iter", type=int, default=40)
    ap.add_argument("--calibrate", default="TIC")
    ap.add_argument("--noise", default="MAD")
    ap.add_argument("--detect-half", type=int, default=20)
    ap.add_argument("--snr", type=float, default=2.0)
    ap.add_argument("--f1-min", type=float, default=0.95)
    ap.add_argument("--out", type=Path, default=None)

    args = ap.parse_args()

    opts = ProcessingOptions(
        transform_method=args.transform,
        smooth_method=args.smooth,
        smooth_half_window_size=args.smooth_half,
        baseline_method=args.baseline,
        baseline_half_window_size=args.baseline_half,
        baseline_iterations=args.baseline_iter,
        calibrate_method=args.calibrate,
        detect_noise_method=args.noise,
        detect_half_window_size=args.detect_half,
        detect_snr=args.snr,
    )

    targets = sorted({p.parent for p in args.fid_root.rglob("fid") if p.is_file()})
    items = []

    for folder in targets:
        sid = sample_id_from_folder(folder)
        ref_file = args.reference_dir / f"{sid}.json"
        if not ref_file.exists():
            continue

        spec = read_bruker_fid(folder)
        proc = process_spectrum(spec, opts)
        mz_pred = proc.peaks.mz if proc.peaks is not None else np.array([])

        payload = json.loads(ref_file.read_text(encoding="utf-8"))
        mz_ref = np.array(payload.get("mz", []), dtype=float)

        m = peak_f1(mz_pred, mz_ref, tolerance_da=args.tolerance_da)
        m["n_pred"] = float(mz_pred.size)
        m["n_ref"] = float(mz_ref.size)

        items.append(
            {
                "sample_id": sid,
                "fid_folder": str(folder),
                "reference": str(ref_file),
                "metrics": m,
                "pass": bool(m["f1"] >= args.f1_min),
            }
        )

    passed = sum(1 for it in items if it["pass"])
    failed = len(items) - passed

    report = {
        "total": len(items),
        "passed": passed,
        "failed": failed,
        "thresholds": {
            "f1_min": args.f1_min,
            "tolerance_da": args.tolerance_da,
        },
        "options": {
            "transform": args.transform,
            "smooth": args.smooth,
            "smooth_half": args.smooth_half,
            "baseline": args.baseline,
            "baseline_half": args.baseline_half,
            "baseline_iter": args.baseline_iter,
            "calibrate": args.calibrate,
            "noise": args.noise,
            "detect_half": args.detect_half,
            "snr": args.snr,
        },
        "items": items,
    }

    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("reports") / f"parity_peaks_{stamp}.json"
    else:
        out = args.out

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"report={out}")
    print(f"total={report['total']} passed={report['passed']} failed={report['failed']}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
