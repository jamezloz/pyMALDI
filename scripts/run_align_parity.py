from __future__ import annotations

import argparse
from pathlib import Path
import json
from datetime import datetime

from pymaldi.io.bruker_fid import read_bruker_fid
from pymaldi.processing import ProcessingOptions, process_batch, peak_f1


def sample_id_from_folder(folder: Path) -> str:
    parts = folder.parts
    if len(parts) >= 4:
        return "__".join(parts[-4:])
    return folder.name


def main() -> int:
    ap = argparse.ArgumentParser(description="Run alignSpectra parity against R alignment reference")
    ap.add_argument("fid_root", type=Path)
    ap.add_argument("reference_json", type=Path)
    ap.add_argument("--tolerance-da", type=float, default=0.5)
    ap.add_argument("--f1-min", type=float, default=0.90)

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
    ap.add_argument("--align-method", default="lowess")
    ap.add_argument("--align-tolerance", type=float, default=0.002)
    ap.add_argument("--align-min-frequency", type=float, default=0.9)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    payload = json.loads(args.reference_json.read_text(encoding="utf-8"))
    ref_items = {it["sample_id"]: it for it in payload.get("items", [])}

    folders = sorted({p.parent for p in args.fid_root.rglob("fid") if p.is_file()})
    specs = []
    sids = []
    for folder in folders:
        sid = sample_id_from_folder(folder)
        if sid not in ref_items:
            continue
        specs.append(read_bruker_fid(folder))
        sids.append(sid)

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
        align_method=args.align_method,
        align_tolerance=args.align_tolerance,
        align_warping_method=args.align_method,
        align_min_frequency=args.align_min_frequency,
    )

    processed = process_batch(specs, opts)

    items = []
    for sid, ps in zip(sids, processed):
        ref_mz = ref_items[sid].get("peaks_after", [])
        m = peak_f1(ps.peaks.mz, __import__('numpy').array(ref_mz, dtype=float), tolerance_da=args.tolerance_da)
        m["n_pred"] = float(ps.peaks.mz.size)
        m["n_ref"] = float(len(ref_mz))
        items.append({"sample_id": sid, "metrics": m, "pass": bool(m["f1"] >= args.f1_min)})

    passed = sum(1 for it in items if it["pass"])
    failed = len(items) - passed
    report = {
        "total": len(items),
        "passed": passed,
        "failed": failed,
        "thresholds": {"f1_min": args.f1_min, "tolerance_da": args.tolerance_da},
        "items": items,
    }

    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("reports") / f"parity_align_{stamp}.json"
    else:
        out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"report={out}")
    print(f"total={report['total']} passed={report['passed']} failed={report['failed']}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
