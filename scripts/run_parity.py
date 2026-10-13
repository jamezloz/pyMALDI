from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import numpy as np

from pymaldi.io.bruker_fid import read_bruker_fid
from pymaldi.parity.metrics import compute_raw_metrics
from pymaldi.parity.report import ParityThresholds, load_reference_json, summarize_results, write_report


def sample_id_from_folder(folder: Path) -> str:
    parts = folder.parts
    if len(parts) >= 4:
        return "__".join(parts[-4:])
    return folder.name


def main() -> int:
    parser = argparse.ArgumentParser(description="Run raw parity checks (Python vs R references).")
    parser.add_argument("fid_root", type=Path, help="Root folder containing Bruker fid folders")
    parser.add_argument("reference_dir", type=Path, help="Folder with R reference JSON files")
    parser.add_argument("--out", type=Path, default=None, help="Output report JSON path")
    args = parser.parse_args()

    fid_root = args.fid_root
    reference_dir = args.reference_dir

    targets = sorted({p.parent for p in fid_root.rglob("fid") if p.is_file()})
    if not targets:
        print(f"No fid folders found in: {fid_root}")
        return 2

    items = []
    for folder in targets:
        sid = sample_id_from_folder(folder)
        ref = reference_dir / f"{sid}.json"
        if not ref.exists():
            continue

        py = read_bruker_fid(folder)
        payload = load_reference_json(ref)
        ref_mz = np.array(payload["mz"], dtype=float)
        ref_i = np.array(payload["intensity"], dtype=float)

        metrics = compute_raw_metrics(py.mz, py.intensity, ref_mz, ref_i)
        items.append(
            {
                "sample_id": sid,
                "fid_folder": str(folder),
                "reference": str(ref),
                "metrics": metrics,
            }
        )

    summary = summarize_results(items, thresholds=ParityThresholds())

    if args.out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("reports") / f"parity_raw_{stamp}.json"
    else:
        out = args.out

    write_report(out, summary)
    print(f"report={out}")
    print(f"total={summary['total']} passed={summary['passed']} failed={summary['failed']}")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
