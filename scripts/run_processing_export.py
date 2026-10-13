from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from pymaldi.io import ImportOptions
from pymaldi.process_api import ProcessingConfig, run_processing_records
from pymaldi.trace import create_operation_manifest, write_manifest


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run processing API and export outputs + manifest.")
    ap.add_argument("input_path", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/processing_exports"))
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--scan-index", type=int, default=0)
    ap.add_argument("--process-peaklists", action="store_true")
    ap.add_argument("--batch-align-raw", action="store_true")
    args = ap.parse_args()

    cfg = ProcessingConfig(
        recursive=bool(args.recursive),
        scan_index=max(0, int(args.scan_index)),
        process_peaklists=bool(args.process_peaklists),
        batch_align_raw=bool(args.batch_align_raw),
    )
    imp = ImportOptions(prefer_indexed=True)

    res = run_processing_records(args.input_path, config=cfg, import_options=imp)

    out = args.out_dir / _safe(args.input_path.stem)
    out.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, str] = {}
    for i, rec in enumerate(res.records):
        stem = _safe(f"{Path(rec.source_path).stem}__{rec.source_id}__{i}")
        csv_path = out / f"{stem}.processed.csv"
        np.savetxt(csv_path, np.column_stack([rec.mz, rec.intensity]), delimiter=",", header="mz,intensity", comments="")
        peak_path = out / f"{stem}.peaks.csv"
        np.savetxt(peak_path, np.column_stack([rec.peak_mz, rec.peak_intensity]) if rec.peak_mz.size else np.empty((0, 2)), delimiter=",", header="mz,intensity", comments="")
        outputs[f"{stem}:processed"] = str(csv_path)
        outputs[f"{stem}:peaks"] = str(peak_path)

    summary = {
        "api_version": res.api_version,
        "input_path": res.input_path,
        "n_records": len(res.records),
        "records": [
            {
                "source_path": r.source_path,
                "source_id": r.source_id,
                "file_type": r.file_type,
                "n_points": int(r.mz.size),
                "n_peaks": int(r.peak_mz.size),
            }
            for r in res.records
        ],
    }
    summary_path = out / "processing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    outputs["summary"] = str(summary_path)

    manifest = create_operation_manifest(
        input_path=args.input_path,
        operation="processing",
        api_version=res.api_version,
        config=asdict(cfg),
        import_options=imp,
        outputs=outputs,
        notes={"n_records": len(res.records)},
    )
    manifest_path = write_manifest(manifest, out / "run_manifest.json")

    print(f"api_version={res.api_version}")
    print(f"records={len(res.records)}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
