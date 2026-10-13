from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pymaldi.io import ImportOptions
from pymaldi.process_api import ProcessingConfig, run_processing_records


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return float("nan")
    n = min(a.size, b.size)
    if n < 4:
        return float("nan")
    x = np.asarray(a[:n], dtype=float)
    y = np.asarray(b[:n], dtype=float)
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx == 0.0 or sy == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _sample_id_from_fid(folder: Path) -> str:
    parts = folder.parts
    if len(parts) >= 4:
        return "__".join(parts[-4:])
    return folder.name


def _discover(root: Path) -> dict[str, list[Path]]:
    out = {
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


def _limit(xs: list[Path], n: int) -> list[Path]:
    if n <= 0 or len(xs) <= n:
        return xs
    return xs[:n]


def main() -> int:
    ap = argparse.ArgumentParser(description="Processing API gate by file type")
    ap.add_argument("--espectros-root", type=Path, required=True)
    ap.add_argument("--r-fid-ref", type=Path, default=Path("references/r_preprocess_profile"))
    ap.add_argument("--corr-min-fid", type=float, default=0.98)
    ap.add_argument("--max-per-group", type=int, default=30)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    groups = _discover(args.espectros_root)
    cfg = ProcessingConfig(recursive=False, scan_index=0, process_peaklists=True, batch_align_raw=False)
    imp = ImportOptions(prefer_indexed=True)

    failures: list[dict[str, object]] = []
    by_group: dict[str, dict[str, object]] = {}

    total_sources = total_processed = 0
    process_errors = 0
    fid_corrs: list[float] = []

    for g, paths in groups.items():
        test_paths = _limit(paths, args.max_per_group)
        total_sources += len(test_paths)
        ok = 0
        for p in test_paths:
            try:
                res = run_processing_records(p, config=cfg, import_options=imp)
            except Exception as exc:
                process_errors += 1
                failures.append({"group": g, "source": str(p), "kind": "process_error", "error": str(exc)})
                continue

            for rec in res.records:
                total_processed += 1
                ok += 1
                if rec.mz.size == 0 or rec.intensity.size != rec.mz.size or np.any(~np.isfinite(rec.intensity)):
                    process_errors += 1
                    failures.append({"group": g, "source": str(p), "kind": "invalid_processed_arrays", "source_id": rec.source_id})

                if g == "fid":
                    sid = _sample_id_from_fid(Path(rec.source_path))
                    ref = args.r_fid_ref / f"{sid}.json"
                    if ref.exists():
                        payload = json.loads(ref.read_text(encoding="utf-8"))
                        r_mz = np.asarray(payload.get("mz", []), dtype=float)
                        r_y = np.asarray(payload.get("intensity", []), dtype=float)
                        if r_mz.size and r_y.size:
                            r_interp = np.interp(rec.mz, r_mz, r_y, left=0.0, right=0.0)
                            c = _pearson(rec.intensity, r_interp)
                            fid_corrs.append(c)
                            if not np.isfinite(c) or c < args.corr_min_fid:
                                failures.append({
                                    "group": g,
                                    "source": str(p),
                                    "kind": "fid_r_parity_fail",
                                    "source_id": rec.source_id,
                                    "corr": c,
                                })

        by_group[g] = {
            "tested_sources": len(test_paths),
            "processed_records": ok,
        }

    fid_fail = sum(1 for x in failures if x.get("kind") == "fid_r_parity_fail")
    go = process_errors == 0 and fid_fail == 0

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "go": go,
        "criteria": {
            "process_errors": 0,
            "fid_r_parity_fail": 0,
            "corr_min_fid": args.corr_min_fid,
        },
        "summary": {
            "total_sources": total_sources,
            "total_processed_records": total_processed,
            "process_errors": process_errors,
            "fid_r_parity_fail": fid_fail,
            "fid_corr_mean": float(np.nanmean(fid_corrs)) if fid_corrs else None,
            "fid_corr_p05": float(np.nanpercentile(np.asarray(fid_corrs), 5)) if fid_corrs else None,
        },
        "by_group": by_group,
        "failures": failures[:200],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"processing_gate_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"report={out}")
    print(
        f"GO={go} sources={total_sources} processed={total_processed} "
        f"process_errors={process_errors} fid_r_parity_fail={fid_fail}"
    )
    return 0 if go else 1


if __name__ == "__main__":
    raise SystemExit(main())
