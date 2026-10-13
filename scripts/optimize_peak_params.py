from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import subprocess


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, out


def latest_report(pattern: str) -> Path:
    files = sorted(Path("reports").glob(pattern))
    if not files:
        raise RuntimeError(f"No report files for pattern {pattern}")
    return files[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description="Grid search for peak parity parameters")
    ap.add_argument("fid_root", type=Path)
    ap.add_argument("ref_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/peak_param_search.json"))
    args = ap.parse_args()

    grid = {
        "smooth": ["SavitzkyGolay", "MovingAverage"],
        "smooth_half": [5, 10, 15],
        "baseline": ["SNIP", "median", "none"],
        "baseline_iter": [20, 40, 80],
        "noise": ["MAD", "SuperSmoother"],
        "detect_half": [10, 20, 30],
        "snr": [1.5, 2.0, 2.5],
    }

    results = []
    py_root = Path.cwd()

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    for combo in itertools.product(*values):
        p = dict(zip(keys, combo))
        cmd = [
            "python3",
            "scripts/run_peak_parity.py",
            str(args.fid_root),
            str(args.ref_dir),
            "--transform",
            "none",
            "--smooth",
            str(p["smooth"]),
            "--smooth-half",
            str(p["smooth_half"]),
            "--baseline",
            str(p["baseline"]),
            "--baseline-half",
            "20",
            "--baseline-iter",
            str(p["baseline_iter"]),
            "--calibrate",
            "TIC",
            "--noise",
            str(p["noise"]),
            "--detect-half",
            str(p["detect_half"]),
            "--snr",
            str(p["snr"]),
            "--f1-min",
            "0.95",
        ]
        code, _ = run_cmd(cmd, cwd=py_root)
        rep = latest_report("parity_peaks_*.json")
        payload = json.loads(rep.read_text(encoding="utf-8"))
        f1 = [it["metrics"]["f1"] for it in payload.get("items", [])]
        p["median_f1"] = sorted(f1)[len(f1) // 2] if f1 else 0.0
        p["passed"] = payload.get("passed", 0)
        p["failed"] = payload.get("failed", 0)
        p["returncode"] = code
        results.append(p)

    results.sort(key=lambda x: (x["passed"], x["median_f1"]), reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"results": results[:50], "best": results[0] if results else None}, indent=2), encoding="utf-8")
    print(f"report={args.out}")
    if results:
        print("best", results[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
