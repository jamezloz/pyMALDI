from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Dict, List


@dataclass
class ParityThresholds:
    mz_median_abs_err_da_max: float = 0.5
    intensity_pearson_r_min: float = 0.99


def load_reference_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "mz" not in payload or "intensity" not in payload:
        raise ValueError(f"Invalid reference file (missing mz/intensity): {path}")
    return payload


def summarize_results(items: List[Dict[str, Any]], thresholds: ParityThresholds) -> Dict[str, Any]:
    passed = 0
    for item in items:
        m = item.get("metrics", {})
        ok = (
            m.get("mz_median_abs_err_da", float("inf")) <= thresholds.mz_median_abs_err_da_max
            and m.get("intensity_pearson_r", float("-inf")) >= thresholds.intensity_pearson_r_min
        )
        item["pass"] = bool(ok)
        passed += 1 if ok else 0

    return {
        "total": len(items),
        "passed": passed,
        "failed": len(items) - passed,
        "thresholds": {
            "mz_median_abs_err_da_max": thresholds.mz_median_abs_err_da_max,
            "intensity_pearson_r_min": thresholds.intensity_pearson_r_min,
        },
        "items": items,
    }


def write_report(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
