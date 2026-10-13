from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SpectrumData:
    mz: np.ndarray
    intensity: np.ndarray
    is_profile: bool = True


def _parse_jcamp_params(path: Path) -> Tuple[Dict[str, str], Dict[str, List[float]]]:
    scalars: Dict[str, str] = {}
    arrays: Dict[str, List[float]] = {}
    if not path.exists():
        return scalars, arrays
    lines = path.read_text(encoding="latin-1", errors="ignore").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m_arr = re.match(r"##\$(\w+)=\s*\(\d+\.\.\d+\)\s*$", line)
        if m_arr:
            key = m_arr.group(1)
            vals: List[float] = []
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if nxt.startswith("##"):
                    i -= 1
                    break
                for tok in nxt.split():
                    try:
                        vals.append(float(tok))
                    except ValueError:
                        pass
                i += 1
            arrays[key] = vals
            i += 1
            continue
        m = re.match(r"##\$(\w+)=\s*(.+)\s*$", line)
        if m:
            scalars[m.group(1)] = m.group(2).strip()
        i += 1
    return scalars, arrays


def _read_int32(path: Path, little_endian_hint: bool = True) -> np.ndarray:
    dtypes = ["<i4", ">i4"] if little_endian_hint else [">i4", "<i4"]

    def score(arr: np.ndarray) -> float:
        if arr.size == 0:
            return -1e18
        nonneg_ratio = float(np.mean(arr >= 0))
        p99 = float(np.percentile(np.maximum(arr, 0.0), 99))
        p999 = float(np.percentile(np.maximum(arr, 0.0), 99.9))
        tail_ratio = p999 / (p99 + 1e-9) if p99 > 0 else 0.0
        return nonneg_ratio * 1000.0 - abs(tail_ratio - 1.5)

    best = np.array([], dtype=float)
    best_score = -1e18
    for dt in dtypes:
        raw = np.fromfile(path, dtype=dt)
        cand = raw.astype(float)
        sc = score(cand)
        if sc > best_score:
            best = cand
            best_score = sc
    if best.size == 0:
        raise RuntimeError(f"Empty or unreadable spectrum file: {path}")
    return np.maximum(best, 0.0)


def _tof_to_mass(tof: np.ndarray, c1: float, c2: float, c3: float) -> np.ndarray:
    """Quadratic time-of-flight calibration of Bruker flex instruments (Titulaer et al., 2006, BMC Bioinformatics 7:403).

    The instrument stores the calibration constants ML1, ML2 and ML3 (c1, c2, c3) in the acqus file and relates the
    flight time t to the mass m as

        t = c2 + sqrt(1e12 / c1) * sqrt(m) + c3 * m

    Writing u = sqrt(m), this is the quadratic c3 * u**2 + k * u + (c2 - t) = 0 with k = sqrt(1e12 / c1), whose positive
    root gives u; for c3 = 0 the relation is linear in u.
    """
    k = np.sqrt(1.0e12 / c1)
    if abs(c3) < 1.0e-15:
        u = (tof - c2) / k
    else:
        u = (-k + np.sqrt(np.maximum(k * k - 4.0 * c3 * (c2 - tof), 0.0))) / (2.0 * c3)
    return u * u


def _build_mz_axis(n: int, procs_scalars: Dict[str, str], procs_arrays: Dict[str, List[float]], acqu_scalars: Dict[str, str]) -> np.ndarray:
    try:
        delay = float(acqu_scalars.get("DELAY", "0").split()[0])
        dw = float(acqu_scalars.get("DW", "1").split()[0])
        c1 = float(acqu_scalars.get("ML1", "0").split()[0])
        c2 = float(acqu_scalars.get("ML2", "0").split()[0])
        c3 = float(acqu_scalars.get("ML3", "0").split()[0])
        if c1 > 0.0:
            tof = delay + np.arange(n, dtype=float) * dw
            mz = _tof_to_mass(tof, c1, c2, c3)
            if np.all(np.isfinite(mz)) and float(np.max(mz)) > float(np.min(mz)):
                return mz
    except Exception:
        pass

    wstart = procs_arrays.get("WSTART", [])
    wend = procs_arrays.get("WEND", [])
    if wstart and wend:
        lo = float(wstart[0])
        hi = float(wend[0])
        if hi > lo:
            return np.linspace(lo, hi, n)

    for source in (procs_scalars, acqu_scalars):
        try:
            lo = float(source.get("WSTART", "0").split()[0])
            hi = float(source.get("WEND", "0").split()[0])
            if hi > lo:
                return np.linspace(lo, hi, n)
        except Exception:
            pass

    return np.arange(n, dtype=float)


def _find_raw_fid_folder(path: Path) -> Path:
    base = path if path.is_dir() else path.parent
    if (base / "fid").is_file():
        return base
    for candidate in base.rglob("fid"):
        if candidate.is_file():
            return candidate.parent
    raise RuntimeError(f"No raw fid file found below: {path}")


def read_bruker_fid(path: Path) -> SpectrumData:
    """Read Bruker Flex `.fid` RAW without conversion."""
    base = _find_raw_fid_folder(path)

    acqu = base / "acqu"
    if not acqu.exists():
        acqu = base / "acqus"
    acqu_scalars, _ = _parse_jcamp_params(acqu)

    procs = base / "pdata" / "1" / "procs"
    if not procs.exists():
        procs = base / "pdata" / "1" / "proc"
    procs_scalars, procs_arrays = _parse_jcamp_params(procs)

    fid = base / "fid"
    bytorda = int(float(acqu_scalars.get("BYTORDA", "0").split()[0])) if acqu_scalars.get("BYTORDA") else 0
    intensity = _read_int32(fid, little_endian_hint=(bytorda == 0))
    mz = _build_mz_axis(intensity.size, procs_scalars, procs_arrays, acqu_scalars)

    return SpectrumData(mz=mz.astype(float), intensity=intensity.astype(float), is_profile=True)
