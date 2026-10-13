from __future__ import annotations

from pathlib import Path
import csv

import numpy as np

from .bruker_fid import SpectrumData


def read_txt_peaklist(path: Path) -> SpectrumData:
    mz_vals: list[float] = []
    int_vals: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", ".").split()
            if len(parts) < 2:
                continue
            try:
                mz_vals.append(float(parts[0]))
                int_vals.append(float(parts[1]))
            except ValueError:
                continue
    return SpectrumData(mz=np.asarray(mz_vals, dtype=float), intensity=np.asarray(int_vals, dtype=float), is_profile=False)


def read_csv_peaklist(path: Path) -> SpectrumData:
    mz_vals: list[float] = []
    int_vals: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as fh:
        sample = fh.read(2048)
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        except csv.Error:
            dialect = csv.get_dialect("excel")
        reader = csv.reader(fh, dialect)
        for row in reader:
            if not row or len(row) < 2:
                continue
            c0 = row[0].strip().replace(",", ".")
            c1 = row[1].strip().replace(",", ".")
            if not c0 or not c1:
                continue
            try:
                mz_vals.append(float(c0))
                int_vals.append(float(c1))
            except ValueError:
                # Skip header/non-numeric lines.
                continue
    return SpectrumData(mz=np.asarray(mz_vals, dtype=float), intensity=np.asarray(int_vals, dtype=float), is_profile=False)
