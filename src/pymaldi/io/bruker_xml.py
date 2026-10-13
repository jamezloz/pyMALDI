from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lxml import etree

from .bruker_fid import SpectrumData


@dataclass
class BrukerXmlEntry:
    source_id: str
    analyte_name: str
    sample_name: str
    type_name: str
    position: str
    spectrum: SpectrumData


def read_bruker_xml_entries(path: Path) -> list[BrukerXmlEntry]:
    tree = etree.parse(str(path))
    root = tree.getroot()
    analytes = root.findall(".//Analyte")
    out: list[BrukerXmlEntry] = []

    if analytes:
        for idx, analyte in enumerate(analytes):
            peaks = analyte.findall(".//Peak")
            mz_vals: list[float] = []
            int_vals: list[float] = []
            for p in peaks:
                mass = p.get("mass")
                inten = p.get("intensity")
                if mass is None or inten is None:
                    continue
                try:
                    mz_vals.append(float(mass))
                    int_vals.append(float(inten))
                except ValueError:
                    continue
            out.append(
                BrukerXmlEntry(
                    source_id=f"analyte:{idx}",
                    analyte_name=analyte.get("name") or f"Analyte_{idx+1}",
                    sample_name=analyte.get("externId") or "",
                    type_name=analyte.get("typeName") or "",
                    position=analyte.get("targetPosition") or "",
                    spectrum=SpectrumData(
                        mz=np.asarray(mz_vals, dtype=float),
                        intensity=np.asarray(int_vals, dtype=float),
                        is_profile=False,
                    ),
                )
            )
        return out

    # Fallback single peaklist structures.
    peaks = root.findall(".//Peaklist/Peaks/Peak")
    mz_vals: list[float] = []
    int_vals: list[float] = []
    for p in peaks:
        mass = p.get("mass")
        inten = p.get("intensity")
        if mass is None or inten is None:
            continue
        try:
            mz_vals.append(float(mass))
            int_vals.append(float(inten))
        except ValueError:
            continue
    out.append(
        BrukerXmlEntry(
            source_id="analyte:0",
            analyte_name=path.stem,
            sample_name="",
            type_name="BrukerXML",
            position="",
            spectrum=SpectrumData(mz=np.asarray(mz_vals, dtype=float), intensity=np.asarray(int_vals, dtype=float), is_profile=False),
        )
    )
    return out
