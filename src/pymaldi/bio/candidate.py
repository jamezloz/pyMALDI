from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
import urllib.parse
import urllib.request


@dataclass(frozen=True)
class CandidateProtein:
    accession: str
    entry_name: str
    protein_name: str
    gene_names: str
    organism: str
    length: int | None
    mass_da: float | None
    has_signal_annotation: bool
    subcellular_location: str
    delta_da: float
    score: float


_ORG_ALIASES = {
    "ecoli": "Escherichia coli",
    "escherichia coli": "Escherichia coli",
    "klebsiella pneumoniae": "Klebsiella pneumoniae",
    "kp": "Klebsiella pneumoniae",
}


def normalize_organism_name(name: str) -> str:
    key = (name or "").strip().lower()
    if key in _ORG_ALIASES:
        return _ORG_ALIASES[key]
    if key:
        return name.strip()
    raise ValueError("Organismo vacío")


def _fetch_uniprot_records(organism: str, reviewed: bool, size: int) -> list[dict[str, Any]]:
    org = normalize_organism_name(organism)
    q = f'(organism_name:"{org}")'
    if reviewed:
        q += " AND (reviewed:true)"
    params = {
        "query": q,
        "fields": ",".join(
            [
                "accession",
                "id",
                "protein_name",
                "gene_names",
                "organism_name",
                "length",
                "mass",
                "cc_subcellular_location",
                "ft_signal",
            ]
        ),
        "format": "json",
        "size": str(max(1, min(int(size), 500))),
    }
    url = "https://rest.uniprot.org/uniprotkb/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "pyMALDI/0.2"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return list(payload.get("results", []))


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_extract_text(x) for x in value)
    if isinstance(value, dict):
        for key in ("value", "fullName", "recommendedName", "submissionNames", "name"):
            if key in value:
                return _extract_text(value.get(key))
        return " ".join(_extract_text(v) for v in value.values())
    return str(value)


def _parse_record(r: dict[str, Any]) -> dict[str, Any]:
    acc = str(r.get("primaryAccession", "")).strip()
    entry = str(r.get("uniProtkbId", "")).strip()

    prot_desc = r.get("proteinDescription", {})
    prot_name = ""
    if isinstance(prot_desc, dict):
        rec = prot_desc.get("recommendedName")
        if isinstance(rec, dict):
            prot_name = _extract_text(rec.get("fullName"))
        if not prot_name:
            subs = prot_desc.get("submissionNames") or []
            if subs:
                prot_name = _extract_text(subs[0])

    genes = r.get("genes") or []
    gene_names: list[str] = []
    for g in genes:
        if not isinstance(g, dict):
            continue
        n = g.get("geneName")
        if isinstance(n, dict):
            v = str(n.get("value", "")).strip()
            if v:
                gene_names.append(v)

    org = ""
    org_obj = r.get("organism", {})
    if isinstance(org_obj, dict):
        org = str(org_obj.get("scientificName", "")).strip()
    length = r.get("sequence", {}).get("length") if isinstance(r.get("sequence"), dict) else None
    mass = r.get("sequence", {}).get("molWeight") if isinstance(r.get("sequence"), dict) else None

    comments = r.get("comments") or []
    subloc_text = ""
    has_signal = False
    for c in comments:
        if not isinstance(c, dict):
            continue
        ctype = str(c.get("commentType", "")).upper()
        if ctype == "SUBCELLULAR LOCATION":
            subloc_text += " " + _extract_text(c)

    feats = r.get("features") or []
    for f in feats:
        if not isinstance(f, dict):
            continue
        if str(f.get("type", "")).upper() == "SIGNAL":
            has_signal = True
            break

    return {
        "accession": acc,
        "entry_name": entry,
        "protein_name": prot_name,
        "gene_names": " ".join(gene_names),
        "organism": org,
        "length": int(length) if isinstance(length, int) else None,
        "mass_da": float(mass) if isinstance(mass, (int, float)) else None,
        "has_signal_annotation": has_signal,
        "subcellular_location": subloc_text.strip(),
    }


def rank_candidates(
    mass_da: float,
    records: list[dict[str, Any]],
    tolerance_da: float = 300.0,
    require_signal: bool | None = None,
    top_k: int = 20,
) -> tuple[CandidateProtein, ...]:
    target = float(mass_da)
    tol = max(1e-9, abs(float(tolerance_da)))
    out: list[CandidateProtein] = []
    for r in records:
        p = _parse_record(r)
        m = p.get("mass_da")
        if m is None:
            continue
        delta = float(m - target)
        if abs(delta) > tol:
            continue
        has_signal = bool(p.get("has_signal_annotation"))
        if require_signal is True and not has_signal:
            continue
        if require_signal is False and has_signal:
            continue
        rel = abs(delta) / tol
        signal_bonus = 0.1 if has_signal else 0.0
        subloc_bonus = 0.05 if p.get("subcellular_location") else 0.0
        score = max(0.0, 1.0 - rel + signal_bonus + subloc_bonus)
        out.append(
            CandidateProtein(
                accession=str(p.get("accession", "")),
                entry_name=str(p.get("entry_name", "")),
                protein_name=str(p.get("protein_name", "")),
                gene_names=str(p.get("gene_names", "")),
                organism=str(p.get("organism", "")),
                length=p.get("length"),
                mass_da=p.get("mass_da"),
                has_signal_annotation=has_signal,
                subcellular_location=str(p.get("subcellular_location", "")),
                delta_da=delta,
                score=score,
            )
        )
    out.sort(key=lambda x: (-x.score, abs(x.delta_da), x.accession))
    return tuple(out[: max(1, int(top_k))])


def find_candidates_by_mass(
    mass_da: float,
    organism: str,
    tolerance_da: float = 300.0,
    require_signal: bool | None = None,
    reviewed: bool = True,
    size: int = 500,
    top_k: int = 20,
) -> tuple[CandidateProtein, ...]:
    recs = _fetch_uniprot_records(organism=organism, reviewed=reviewed, size=size)
    return rank_candidates(
        mass_da=mass_da,
        records=recs,
        tolerance_da=tolerance_da,
        require_signal=require_signal,
        top_k=top_k,
    )
