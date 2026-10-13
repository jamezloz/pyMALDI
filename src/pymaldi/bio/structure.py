from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin
from pathlib import Path
from typing import Any

import numpy as np

from .protein import ProteinAnalysis


@dataclass(frozen=True)
class StructureResidue:
    residue_index: int
    chain_id: str
    residue_name: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class PredictedStructureSummary:
    method: str
    residue_count: int
    mean_confidence: float


def _parse_pdb_ca(path: Path) -> tuple[StructureResidue, ...]:
    residues: list[StructureResidue] = []
    seen: set[tuple[str, int]] = set()
    with Path(path).open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom = line[12:16].strip()
            if atom != "CA":
                continue
            chain = line[21].strip() or "A"
            try:
                resseq = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
            except Exception:
                continue
            key = (chain, resseq)
            if key in seen:
                continue
            seen.add(key)
            residues.append(
                StructureResidue(
                    residue_index=resseq,
                    chain_id=chain,
                    residue_name=line[17:20].strip(),
                    x=x,
                    y=y,
                    z=z,
                )
            )
    residues.sort(key=lambda r: (r.chain_id, r.residue_index))
    return tuple(residues)


def parse_structure_residues(path: str | Path) -> tuple[StructureResidue, ...]:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".pdb", ".ent", ".cif", ".mmcif"}:
        # Current parser supports PDB-style ATOM rows; this also works for many converted mmCIF exports.
        return _parse_pdb_ca(p)
    raise ValueError(f"Unsupported structure format: {p.suffix}")


def build_pseudo_structure_from_sequence(sequence: str, out_pdb: str | Path, chain_id: str = "A") -> Path:
    seq = "".join(ch for ch in (sequence or "").upper() if ch.isalpha())
    if not seq:
        raise RuntimeError("Cannot build pseudo-structure from empty sequence")
    out = Path(out_pdb)
    out.parent.mkdir(parents=True, exist_ok=True)

    radius = 8.0
    rise = 1.5
    deg_per_res = 100.0
    start_x = 0.0
    start_y = 0.0
    start_z = 0.0

    lines: list[str] = []
    for i, aa in enumerate(seq, 1):
        angle = radians((i - 1) * deg_per_res)
        x = start_x + radius * cos(angle)
        y = start_y + radius * sin(angle)
        z = start_z + (i - 1) * rise
        line = (
            f"ATOM  {i:5d}  CA  {aa:>3s} {chain_id:1s}{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        )
        lines.append(line)
    lines.append("TER")
    lines.append("END")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


_HELIX_FAVOR = frozenset({"A", "L", "M", "Q", "E", "K", "R", "H"})
_STRAND_FAVOR = frozenset({"V", "I", "Y", "F", "W", "T"})
_COIL_BREAK = frozenset({"P", "G"})


def predict_structure_boltz_like(
    sequence: str,
    out_pdb: str | Path,
    chain_id: str = "A",
) -> PredictedStructureSummary:
    """
    Lightweight local backbone generator inspired by secondary-structure propensities.
    It is not a physical/ML structure predictor, but gives a deterministic 3D model
    suitable for visualization and residue-level annotation workflows.
    """
    seq = "".join(ch for ch in (sequence or "").upper() if ch.isalpha())
    if not seq:
        raise RuntimeError("Cannot predict structure from empty sequence")
    out = Path(out_pdb)
    out.parent.mkdir(parents=True, exist_ok=True)

    x, y, z = 0.0, 0.0, 0.0
    theta = 0.0
    phi = 0.0
    lines: list[str] = []
    conf: list[float] = []

    for i, aa in enumerate(seq, 1):
        if aa in _COIL_BREAK:
            turn_deg = 122.0
            rise = 1.0
            c = 0.45
        elif aa in _HELIX_FAVOR:
            turn_deg = 99.0
            rise = 1.5
            c = 0.72
        elif aa in _STRAND_FAVOR:
            turn_deg = 179.0 if (i % 2 == 0) else 181.0
            rise = 3.2
            c = 0.66
        else:
            turn_deg = 110.0
            rise = 1.3
            c = 0.58

        theta += radians(turn_deg)
        phi += radians(13.0 if aa not in _COIL_BREAK else 33.0)
        dx = cos(theta) * cos(phi)
        dy = sin(theta) * cos(phi)
        dz = sin(phi)
        norm = max(1e-9, (dx * dx + dy * dy + dz * dz) ** 0.5)
        x += rise * dx / norm
        y += rise * dy / norm
        z += rise * dz / norm

        line = (
            f"ATOM  {i:5d}  CA  {aa:>3s} {chain_id:1s}{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C"
        )
        lines.append(line)
        conf.append(c)

    lines.append("TER")
    lines.append("END")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return PredictedStructureSummary(
        method="pymaldi_boltz_like_v1",
        residue_count=len(seq),
        mean_confidence=float(np.mean(np.asarray(conf, dtype=float))),
    )


def _import_matplotlib() -> Any:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def plot_structure_architecture_png(
    structure_path: str | Path,
    analysis: ProteinAnalysis,
    out_file: str | Path,
    title: str | None = None,
) -> Path:
    residues = parse_structure_residues(structure_path)
    if not residues:
        raise RuntimeError("No CA residues found in structure")

    plt = _import_matplotlib()
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    xs = [r.x for r in residues]
    ys = [r.y for r in residues]
    zs = [r.z for r in residues]
    idxs = [r.residue_index for r in residues]

    colors = ["#9e9e9e"] * len(residues)

    sp = analysis.signal_peptide
    if sp.has_signal_peptide and sp.cleavage_site is not None:
        for i, idx in enumerate(idxs):
            if 1 <= idx <= sp.cleavage_site:
                colors[i] = "#42a5f5"

    for hit in analysis.lipobox_hits:
        for i, idx in enumerate(idxs):
            if hit.start <= idx <= hit.end:
                colors[i] = "#ef5350"

    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xs, ys, zs, color="#bdbdbd", linewidth=1.2, alpha=0.7)
    ax.scatter(xs, ys, zs, c=colors, s=18, depthshade=True)
    ax.set_title(title or "Protein 3D architecture")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    fig.savefig(out, dpi=170)
    plt.close(fig)
    return out


def export_structure_view_html(
    structure_path: str | Path,
    analysis: ProteinAnalysis,
    out_file: str | Path,
    title: str | None = None,
) -> Path:
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    p = Path(structure_path).resolve()
    pdb_text = p.read_text(encoding="utf-8", errors="ignore")
    pdb_text_js = pdb_text.replace("\\", "\\\\").replace("`", "\\`")

    sp = analysis.signal_peptide
    signal_end = sp.cleavage_site if (sp.has_signal_peptide and sp.cleavage_site is not None) else 0
    lipobox_ranges = ", ".join(f"{h.start}-{h.end}" for h in analysis.lipobox_hits) or "none"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>{title or "Protein 3D viewer"}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f7f8fa; }}
    .wrap {{ max-width: 1100px; margin: 20px auto; padding: 0 16px; }}
    .meta {{ margin: 8px 0 14px; color: #263238; }}
    #viewer {{ width: 100%; height: 680px; border: 1px solid #cfd8dc; border-radius: 10px; background: #fff; }}
    .note {{ font-size: 13px; color: #455a64; margin-top: 10px; }}
  </style>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
</head>
<body>
  <div class="wrap">
    <h2>{title or "Protein 3D architecture"}</h2>
    <div class="meta">
      signal_class={sp.prediction_class}; signal_end={signal_end}; lipobox={lipobox_ranges}
    </div>
    <div id="viewer"></div>
    <div class="note">Gray=protein backbone, Blue=signal peptide, Red=lipobox.</div>
  </div>
  <script>
    (async function() {{
      const viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
      const txt = `{pdb_text_js}`;
      viewer.addModel(txt, "pdb");
      viewer.setStyle({{}}, {{cartoon: {{color: "#b0bec5"}}}});
"""

    if signal_end > 0:
        html += f'    viewer.setStyle({{resi: "1-{signal_end}"}}, {{cartoon: {{color: "#42a5f5"}}}});\n'
    for hit in analysis.lipobox_hits:
        html += (
            f'    viewer.setStyle({{resi: "{hit.start}-{hit.end}"}}, '
            '{cartoon: {color: "#ef5350"}, stick: {radius: 0.15, color: "#ef5350"}});\n'
        )
    html += """      viewer.zoomTo();
      viewer.render();
    })();
  </script>
</body>
</html>
"""
    out.write_text(html, encoding="utf-8")
    return out
