from pathlib import Path

from pymaldi import (
    analyze_protein,
    predict_structure_boltz_like,
    export_structure_view_html,
    parse_structure_residues,
    plot_structure_architecture_png,
)


def _write_toy_pdb(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "ATOM      1  CA  MET A   1      10.000  10.000  10.000  1.00 20.00           C",
                "ATOM      2  CA  LYS A   2      11.500  10.500  10.100  1.00 20.00           C",
                "ATOM      3  CA  LEU A   3      13.000  11.000  10.200  1.00 20.00           C",
                "ATOM      4  CA  ALA A   4      14.500  11.500  10.300  1.00 20.00           C",
                "ATOM      5  CA  SER A   5      16.000  12.000  10.400  1.00 20.00           C",
                "TER",
                "END",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_parse_structure_residues_reads_ca(tmp_path: Path) -> None:
    pdb = tmp_path / "toy.pdb"
    _write_toy_pdb(pdb)
    residues = parse_structure_residues(pdb)
    assert len(residues) == 5
    assert residues[0].residue_index == 1
    assert residues[-1].residue_index == 5


def test_structure_outputs_png_and_html(tmp_path: Path) -> None:
    pdb = tmp_path / "toy.pdb"
    _write_toy_pdb(pdb)

    analysis = analyze_protein("MKKIWLALAGLVLAFSLLASCQAAKPEPTIDEK")
    png = plot_structure_architecture_png(pdb, analysis, tmp_path / "toy.png")
    html = export_structure_view_html(pdb, analysis, tmp_path / "toy.html")

    assert png.exists()
    assert png.stat().st_size > 0
    assert html.exists()
    assert "3Dmol" in html.read_text(encoding="utf-8")


def test_predict_structure_boltz_like_builds_pdb(tmp_path: Path) -> None:
    out = tmp_path / "pred.pdb"
    summary = predict_structure_boltz_like("MKKIWLALAGLVLAFSLLASCQAAKPEPTIDEK", out)
    assert out.exists()
    assert summary.residue_count == 33
    text = out.read_text(encoding="utf-8")
    assert "ATOM" in text
