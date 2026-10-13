from pathlib import Path

from pymaldi import analyze_protein, plot_protein_architecture


def test_plot_protein_architecture_writes_png(tmp_path: Path) -> None:
    seq = "MKKIWLALAGLVLAFSLLASCQAAKPEPTIDEK"
    analysis = analyze_protein(seq)
    out = plot_protein_architecture(analysis, tmp_path / "protein.png", title="Protein Test")
    assert out.exists()
    assert out.stat().st_size > 0
