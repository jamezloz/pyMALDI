import numpy as np

from pymaldi.bio.protein import analyze_protein, estimate_protein_mass, match_masses_to_peaks


def test_estimate_protein_mass_simple() -> None:
    mono, avg = estimate_protein_mass("ACD")
    assert mono > 0.0
    assert avg > 0.0
    # Spot-check stable expected window.
    assert 300.0 < mono < 310.0
    assert 300.0 < avg < 311.0


def test_analyze_protein_detects_signal_peptide_and_lipobox() -> None:
    # Synthetic Sec-type signal peptide with hydrophobic region and LASC lipobox.
    seq = "MKKIWLALAGLVLAFSLLASCQAAKPEPTIDEK"
    res = analyze_protein(seq)
    assert res.signal_peptide.has_signal_peptide
    assert res.signal_peptide.cleavage_site is not None
    assert res.signal_peptide.prediction_class in {"sec_spi", "sec_spii"}
    assert len(res.lipobox_hits) >= 1
    assert res.mass_mono_mature is not None
    assert res.mass_mono_mature < res.mass_mono_full


def test_match_masses_to_peaks_hits_expected_form() -> None:
    seq = "MKKIWLALAGLVLAFSLLASCQAAKPEPTIDEK"
    res = analyze_protein(seq)
    assert res.mass_mono_mature is not None

    mz = np.array([1200.0, res.mass_mono_mature + 0.7, 9999.0], dtype=float)
    matches = match_masses_to_peaks(res, mz, tolerance_da=1.0)
    labels = {m.label for m in matches}
    assert "mature_mono" in labels
