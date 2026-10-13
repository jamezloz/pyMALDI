from pymaldi.bio.dna import analyze_dna_sequence, find_orfs, reverse_complement, translate_frame


def test_reverse_complement() -> None:
    assert reverse_complement("ATGCCCTAA") == "TTAGGGCAT"


def test_translate_frame_known_sequence() -> None:
    dna = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
    aa = translate_frame(dna, frame=0, stop_at_stop=False)
    assert aa.startswith("MAIVMGR")
    assert "*" in aa


def test_find_orfs_detects_basic_orf() -> None:
    dna = "CCCATGAAACCCGGGTTTTAAACC"
    orfs = find_orfs(dna, min_aa_length=3, include_reverse=False)
    assert len(orfs) >= 1
    assert orfs[0].protein_sequence.startswith("MK")


def test_analyze_dna_sequence_returns_protein_analysis() -> None:
    dna = "CCCATGAAACCCGGGTTTTAAACC"
    out = analyze_dna_sequence(dna, min_aa_length=3, include_reverse=False, top_k=1)
    assert len(out) == 1
    item = out[0]
    assert item.orf.protein_length >= 3
    assert item.protein.mass_mono_full > 0.0
