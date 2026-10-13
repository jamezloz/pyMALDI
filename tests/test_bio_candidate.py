from pymaldi.bio.candidate import normalize_organism_name, rank_candidates


def test_normalize_organism_name_aliases() -> None:
    assert normalize_organism_name("ecoli") == "Escherichia coli"
    assert normalize_organism_name("kp") == "Klebsiella pneumoniae"


def test_rank_candidates_by_mass_window() -> None:
    records = [
        {
            "primaryAccession": "P1",
            "uniProtkbId": "X_P1",
            "proteinDescription": {"recommendedName": {"fullName": {"value": "Prot1"}}},
            "genes": [{"geneName": {"value": "g1"}}],
            "organism": {"scientificName": "Escherichia coli"},
            "sequence": {"length": 250, "molWeight": 28100},
            "features": [{"type": "SIGNAL"}],
            "comments": [{"commentType": "SUBCELLULAR LOCATION", "subcellularLocations": [{"location": {"value": "Periplasm"}}]}],
        },
        {
            "primaryAccession": "P2",
            "uniProtkbId": "X_P2",
            "proteinDescription": {"recommendedName": {"fullName": {"value": "Prot2"}}},
            "genes": [{"geneName": {"value": "g2"}}],
            "organism": {"scientificName": "Escherichia coli"},
            "sequence": {"length": 280, "molWeight": 30000},
            "features": [],
            "comments": [],
        },
    ]
    ranked = rank_candidates(mass_da=28200, records=records, tolerance_da=500, top_k=10)
    assert len(ranked) == 1
    assert ranked[0].accession == "P1"
