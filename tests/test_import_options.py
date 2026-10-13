from pathlib import Path

from pymaldi.io import ImportOptions
from pymaldi.io.universal import list_supported_sources, read_spectra


def test_list_supported_sources_pattern_filters(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("100 1\n", encoding="utf-8")
    (tmp_path / "b.csv").write_text("mz,intensity\n100,1\n", encoding="utf-8")

    all_items = list_supported_sources(tmp_path, recursive=False)
    assert len(all_items) == 2

    only_txt = list_supported_sources(tmp_path, recursive=False, options=ImportOptions(include_patterns=["*.txt"]))
    assert len(only_txt) == 1
    assert only_txt[0].name == "a.txt"


def test_read_spectra_applies_min_intensity_filter(tmp_path: Path) -> None:
    src = tmp_path / "x.csv"
    src.write_text("mz,intensity\n100,1\n200,10\n300,100\n", encoding="utf-8")

    out = read_spectra(src, options=ImportOptions(min_intensity=10.0))
    sp = out[0].spectrum
    assert sp.mz.size == 2
    assert float(sp.intensity.min()) >= 10.0
