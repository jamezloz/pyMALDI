from pathlib import Path

from pyteomics.auxiliary.file_helpers import HierarchicalOffsetIndex

from pymaldi.io import mass_formats


def test_mzml_index_cache_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "sample.mzML"
    src.write_text("<mzML/>", encoding="utf-8")
    cache = tmp_path / "sample.mzML.pymaldi_idx.json"

    idx = HierarchicalOffsetIndex({"spectrum": [("scan=1", 111), ("scan=2", 222)]})
    mass_formats._save_cached_index(src, cache, idx)

    restored = mass_formats._load_cached_index(src, cache)
    assert restored is not None
    assert list(restored["spectrum"].items()) == [("scan=1", 111), ("scan=2", 222)]


def test_mzml_index_cache_invalidated_when_source_changes(tmp_path: Path) -> None:
    src = tmp_path / "sample.mzML"
    src.write_text("<mzML/>", encoding="utf-8")
    cache = tmp_path / "sample.mzML.pymaldi_idx.json"
    idx = HierarchicalOffsetIndex({"spectrum": [("scan=1", 111)]})
    mass_formats._save_cached_index(src, cache, idx)

    src.write_text("<mzML>changed</mzML>", encoding="utf-8")
    assert mass_formats._load_cached_index(src, cache) is None


def test_default_mzml_cache_path_with_cache_dir(tmp_path: Path) -> None:
    src = tmp_path / "a" / "sample.mzML"
    cache_dir = tmp_path / "index-cache"
    out = mass_formats._default_mzml_cache_path(src, cache_dir=cache_dir)
    assert out.parent == cache_dir
    assert out.name.endswith(".pymaldi_idx.json")


def test_effective_cache_dir_uses_override_env(tmp_path: Path, monkeypatch) -> None:
    custom = tmp_path / "custom-cache"
    monkeypatch.setenv("PYMALDI_MZML_INDEX_CACHE_DIR", str(custom))
    opts = mass_formats.ImportOptions()
    assert mass_formats._effective_mzml_cache_dir(opts) == custom
