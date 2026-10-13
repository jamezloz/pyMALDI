from pathlib import Path

from conftest import dato

import pytest

from pymaldi.io.universal import detect_file_type, load_from_path, read_spectra


def test_detect_file_types() -> None:
    assert detect_file_type(Path("sample.mzML")) == "mzml"
    assert detect_file_type(Path("sample.mzXML")) == "mzxml"
    assert detect_file_type(Path("sample.xml")) == "bruker_xml"
    assert detect_file_type(Path("sample.txt")) == "txt_peaklist"
    assert detect_file_type(Path("sample.csv")) == "csv_peaklist"


def test_read_txt_peaklist_smoke() -> None:
    src = dato("TXT/EV40_col_sen-1.txt")
    out = read_spectra(src)
    assert len(out) == 1
    assert out[0].spectrum.mz.size > 10
    assert out[0].spectrum.mz.size == out[0].spectrum.intensity.size


def test_read_bruker_fid_smoke() -> None:
    src = dato("fid/EV56COL/0_B6/1/1SLin")
    out = read_spectra(src)
    assert len(out) == 1
    assert out[0].file_type == "bruker_fid"
    assert out[0].spectrum.mz.size > 100


def test_read_bruker_xml_multianalyte_smoke() -> None:
    src = dato("XML/240802-1602-1011028337.xml")
    out = read_spectra(src)
    assert len(out) > 1
    assert out[0].spectrum.mz.size > 10


def test_read_mzml_smoke() -> None:
    pytest.importorskip("pyteomics")
    src = dato("mzML RAW/7H_COL_KPC_ST25_A1_AVG_av-IVD.mzML")
    out = read_spectra(src)
    assert len(out) == 1
    assert out[0].spectrum.mz.size > 100


def test_read_mzxml_smoke() -> None:
    pytest.importorskip("pyteomics")
    src = dato("mzXML/spectrum_CAN.LFB1.5_1.mzXML")
    out = read_spectra(src)
    assert len(out) == 1
    assert out[0].spectrum.mz.size > 100


def test_load_from_path_directory_smoke() -> None:
    root = dato("mzML RUO IVD")
    out = load_from_path(root, recursive=False)
    assert len(out) > 0
