from pathlib import Path

from conftest import dato

from pymaldi.io.bruker_fid import read_bruker_fid


def test_read_single_bruker_fid_smoke() -> None:
    sample = dato("fid/EV56COL/0_B6/1/1SLin")

    spec = read_bruker_fid(sample)

    assert spec.mz.size > 0
    assert spec.intensity.size == spec.mz.size
    assert float(spec.intensity.max()) > 0.0
