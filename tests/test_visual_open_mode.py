from pathlib import Path

import numpy as np

from pymaldi.visualization import open_for_visualization
from pymaldi.io.universal import read_spectra


def test_open_for_visualization_preserves_arrays(tmp_path: Path) -> None:
    src = tmp_path / "s.csv"
    src.write_text("mz,intensity\n100,1\n200,2\n300,3\n", encoding="utf-8")

    payloads = open_for_visualization(src)
    assert len(payloads) == 1
    p = payloads[0]

    loaded = read_spectra(src)[0].spectrum
    assert np.array_equal(p.mz, loaded.mz)
    assert np.array_equal(p.intensity, loaded.intensity)
