import numpy as np

from pymaldi.io.bruker_fid import SpectrumData
from pymaldi.matrix import build_peak_matrix, build_profile_matrix, intensity_to_binary


def _spec_with_peaks(shift: float = 0.0) -> SpectrumData:
    mz = np.linspace(2000.0, 12000.0, 3000)
    y = (
        np.exp(-0.5 * ((mz - (4000.0 + shift)) / 18.0) ** 2) * 1000.0
        + np.exp(-0.5 * ((mz - (7000.0 + shift)) / 22.0) ** 2) * 800.0
        + 10.0
    )
    return SpectrumData(mz=mz, intensity=y, is_profile=True)


def test_build_peak_matrix_and_binary() -> None:
    s1 = _spec_with_peaks(0.0)
    s2 = _spec_with_peaks(2.0)
    pm = build_peak_matrix([s1, s2], tolerance=0.002, min_frequency=0.5, normalize="tic")

    assert pm.matrix.shape[0] == 2
    assert pm.matrix.shape[1] > 0
    assert pm.feature_mz.size == pm.matrix.shape[1]

    b = intensity_to_binary(pm.matrix, threshold=0.1, relative=True)
    assert b.shape == pm.matrix.shape
    assert b.dtype == np.uint8


def test_build_profile_matrix() -> None:
    s1 = _spec_with_peaks(0.0)
    s2 = _spec_with_peaks(1.5)
    pm = build_profile_matrix([s1, s2], grid_size=1024, normalize="max")
    assert pm.matrix.shape == (2, 1024)
    assert np.all(pm.matrix >= 0.0)
