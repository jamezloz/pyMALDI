import numpy as np
from pathlib import Path

from pymaldi.io.bruker_fid import SpectrumData
from pymaldi.io.universal import LoadedSpectrum
from pymaldi.visualization import build_multiresolution_payloads, build_plot_payload, downsample_peak_preserving


def _mock_loaded(n: int = 20000) -> LoadedSpectrum:
    mz = np.linspace(2000.0, 20000.0, n)
    y = np.sin(np.linspace(0, 60, n)) ** 2 * 100.0 + 1.0
    spec = SpectrumData(mz=mz, intensity=y, is_profile=True)
    return LoadedSpectrum(source_path=Path("mock.mzML"), source_id="scan:0", file_type="mzml", spectrum=spec, metadata={})


def test_downsample_peak_preserving_reduces_points() -> None:
    x = np.linspace(0, 1, 10000)
    y = np.sin(np.linspace(0, 100, 10000))
    xs, ys = downsample_peak_preserving(x, y, target_points=1000)
    assert xs.size <= 1200
    assert ys.size == xs.size


def test_build_plot_payload_clip_normalize_downsample() -> None:
    loaded = _mock_loaded()
    p = build_plot_payload(loaded, normalize="max", clip_mz_min=5000.0, clip_mz_max=10000.0, downsample_to=1500)
    assert p.mz.size <= 1800
    assert p.intensity.size == p.mz.size
    assert float(np.max(p.intensity)) <= 1.0 + 1e-9


def test_multiresolution_payloads() -> None:
    loaded = _mock_loaded()
    payloads = build_multiresolution_payloads(loaded, levels=(4000, 2000, 1000))
    assert set(payloads.keys()) == {4000, 2000, 1000}
    assert payloads[1000].mz.size <= payloads[2000].mz.size <= payloads[4000].mz.size
