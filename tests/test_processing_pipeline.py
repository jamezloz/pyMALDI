import numpy as np

from pymaldi.io.bruker_fid import SpectrumData
from pymaldi.processing import ProcessingOptions, process_spectrum


def test_process_spectrum_with_options_runs() -> None:
    x = np.linspace(2000.0, 20000.0, 2048)
    # synthetic with peaks
    y = (
        np.exp(-0.5 * ((x - 5000.0) / 20.0) ** 2) * 1000.0
        + np.exp(-0.5 * ((x - 10000.0) / 30.0) ** 2) * 1500.0
        + np.random.default_rng(0).normal(0, 5, size=x.size)
        + 50.0
    )
    y = np.maximum(y, 0.0)

    spec = SpectrumData(mz=x, intensity=y, is_profile=True)
    opts = ProcessingOptions(
        transform_method="sqrt",
        smooth_method="SavitzkyGolay",
        baseline_method="SNIP",
        calibrate_method="TIC",
        detect_noise_method="MAD",
        detect_snr=2.0,
    )

    out = process_spectrum(spec, opts)
    assert out.spectrum.intensity.size == x.size
    assert out.peaks is not None
    assert out.peaks.mz.size > 0
