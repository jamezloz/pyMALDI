import numpy as np

from pymaldi.io.bruker_fid import SpectrumData
from pymaldi.processing import ProcessingOptions, process_batch


def _make_spec(shift: float) -> SpectrumData:
    mz = np.linspace(2000, 21000, 4096) + shift
    y = (
        np.exp(-0.5 * ((mz - (5000 + shift)) / 20.0) ** 2) * 1000.0
        + np.exp(-0.5 * ((mz - (10000 + shift)) / 30.0) ** 2) * 1500.0
        + 10.0
    )
    return SpectrumData(mz=mz, intensity=y, is_profile=True)


def test_process_batch_align_linear_runs() -> None:
    specs = [_make_spec(0.0), _make_spec(5.0)]
    opts = ProcessingOptions(align_method="linear", align_tolerance=10.0)
    out = process_batch(specs, opts)
    assert len(out) == 2
    assert out[0].spectrum.mz.size == specs[0].mz.size
    assert out[1].spectrum.mz.size == specs[1].mz.size
