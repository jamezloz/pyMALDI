from pathlib import Path

import numpy as np

from pymaldi import ImportOptions
from pymaldi.open_plot import OPEN_PLOT_API_VERSION, OpenPlotConfig, open_plot_records


def test_open_plot_records_returns_api_version(tmp_path: Path) -> None:
    src = tmp_path / "a.csv"
    src.write_text("mz,intensity\n100,1\n200,2\n", encoding="utf-8")

    res = open_plot_records(src, config=OpenPlotConfig(), import_options=ImportOptions())
    assert res.api_version == OPEN_PLOT_API_VERSION
    assert len(res.payloads) == 1


def test_open_plot_records_fidelity_defaults(tmp_path: Path) -> None:
    src = tmp_path / "b.csv"
    src.write_text("mz,intensity\n100,1\n200,2\n300,3\n", encoding="utf-8")

    res = open_plot_records(src)
    p = res.payloads[0]
    assert np.array_equal(p.mz, np.array([100.0, 200.0, 300.0]))
    assert np.array_equal(p.intensity, np.array([1.0, 2.0, 3.0]))
