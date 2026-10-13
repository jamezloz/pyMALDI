from pathlib import Path
import json

import numpy as np
import pytest

from pymaldi.open_plot import OpenPlotConfig
from pymaldi.io.options import ImportOptions
from pymaldi.visualization import PlotPayload
from pymaldi.export import export_plot_payload_bundle, export_plot_payload_csv, export_plot_payload_image
from pymaldi.trace import create_run_manifest, write_manifest


def _payload() -> PlotPayload:
    return PlotPayload(
        mz=np.array([100.0, 200.0, 300.0]),
        intensity=np.array([1.0, 2.0, 3.0]),
        is_profile=False,
        file_type="csv_peaklist",
        source_id="scan:0",
        source_path="/tmp/mock.csv",
        metadata={},
    )


def test_export_csv_and_manifest(tmp_path: Path) -> None:
    p = _payload()
    csv_path = export_plot_payload_csv(p, tmp_path / "x.csv")
    assert csv_path.exists()

    manifest = create_run_manifest(
        input_path=tmp_path / "input.csv",
        api_version="0.2.0",
        open_plot_config=OpenPlotConfig(),
        import_options=ImportOptions(),
        outputs={"csv": str(csv_path)},
        notes={"k": 1},
    )
    out = write_manifest(manifest, tmp_path / "run_manifest.json")
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["api_version"] == "0.2.0"


def test_export_bundle_writes_meta(tmp_path: Path) -> None:
    p = _payload()
    written = export_plot_payload_bundle(p, out_dir=tmp_path, stem="sample", include_png=False, include_svg=False, include_csv=True)
    assert "csv" in written
    assert "meta" in written
    assert Path(written["meta"]).exists()


def test_export_image_requires_matplotlib(tmp_path: Path) -> None:
    p = _payload()
    try:
        import matplotlib  # noqa: F401
    except Exception:
        with pytest.raises(RuntimeError):
            export_plot_payload_image(p, tmp_path / "z.png")
    else:
        out = export_plot_payload_image(p, tmp_path / "z.png")
        assert out.exists()
