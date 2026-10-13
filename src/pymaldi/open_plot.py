from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pymaldi.io import ImportOptions, load_from_path
from pymaldi.visualization import PlotPayload, build_plot_payload


OPEN_PLOT_API_VERSION = "0.2.0"


@dataclass(frozen=True)
class OpenPlotConfig:
    # Keep defaults strictly faithful to file content.
    normalize: str = "none"
    clip_mz_min: float | None = None
    clip_mz_max: float | None = None
    downsample_to: int | None = None
    recursive: bool = False
    scan_index: int = 0


@dataclass(frozen=True)
class OpenPlotResult:
    api_version: str
    input_path: str
    payloads: tuple[PlotPayload, ...]


def open_plot_records(
    path: str | Path,
    config: OpenPlotConfig | None = None,
    import_options: ImportOptions | None = None,
) -> OpenPlotResult:
    cfg = config or OpenPlotConfig()
    imp = import_options or ImportOptions(
        mass_min=None,
        mass_max=None,
        min_intensity=None,
        centroided_only=False,
        prefer_indexed=True,
        include_patterns=None,
        exclude_patterns=None,
    )

    loaded = load_from_path(
        Path(path),
        recursive=cfg.recursive,
        scan_index=cfg.scan_index,
        options=imp,
    )

    payloads = tuple(
        build_plot_payload(
            item,
            normalize=cfg.normalize,
            clip_mz_min=cfg.clip_mz_min,
            clip_mz_max=cfg.clip_mz_max,
            downsample_to=cfg.downsample_to,
        )
        for item in loaded
    )

    return OpenPlotResult(
        api_version=OPEN_PLOT_API_VERSION,
        input_path=str(Path(path)),
        payloads=payloads,
    )
