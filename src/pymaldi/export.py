from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from pymaldi.visualization import PlotPayload


def export_plot_payload_csv(payload: PlotPayload, out_csv: str | Path) -> Path:
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.column_stack([payload.mz, payload.intensity])
    np.savetxt(out, arr, delimiter=",", header="mz,intensity", comments="")
    return out


def _render_with_matplotlib(payload: PlotPayload, out_file: Path, title: str | None = None) -> Path:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for PNG/SVG export") from exc

    fig = plt.figure(figsize=(10, 4), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(payload.mz, payload.intensity, linewidth=1.0)
    ax.set_xlabel("m/z")
    ax.set_ylabel("intensity")
    ax.set_title(title or f"{payload.file_type} | {payload.source_id}")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)
    return out_file


def export_plot_payload_image(payload: PlotPayload, out_file: str | Path, title: str | None = None) -> Path:
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    if suffix not in {".png", ".svg"}:
        raise ValueError("Image export supports only .png or .svg")
    return _render_with_matplotlib(payload, out, title=title)


def export_plot_payload_bundle(
    payload: PlotPayload,
    out_dir: str | Path,
    stem: str,
    include_png: bool = True,
    include_svg: bool = True,
    include_csv: bool = True,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    if include_csv:
        csv_path = export_plot_payload_csv(payload, out / f"{stem}.csv")
        written["csv"] = str(csv_path)
    if include_png:
        png_path = export_plot_payload_image(payload, out / f"{stem}.png")
        written["png"] = str(png_path)
    if include_svg:
        svg_path = export_plot_payload_image(payload, out / f"{stem}.svg")
        written["svg"] = str(svg_path)

    meta = {
        "file_type": payload.file_type,
        "source_id": payload.source_id,
        "source_path": payload.source_path,
        "is_profile": payload.is_profile,
        "n_points": int(payload.mz.size),
        "outputs": written,
    }
    meta_path = out / f"{stem}.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    written["meta"] = str(meta_path)
    return written
