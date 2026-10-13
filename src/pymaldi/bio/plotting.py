from __future__ import annotations

from pathlib import Path
from typing import Any

from .protein import ProteinAnalysis, hydropathy_profile


def _import_matplotlib() -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return matplotlib, plt


def plot_protein_architecture(
    analysis: ProteinAnalysis,
    out_file: str | Path,
    title: str | None = None,
    hydropathy_window: int = 7,
) -> Path:
    _, plt = _import_matplotlib()
    out = Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.4])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)

    n = analysis.length
    ax0.hlines(0.5, 1, n, color="#444444", linewidth=7, alpha=0.25)
    ax0.set_ylim(0.0, 1.0)
    ax0.set_yticks([])
    ax0.set_xlim(1, n)
    ax0.set_title(title or "Protein Architecture")

    sp = analysis.signal_peptide
    if sp.has_signal_peptide and sp.cleavage_site is not None:
        ax0.axvspan(1, sp.cleavage_site, color="#90caf9", alpha=0.6, label=f"Signal peptide ({sp.prediction_class})")
        ax0.axvline(sp.cleavage_site, color="#1565c0", linestyle="--", linewidth=1.5, label="Cleavage")
    elif sp.h_region_start and sp.h_region_end:
        ax0.axvspan(sp.h_region_start, sp.h_region_end, color="#ffe082", alpha=0.6, label="Hydrophobic N-region")

    for i, hit in enumerate(analysis.lipobox_hits):
        label = "Lipobox" if i == 0 else None
        ax0.axvspan(hit.start, hit.end, color="#ef9a9a", alpha=0.8, label=label)
        ax0.text((hit.start + hit.end) / 2.0, 0.8, hit.motif, ha="center", va="center", fontsize=8)

    if analysis.mass_mono_mature is not None:
        ax0.text(
            n,
            0.1,
            f"full={analysis.mass_mono_full:.2f} Da  mature={analysis.mass_mono_mature:.2f} Da",
            ha="right",
            va="center",
            fontsize=9,
        )
    else:
        ax0.text(n, 0.1, f"full={analysis.mass_mono_full:.2f} Da", ha="right", va="center", fontsize=9)

    x, y = hydropathy_profile(analysis.sequence, window=hydropathy_window)
    ax1.plot(x, y, color="#2e7d32", linewidth=1.5, label=f"Kyte-Doolittle (w={hydropathy_window})")
    ax1.axhline(1.6, color="#616161", linestyle="--", linewidth=1.0, label="Hydrophobic threshold")
    ax1.set_ylabel("Hydropathy")
    ax1.set_xlabel("Residue position")
    ax1.grid(alpha=0.25)

    handles0, labels0 = ax0.get_legend_handles_labels()
    handles1, labels1 = ax1.get_legend_handles_labels()
    if handles0 or handles1:
        ax1.legend(handles0 + handles1, labels0 + labels1, loc="upper right", fontsize=8, frameon=True)

    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
