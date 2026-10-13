from __future__ import annotations

import argparse
from pathlib import Path

from pymaldi.export import export_plot_payload_bundle
from pymaldi.io import ImportOptions
from pymaldi.open_plot import OpenPlotConfig, open_plot_records
from pymaldi.trace import create_run_manifest, write_manifest


def _safe_stem(s: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in s)


def main() -> int:
    ap = argparse.ArgumentParser(description="Open spectra and export plot assets + manifest.")
    ap.add_argument("input_path", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("reports/open_plot_exports"))
    ap.add_argument("--downsample", type=int, default=5000)
    ap.add_argument("--scan-index", type=int, default=0)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--no-svg", action="store_true")
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    cfg = OpenPlotConfig(
        normalize="none",
        clip_mz_min=None,
        clip_mz_max=None,
        downsample_to=None if args.downsample <= 0 else int(args.downsample),
        recursive=False,
        scan_index=max(0, int(args.scan_index)),
    )
    imp = ImportOptions(prefer_indexed=True)

    result = open_plot_records(args.input_path, config=cfg, import_options=imp)

    base = args.out_dir / _safe_stem(args.input_path.stem)
    base.mkdir(parents=True, exist_ok=True)

    all_outputs: dict[str, str] = {}
    for i, payload in enumerate(result.payloads):
        stem = _safe_stem(f"{args.input_path.stem}__{payload.source_id}__{i}")
        written = export_plot_payload_bundle(
            payload,
            out_dir=base,
            stem=stem,
            include_png=not args.no_png,
            include_svg=not args.no_svg,
            include_csv=not args.no_csv,
        )
        for k, v in written.items():
            all_outputs[f"{stem}:{k}"] = v

    manifest = create_run_manifest(
        input_path=args.input_path,
        api_version=result.api_version,
        open_plot_config=cfg,
        import_options=imp,
        outputs=all_outputs,
        notes={"n_payloads": len(result.payloads)},
    )
    manifest_path = write_manifest(manifest, base / "run_manifest.json")

    print(f"api_version={result.api_version}")
    print(f"payloads={len(result.payloads)}")
    print(f"outputs={len(all_outputs)}")
    print(f"manifest={manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
