# pyMALDI Open+Plot API v0.2 (Frozen)

This document freezes the public interface for file opening and plotting payload generation.

## Version
- `OPEN_PLOT_API_VERSION = "0.2.0"`

## Public API

```python
from pymaldi.open_plot import OpenPlotConfig, open_plot_records
from pymaldi.io.options import ImportOptions
```

- `open_plot_records(path, config=None, import_options=None) -> OpenPlotResult`
- `OpenPlotResult.api_version`
- `OpenPlotResult.payloads` (tuple of `PlotPayload`)

## Fidelity Defaults
`OpenPlotConfig()` defaults are strict-fidelity for visualization:
- `normalize="none"`
- no clipping
- no downsampling
- no hidden transformations

## Traceability
Use:
- `pymaldi.trace.create_run_manifest(...)`
- `pymaldi.trace.write_manifest(...)`

## Export
Use:
- `pymaldi.export.export_plot_payload_bundle(...)`
- Supports `csv` always, `png/svg` when `matplotlib` is available.

## Compatibility rule
Minor/patch changes in v0.2.x must keep this API backward compatible.
