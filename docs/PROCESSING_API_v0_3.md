# pyMALDI Processing API v0.3

Scope: configurable preprocessing/processing, separate from open+plot fidelity layer.

## Version
- `PROCESSING_API_VERSION = "0.3.0"`

## Public API

```python
from pymaldi.process_api import ProcessingConfig, run_processing_records
from pymaldi.io.options import ImportOptions
```

- `run_processing_records(path, config=None, import_options=None, options_override=None)`
- Returns `ProcessingResult` with list of `ProcessedRecord`.

## Design
- Open+plot fidelity remains independent and untouched.
- Processing options can be default-by-file-type or explicitly overridden.
- Optional raw batch alignment is available (`batch_align_raw=True`).

## Gate by file type
Use `scripts/run_processing_gate.py` to validate:
- processing array integrity on all tested groups,
- FID parity against R preprocessed profile references.

## Command

```bash
PYTHONPATH=src python scripts/run_processing_gate.py \
  --espectros-root /path/to/spectra \
  --r-fid-ref references/r_preprocess_profile \
  --max-per-group 30
```
