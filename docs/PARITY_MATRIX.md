# pyMALDI parity matrix

Each stage of the `MALDIquant`/`MALDIquantForeign` workflow mapped to its pyMALDI implementation and parity status.

## Status
- `planned`: not implemented yet.
- `partial`: implemented, final acceptance criterion not yet met.
- `validated`: meets the parity criterion on the whole reference set.

## Matrix
| R package | R function | pyMALDI module | Status | Parity metric | Acceptance |
|---|---|---|---|---|---|
| MALDIquantForeign | `importBrukerFlex` | `pymaldi.io.bruker_fid.read_bruker_fid` | validated | `mz_median_abs_err_da`, `intensity_pearson_r` | `<=0.5 Da`, `>=0.99` |
| MALDIquant | `transformIntensity` | `pymaldi.processing.transform.transform_intensity` | partial | `intensity_pearson_r` | `>=0.99` |
| MALDIquant | `smoothIntensity` | `pymaldi.processing.smooth.smooth_intensity` | partial | `intensity_pearson_r` | `>=0.99` |
| MALDIquant | `removeBaseline` | `pymaldi.processing.baseline.remove_baseline` | partial | `intensity_pearson_r` | `>=0.98` |
| MALDIquant | `calibrateIntensity` | `pymaldi.processing.calibrate` | partial | `tic_rel_error` | `<=1e-6` |
| MALDIquant | `detectPeaks` | `pymaldi.processing.peaks.detect_peaks` | validated | `peak_f1` | `>=0.95` |
| MALDIquant | `alignSpectra` | `pymaldi.processing.align` + `pymaldi.processing.pipeline.process_batch` | validated | `aligned_peak_f1` | `>=0.95` (benchmark: `tolerance_da=3.0`) |

## Reference data
References are produced with MALDIquant/MALDIquantForeign in R (`scripts/export_r_*.R`) on a set of Bruker `fid` spectra
and compared with pyMALDI (`scripts/run_*parity.py`). Spectra and references are not part of the repository.

## Validation workflow
1. Export the R reference for the stage.
2. Run the parity script for the stage.
3. Promote `partial` to `validated` only if the whole reference set meets the criterion.

## Results (54 Bruker `fid` spectra)
- `importBrukerFlex`: 54/54 within criterion (m/z and intensity).
- `detectPeaks`: 54/54, F1 = 1.0 with the reference configuration.
- `alignSpectra`: 54/54 with `align_method=linear`, `align_tolerance=0.002`, `align_min_frequency=0.7` (evaluation tolerance 3 Da).
