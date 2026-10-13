# pyMALDI

Reading and preprocessing of MALDI-TOF mass spectra in Python, with numerical parity against the R packages
**MALDIquant** and **MALDIquantForeign**.

pyMALDI reads Bruker flex `fid` acquisitions directly (no conversion step), as well as mzML, mzXML, Bruker XML and
peak lists in TXT/CSV, and implements the MALDIquant preprocessing workflow: intensity transformation, smoothing,
baseline correction, intensity calibration, peak detection and alignment. Each stage was validated against MALDIquant
outputs on the same spectra (see [Validation](#validation)).

Developed by the **MALDI-TEAM**: Jerson Andrés Martínez, Roque Figueroa-Espinosa, Lautaro Barraza and José Alejandro
Di Conza. pyMALDI is the reading and preprocessing engine of MALDI-ARC (web classifier of carbapenemase mechanism) and
MALDI-ONE (desktop application for spectra analysis).

## Installation

```bash
pip install .              # from a clone of this repository
pip install ".[dev]"       # with test tools
```

Requires Python 3.10 or later, NumPy, SciPy, lxml and pyteomics. matplotlib is optional (image export and plots).

## Quick start

```python
from pathlib import Path
from pymaldi import load_from_path, ProcessingOptions, process_batch, build_peak_matrix

# Read every spectrum under a folder (Bruker fid acquisitions, mzML, mzXML, XML, TXT, CSV)
spectra = load_from_path(Path("spectra/"), recursive=True)

# MALDIquant-style preprocessing
options = ProcessingOptions(
    transform_method="sqrt",
    smooth_method="SavitzkyGolay", smooth_half_window_size=16, smooth_polyorder=4,
    baseline_method="SNIP", baseline_iterations=100,
    calibrate_method="TIC",
    detect_noise_method="MAD", detect_half_window_size=20, detect_snr=3.0,
    align_method="none",
)
processed = process_batch([s.spectrum for s in spectra], options)

# Peak matrix for statistics or machine learning
matrix = build_peak_matrix(processed, tolerance=0.002, min_frequency=0.5)
```

A whole folder can also be processed in one call with `run_dataset_pipeline(input_path, options)`, which returns the
intensity and binary (presence/absence) matrices.

### Bruker `fid`

```python
from pymaldi import read_bruker_fid
spectrum = read_bruker_fid(Path("sample/0_A1/1/1SLin"))   # folder containing fid and acqus
spectrum.mz, spectrum.intensity
```

The time-of-flight to mass conversion uses the quadratic calibration of Bruker flex instruments with the ML1, ML2 and
ML3 constants stored in `acqus` (Titulaer et al., 2006).

## Modules

| Module | Content |
|---|---|
| `pymaldi.io` | readers for Bruker `fid`, mzML (with index cache), mzXML, Bruker XML, TXT and CSV; import options and quality summaries |
| `pymaldi.processing` | transformation, smoothing, baseline (SNIP, TopHat, convex hull, median), calibration (TIC, median, PQN), peak detection, alignment |
| `pymaldi.matrix`, `pymaldi.dataset` | peak and profile matrices, binarization, end-to-end dataset pipeline |
| `pymaldi.open_plot`, `pymaldi.visualization`, `pymaldi.export` | lossless opening for visualization, plotting payloads, CSV/PNG/SVG export |
| `pymaldi.trace` | run manifests with file hashes for traceability |
| `pymaldi.parity` | parity metrics and reports against MALDIquant |
| `pymaldi.bio` | experimental helpers: theoretical protein mass, signal peptide and lipobox prediction, matching of theoretical masses to observed peaks, UniProt search by mass |

## Validation

Each stage was compared with MALDIquant/MALDIquantForeign on 54 Bruker `fid` spectra; R references are produced with
`scripts/export_r_*.R` and compared with `scripts/run_*parity.py`.

| Stage | Result |
|---|---|
| `importBrukerFlex` (reading `fid`) | 54/54 within criterion (m/z and intensity) |
| `detectPeaks` | 54/54, F1 = 1.0 |
| `alignSpectra` | 54/54 |

Transformation, smoothing, baseline and calibration are implemented and in use; their formal parity criterion is still
`partial`. Details in [`docs/PARITY_MATRIX.md`](docs/PARITY_MATRIX.md).

## Tests

```bash
pytest
PYMALDI_TEST_DATA=/path/to/spectra pytest     # also runs the tests that read real spectra
```

## How to cite

Martínez JA, Figueroa-Espinosa R, Barraza L, Di Conza JA. *pyMALDI: reading and preprocessing of MALDI-TOF spectra in
Python* (version 0.2.0). MALDI-TEAM, 2026. See `CITATION.cff`.

## References

- Gibb S, Strimmer K. MALDIquant: a versatile R package for the analysis of mass spectrometry data. *Bioinformatics*
  2012;28:2270–2271. doi:10.1093/bioinformatics/bts447
- Titulaer MK, et al. A database application for pre-processing, storage and comparison of mass spectra derived from
  patients and controls. *BMC Bioinformatics* 2006;7:403. doi:10.1186/1471-2105-7-403

## Use of artificial intelligence tools

Artificial intelligence tools were used as assistance in writing part of the code, correcting errors and auditing, under
the direction and review of the authors. The design, methodological decisions and validation are the responsibility of
the authors.

## License

MIT. See `LICENSE`.
