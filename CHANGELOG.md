# Changelog

## 0.2.0 — 2026-10-13 (first public release)
- Readers for Bruker flex `fid`, mzML (with index cache), mzXML, Bruker XML, TXT and CSV peak lists.
- MALDIquant-style preprocessing pipeline with batch alignment; peak and profile matrices.
- Parity with MALDIquant/MALDIquantForeign validated for reading `fid`, peak detection and alignment (54/54 spectra).
- Time-of-flight to mass conversion written from the published quadratic calibration of Bruker instruments
  (Titulaer et al., 2006).
- Lossless opening for visualization, plot payloads, export and run manifests.
- Experimental `bio` helpers.
