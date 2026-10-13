from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# Monoisotopic residue masses (residue form, i.e. peptide chain without H2O).
_AA_MONO = {
    "A": 71.037113805,
    "R": 156.10111105,
    "N": 114.04292747,
    "D": 115.026943065,
    "C": 103.009184505,
    "E": 129.042593135,
    "Q": 128.05857754,
    "G": 57.021463735,
    "H": 137.058911875,
    "I": 113.084064015,
    "L": 113.084064015,
    "K": 128.09496305,
    "M": 131.040484645,
    "F": 147.068413945,
    "P": 97.052763875,
    "S": 87.032028435,
    "T": 101.047678505,
    "W": 186.07931298,
    "Y": 163.063328575,
    "V": 99.068413945,
}

# Average residue masses.
_AA_AVG = {
    "A": 71.0788,
    "R": 156.1875,
    "N": 114.1038,
    "D": 115.0886,
    "C": 103.1388,
    "E": 129.1155,
    "Q": 128.1307,
    "G": 57.0519,
    "H": 137.1411,
    "I": 113.1594,
    "L": 113.1594,
    "K": 128.1741,
    "M": 131.1926,
    "F": 147.1766,
    "P": 97.1167,
    "S": 87.0782,
    "T": 101.1051,
    "W": 186.2132,
    "Y": 163.1760,
    "V": 99.1326,
}

# Kyte-Doolittle hydropathy scale.
_KD = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "E": -3.5,
    "Q": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

_WATER_MONO = 18.010564684
_WATER_AVG = 18.01528
_VALID_AA = frozenset(_AA_MONO)
_HYDROPHOBIC = frozenset({"A", "V", "I", "L", "M", "F", "W", "Y", "C"})
_SMALL_NEUTRAL = frozenset({"A", "S", "G", "T", "V", "C"})


@dataclass(frozen=True)
class SignalPeptidePrediction:
    has_signal_peptide: bool
    cleavage_site: int | None
    score: float
    n_region_basic_count: int
    h_region_hydrophobic_ratio: float
    motif: str
    prediction_class: str = "none"  # "none", "sec_spi", "sec_spii"
    n_region_start: int | None = None
    n_region_end: int | None = None
    h_region_start: int | None = None
    h_region_end: int | None = None
    c_region_start: int | None = None
    c_region_end: int | None = None


@dataclass(frozen=True)
class LipoboxHit:
    motif: str
    start: int
    end: int
    cleavage_site: int
    cysteine_position: int
    score: float


@dataclass(frozen=True)
class MassMatch:
    label: str
    theoretical_mass: float
    observed_mz: float
    delta_da: float
    delta_ppm: float


@dataclass(frozen=True)
class ProteinAnalysis:
    sequence: str
    length: int
    mass_mono_full: float
    mass_avg_full: float
    signal_peptide: SignalPeptidePrediction
    lipobox_hits: tuple[LipoboxHit, ...]
    mature_sequence: str
    mass_mono_mature: float | None
    mass_avg_mature: float | None


def _normalize_sequence(sequence: str) -> str:
    seq = "".join(ch for ch in sequence.upper() if ch.isalpha())
    if not seq:
        raise ValueError("Protein sequence is empty")
    unknown = sorted(set(seq) - _VALID_AA)
    if unknown:
        raise ValueError(f"Unsupported amino acids in sequence: {''.join(unknown)}")
    return seq


def estimate_protein_mass(sequence: str) -> tuple[float, float]:
    seq = _normalize_sequence(sequence)
    mono = _WATER_MONO + float(sum(_AA_MONO[aa] for aa in seq))
    avg = _WATER_AVG + float(sum(_AA_AVG[aa] for aa in seq))
    return mono, avg


def _window_hydrophobic_ratio(seq: str, start: int, end: int) -> float:
    frag = seq[start:end]
    if not frag:
        return 0.0
    hydrophobic = sum(1 for aa in frag if aa in _HYDROPHOBIC)
    return float(hydrophobic / len(frag))


def _kyte_doolittle(seq: str, window: int = 7) -> np.ndarray:
    values = np.array([_KD[aa] for aa in seq], dtype=float)
    if values.size == 0:
        return values
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(values, kernel, mode="same")


def detect_lipobox(sequence: str) -> tuple[LipoboxHit, ...]:
    seq = _normalize_sequence(sequence)
    hits: list[LipoboxHit] = []
    # Lipobox usually around signal peptide C-region in bacterial lipoproteins.
    # Pattern near cleavage: [LVI][ASTVI][GAS]C (+1 Cys).
    for i in range(10, max(10, min(len(seq) - 3, 45))):
        a, b, c, d = seq[i], seq[i + 1], seq[i + 2], seq[i + 3]
        if d != "C":
            continue
        if a not in {"L", "I", "V", "M", "F", "T"}:
            continue
        if b not in {"A", "S", "T", "V", "I", "G"}:
            continue
        if c not in {"G", "A", "S"}:
            continue
        motif = a + b + c + d
        # cleavage is before invariant Cys (+1).
        cleavage = i + 3  # 1-based index of last residue before cleavage
        score = 1.0
        hits.append(
            LipoboxHit(
                motif=motif,
                start=i + 1,
                end=i + 4,
                cleavage_site=cleavage,
                cysteine_position=i + 4,
                score=score,
            )
        )
    return tuple(hits)


def _best_h_region(seq: str, kd: np.ndarray) -> tuple[int, int, float]:
    # Search expected signal peptide hydrophobic core mostly in first ~40 aa.
    lo = 6
    hi = min(len(seq), 40)
    if hi <= lo:
        return 0, 0, 0.0
    peak = int(lo + np.argmax(kd[lo:hi]))
    threshold = 1.6
    left = peak
    right = peak
    while left > 0 and kd[left] >= threshold:
        left -= 1
    while right < len(seq) - 1 and kd[right] >= threshold:
        right += 1
    start = max(1, left + 1)  # 1-based
    end = min(len(seq), right + 1)  # 1-based inclusive
    ratio = _window_hydrophobic_ratio(seq, start - 1, end)
    return start, end, ratio


def predict_signal_peptide(sequence: str) -> SignalPeptidePrediction:
    seq = _normalize_sequence(sequence)
    if len(seq) < 15:
        return SignalPeptidePrediction(False, None, 0.0, 0, 0.0, "")

    n_region = seq[:8]
    basic_count = sum(1 for aa in n_region if aa in {"K", "R"})
    kd = _kyte_doolittle(seq, window=7)
    h_start, h_end, h_ratio = _best_h_region(seq, kd)
    if h_start == 0:
        return SignalPeptidePrediction(False, None, 0.0, basic_count, 0.0, "")

    # Candidates for Sec/SPI cleavage after H-region.
    best_score = -1.0
    best_site: int | None = None
    best_motif = ""
    lo = max(16, h_end + 1)
    hi = min(len(seq) - 1, 40)
    for cleavage in range(lo, hi + 1):
        # -3/-1 rule (von Heijne).
        c_minus_3 = seq[cleavage - 3] if cleavage - 3 >= 0 else ""
        c_minus_1 = seq[cleavage - 1] if cleavage - 1 >= 0 else ""
        motif_ok = c_minus_3 in _SMALL_NEUTRAL and c_minus_1 in _SMALL_NEUTRAL
        score = (0.4 if basic_count > 0 else 0.0) + h_ratio + (0.6 if motif_ok else 0.0)
        if score > best_score:
            best_score = score
            best_site = cleavage
            best_motif = f"{c_minus_3}x{c_minus_1}"

    # Candidate for Sec/SPII (lipoprotein signal peptide).
    lipobox_hits = detect_lipobox(seq)
    best_lipobox = None
    for hit in lipobox_hits:
        if h_end <= hit.cleavage_site <= 40:
            best_lipobox = hit
            break

    is_spi = bool(best_site is not None and h_ratio >= 0.6 and basic_count >= 1 and best_score >= 1.6)
    is_spii = bool(best_lipobox is not None and h_ratio >= 0.6 and basic_count >= 1)

    if is_spii and best_lipobox is not None:
        cleavage = best_lipobox.cleavage_site
        return SignalPeptidePrediction(
            has_signal_peptide=True,
            cleavage_site=cleavage,
            score=float(max(0.0, best_score + 0.2)),
            n_region_basic_count=basic_count,
            h_region_hydrophobic_ratio=float(h_ratio),
            motif=best_lipobox.motif,
            prediction_class="sec_spii",
            n_region_start=1,
            n_region_end=8,
            h_region_start=h_start,
            h_region_end=h_end,
            c_region_start=max(h_end + 1, 1),
            c_region_end=cleavage,
        )

    if is_spi and best_site is not None:
        return SignalPeptidePrediction(
            has_signal_peptide=True,
            cleavage_site=best_site,
            score=float(max(0.0, best_score)),
            n_region_basic_count=basic_count,
            h_region_hydrophobic_ratio=float(h_ratio),
            motif=best_motif,
            prediction_class="sec_spi",
            n_region_start=1,
            n_region_end=8,
            h_region_start=h_start,
            h_region_end=h_end,
            c_region_start=max(h_end + 1, 1),
            c_region_end=best_site,
        )

    return SignalPeptidePrediction(
        has_signal_peptide=False,
        cleavage_site=None,
        score=float(max(0.0, best_score)),
        n_region_basic_count=basic_count,
        h_region_hydrophobic_ratio=float(h_ratio),
        motif="",
        prediction_class="none",
        n_region_start=1,
        n_region_end=8,
        h_region_start=h_start,
        h_region_end=h_end,
    )


def analyze_protein(sequence: str) -> ProteinAnalysis:
    seq = _normalize_sequence(sequence)
    mono_full, avg_full = estimate_protein_mass(seq)
    lipobox = detect_lipobox(seq)
    sp = predict_signal_peptide(seq)

    mature = seq
    mono_mature: float | None = None
    avg_mature: float | None = None
    if sp.has_signal_peptide and sp.cleavage_site is not None and sp.cleavage_site < len(seq):
        mature = seq[sp.cleavage_site :]
        mono_mature, avg_mature = estimate_protein_mass(mature)

    return ProteinAnalysis(
        sequence=seq,
        length=len(seq),
        mass_mono_full=mono_full,
        mass_avg_full=avg_full,
        signal_peptide=sp,
        lipobox_hits=lipobox,
        mature_sequence=mature,
        mass_mono_mature=mono_mature,
        mass_avg_mature=avg_mature,
    )


def _iter_theoretical_masses(analysis: ProteinAnalysis) -> Iterable[tuple[str, float]]:
    yield "full_mono", analysis.mass_mono_full
    yield "full_avg", analysis.mass_avg_full
    if analysis.mass_mono_mature is not None:
        yield "mature_mono", analysis.mass_mono_mature
    if analysis.mass_avg_mature is not None:
        yield "mature_avg", analysis.mass_avg_mature


def match_masses_to_peaks(
    analysis: ProteinAnalysis,
    mz_values: np.ndarray | list[float],
    tolerance_da: float = 2.0,
    max_ppm: float | None = None,
) -> tuple[MassMatch, ...]:
    mz = np.asarray(mz_values, dtype=float)
    if mz.size == 0:
        return tuple()
    out: list[MassMatch] = []
    abs_tol = abs(float(tolerance_da))
    for label, theo in _iter_theoretical_masses(analysis):
        idx = int(np.argmin(np.abs(mz - theo)))
        obs = float(mz[idx])
        delta = obs - theo
        ppm = (delta / theo) * 1e6 if theo else np.inf
        if abs(delta) <= abs_tol:
            if max_ppm is None or abs(ppm) <= float(max_ppm):
                out.append(
                    MassMatch(
                        label=label,
                        theoretical_mass=float(theo),
                        observed_mz=obs,
                        delta_da=float(delta),
                        delta_ppm=float(ppm),
                    )
                )
    return tuple(out)


def hydropathy_profile(sequence: str, window: int = 7) -> tuple[np.ndarray, np.ndarray]:
    seq = _normalize_sequence(sequence)
    x = np.arange(1, len(seq) + 1, dtype=float)
    y = _kyte_doolittle(seq, window=window)
    return x, y
