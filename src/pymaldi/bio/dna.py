from __future__ import annotations

from dataclasses import dataclass

from .protein import ProteinAnalysis, analyze_protein

_DNA = frozenset({"A", "C", "G", "T", "N"})
_RC = str.maketrans("ACGTN", "TGCAN")

_CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

_START_CODONS = frozenset({"ATG", "GTG", "TTG"})
_STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})


@dataclass(frozen=True)
class OrfCandidate:
    strand: str
    frame: int
    start_nt: int
    end_nt: int
    start_codon: str
    has_stop_codon: bool
    protein_sequence: str
    protein_length: int


@dataclass(frozen=True)
class DnaProteinAnalysis:
    dna_sequence: str
    orf: OrfCandidate
    protein: ProteinAnalysis


def normalize_dna(sequence: str) -> str:
    seq = "".join(ch for ch in sequence.upper() if ch.isalpha())
    if not seq:
        raise ValueError("DNA sequence is empty")
    bad = sorted(set(seq) - _DNA)
    if bad:
        raise ValueError(f"Unsupported DNA letters: {''.join(bad)}")
    return seq


def reverse_complement(sequence: str) -> str:
    seq = normalize_dna(sequence)
    return seq.translate(_RC)[::-1]


def translate_frame(sequence: str, frame: int = 0, stop_at_stop: bool = False) -> str:
    dna = normalize_dna(sequence)
    frm = int(frame)
    if frm not in (0, 1, 2):
        raise ValueError("frame must be 0, 1, or 2")
    aa: list[str] = []
    for i in range(frm, len(dna) - 2, 3):
        codon = dna[i : i + 3]
        a = _CODON_TABLE.get(codon, "X")
        if a == "*" and stop_at_stop:
            break
        aa.append(a)
    return "".join(aa)


def _find_orfs_on_strand(
    dna: str,
    strand: str,
    min_aa_length: int,
) -> list[OrfCandidate]:
    out: list[OrfCandidate] = []
    n = len(dna)
    for frame in (0, 1, 2):
        i = frame
        while i <= n - 3:
            codon = dna[i : i + 3]
            if codon not in _START_CODONS:
                i += 3
                continue
            start = i
            j = i
            aa: list[str] = []
            has_stop = False
            while j <= n - 3:
                c = dna[j : j + 3]
                if c in _STOP_CODONS:
                    has_stop = True
                    break
                aa.append(_CODON_TABLE.get(c, "X"))
                j += 3
            prot = "".join(aa)
            if len(prot) >= int(min_aa_length) and "X" not in prot:
                end_nt = j + 3 if has_stop else n
                out.append(
                    OrfCandidate(
                        strand=strand,
                        frame=frame,
                        start_nt=start + 1,
                        end_nt=end_nt,
                        start_codon=codon,
                        has_stop_codon=has_stop,
                        protein_sequence=prot,
                        protein_length=len(prot),
                    )
                )
            i = start + 3
    return out


def find_orfs(
    sequence: str,
    min_aa_length: int = 50,
    include_reverse: bool = True,
) -> tuple[OrfCandidate, ...]:
    dna = normalize_dna(sequence)
    out = _find_orfs_on_strand(dna, "+", min_aa_length=min_aa_length)
    if include_reverse:
        out.extend(_find_orfs_on_strand(reverse_complement(dna), "-", min_aa_length=min_aa_length))
    out.sort(key=lambda x: (-x.protein_length, x.strand, x.frame, x.start_nt))
    return tuple(out)


def analyze_dna_sequence(
    sequence: str,
    min_aa_length: int = 50,
    include_reverse: bool = True,
    top_k: int = 3,
) -> tuple[DnaProteinAnalysis, ...]:
    dna = normalize_dna(sequence)
    orfs = find_orfs(dna, min_aa_length=min_aa_length, include_reverse=include_reverse)
    out: list[DnaProteinAnalysis] = []
    for orf in orfs[: max(0, int(top_k))]:
        out.append(DnaProteinAnalysis(dna_sequence=dna, orf=orf, protein=analyze_protein(orf.protein_sequence)))
    return tuple(out)
