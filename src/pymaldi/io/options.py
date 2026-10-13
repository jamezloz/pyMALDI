from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class ImportOptions:
    # Keep default behavior fully faithful to source files.
    mass_min: float | None = None
    mass_max: float | None = None
    min_intensity: float | None = None
    centroided_only: bool = False
    prefer_indexed: bool = True
    mzml_cache_index: bool = True
    mzml_force_reindex: bool = False
    mzml_index_cache_dir: str | None = None
    include_patterns: Sequence[str] | None = None
    exclude_patterns: Sequence[str] | None = None

    def mass_range(self) -> tuple[float | None, float | None]:
        return self.mass_min, self.mass_max

    def mzml_cache_dir_path(self) -> Path | None:
        if not self.mzml_index_cache_dir:
            return None
        return Path(self.mzml_index_cache_dir).expanduser()
