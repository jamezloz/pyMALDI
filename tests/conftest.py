"""Datos de prueba: las pruebas que leen espectros reales usan la carpeta indicada en PYMALDI_TEST_DATA y se omiten si no
está definida. Estructura esperada: TXT/, fid/, XML/, mzML RAW/, mzXML/, mzML RUO IVD/."""
import os
from pathlib import Path

import pytest

DATOS = os.environ.get("PYMALDI_TEST_DATA")


def dato(relativa: str) -> Path:
    if not DATOS:
        pytest.skip("PYMALDI_TEST_DATA no está definida")
    p = Path(DATOS) / relativa
    if not p.exists():
        pytest.skip(f"no se encontró {relativa} en PYMALDI_TEST_DATA")
    return p
