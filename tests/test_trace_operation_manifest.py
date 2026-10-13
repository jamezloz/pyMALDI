from pathlib import Path
import json

from pymaldi.io.options import ImportOptions
from pymaldi.trace import create_operation_manifest, write_manifest


def test_operation_manifest_write(tmp_path: Path) -> None:
    src = tmp_path / "input.txt"
    src.write_text("abc", encoding="utf-8")

    m = create_operation_manifest(
        input_path=src,
        operation="processing",
        api_version="0.3.0",
        config={"x": 1},
        import_options=ImportOptions(),
        outputs={"a": "b"},
        notes={"k": "v"},
    )
    out = write_manifest(m, tmp_path / "manifest.json")
    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["operation"] == "processing"
