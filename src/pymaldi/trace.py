from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import uuid

from pymaldi.io.options import ImportOptions
from pymaldi.open_plot import OpenPlotConfig


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_at_utc: str
    input_path: str
    input_sha256: str | None
    api_version: str
    open_plot_config: dict[str, object]
    import_options: dict[str, object]
    outputs: dict[str, str]
    notes: dict[str, object]


@dataclass(frozen=True)
class OperationManifest:
    run_id: str
    created_at_utc: str
    input_path: str
    input_sha256: str | None
    operation: str
    api_version: str
    config: dict[str, object]
    import_options: dict[str, object]
    outputs: dict[str, str]
    notes: dict[str, object]


def create_run_manifest(
    input_path: str | Path,
    api_version: str,
    open_plot_config: OpenPlotConfig,
    import_options: ImportOptions,
    outputs: dict[str, str],
    notes: dict[str, object] | None = None,
) -> RunManifest:
    p = Path(input_path)
    checksum: str | None = None
    if p.is_file():
        try:
            checksum = sha256_file(p)
        except Exception:
            checksum = None

    return RunManifest(
        run_id=uuid.uuid4().hex,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        input_path=str(p),
        input_sha256=checksum,
        api_version=api_version,
        open_plot_config=dict(asdict(open_plot_config)),
        import_options=dict(asdict(import_options)),
        outputs=dict(outputs),
        notes=dict(notes or {}),
    )


def create_operation_manifest(
    input_path: str | Path,
    operation: str,
    api_version: str,
    config: dict[str, object],
    import_options: ImportOptions,
    outputs: dict[str, str],
    notes: dict[str, object] | None = None,
) -> OperationManifest:
    p = Path(input_path)
    checksum: str | None = None
    if p.is_file():
        try:
            checksum = sha256_file(p)
        except Exception:
            checksum = None

    return OperationManifest(
        run_id=uuid.uuid4().hex,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        input_path=str(p),
        input_sha256=checksum,
        operation=str(operation),
        api_version=api_version,
        config=dict(config),
        import_options=dict(asdict(import_options)),
        outputs=dict(outputs),
        notes=dict(notes or {}),
    )


def write_manifest(manifest: RunManifest | OperationManifest, out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(manifest), indent=2), encoding="utf-8")
    return out
