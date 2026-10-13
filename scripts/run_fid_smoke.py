from __future__ import annotations

import sys
from pathlib import Path

from pymaldi.io.bruker_fid import read_bruker_fid


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_fid_smoke.py <fid_root_or_folder>")
        return 2

    root = Path(sys.argv[1])
    if not root.exists():
        print(f"Path not found: {root}")
        return 2

    targets = sorted({p.parent for p in root.rglob("fid") if p.is_file()}) if root.is_dir() else []
    if (root / "fid").is_file():
        targets = [root]
    if not targets and root.name.lower() == "fid" and root.is_file():
        targets = [root.parent]

    ok = 0
    for t in targets:
        spec = read_bruker_fid(t)
        print(f"OK\t{t}\tn={spec.mz.size}\tmz=[{spec.mz.min():.3f},{spec.mz.max():.3f}]")
        ok += 1

    print(f"summary ok={ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
