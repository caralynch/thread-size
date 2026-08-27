#!/usr/bin/env python3
"""Create a SHA-256 manifest for the endpoint-audit package."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "outputs" / "package_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    records = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path == MANIFEST or "__pycache__" in path.parts:
            continue
        records.append(
            {
                "relative_path": path.relative_to(HERE).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"manifested {len(records)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
