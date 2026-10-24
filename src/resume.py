from __future__ import annotations

from pathlib import Path


def is_complete_file(path: str | Path) -> bool:
    path = Path(path)
    return path.exists() and path.is_file() and path.stat().st_size > 0

