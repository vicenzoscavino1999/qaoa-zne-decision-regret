"""Portable file display helper used by the Makefile."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Print a UTF-8 text file.")
    parser.add_argument("path", help="Path to print")
    args = parser.parse_args()

    path = Path(args.path)
    if not path.is_file():
        print(f"show_file_status=missing path={path.as_posix()}")
        return 1
    print(path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
