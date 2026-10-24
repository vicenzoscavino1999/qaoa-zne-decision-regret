from __future__ import annotations

from pathlib import Path

from .io_utils import sha256_file, write_json


def build_manifest(root: str | Path, output_path: str | Path) -> dict[str, object]:
    root = Path(root)
    include_suffixes = {".py", ".md", ".yaml", ".yml", ".txt", ".json", ".csv", ".npy", ".png", ".log"}
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if ".venv" in path.parts or not path.is_file():
            continue
        if path.suffix.lower() not in include_suffixes:
            continue
        rel = path.relative_to(root).as_posix()
        if rel == Path(output_path).name:
            continue
        files[rel] = sha256_file(path)
    manifest = {"root": str(root), "files": files}
    write_json(output_path, manifest)
    return manifest
