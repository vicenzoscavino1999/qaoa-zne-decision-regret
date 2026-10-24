from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .phase_c_120_kernel_runner import (
    DRY_RUN_NAME,
    OFFICIAL_RUN_NAME,
    SELF_TEST_RUN_NAME,
)


BACKUP_VERSION = "phase_c_120_backup_v1"
DEFAULT_BACKUP_ROOT = Path("C:/phase_c_120_backups")
RUN_NAMES = {
    "official": OFFICIAL_RUN_NAME,
    "dry_run": DRY_RUN_NAME,
    "self_test": SELF_TEST_RUN_NAME,
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _stable_files(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return [
        path
        for path in sorted(base.rglob("*"))
        if path.is_file()
        and not path.name.startswith(".")
        and ".tmp." not in path.name
    ]


def _collect_source_hashes(root: Path, run_name: str) -> dict[str, str]:
    source_dirs = [
        root / "data" / "raw" / run_name,
        root / "results" / run_name,
    ]
    hashes: dict[str, str] = {}
    for base in source_dirs:
        for path in _stable_files(base):
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def backup_phase_c_120_run(
    mode: str,
    backup_root: Path = DEFAULT_BACKUP_ROOT,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if mode not in RUN_NAMES:
        raise ValueError(f"Unknown backup mode: {mode}")
    run_name = RUN_NAMES[mode]
    backup_dir = backup_root / run_name
    ensure_dir(backup_dir)

    source_hashes = _collect_source_hashes(root, run_name)
    copied: dict[str, str] = {}
    for rel, source_hash in sorted(source_hashes.items()):
        source = root / rel
        dest = backup_dir / rel
        ensure_dir(dest.parent)
        shutil.copy2(source, dest)
        copied[rel] = sha256_file(dest)

    missing = sorted(rel for rel in source_hashes if rel not in copied)
    mismatches = sorted(
        rel for rel, digest in source_hashes.items()
        if copied.get(rel) != digest
    )
    verified = bool(source_hashes) and not missing and not mismatches
    report = {
        "backup_version": BACKUP_VERSION,
        "created_at_utc": _utc_now(),
        "mode": mode,
        "run_name": run_name,
        "scope": "verified_file_copy_only_no_scientific_transform",
        "source_root": str(root),
        "backup_dir": str(backup_dir),
        "source_files": len(source_hashes),
        "files_copied": len(copied),
        "missing": missing,
        "mismatches": mismatches,
        "verified": verified,
        "notes": [
            "Temporary atomic-write files are ignored.",
            "Run after each completed shard or after a controlled pause.",
            "This script copies files only and does not modify scientific arrays.",
        ],
    }
    write_json_atomic(backup_dir / "backup_report.json", report, overwrite=True)
    result_dir = root / "results" / run_name
    if result_dir.exists():
        write_json_atomic(result_dir / "backup_report.json", report, overwrite=True)
    return report
