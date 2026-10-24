from __future__ import annotations

import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic


PHASE_E_STAGE_C0_STORAGE_VERSION = "phase_e_stage_c0_storage_v0_1_draft"
DISK_POST_RUN_USAGE_GATE_FRACTION = 0.80
DISK_FREE_SPACE_USE_GATE_FRACTION = 0.50
DEFAULT_NOMINAL_SHARDS = 8
DEFAULT_BACKUP_OVERHEAD_FACTOR = 1.0
DEFAULT_CHECKPOINT_SAFETY_FACTOR = 2.0
DEFAULT_TABULAR_SAFETY_FACTOR = 4.0
DEFAULT_JSON_SAFETY_FACTOR = 4.0
DEFAULT_LOG_BYTES_PER_SHARD = 64 * 1024 * 1024
DEFAULT_HASH_BYTES = 4 * 1024 * 1024
DEFAULT_OPTIMIZER_PILOT_CASES = 240
DEFAULT_OPTIMIZER_OBJECTIVE_FAMILIES = 2
DEFAULT_OPTIMIZER_MAX_K = 100
DEFAULT_T_CHECKPOINT_COUNT = 5
DEFAULT_JITTER_CASES = 30
DEFAULT_JITTER_DUPLICATE_SEEDS = 3
DEFAULT_JITTER_STARTS = 20
DEFAULT_RESUME_CASES = 2
DEFAULT_RESUME_STARTS = 20
DEFAULT_LHS_PILOT_CASES = 160
DEFAULT_N_LHS_CANDIDATES = 5


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _iter_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    if path.is_file():
        return [path]
    return [item for item in path.rglob("*") if item.is_file()]


def _inventory_path(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    files = _iter_files(path)
    total_bytes = int(sum(item.stat().st_size for item in files))
    return {
        "relative_path": relative_path,
        "exists": path.exists(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "csv_bytes": int(sum(item.stat().st_size for item in files if item.suffix.lower() == ".csv")),
        "json_bytes": int(sum(item.stat().st_size for item in files if item.suffix.lower() == ".json")),
        "checkpoint_bytes": int(sum(item.stat().st_size for item in files if "checkpoint" in str(item).replace("\\", "/").lower())),
    }


def _observed_inventory(root: Path) -> list[dict[str, Any]]:
    paths = [
        "results/phase_e/calibration/c0_benchmark",
        "results/phase_e/calibration/c0_jitter",
        "results/phase_e/calibration/c0_resume",
        "logs/phase_e",
    ]
    return [_inventory_path(root, relative_path) for relative_path in paths]


def _count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def _observed_csv_stats(root: Path) -> dict[str, Any]:
    csv_files = [
        path
        for relative_path in (
            "results/phase_e/calibration/c0_benchmark",
            "results/phase_e/calibration/c0_jitter",
            "results/phase_e/calibration/c0_resume",
        )
        for path in _iter_files(root / relative_path)
        if path.suffix.lower() == ".csv"
    ]
    rows = int(sum(_count_csv_rows(path) for path in csv_files))
    bytes_total = int(sum(path.stat().st_size for path in csv_files))
    return {
        "csv_file_count": len(csv_files),
        "csv_rows": rows,
        "csv_bytes": bytes_total,
        "csv_bytes_per_row": float(bytes_total / rows) if rows > 0 else 1024.0,
    }


def _checkpoint_sample_stats(root: Path) -> dict[str, Any]:
    checkpoint_files = _iter_files(root / "results" / "phase_e" / "calibration" / "c0_resume" / "checkpoints")
    sizes = [int(path.stat().st_size) for path in checkpoint_files if path.suffix.lower() == ".json"]
    if not sizes:
        sizes = [4096]
    return {
        "sample_checkpoint_files": len(checkpoint_files),
        "checkpoint_bytes_min": int(min(sizes)),
        "checkpoint_bytes_mean": float(np.mean(sizes)),
        "checkpoint_bytes_p95": float(np.percentile(sizes, 95)),
        "checkpoint_bytes_max": int(max(sizes)),
    }


def evaluate_storage_thresholds(total_bytes: int, used_bytes: int, free_bytes: int, required_bytes: int) -> dict[str, Any]:
    projected_used_bytes = int(used_bytes + required_bytes)
    required_fraction_of_free = float(required_bytes / free_bytes) if free_bytes > 0 else math.inf
    projected_used_fraction = float(projected_used_bytes / total_bytes) if total_bytes > 0 else math.inf
    return {
        "required_bytes": int(required_bytes),
        "projected_used_bytes": projected_used_bytes,
        "required_fraction_of_free": required_fraction_of_free,
        "projected_used_fraction": projected_used_fraction,
        "passes_required_le_50pct_free": bool(required_fraction_of_free <= DISK_FREE_SPACE_USE_GATE_FRACTION),
        "passes_projected_used_le_80pct_total": bool(projected_used_fraction <= DISK_POST_RUN_USAGE_GATE_FRACTION),
        "passes_storage_thresholds": bool(
            required_fraction_of_free <= DISK_FREE_SPACE_USE_GATE_FRACTION
            and projected_used_fraction <= DISK_POST_RUN_USAGE_GATE_FRACTION
        ),
    }


def _disk_report(path: Path, required_bytes: int) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    thresholds = evaluate_storage_thresholds(
        total_bytes=int(usage.total),
        used_bytes=int(usage.used),
        free_bytes=int(usage.free),
        required_bytes=int(required_bytes),
    )
    return {
        "path": str(path),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        **thresholds,
    }


def _projected_counts() -> dict[str, int]:
    optimizer_checkpoint_files = DEFAULT_OPTIMIZER_PILOT_CASES * DEFAULT_OPTIMIZER_OBJECTIVE_FAMILIES * DEFAULT_OPTIMIZER_MAX_K
    jitter_checkpoint_files = DEFAULT_JITTER_CASES * DEFAULT_JITTER_DUPLICATE_SEEDS * DEFAULT_JITTER_STARTS
    resume_checkpoint_files = DEFAULT_RESUME_CASES * DEFAULT_RESUME_STARTS
    optimizer_endpoint_rows = optimizer_checkpoint_files * DEFAULT_T_CHECKPOINT_COUNT
    nested_k_rows = DEFAULT_OPTIMIZER_PILOT_CASES * DEFAULT_OPTIMIZER_OBJECTIVE_FAMILIES * 3 * DEFAULT_T_CHECKPOINT_COUNT
    jitter_endpoint_rows = DEFAULT_JITTER_CASES * DEFAULT_JITTER_DUPLICATE_SEEDS * DEFAULT_JITTER_STARTS
    jitter_pairwise_rows = DEFAULT_JITTER_CASES * 3 * DEFAULT_JITTER_STARTS
    resume_rows = DEFAULT_RESUME_CASES * DEFAULT_RESUME_STARTS
    lhs_rows = DEFAULT_LHS_PILOT_CASES * DEFAULT_N_LHS_CANDIDATES
    return {
        "optimizer_checkpoint_files": int(optimizer_checkpoint_files),
        "jitter_checkpoint_files": int(jitter_checkpoint_files),
        "resume_checkpoint_files": int(resume_checkpoint_files),
        "total_checkpoint_files": int(optimizer_checkpoint_files + jitter_checkpoint_files + resume_checkpoint_files),
        "optimizer_endpoint_rows": int(optimizer_endpoint_rows),
        "nested_k_rows": int(nested_k_rows),
        "jitter_endpoint_rows": int(jitter_endpoint_rows),
        "jitter_pairwise_rows": int(jitter_pairwise_rows),
        "resume_rows": int(resume_rows),
        "lhs_rows": int(lhs_rows),
        "total_projected_tabular_rows": int(
            optimizer_endpoint_rows + nested_k_rows + jitter_endpoint_rows + jitter_pairwise_rows + resume_rows + lhs_rows
        ),
    }


def _component_projection(
    checkpoint_stats: dict[str, Any],
    csv_stats: dict[str, Any],
    observed_json_bytes: int,
    projected_counts: dict[str, int],
    nominal_shards: int,
) -> list[dict[str, Any]]:
    checkpoint_bytes_per_file = max(float(checkpoint_stats["checkpoint_bytes_p95"]), 4096.0)
    checkpoint_bytes = int(
        math.ceil(checkpoint_bytes_per_file * projected_counts["total_checkpoint_files"] * DEFAULT_CHECKPOINT_SAFETY_FACTOR)
    )
    tabular_bytes = int(
        math.ceil(float(csv_stats["csv_bytes_per_row"]) * projected_counts["total_projected_tabular_rows"] * DEFAULT_TABULAR_SAFETY_FACTOR)
    )
    json_bytes_observed = max(1, int(observed_json_bytes))
    json_bytes = int(math.ceil(json_bytes_observed * DEFAULT_JSON_SAFETY_FACTOR))
    log_bytes = int(DEFAULT_LOG_BYTES_PER_SHARD * int(nominal_shards))
    hash_bytes = int(DEFAULT_HASH_BYTES)
    rows = [
        {
            "component": "checkpoint_state_files",
            "basis": "p95 observed resume checkpoint bytes times projected C0 checkpoint files",
            "unit_count": projected_counts["total_checkpoint_files"],
            "bytes_per_unit": checkpoint_bytes_per_file,
            "safety_factor": DEFAULT_CHECKPOINT_SAFETY_FACTOR,
            "projected_bytes": checkpoint_bytes,
        },
        {
            "component": "tabular_outputs",
            "basis": "observed CSV bytes per row times projected C0 diagnostic rows",
            "unit_count": projected_counts["total_projected_tabular_rows"],
            "bytes_per_unit": float(csv_stats["csv_bytes_per_row"]),
            "safety_factor": DEFAULT_TABULAR_SAFETY_FACTOR,
            "projected_bytes": tabular_bytes,
        },
        {
            "component": "summary_json_outputs",
            "basis": "observed C0 JSON outputs with safety factor",
            "unit_count": 1,
            "bytes_per_unit": json_bytes_observed,
            "safety_factor": DEFAULT_JSON_SAFETY_FACTOR,
            "projected_bytes": json_bytes,
        },
        {
            "component": "logs",
            "basis": "fixed log budget per nominal shard",
            "unit_count": int(nominal_shards),
            "bytes_per_unit": DEFAULT_LOG_BYTES_PER_SHARD,
            "safety_factor": 1.0,
            "projected_bytes": log_bytes,
        },
        {
            "component": "hashes_and_manifests",
            "basis": "fixed hash/manifest budget",
            "unit_count": 1,
            "bytes_per_unit": DEFAULT_HASH_BYTES,
            "safety_factor": 1.0,
            "projected_bytes": hash_bytes,
        },
    ]
    primary_bytes = int(sum(int(row["projected_bytes"]) for row in rows))
    rows.append(
        {
            "component": "backup_overhead",
            "basis": "one full copy of projected primary C0 artifacts",
            "unit_count": 1,
            "bytes_per_unit": primary_bytes,
            "safety_factor": DEFAULT_BACKUP_OVERHEAD_FACTOR,
            "projected_bytes": int(math.ceil(primary_bytes * DEFAULT_BACKUP_OVERHEAD_FACTOR)),
        }
    )
    return rows


def run_phase_e_stage_c0_storage(
    root: Path = PROJECT_ROOT,
    force: bool = False,
    nominal_shards: int = DEFAULT_NOMINAL_SHARDS,
) -> dict[str, Any]:
    if int(nominal_shards) < 1:
        raise ValueError("nominal_shards must be positive.")

    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration" / "c0_storage")
    paths = {
        "summary": output_dir / "phase_e_stage_c0_storage_summary.json",
        "projection": output_dir / "storage_projection.json",
        "components": output_dir / "storage_component_projection.csv",
        "inventory": output_dir / "observed_c0_storage_inventory.csv",
        "hashes": output_dir / "phase_e_stage_c0_storage_hashes.json",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase E C0 storage outputs.")

    started = time.perf_counter()
    inventory_rows = _observed_inventory(root)
    checkpoint_stats = _checkpoint_sample_stats(root)
    csv_stats = _observed_csv_stats(root)
    observed_json_bytes = int(sum(row["json_bytes"] for row in inventory_rows))
    projected_counts = _projected_counts()
    component_rows = _component_projection(checkpoint_stats, csv_stats, observed_json_bytes, projected_counts, int(nominal_shards))
    primary_required_bytes = int(sum(int(row["projected_bytes"]) for row in component_rows if row["component"] != "backup_overhead"))
    backup_required_bytes = int(sum(int(row["projected_bytes"]) for row in component_rows if row["component"] == "backup_overhead"))
    total_required_bytes = int(primary_required_bytes + backup_required_bytes)
    checkpoint_bytes_total = int(next(row["projected_bytes"] for row in component_rows if row["component"] == "checkpoint_state_files"))
    checkpoint_bytes_per_shard = int(math.ceil(checkpoint_bytes_total / int(nominal_shards)))
    project_disk = _disk_report(root, total_required_bytes)
    backup_root = Path("C:/") if Path("C:/").exists() else root
    backup_disk = _disk_report(backup_root, backup_required_bytes)
    components_present = {
        "checkpoints": checkpoint_bytes_total > 0,
        "outputs": any(row["component"] in ("tabular_outputs", "summary_json_outputs") and int(row["projected_bytes"]) > 0 for row in component_rows),
        "logs": any(row["component"] == "logs" and int(row["projected_bytes"]) > 0 for row in component_rows),
        "hashes": any(row["component"] == "hashes_and_manifests" and int(row["projected_bytes"]) > 0 for row in component_rows),
        "backup": backup_required_bytes > 0,
    }
    all_components_included = all(components_present.values())
    storage_gate_pass = bool(project_disk["passes_storage_thresholds"] and all_components_included)
    projection = {
        "created_at_utc": _utc_now(),
        "storage_version": PHASE_E_STAGE_C0_STORAGE_VERSION,
        "status": "pass" if storage_gate_pass else "fail",
        "scope": "protocol_scale_c0_storage_projection_no_optimizer_execution",
        "official_execution_authorized": False,
        "full_storage_gate_complete": True,
        "storage_gate_pass": storage_gate_pass,
        "nominal_shards": int(nominal_shards),
        "checkpoint_bytes_per_shard": checkpoint_bytes_per_shard,
        "checkpoint_bytes_total": checkpoint_bytes_total,
        "primary_required_bytes": primary_required_bytes,
        "backup_required_bytes": backup_required_bytes,
        "total_required_bytes_including_backup": total_required_bytes,
        "components_present": components_present,
        "all_required_components_included": all_components_included,
        "observed_inventory": inventory_rows,
        "checkpoint_sample_stats": checkpoint_stats,
        "csv_sample_stats": csv_stats,
        "projected_counts": projected_counts,
        "component_projection": component_rows,
        "project_disk_report": project_disk,
        "backup_disk_report_diagnostic": backup_disk,
        "thresholds": {
            "projected_post_run_disk_use_max_fraction": DISK_POST_RUN_USAGE_GATE_FRACTION,
            "required_bytes_max_fraction_of_free_space": DISK_FREE_SPACE_USE_GATE_FRACTION,
        },
        "limitations": [
            "Storage projection does not execute Stage C or the full optimizer workload.",
            "Checkpoint byte estimates are based on observed resume checkpoint JSON sizes with a safety factor.",
            "The project-disk gate conservatively includes backup overhead in the required bytes.",
        ],
    }
    _write_csv(paths["components"], component_rows)
    _write_csv(paths["inventory"], inventory_rows)
    write_json_atomic(paths["projection"], projection, overwrite=True)

    summary = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_stage_c0_storage_projection_passed" if storage_gate_pass else "phase_e_stage_c0_storage_projection_failed",
        "storage_version": PHASE_E_STAGE_C0_STORAGE_VERSION,
        "scope": projection["scope"],
        "official_execution_authorized": False,
        "full_storage_gate_complete": True,
        "storage_gate_pass": storage_gate_pass,
        "nominal_shards": int(nominal_shards),
        "checkpoint_bytes_per_shard": checkpoint_bytes_per_shard,
        "checkpoint_bytes_total": checkpoint_bytes_total,
        "primary_required_bytes": primary_required_bytes,
        "backup_required_bytes": backup_required_bytes,
        "total_required_bytes_including_backup": total_required_bytes,
        "project_disk_projected_used_fraction": project_disk["projected_used_fraction"],
        "project_disk_required_fraction_of_free": project_disk["required_fraction_of_free"],
        "all_required_components_included": all_components_included,
        "all_checked_numeric_finite": all(
            math.isfinite(float(value))
            for value in (
                checkpoint_bytes_per_shard,
                checkpoint_bytes_total,
                primary_required_bytes,
                backup_required_bytes,
                total_required_bytes,
                project_disk["projected_used_fraction"],
                project_disk["required_fraction_of_free"],
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {key: str(path.relative_to(root)).replace("\\", "/") for key, path in paths.items()},
        "next_gate": "shard_balance",
    }
    write_json_atomic(paths["summary"], summary, overwrite=True)
    hashes = {
        "created_at_utc": _utc_now(),
        "storage_version": PHASE_E_STAGE_C0_STORAGE_VERSION,
        "hashes": {key: sha256_file(path) for key, path in paths.items() if key != "hashes"},
    }
    write_json_atomic(paths["hashes"], hashes, overwrite=True)
    summary["output_hashes"] = hashes["hashes"]
    return summary
