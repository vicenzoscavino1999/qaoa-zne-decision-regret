from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic


PHASE_E_STAGE_C0_SHARD_BALANCE_VERSION = "phase_e_stage_c0_shard_balance_v0_1_draft"
SHARD_RUNTIME_CV_GATE = 0.20
SHARD_MAX_TO_MEDIAN_GATE = 1.50
DEFAULT_NOMINAL_SHARDS = 8
PROTOCOL_MAX_T = 4000
SMOKE_T_ACTUAL_EVALUATIONS = 10
OPTIMIZER_PILOT_CASES = 240
OPTIMIZER_FAMILIES = ("noisy", "zne")
OPTIMIZER_MAX_K = 100
JITTER_CASES = 30
JITTER_DUPLICATE_SEEDS = 3
JITTER_STARTS = 20
RESUME_CASES = 2
RESUME_STARTS = 20


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _projected_seconds_t4000_k1(
    rows: list[dict[str, str]],
    operation: str,
    family: str | None = None,
    default_t_requested: int = 12,
    starts_field: str | None = None,
) -> float:
    values = []
    target_evaluations = actual_spsa_evaluations(PROTOCOL_MAX_T)
    for row in rows:
        if row.get("operation") != operation:
            continue
        if family is not None and row.get("objective_family") != family:
            continue
        try:
            elapsed = float(row["elapsed_seconds"])
        except (KeyError, TypeError, ValueError):
            continue
        observed_t = _as_int(row.get("T_requested", row.get("max_T", row.get("T"))), default_t_requested)
        observed_evaluations = actual_spsa_evaluations(observed_t)
        starts = _as_int(row.get(starts_field), 1) if starts_field is not None else 1
        if observed_evaluations <= 0 or starts <= 0:
            continue
        values.append(float((elapsed / starts) * target_evaluations / observed_evaluations))
    if not values:
        raise ValueError(f"No elapsed_seconds rows for operation={operation!r} family={family!r}.")
    return float(np.mean(values))


def actual_spsa_evaluations(max_evaluations: int) -> int:
    return int(1 + 3 * max(0, (int(max_evaluations) - 1) // 3))


def evaluate_shard_balance(totals: list[float]) -> dict[str, Any]:
    finite = np.asarray([float(value) for value in totals if math.isfinite(float(value))], dtype=float)
    if finite.size == 0:
        return {
            "shard_count": 0,
            "mean_seconds": math.nan,
            "sd_seconds": math.nan,
            "cv": math.inf,
            "median_seconds": math.nan,
            "max_seconds": math.nan,
            "max_to_median": math.inf,
            "passes_shard_balance": False,
        }
    mean = float(np.mean(finite))
    sd = float(np.std(finite, ddof=0))
    median = float(np.median(finite))
    max_value = float(np.max(finite))
    cv = float(sd / mean) if mean > 0.0 else math.inf
    max_to_median = float(max_value / median) if median > 0.0 else math.inf
    return {
        "shard_count": int(finite.size),
        "mean_seconds": mean,
        "sd_seconds": sd,
        "cv": cv,
        "median_seconds": median,
        "max_seconds": max_value,
        "max_to_median": max_to_median,
        "passes_shard_balance": bool(cv < SHARD_RUNTIME_CV_GATE and max_to_median <= SHARD_MAX_TO_MEDIAN_GATE),
    }


def _runtime_templates(root: Path) -> dict[str, Any]:
    benchmark = _read_csv(root / "results" / "phase_e" / "calibration" / "c0_benchmark" / "runtime_profile_by_phase.csv")
    jitter = _read_csv(root / "results" / "phase_e" / "calibration" / "c0_jitter" / "duplicate_endpoint_jitter_runtime.csv")
    resume = _read_csv(root / "results" / "phase_e" / "calibration" / "c0_resume" / "resume_integrity_runtime.csv")
    out: dict[str, Any] = {
        "scale_factor_T12_to_T4000": float(actual_spsa_evaluations(PROTOCOL_MAX_T) / SMOKE_T_ACTUAL_EVALUATIONS),
        "normalization": "observed_elapsed_seconds_normalized_by_actual_spsa_evaluations_and_starts_to_T4000_K1",
        "templates": [],
    }
    for family in OPTIMIZER_FAMILIES:
        checkpointed = _projected_seconds_t4000_k1(benchmark, "checkpointed_spsa", family, default_t_requested=12)
        jitter_elapsed = _projected_seconds_t4000_k1(
            jitter,
            "duplicate_endpoint_optimizer",
            family,
            default_t_requested=12,
            starts_field="starts_per_duplicate",
        )
        resume_elapsed = _projected_seconds_t4000_k1(resume, "checkpoint_write_read_resume_spsa", family, default_t_requested=12)
        out["templates"].extend(
            [
                {
                    "component": "optimizer_checkpointed",
                    "objective_family": family,
                    "observed_seconds_T12_K1": "",
                    "projected_seconds_T4000_K1": checkpointed,
                },
                {
                    "component": "duplicate_endpoint_jitter",
                    "objective_family": family,
                    "observed_seconds_T12_K1": "",
                    "projected_seconds_T4000_K1": jitter_elapsed,
                },
                {
                    "component": "resume_integrity",
                    "objective_family": family,
                    "observed_seconds_T12_K1": "",
                    "projected_seconds_T4000_K1": resume_elapsed,
                },
            ]
        )
    return out


def _template_seconds(templates: dict[str, Any], component: str, family: str) -> float:
    for row in templates["templates"]:
        if row["component"] == component and row["objective_family"] == family:
            return float(row["projected_seconds_T4000_K1"])
    raise KeyError(f"Missing template for {component}/{family}")


def _task_groups(templates: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in OPTIMIZER_FAMILIES:
        rows.append(
            {
                "component": "optimizer_checkpointed",
                "objective_family": family,
                "task_count": int(OPTIMIZER_PILOT_CASES * OPTIMIZER_MAX_K),
                "estimated_seconds_per_task": _template_seconds(templates, "optimizer_checkpointed", family),
            }
        )
    for family in OPTIMIZER_FAMILIES:
        rows.append(
            {
                "component": "duplicate_endpoint_jitter",
                "objective_family": family,
                "task_count": int((JITTER_CASES * JITTER_DUPLICATE_SEEDS * JITTER_STARTS) // len(OPTIMIZER_FAMILIES)),
                "estimated_seconds_per_task": _template_seconds(templates, "duplicate_endpoint_jitter", family),
            }
        )
    for family in OPTIMIZER_FAMILIES:
        rows.append(
            {
                "component": "resume_integrity",
                "objective_family": family,
                "task_count": int(RESUME_CASES * RESUME_STARTS // len(OPTIMIZER_FAMILIES)),
                "estimated_seconds_per_task": _template_seconds(templates, "resume_integrity", family),
            }
        )
    for row in rows:
        row["estimated_total_seconds"] = float(row["task_count"] * row["estimated_seconds_per_task"])
    return rows


def _assign_lpt(task_groups: list[dict[str, Any]], nominal_shards: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shards = [
        {
            "shard_index": index,
            "estimated_seconds": 0.0,
            "task_count": 0,
            "optimizer_checkpointed_tasks": 0,
            "duplicate_endpoint_jitter_tasks": 0,
            "resume_integrity_tasks": 0,
            "noisy_tasks": 0,
            "zne_tasks": 0,
        }
        for index in range(int(nominal_shards))
    ]
    assignment_summary: list[dict[str, Any]] = []
    expanded: list[tuple[float, str, str]] = []
    for group in task_groups:
        expanded.extend(
            [(float(group["estimated_seconds_per_task"]), str(group["component"]), str(group["objective_family"]))]
            * int(group["task_count"])
        )
    expanded.sort(key=lambda item: (-item[0], item[1], item[2]))
    for seconds, component, family in expanded:
        shard = min(shards, key=lambda item: (float(item["estimated_seconds"]), int(item["task_count"]), int(item["shard_index"])))
        shard["estimated_seconds"] = float(shard["estimated_seconds"]) + float(seconds)
        shard["task_count"] = int(shard["task_count"]) + 1
        shard[f"{component}_tasks"] = int(shard[f"{component}_tasks"]) + 1
        shard[f"{family}_tasks"] = int(shard[f"{family}_tasks"]) + 1
    for shard in shards:
        shard["estimated_hours"] = float(shard["estimated_seconds"] / 3600.0)
        assignment_summary.append(dict(shard))
    return assignment_summary, expanded


def run_phase_e_stage_c0_shard_balance(
    root: Path = PROJECT_ROOT,
    force: bool = False,
    nominal_shards: int = DEFAULT_NOMINAL_SHARDS,
) -> dict[str, Any]:
    if int(nominal_shards) < 1:
        raise ValueError("nominal_shards must be positive.")
    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration" / "c0_shard_balance")
    paths = {
        "summary": output_dir / "phase_e_stage_c0_shard_balance_summary.json",
        "balance": output_dir / "shard_runtime_balance.csv",
        "task_groups": output_dir / "shard_task_group_projection.csv",
        "templates": output_dir / "runtime_template_estimates.csv",
        "hashes": output_dir / "phase_e_stage_c0_shard_balance_hashes.json",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase E C0 shard-balance outputs.")

    started = time.perf_counter()
    templates = _runtime_templates(root)
    task_groups = _task_groups(templates)
    shard_rows, expanded = _assign_lpt(task_groups, int(nominal_shards))
    balance = evaluate_shard_balance([float(row["estimated_seconds"]) for row in shard_rows])
    projection_pass = bool(balance["passes_shard_balance"])
    _write_csv(paths["templates"], templates["templates"])
    _write_csv(paths["task_groups"], task_groups)
    _write_csv(paths["balance"], shard_rows)

    full_gate_complete = False
    summary = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "phase_e_stage_c0_shard_balance_projection_passed_not_full_gate" if projection_pass else "phase_e_stage_c0_shard_balance_projection_failed",
        "shard_balance_version": PHASE_E_STAGE_C0_SHARD_BALANCE_VERSION,
        "scope": "protocol_scale_c0_shard_balance_projection_no_parallel_execution",
        "official_execution_authorized": False,
        "full_shard_balance_gate_complete": full_gate_complete,
        "shard_balance_projection_pass": projection_pass,
        "nominal_shards": int(nominal_shards),
        "task_count": len(expanded),
        "projected_total_serial_seconds": float(sum(seconds for seconds, _, _ in expanded)),
        "projected_total_serial_hours": float(sum(seconds for seconds, _, _ in expanded) / 3600.0),
        "projected_wall_seconds": float(balance["max_seconds"]),
        "projected_wall_hours": float(balance["max_seconds"] / 3600.0),
        "shard_runtime_cv": balance["cv"],
        "shard_max_to_median": balance["max_to_median"],
        "shard_mean_hours": float(balance["mean_seconds"] / 3600.0),
        "shard_median_hours": float(balance["median_seconds"] / 3600.0),
        "shard_max_hours": float(balance["max_seconds"] / 3600.0),
        "thresholds": {
            "shard_runtime_cv_lt": SHARD_RUNTIME_CV_GATE,
            "shard_max_to_median_lte": SHARD_MAX_TO_MEDIAN_GATE,
        },
        "runtime_template_scale_factor_T12_to_T4000": templates["scale_factor_T12_to_T4000"],
        "all_checked_numeric_finite": all(
            math.isfinite(float(value))
            for value in (
                balance["cv"],
                balance["max_to_median"],
                balance["mean_seconds"],
                balance["median_seconds"],
                balance["max_seconds"],
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {key: str(path.relative_to(root)).replace("\\", "/") for key, path in paths.items()},
        "limitations": [
            "This is a deterministic shard-balance projection using smoke-measured runtime templates; it is not an actual parallel shard execution.",
            "The full shard-balance gate must be confirmed during the fixed C0 runtime probe or protocol-scale dry run.",
            "No final T, K, K_ref, or N_LHS values are selected.",
        ],
        "next_gate": "runtime",
    }
    write_json_atomic(paths["summary"], summary, overwrite=True)
    hashes = {
        "created_at_utc": _utc_now(),
        "shard_balance_version": PHASE_E_STAGE_C0_SHARD_BALANCE_VERSION,
        "hashes": {key: sha256_file(path) for key, path in paths.items() if key != "hashes"},
    }
    write_json_atomic(paths["hashes"], hashes, overwrite=True)
    summary["output_hashes"] = hashes["hashes"]
    return summary
