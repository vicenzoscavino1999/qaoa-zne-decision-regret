from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic


PHASE_E_STAGE_C0_RUNTIME_VERSION = "phase_e_stage_c0_runtime_v0_1_draft"
ACCEPTED_LOCAL_WALL_HOURS = 72.0
CONSERVATIVE_EFFECTIVE_SHARD_CAP = 7.0
EXACT_NESTED_K_FACTOR = 1.7
EXACT_CHECKPOINT_FACTOR_SUM_T_OVER_MAX_T = 1.9375


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_runtime_gate(projected_wall_hours: float, accepted_wall_hours: float = ACCEPTED_LOCAL_WALL_HOURS) -> dict[str, Any]:
    projected = float(projected_wall_hours)
    accepted = float(accepted_wall_hours)
    return {
        "projected_wall_hours": projected,
        "accepted_wall_hours": accepted,
        "runtime_gate_pass": bool(math.isfinite(projected) and projected <= accepted),
        "wall_hours_over_budget": float(projected - accepted) if math.isfinite(projected) else math.inf,
        "budget_multiplier": float(projected / accepted) if accepted > 0.0 and math.isfinite(projected) else math.inf,
    }


def _calibration_baseline(root: Path) -> float:
    status_path = root / "results" / "phase_e" / "calibration" / "phase_e_calibration_status.json"
    payload = _read_json(status_path)
    runtime = payload.get("runtime_projection_summary", {})
    if "protocol_T_candidate_total_serial_hours_no_reuse" in runtime:
        return float(runtime["protocol_T_candidate_total_serial_hours_no_reuse"])
    return float(payload["next_stage"]["estimated_or_actual_hours"])


def _plan_accounting(root: Path) -> dict[str, Any]:
    path = root / "results" / "phase_e" / "calibration" / "phase_e_stage_c0_optimization_plan.json"
    payload = _read_json(path)
    accounting = payload.get("runtime_accounting", {})
    external = accounting.get("external_feedback_target", {})
    return {
        "baseline_stage_c_T_only_serial_hours": float(
            accounting.get("baseline_stage_c_T_only_serial_hours", _calibration_baseline(root))
        ),
        "accepted_local_wall_hours": float(accounting.get("accepted_local_wall_hours", ACCEPTED_LOCAL_WALL_HOURS)),
        "exact_checkpoint_factor": float(accounting.get("exact_checkpoint_factor_sum_T_over_max_T", EXACT_CHECKPOINT_FACTOR_SUM_T_OVER_MAX_T)),
        "exact_nested_k_factor": float(accounting.get("exact_K_nested_factor_sum_K_over_max_K", EXACT_NESTED_K_FACTOR)),
        "external_target_wall_hours": float(external.get("target_wall_hours", 66.51533349559897)),
        "external_checkpoint_factor_assumed": float(external.get("checkpoint_factor_assumed", 4.0)),
        "external_k_factor_assumed": float(external.get("K_nested_factor_assumed", 1.8)),
    }


def _checkpoint_factor_stats(root: Path) -> dict[str, Any]:
    rows = _read_csv(root / "results" / "phase_e" / "calibration" / "c0_benchmark" / "checkpoint_factor_measurements.csv")
    values = []
    for row in rows:
        try:
            value = float(row["measured_checkpoint_factor"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise ValueError("No finite checkpoint factor measurements available.")
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _shard_accounting(root: Path) -> dict[str, Any]:
    path = root / "results" / "phase_e" / "calibration" / "c0_shard_balance" / "phase_e_stage_c0_shard_balance_summary.json"
    payload = _read_json(path)
    serial = float(payload["projected_total_serial_hours"])
    wall = float(payload["projected_wall_hours"])
    nominal = int(payload["nominal_shards"])
    effective = float(serial / wall) if wall > 0.0 else math.inf
    used = float(min(effective, CONSERVATIVE_EFFECTIVE_SHARD_CAP))
    return {
        "nominal_shards": nominal,
        "projected_total_serial_hours_from_templates": serial,
        "projected_wall_hours_from_templates": wall,
        "effective_shards_from_balance_projection": effective,
        "effective_shards_used_conservative": used,
        "shard_efficiency_from_balance_projection": float(effective / nominal) if nominal > 0 else math.nan,
        "shard_efficiency_used_conservative": float(used / nominal) if nominal > 0 else math.nan,
        "shard_runtime_cv": float(payload["shard_runtime_cv"]),
        "shard_max_to_median": float(payload["shard_max_to_median"]),
        "shard_balance_projection_pass": bool(payload["shard_balance_projection_pass"]),
        "full_shard_balance_gate_complete": bool(payload["full_shard_balance_gate_complete"]),
    }


def _gate_statuses(root: Path) -> list[dict[str, Any]]:
    def load(relative: str) -> dict[str, Any] | None:
        path = root / relative
        return _read_json(path) if path.exists() else None

    jitter = load("results/phase_e/calibration/c0_jitter/phase_e_stage_c0_jitter_summary.json")
    resume = load("results/phase_e/calibration/c0_resume/phase_e_stage_c0_resume_summary.json")
    storage = load("results/phase_e/calibration/c0_storage/phase_e_stage_c0_storage_summary.json")
    shard = load("results/phase_e/calibration/c0_shard_balance/phase_e_stage_c0_shard_balance_summary.json")
    rows = [
        {
            "order": 1,
            "gate": "duplicate_endpoint_jitter",
            "complete": bool(jitter and jitter.get("full_jitter_gate_complete") is True),
            "pass": bool(jitter and jitter.get("full_jitter_gate_complete") is True and jitter.get("threshold_classification") in ("pass", "warning")),
            "status": None if jitter is None else jitter.get("status"),
        },
        {
            "order": 2,
            "gate": "resume_integrity",
            "complete": bool(resume and resume.get("full_resume_gate_complete") is True),
            "pass": bool(resume and resume.get("full_resume_gate_complete") is True and resume.get("resume_integrity_status") == "pass"),
            "status": None if resume is None else resume.get("status"),
        },
        {
            "order": 3,
            "gate": "storage",
            "complete": bool(storage and storage.get("full_storage_gate_complete") is True),
            "pass": bool(storage and storage.get("full_storage_gate_complete") is True and storage.get("storage_gate_pass") is True),
            "status": None if storage is None else storage.get("status"),
        },
        {
            "order": 4,
            "gate": "shard_balance",
            "complete": bool(shard and shard.get("full_shard_balance_gate_complete") is True),
            "pass": bool(shard and shard.get("full_shard_balance_gate_complete") is True and shard.get("shard_balance_projection_pass") is True),
            "status": None if shard is None else shard.get("status"),
        },
    ]
    return rows


def _runtime_projection_rows(accounting: dict[str, Any], checkpoint: dict[str, Any], shard: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = float(accounting["baseline_stage_c_T_only_serial_hours"])
    accepted = float(accounting["accepted_local_wall_hours"])
    effective = float(shard["effective_shards_used_conservative"])
    measured_checkpoint = float(checkpoint["mean"])
    exact_checkpoint = float(accounting["exact_checkpoint_factor"])
    exact_k = float(accounting["exact_nested_k_factor"])
    external_checkpoint = float(accounting["external_checkpoint_factor_assumed"])
    external_k = float(accounting["external_k_factor_assumed"])
    rows = [
        {
            "model": "measured_checkpoint_plus_exact_nested_K_conservative_shards",
            "baseline_serial_hours": baseline,
            "checkpoint_factor": measured_checkpoint,
            "nested_K_factor": exact_k,
            "effective_shards": effective,
            "projected_wall_hours": baseline / (measured_checkpoint * exact_k * effective),
            "interpretation": "Optimistic combined-credit model requested by C0 accounting; checkpoint factor comes from smoke benchmark.",
        },
        {
            "model": "measured_checkpoint_no_K_credit_conservative_shards",
            "baseline_serial_hours": baseline,
            "checkpoint_factor": measured_checkpoint,
            "nested_K_factor": 1.0,
            "effective_shards": effective,
            "projected_wall_hours": baseline / (measured_checkpoint * effective),
            "interpretation": "Conservative model because the 3352h baseline is already Stage-C T-only at fixed K.",
        },
        {
            "model": "exact_checkpoint_plus_exact_nested_K_conservative_shards",
            "baseline_serial_hours": baseline,
            "checkpoint_factor": exact_checkpoint,
            "nested_K_factor": exact_k,
            "effective_shards": effective,
            "projected_wall_hours": baseline / (exact_checkpoint * exact_k * effective),
            "interpretation": "Strict accounting from frozen T and K grids, not measured speedup.",
        },
        {
            "model": "external_aggressive_target_assumptions",
            "baseline_serial_hours": baseline,
            "checkpoint_factor": external_checkpoint,
            "nested_K_factor": external_k,
            "effective_shards": effective,
            "projected_wall_hours": baseline / (external_checkpoint * external_k * effective),
            "interpretation": "External-feedback aggressive target, reported as sensitivity only.",
        },
        {
            "model": "task_template_projection_from_smoke_timings",
            "baseline_serial_hours": float(shard["projected_total_serial_hours_from_templates"]),
            "checkpoint_factor": 1.0,
            "nested_K_factor": 1.0,
            "effective_shards": float(shard["effective_shards_from_balance_projection"]),
            "projected_wall_hours": float(shard["projected_wall_hours_from_templates"]),
            "interpretation": "Direct task-template projection from smoke timings scaled to T=4000; not a full runtime probe.",
        },
    ]
    for row in rows:
        decision = evaluate_runtime_gate(float(row["projected_wall_hours"]), accepted)
        row.update(
            {
                "accepted_wall_hours": accepted,
                "runtime_gate_pass": decision["runtime_gate_pass"],
                "wall_hours_over_budget": decision["wall_hours_over_budget"],
                "budget_multiplier": decision["budget_multiplier"],
            }
        )
    return rows


def run_phase_e_stage_c0_runtime(
    root: Path = PROJECT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration" / "c0_runtime")
    paths = {
        "summary": output_dir / "phase_e_stage_c0_runtime_summary.json",
        "projection": output_dir / "runtime_projection_summary.csv",
        "gate_composition": output_dir / "gate_composition_decision.json",
        "hashes": output_dir / "phase_e_stage_c0_runtime_hashes.json",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase E C0 runtime outputs.")

    started = time.perf_counter()
    accounting = _plan_accounting(root)
    checkpoint = _checkpoint_factor_stats(root)
    shard = _shard_accounting(root)
    projection_rows = _runtime_projection_rows(accounting, checkpoint, shard)
    _write_csv(paths["projection"], projection_rows)

    primary = next(row for row in projection_rows if row["model"] == "measured_checkpoint_plus_exact_nested_K_conservative_shards")
    conservative = next(row for row in projection_rows if row["model"] == "measured_checkpoint_no_K_credit_conservative_shards")
    template = next(row for row in projection_rows if row["model"] == "task_template_projection_from_smoke_timings")
    full_runtime_gate_complete = False
    runtime_gate_pass = bool(primary["runtime_gate_pass"] and full_runtime_gate_complete)
    gate_rows = _gate_statuses(root)
    gate_rows.append(
        {
            "order": 5,
            "gate": "runtime",
            "complete": full_runtime_gate_complete,
            "pass": runtime_gate_pass,
            "status": "projection_failed" if not primary["runtime_gate_pass"] else "projection_passed_not_full_gate",
        }
    )
    strict_and_pass = all(bool(row["complete"]) and bool(row["pass"]) for row in gate_rows)
    gate_decision = {
        "created_at_utc": _utc_now(),
        "runtime_version": PHASE_E_STAGE_C0_RUNTIME_VERSION,
        "strict_AND": True,
        "stage_c_authorized": False,
        "stage_c_blocked": not strict_and_pass,
        "ordered_gates": gate_rows,
        "first_blocking_gate": next((row["gate"] for row in gate_rows if not (bool(row["complete"]) and bool(row["pass"]))), None),
        "interpretation": "No gate may be relaxed post-hoc; failed or incomplete gates require amendment or re-execution before Stage C.",
    }
    write_json_atomic(paths["gate_composition"], gate_decision, overwrite=True)

    summary = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_stage_c0_runtime_projection_failed_stage_c_blocked"
        if not primary["runtime_gate_pass"]
        else "phase_e_stage_c0_runtime_projection_passed_not_full_gate",
        "runtime_version": PHASE_E_STAGE_C0_RUNTIME_VERSION,
        "scope": "c0_runtime_projection_no_protocol_scale_execution",
        "official_execution_authorized": False,
        "full_runtime_gate_complete": full_runtime_gate_complete,
        "runtime_gate_pass": runtime_gate_pass,
        "runtime_projection_pass_under_primary_model": bool(primary["runtime_gate_pass"]),
        "accepted_local_wall_hours": accounting["accepted_local_wall_hours"],
        "baseline_stage_c_T_only_serial_hours": accounting["baseline_stage_c_T_only_serial_hours"],
        "measured_checkpoint_factor_mean": checkpoint["mean"],
        "measured_checkpoint_factor_min": checkpoint["min"],
        "measured_checkpoint_factor_max": checkpoint["max"],
        "exact_nested_K_factor": accounting["exact_nested_k_factor"],
        "effective_shards_from_balance_projection": shard["effective_shards_from_balance_projection"],
        "effective_shards_used_conservative": shard["effective_shards_used_conservative"],
        "shard_efficiency_used_conservative": shard["shard_efficiency_used_conservative"],
        "primary_projected_wall_hours": primary["projected_wall_hours"],
        "primary_wall_hours_over_budget": primary["wall_hours_over_budget"],
        "primary_budget_multiplier": primary["budget_multiplier"],
        "conservative_no_K_projected_wall_hours": conservative["projected_wall_hours"],
        "task_template_projected_wall_hours": template["projected_wall_hours"],
        "external_target_wall_hours": accounting["external_target_wall_hours"],
        "stage_c_blocked": True,
        "recommended_next_action": "write_runtime_or_scope_amendment_before_stage_c",
        "all_checked_numeric_finite": all(
            math.isfinite(float(value))
            for value in (
                checkpoint["mean"],
                checkpoint["min"],
                checkpoint["max"],
                primary["projected_wall_hours"],
                conservative["projected_wall_hours"],
                template["projected_wall_hours"],
                shard["effective_shards_used_conservative"],
            )
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {key: str(path.relative_to(root)).replace("\\", "/") for key, path in paths.items()},
        "limitations": [
            "This is a runtime projection from existing C0 smoke/projection artifacts, not a fixed C0 runtime probe.",
            "The full runtime gate remains incomplete until a pre-specified C0 runtime probe or accepted amendment is executed.",
            "No final T, K, K_ref, or N_LHS values are selected.",
        ],
    }
    write_json_atomic(paths["summary"], summary, overwrite=True)
    hashes = {
        "created_at_utc": _utc_now(),
        "runtime_version": PHASE_E_STAGE_C0_RUNTIME_VERSION,
        "hashes": {key: sha256_file(path) for key, path in paths.items() if key != "hashes"},
    }
    write_json_atomic(paths["hashes"], hashes, overwrite=True)
    summary["output_hashes"] = hashes["hashes"]
    return summary
