from __future__ import annotations

import contextlib
import compileall
import csv
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from .circuits import GraphInstance
from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .mitiq_wrapper import allocate_shots, deterministic_seed, richardson_weights
from .phase_c_120_custom_kernel import (
    _apply_local_depolarizing,
    _apply_local_depolarizing_reference,
    _apply_local_unitary_density,
    _apply_local_unitary_density_reference,
    _gate_matrix,
    _qargs_to_indices,
)
from .phase_e_optimizers import COBYLAConfig, cobyla_minimize, make_paired_initial_thetas
from .phase_e_sim_backend import (
    PHASE_E_BACKEND_VERSION,
    PhaseENoiseRates,
    build_folded_transpiled_qaoa_depth,
    evaluate_lhs_point_cross_order_probability_cache,
    evaluate_lhs_point_with_scale1_probability_reuse,
    evaluate_custom_density_objective,
    richardson_zne_objective,
    theta_periods,
    wrap_theta,
)


COBYLA_RUNTIME_OPTIMIZATION_VERSION = "phase_e_cobyla_runtime_optimization_v0_1"
OUTPUT_DIR_RELATIVE = Path("results") / "phase_e" / "calibration" / "cobyla_runtime_optimization"

PLAN_FILE = "PHASE_E_COBYLA_RUNTIME_OPTIMIZATION_AND_CHUNKED_EXECUTION_PLAN_v0.1-draft.md"
PROTOCOL_FILE = "PHASE_E_COBYLA_ENDPOINT_PROTOCOL_v0.1-draft.md"
COMPUTE_FILE = "computacion.md"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _output_dir(root: Path) -> Path:
    return ensure_dir(root / OUTPUT_DIR_RELATIVE)


def _run_capture(args: list[str], root: Path, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(args, cwd=root, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "args": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"args": args, "returncode": None, "stdout": "", "stderr": repr(exc)}


def _git_status(root: Path) -> dict[str, Any]:
    branch = _run_capture(["git", "branch", "--show-current"], root)
    status = _run_capture(["git", "status", "--short", "--branch"], root)
    head = _run_capture(["git", "rev-parse", "HEAD"], root)
    return {
        "branch": branch["stdout"].strip(),
        "head": head["stdout"].strip(),
        "status_text": status["stdout"],
        "status_returncode": status["returncode"],
    }


def _hash_inventory_path(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.exists():
        return {"path": relative_path, "exists": False, "kind": "missing", "bytes": 0, "sha256": ""}
    if path.is_file():
        return {
            "path": relative_path,
            "exists": True,
            "kind": "file",
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if path.is_dir():
        records = []
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            records.append(
                {
                    "path": str(child.relative_to(root)).replace("\\", "/"),
                    "bytes": child.stat().st_size,
                    "sha256": sha256_file(child),
                }
            )
        return {
            "path": relative_path,
            "exists": True,
            "kind": "directory",
            "bytes": sum(int(row["bytes"]) for row in records),
            "sha256": _canonical_hash(records),
            "file_count": len(records),
        }
    return {"path": relative_path, "exists": True, "kind": "other", "bytes": 0, "sha256": ""}


def _dirty_inventory(root: Path, status_text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in status_text.splitlines():
        if not raw or raw.startswith("## "):
            continue
        status = raw[:2].strip()
        path_text = raw[3:].strip()
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        rows.append({"git_status": status, **_hash_inventory_path(root, path_text)})
    return rows


def _hashes_for_existing(root: Path, relatives: list[str]) -> dict[str, str]:
    return {relative: sha256_file(root / relative) for relative in relatives if (root / relative).exists()}


def _common_inputs(root: Path) -> dict[str, Any]:
    config = _read_json(root / "config" / "phase_e_config_v0.1-draft.json")
    calibration = _read_json(root / "results" / "phase_e" / "calibration" / "phase_e_calibration_status.json")
    freeze_gate = _read_json(root / "results" / "phase_e" / "calibration" / "freeze_gates" / "phase_e_bias_variance_runtime_gate.json")
    return {"config": config, "calibration": calibration, "freeze_gate": freeze_gate}


def _graph_from_selected_instance(root: Path, label: str) -> GraphInstance:
    payload = _read_json(root / "config" / "phase_e_instances_v0.1-draft.json")
    instance = payload["selected_instances"][label]
    edges = tuple((int(u), int(v), float(w)) for u, v, w in instance["edges"])
    return GraphInstance(name=label, edges=edges)


def _selected_values(config: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    optimizer = config["optimizer"]
    return {
        "E_COBYLA": int(optimizer.get("selected_cobyla_evaluation_budget") or calibration["selected_cobyla_evaluation_budget"]),
        "K": int(optimizer.get("selected_random_starts") or calibration["selected_random_start_count"]),
        "N_LHS": int(optimizer.get("selected_lhs_size") or calibration["selected_N_LHS"]),
        "reference_strategy": optimizer.get("selected_reference_strategy") or calibration.get("selected_reference_strategy"),
    }


def _expected_schema() -> dict[str, Any]:
    fields = [
        ("task_id", "string"),
        ("instance", "string"),
        ("p_dep", "float"),
        ("shots_total", "integer"),
        ("richardson_order", "string"),
        ("seed", "integer"),
        ("lhs_design_id", "integer"),
        ("K", "integer"),
        ("E_COBYLA", "integer"),
        ("noisy_objective", "float_or_null"),
        ("zne_objective", "float_or_null"),
        ("ideal_objective", "float_or_null"),
        ("reference_objective", "float_or_null"),
        ("selected_endpoint", "object"),
        ("d_R", "float_or_null"),
        ("r_MSE", "float_or_null"),
        ("endpoint_classification", "string"),
        ("actual_cobyla_evaluations", "integer_or_null"),
        ("wall_time_s", "float"),
        ("memory_peak_mb", "float"),
        ("code_version_hash", "string"),
    ]
    return {
        "schema_version": "phase_e_cobyla_official_task_output_schema_v0_1",
        "created_at_utc": _utc_now(),
        "minimum_fields": [{"name": name, "type": type_name, "required": True} for name, type_name in fields],
        "notes": [
            "This schema is a pre-execution minimum contract. The final official runner may add fields but must not remove these.",
            "Null objective/effect fields are allowed only for task families where the quantity is not produced by that task.",
        ],
    }


def write_deviation_log_entry(
    root: Path = PROJECT_ROOT,
    *,
    step: str,
    entry_type: str,
    description: str,
    action_taken: str,
    blocking: bool = False,
    resolved: bool = True,
) -> dict[str, Any]:
    out = _output_dir(root)
    entry = {
        "timestamp_utc": _utc_now(),
        "step": step,
        "type": entry_type,
        "description": description,
        "action_taken": action_taken,
        "blocking": bool(blocking),
        "resolved": bool(resolved),
        "resolved_timestamp_utc": _utc_now() if resolved else None,
    }
    path = out / "deviation_log.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def write_official_task_output_schema(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    path = out / "official_task_output_schema.json"
    schema = _expected_schema()
    schema["schema_hash"] = _canonical_hash(schema["minimum_fields"])
    write_json_atomic(path, schema, overwrite=force or not path.exists())
    return schema


def write_manifest_task_count_check(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    path = out / "manifest_task_count_check.json"
    inputs = _common_inputs(root)
    config = inputs["config"]
    gate = inputs["freeze_gate"]["runtime_storage_projection"]
    selected = _selected_values(config, inputs["calibration"])
    orders = config["zne"]["scale_factors"]

    instances = int(config["depth2_design"]["selected_instance_count"])
    noise_rates = len(config["noise_rates"])
    shot_budgets = len(config["shot_budgets"])
    seeds = int(config["depth2_design"]["seed_count"])
    order_count = len(orders)
    lhs_designs = 1
    base_cells = instances * noise_rates * shot_budgets
    full_cells = base_cells * order_count
    scale_records_per_base_cell = 1 + 1 + sum(len(scales) for scales in orders.values())
    lhs_point_scale_records = base_cells * int(selected["N_LHS"]) * scale_records_per_base_cell
    noisy_optimizer_start_records = base_cells * seeds * int(selected["K"])
    zne_optimizer_start_records = base_cells * seeds * int(selected["K"]) * order_count
    reference_optimizer_runs = instances * 16 * 2
    endpoint_summary_records = noisy_optimizer_start_records + zne_optimizer_start_records + reference_optimizer_runs

    assumed = gate["design_counts"]
    checks = {
        "selected_instances_match": instances == int(assumed["selected_instances"]),
        "noise_rates_match": noise_rates == int(assumed["noise_rates"]),
        "shot_budgets_match": shot_budgets == int(assumed["shot_budgets"]),
        "orders_match": order_count == int(assumed["richardson_orders"]),
        "seeds_match": seeds == int(assumed["seeds"]),
        "base_cells_match": base_cells == int(assumed["base_cells_without_order"]),
        "full_cells_match": full_cells == int(assumed["full_cells_with_order"]),
        "selected_K_match": int(selected["K"]) == int(assumed["selected_K"]),
        "selected_E_match": int(selected["E_COBYLA"]) == int(assumed["selected_E_COBYLA"]),
        "selected_N_LHS_match": int(selected["N_LHS"]) == int(assumed["selected_N_LHS"]),
    }
    report = {
        "artifact": "manifest_task_count_check",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "passed" if all(checks.values()) else "failed",
        "match": bool(all(checks.values())),
        "checks": checks,
        "dimension_table": {
            "instances": instances,
            "noise_rates": noise_rates,
            "shot_budgets": shot_budgets,
            "richardson_orders": order_count,
            "seeds": seeds,
            "lhs_designs": lhs_designs,
            "K": int(selected["K"]),
            "E_COBYLA": int(selected["E_COBYLA"]),
            "N_LHS": int(selected["N_LHS"]),
            "scale_records_per_base_cell": scale_records_per_base_cell,
        },
        "independently_derived_counts": {
            "base_cells_without_order": base_cells,
            "full_cells_with_order": full_cells,
            "lhs_point_scale_records": lhs_point_scale_records,
            "noisy_optimizer_start_records": noisy_optimizer_start_records,
            "zne_optimizer_start_records": zne_optimizer_start_records,
            "reference_optimizer_runs": reference_optimizer_runs,
            "endpoint_summary_records": endpoint_summary_records,
        },
        "projection_assumed_counts": assumed,
        "invalidates_projection_if_failed": [
            "287.6 h projection",
            "freeze-gate runtime report derived thresholds",
            "Case A/B/C speedup thresholds",
        ],
    }
    write_json_atomic(path, report, overwrite=force or not path.exists())
    return report


def write_baseline_lock(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    paths = {
        "lock": out / "baseline_lock.json",
        "git_status": out / "git_status.txt",
        "dirty_inventory": out / "dirty_tree_inventory.csv",
        "pre_o1_hashes": out / "pre_o1_hashes.json",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise FileExistsError("Refusing to overwrite existing COBYLA runtime O0 artifacts.")

    inputs = _common_inputs(root)
    selected = _selected_values(inputs["config"], inputs["calibration"])
    schema = write_official_task_output_schema(root, force=True)
    manifest_check = write_manifest_task_count_check(root, force=True)
    git = _git_status(root)
    inventory = _dirty_inventory(root, git["status_text"])
    code_files = [
        "scripts/79_phase_e_cobyla_runtime_o0_o0_75.py",
        "src/phase_e_cobyla_runtime_optimization.py",
        "src/phase_e_sim_backend.py",
        "src/phase_e_optimizers.py",
        "src/phase_e_freeze_gates.py",
        "src/phase_e_config.py",
    ]
    input_files = [
        PLAN_FILE,
        PROTOCOL_FILE,
        COMPUTE_FILE,
        "config/phase_e_config_v0.1-draft.json",
        "config/phase_e_instances_v0.1-draft.json",
        "config/phase_e_noise_model_v0.1-draft.json",
        "results/phase_e/calibration/phase_e_calibration_status.json",
        "results/phase_e/calibration/freeze_gates/phase_e_bias_variance_runtime_gate.json",
        str(OUTPUT_DIR_RELATIVE / "official_task_output_schema.json").replace("\\", "/"),
        str(OUTPUT_DIR_RELATIVE / "manifest_task_count_check.json").replace("\\", "/"),
    ]
    pre_o1_hashes = {
        "artifact": "pre_o1_hashes",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "input_hashes": _hashes_for_existing(root, input_files),
        "code_hashes": _hashes_for_existing(root, code_files),
        "dirty_tree_inventory_sha256": _canonical_hash(inventory),
    }
    write_json_atomic(paths["pre_o1_hashes"], pre_o1_hashes, overwrite=True)
    _write_csv(
        paths["dirty_inventory"],
        inventory,
        fieldnames=["git_status", "path", "exists", "kind", "bytes", "sha256", "file_count"],
    )
    paths["git_status"].write_text(git["status_text"], encoding="utf-8")
    lock = {
        "artifact": "baseline_lock",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "baseline_locked_official_execution_false",
        "official_execution_authorized": False,
        "git_branch": git["branch"],
        "git_head": git["head"],
        "git_status_short_sha256": hashlib.sha256(git["status_text"].encode("utf-8")).hexdigest(),
        "selected_values": selected,
        "plan_content_hash": sha256_file(root / PLAN_FILE),
        "official_task_output_schema_hash": schema["schema_hash"],
        "manifest_task_count_check_status": manifest_check["status"],
        "manifest_task_count_check_hash": sha256_file(out / "manifest_task_count_check.json"),
        "pre_o1_hashes_path": str(paths["pre_o1_hashes"].relative_to(root)).replace("\\", "/"),
        "dirty_inventory_path": str(paths["dirty_inventory"].relative_to(root)).replace("\\", "/"),
        "tracked_modified_files": [row["path"] for row in inventory if row["git_status"] != "??"],
        "untracked_files": [row["path"] for row in inventory if row["git_status"] == "??"],
        "execution_blockers": {
            "O0_baseline_lock": "defined",
            "manifest_task_count_check": manifest_check["status"],
            "O0_5_instrumentation_hooks": "pending",
            "O0_75_analytical_cost_model": "pending",
            "O1_runtime_report": "pending",
            "official_execution_authorized": False,
        },
    }
    write_json_atomic(paths["lock"], lock, overwrite=True)
    write_deviation_log_entry(
        root,
        step="O0",
        entry_type="user_review",
        description="O0 baseline lock and manifest task-count check written; official execution remains blocked.",
        action_taken="Recorded baseline artifacts only.",
        blocking=False,
        resolved=True,
    )
    return lock


def _cpu_freq_snapshot() -> dict[str, Any]:
    try:
        import psutil  # type: ignore

        freq = psutil.cpu_freq()
        return {
            "available": freq is not None,
            "current_mhz": None if freq is None else float(freq.current),
            "min_mhz": None if freq is None else float(freq.min),
            "max_mhz": None if freq is None else float(freq.max),
        }
    except Exception as exc:  # pragma: no cover - platform diagnostic
        return {"available": False, "error": repr(exc)}


def _timed_call(label: str, func: Any) -> tuple[Any, dict[str, Any]]:
    start_freq = _cpu_freq_snapshot()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_freq = _cpu_freq_snapshot()
    return result, {
        "phase": label,
        "wall_time_s": float(elapsed),
        "memory_peak_mb": float(peak / 1024**2),
        "memory_current_mb": float(current / 1024**2),
        "cpu_frequency_start": start_freq,
        "cpu_frequency_end": end_freq,
    }


def run_o0_5_instrumentation_dry_run(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    json_path = out / "O0_5_instrumentation_dry_run.json"
    md_path = out / "O0_5_instrumentation_dry_run.md"
    if (json_path.exists() or md_path.exists()) and not force:
        raise FileExistsError("Refusing to overwrite existing O0.5 dry-run artifacts.")

    graph = _graph_from_selected_instance(root, "H1")
    theta = (0.11, 0.23, 0.37, 0.41)
    p_dep = 0.001
    shots = 64
    seed = 42

    noisy, noisy_timing = _timed_call(
        "single_noisy_objective",
        lambda: evaluate_custom_density_objective(
            graph,
            theta,
            p_dep=p_dep,
            scale_factor=1,
            shots=shots,
            seed=deterministic_seed("o0_5_noisy", graph.name, p_dep, shots, seed),
        ),
    )
    zne, zne_timing = _timed_call(
        "single_linear_zne_objective",
        lambda: richardson_zne_objective(
            graph,
            theta,
            p_dep=p_dep,
            scale_factors=[1, 3],
            shots_total=shots,
            seed=deterministic_seed("o0_5_zne", graph.name, p_dep, shots, seed),
        ),
    )

    eval_counter = {"count": 0}

    def quadratic_objective(x: tuple[float, ...]) -> float:
        eval_counter["count"] += 1
        return float(np.sum((np.asarray(x, dtype=float) - 0.25) ** 2))

    cobyla, cobyla_timing = _timed_call(
        "trivial_cobyla_counter",
        lambda: cobyla_minimize(
            quadratic_objective,
            initial_theta=(0.1, 0.2, 0.3, 0.4),
            config=COBYLAConfig(max_evaluations=12, rhobeg=0.2, tol=1e-6),
        ),
    )
    required_fields = [
        "wall_time_s",
        "memory_peak_mb",
        "cpu_frequency_start",
        "cpu_frequency_end",
    ]
    timings = [noisy_timing, zne_timing, cobyla_timing]
    report = {
        "artifact": "O0_5_instrumentation_dry_run",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "passed"
        if all(all(field in timing for field in required_fields) for timing in timings)
        and int(cobyla.num_evaluations) == int(eval_counter["count"])
        else "failed",
        "official_execution_authorized": False,
        "scope": "non_official_single_task_instrumentation_dry_run",
        "case": {
            "instance": graph.name,
            "p_dep": p_dep,
            "shots_total": shots,
            "seed": seed,
            "theta": list(theta),
            "cobyla_max_evaluations": 12,
        },
        "timings": timings,
        "objective_results": {
            "noisy_objective": noisy.objective_value,
            "zne_objective": zne.objective_value,
            "zne_scale_factors": zne.scale_factors,
        },
        "cobyla_counter": {
            "reported_num_evaluations": int(cobyla.num_evaluations),
            "instrumented_objective_calls": int(eval_counter["count"]),
            "matches": int(cobyla.num_evaluations) == int(eval_counter["count"]),
        },
        "behavioral_change": False,
        "notes": [
            "This dry-run verifies timing/memory/evaluation-count instrumentation fields only.",
            "It is not O1 profiling and does not inform scientific conclusions.",
        ],
    }
    write_json_atomic(json_path, report, overwrite=True)
    md = [
        "# O0.5 Instrumentation Dry Run",
        "",
        f"Created UTC: `{report['created_at_utc']}`",
        f"Status: `{report['status']}`",
        "",
        "This is a non-official single-task instrumentation check. It does not authorize O1 or official execution.",
        "",
        f"- Instance: `{graph.name}`",
        f"- Noisy objective wall time: `{noisy_timing['wall_time_s']:.6f}` s",
        f"- Linear ZNE wall time: `{zne_timing['wall_time_s']:.6f}` s",
        f"- COBYLA reported evaluations: `{cobyla.num_evaluations}`",
        f"- Instrumented objective calls: `{eval_counter['count']}`",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    write_deviation_log_entry(
        root,
        step="O0.5",
        entry_type="amendment",
        description="O0.5 instrumentation dry-run written; no scientific profiling executed.",
        action_taken=f"Recorded {json_path.relative_to(root)}.",
        blocking=False,
        resolved=True,
    )
    return report


def write_o0_75_analytical_cost_model(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    json_path = out / "O0_75_analytical_cost_model.json"
    md_path = out / "O0_75_analytical_cost_model.md"
    if (json_path.exists() or md_path.exists()) and not force:
        raise FileExistsError("Refusing to overwrite existing O0.75 artifacts.")

    inputs = _common_inputs(root)
    config = inputs["config"]
    gate = inputs["freeze_gate"]["runtime_storage_projection"]
    selected = _selected_values(config, inputs["calibration"])
    instances = _read_json(root / "config" / "phase_e_instances_v0.1-draft.json")["selected_instances"]
    orders = config["zne"]["scale_factors"]
    depth = int(config["qaoa_depths"]["depth2_primary"])
    seed_count = int(config["depth2_design"]["seed_count"])
    base_multiplier = len(config["noise_rates"]) * len(config["shot_budgets"])
    k = int(selected["K"])
    n_lhs = int(selected["N_LHS"])
    eval_stats = gate["eval_stats"]
    e_actual = max(float(eval_stats["noisy_eval_mean_per_start"]), float(eval_stats["zne_eval_mean_per_start"]))

    rows = []
    total_work = 0.0
    for label, meta in sorted(instances.items()):
        n_qubits = int(meta["n_nodes"])
        edge_count = int(meta["n_edges"])
        density_work = float((4**n_qubits) * depth)
        for order, scales in orders.items():
            scale_count = len(scales)
            objective_work = density_work * scale_count
            optimizer_work = base_multiplier * seed_count * k * e_actual * objective_work
            lhs_work = base_multiplier * n_lhs * (2 + scale_count) * density_work
            total = optimizer_work + lhs_work
            total_work += total
            rows.append(
                {
                    "instance": label,
                    "n_qubits": n_qubits,
                    "edge_count": edge_count,
                    "richardson_order": order,
                    "scale_factors": list(scales),
                    "density_matrix_asymptotic": "O(4^n * depth)",
                    "density_work_units": density_work,
                    "objective_work_units": objective_work,
                    "optimizer_work_units": optimizer_work,
                    "lhs_work_units": lhs_work,
                    "total_work_units": total,
                }
            )
    for row in rows:
        row["fraction_of_total_work"] = float(row["total_work_units"] / total_work) if total_work else 0.0

    by_order: dict[str, float] = {}
    by_instance: dict[str, float] = {}
    for row in rows:
        by_order[row["richardson_order"]] = by_order.get(row["richardson_order"], 0.0) + float(row["total_work_units"])
        by_instance[row["instance"]] = by_instance.get(row["instance"], 0.0) + float(row["total_work_units"])
    summary = {
        "artifact": "O0_75_analytical_cost_model",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "written_not_profiled",
        "official_execution_authorized": False,
        "model_scope": "order_of_magnitude_sanity_check_not_runtime_truth",
        "asymptotic_density_matrix_cost": "O(4^n * depth)",
        "selected_values": selected,
        "assumptions": {
            "depth": depth,
            "seed_count": seed_count,
            "noise_rates": len(config["noise_rates"]),
            "shot_budgets": len(config["shot_budgets"]),
            "E_actual_proxy": e_actual,
            "E_actual_proxy_source": "max(noisy_eval_mean_per_start, zne_eval_mean_per_start) from freeze gate",
            "N_LHS": n_lhs,
            "K": k,
        },
        "total_work_units": total_work,
        "work_fraction_by_order": {key: value / total_work for key, value in by_order.items()},
        "work_fraction_by_instance": {key: value / total_work for key, value in by_instance.items()},
        "rows": rows,
        "post_o1_comparison_rule": {
            "divergence_threshold": "3x",
            "action": "pause optimization planning and investigate manifest count, implementation inefficiency, or model error",
        },
    }
    write_json_atomic(json_path, summary, overwrite=True)
    md_lines = [
        "# O0.75 Analytical Cost Model",
        "",
        f"Created UTC: `{summary['created_at_utc']}`",
        "Status: `written_not_profiled`",
        "",
        "This is an order-of-magnitude sanity check. It is not O1 profiling and does not authorize official execution.",
        "",
        "## Assumptions",
        "",
        f"- Density-matrix asymptotic cost: `{summary['asymptotic_density_matrix_cost']}`",
        f"- Depth: `{depth}`",
        f"- E_actual proxy: `{e_actual:.6g}`",
        f"- K: `{k}`",
        f"- N_LHS: `{n_lhs}`",
        "",
        "## Work Fractions By Order",
        "",
    ]
    for order, fraction in summary["work_fraction_by_order"].items():
        md_lines.append(f"- {order}: `{fraction:.6f}`")
    md_lines.extend(["", "## Work Fractions By Instance", ""])
    for instance, fraction in summary["work_fraction_by_instance"].items():
        md_lines.append(f"- {instance}: `{fraction:.6f}`")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return summary


def run_o0_bundle(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    lock = write_baseline_lock(root, force=force)
    return {
        "baseline_lock": lock,
        "manifest_task_count_check": _read_json(_output_dir(root) / "manifest_task_count_check.json"),
        "official_task_output_schema": _read_json(_output_dir(root) / "official_task_output_schema.json"),
    }


def write_pre_o1_status(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    json_path = out / "pre_o1_status.json"
    md_path = out / "pre_o1_status.md"
    if (json_path.exists() or md_path.exists()) and not force:
        raise FileExistsError("Refusing to overwrite existing pre-O1 status artifacts.")

    required = {
        "baseline_lock": out / "baseline_lock.json",
        "manifest_task_count_check": out / "manifest_task_count_check.json",
        "official_task_output_schema": out / "official_task_output_schema.json",
        "O0_5_instrumentation_dry_run": out / "O0_5_instrumentation_dry_run.json",
        "O0_75_analytical_cost_model": out / "O0_75_analytical_cost_model.json",
    }
    optional = {"O1_prerun_feasibility_estimate": out / "O1_prerun_feasibility_estimate.json"}
    artifact_status = {}
    for name, path in {**required, **optional}.items():
        if not path.exists():
            artifact_status[name] = {"exists": False, "status": "missing", "sha256": None}
            continue
        payload = _read_json(path)
        artifact_status[name] = {
            "exists": True,
            "status": payload.get("status", "written"),
            "sha256": sha256_file(path),
        }

    blockers = {
        "O0_baseline_lock": artifact_status["baseline_lock"]["status"] == "baseline_locked_official_execution_false",
        "manifest_task_count_check": artifact_status["manifest_task_count_check"]["status"] == "passed",
        "official_task_output_schema": artifact_status["official_task_output_schema"]["exists"],
        "O0_5_instrumentation": artifact_status["O0_5_instrumentation_dry_run"]["status"] == "passed",
        "O0_75_analytical_cost_model": artifact_status["O0_75_analytical_cost_model"]["status"] == "written_not_profiled",
    }
    o1_prerun = artifact_status["O1_prerun_feasibility_estimate"]
    if not all(blockers.values()):
        status = "pre_o1_blocked"
        next_action = "resolve_missing_or_failed_pre_o1_artifacts"
        o1_profiling_authorized = False
    elif not o1_prerun["exists"]:
        status = "pre_o1_ready_for_o1_feasibility_estimate"
        next_action = "write_O1_prerun_feasibility_estimate"
        o1_profiling_authorized = False
    elif o1_prerun["status"] == "native_o1_feasible":
        status = "pre_o1_ready_for_native_o1_profiling"
        next_action = "launch_native_controlled_thread_O1_mandatory_slice_after_user_confirmation"
        o1_profiling_authorized = True
    elif o1_prerun["status"] == "reduced_o1_fallback_required":
        status = "pre_o1_ready_for_reduced_o1_fallback"
        next_action = "launch_reduced_O1_fallback_after_user_confirmation"
        o1_profiling_authorized = True
    else:
        status = "pre_o1_prerun_estimate_failed_or_inconclusive"
        next_action = "review_O1_prerun_feasibility_estimate_before_profiling"
        o1_profiling_authorized = False
    report = {
        "artifact": "pre_o1_status",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": status,
        "official_execution_authorized": False,
        "o1_authorized": bool(status != "pre_o1_blocked"),
        "o1_profiling_authorized": o1_profiling_authorized,
        "artifact_status": artifact_status,
        "blockers": blockers,
        "next_action": next_action,
    }
    write_json_atomic(json_path, report, overwrite=True)
    md_lines = [
        "# COBYLA Runtime Pre-O1 Status",
        "",
        f"Created UTC: `{report['created_at_utc']}`",
        f"Status: `{status}`",
        "",
        "Official execution remains blocked.",
        "",
        "## Artifacts",
        "",
    ]
    for name, row in artifact_status.items():
        md_lines.append(f"- {name}: `{row['status']}`")
    md_lines.extend(["", f"Next action: `{report['next_action']}`"])
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return report


def _slice_work_fraction_from_cost_model(
    cost_model: dict[str, Any],
    *,
    instance: str,
    orders: list[str],
    noise_rate_count: int,
    shot_budget_count: int,
    seed_count: int,
    slice_noise_rates: int,
    slice_shot_budgets: int,
    slice_seeds: int,
) -> float:
    denominator = float(noise_rate_count * shot_budget_count * seed_count)
    if denominator <= 0:
        raise ValueError("Invalid full-factorial dimensions for O1 feasibility estimate.")
    fraction = 0.0
    for row in cost_model["rows"]:
        if row["instance"] != instance or row["richardson_order"] not in orders:
            continue
        fraction += float(row["fraction_of_total_work"]) * float(slice_noise_rates * slice_shot_budgets * slice_seeds) / denominator
    return fraction


def write_o1_prerun_feasibility_estimate(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    json_path = out / "O1_prerun_feasibility_estimate.json"
    md_path = out / "O1_prerun_feasibility_estimate.md"
    if (json_path.exists() or md_path.exists()) and not force:
        raise FileExistsError("Refusing to overwrite existing O1 pre-run feasibility estimate.")

    pre_status_path = out / "pre_o1_status.json"
    if not pre_status_path.exists():
        write_pre_o1_status(root, force=True)
    pre_status = _read_json(pre_status_path)
    if pre_status.get("status") == "pre_o1_blocked":
        raise RuntimeError("Cannot write O1 pre-run estimate while pre-O1 blockers remain unresolved.")

    inputs = _common_inputs(root)
    config = inputs["config"]
    gate = inputs["freeze_gate"]["runtime_storage_projection"]
    manifest = _read_json(out / "manifest_task_count_check.json")
    cost_model = _read_json(out / "O0_75_analytical_cost_model.json")
    selected = _selected_values(config, inputs["calibration"])

    original_hours = float(gate["primary_projected_wall_hours"])
    native_fit_threshold_hours = 2.5
    native_timebox_hours = 3.0
    optional_second_slice_trigger_hours = 2.0
    pre_run_safety_factor = 1.25

    orders = [order for order in ("linear", "quadratic", "cubic") if order in config["zne"]["scale_factors"]]
    shot_budgets = list(config["shot_budgets"])
    noise_rates = list(config["noise_rates"])
    full_units = (
        int(manifest["dimension_table"]["instances"])
        * len(noise_rates)
        * int(config["depth2_design"]["seed_count"])
        * int(manifest["dimension_table"]["lhs_designs"])
        * len(shot_budgets)
        * len(orders)
    )
    native_units = 1 * 1 * 1 * 1 * len(shot_budgets) * len(orders)
    simple_fraction = native_units / full_units
    simple_hours = original_hours * simple_fraction

    mandatory_instance = "H4"
    mandatory_p_dep = 0.01
    mandatory_seed = 42
    optional_instance = "H1"
    optional_p_dep = 0.003
    lhs_design_id = 0
    weighted_fraction = _slice_work_fraction_from_cost_model(
        cost_model,
        instance=mandatory_instance,
        orders=orders,
        noise_rate_count=len(noise_rates),
        shot_budget_count=len(shot_budgets),
        seed_count=int(config["depth2_design"]["seed_count"]),
        slice_noise_rates=1,
        slice_shot_budgets=len(shot_budgets),
        slice_seeds=1,
    )
    warmup_fraction = _slice_work_fraction_from_cost_model(
        cost_model,
        instance=mandatory_instance,
        orders=["cubic"],
        noise_rate_count=len(noise_rates),
        shot_budget_count=len(shot_budgets),
        seed_count=int(config["depth2_design"]["seed_count"]),
        slice_noise_rates=1,
        slice_shot_budgets=1,
        slice_seeds=1,
    )
    optional_fraction = _slice_work_fraction_from_cost_model(
        cost_model,
        instance=optional_instance,
        orders=orders,
        noise_rate_count=len(noise_rates),
        shot_budget_count=len(shot_budgets),
        seed_count=int(config["depth2_design"]["seed_count"]),
        slice_noise_rates=1,
        slice_shot_budgets=len(shot_budgets),
        slice_seeds=1,
    )
    weighted_hours_without_warmup = original_hours * weighted_fraction
    warmup_hours = original_hours * warmup_fraction
    weighted_hours_with_warmup = weighted_hours_without_warmup + warmup_hours
    weighted_hours_with_safety = weighted_hours_with_warmup * pre_run_safety_factor
    optional_hours = original_hours * optional_fraction
    optional_hours_with_safety = optional_hours * pre_run_safety_factor

    native_allowed = weighted_hours_with_safety <= native_fit_threshold_hours
    first_slice_predicted_under_second_slice_trigger = weighted_hours_with_safety < optional_second_slice_trigger_hours
    status = "native_o1_feasible" if native_allowed else "reduced_o1_fallback_required"
    recommended_next_action = (
        "launch_native_controlled_thread_O1_mandatory_slice"
        if native_allowed
        else "launch_reduced_O1_fallback_with_K_and_E_scaling_checks"
    )
    report = {
        "artifact": "O1_prerun_feasibility_estimate",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": status,
        "official_execution_authorized": False,
        "native_o1_allowed": native_allowed,
        "fallback_required": not native_allowed,
        "recommended_next_action": recommended_next_action,
        "decision_basis": "analytical_instance_order_weighted_estimate_with_pre_run_safety_factor",
        "estimate_confidence": "approximate_pre_measurement",
        "source_projection": {
            "original_full_projection_hours": original_hours,
            "accepted_local_wall_hours": float(gate["accepted_local_wall_hours"]),
            "projection_source": "phase_e_bias_variance_runtime_gate runtime_storage_projection",
        },
        "timebox_rules": {
            "native_fit_threshold_hours": native_fit_threshold_hours,
            "native_timebox_hours": native_timebox_hours,
            "optional_second_slice_trigger_hours": optional_second_slice_trigger_hours,
            "pre_run_safety_factor": pre_run_safety_factor,
        },
        "profiling_workload": {
            "mandatory_validation_slice": {
                "instance": mandatory_instance,
                "p_dep": mandatory_p_dep,
                "shot_budgets": shot_budgets,
                "richardson_orders": orders,
                "seed_count": 1,
                "seed": mandatory_seed,
                "lhs_design_id": lhs_design_id,
                "K": int(selected["K"]),
                "E_COBYLA": int(selected["E_COBYLA"]),
                "N_LHS": int(selected["N_LHS"]),
            },
            "warmup_task": {
                "instance": mandatory_instance,
                "p_dep": mandatory_p_dep,
                "shots_total": max(shot_budgets),
                "richardson_order": "cubic",
                "seed": mandatory_seed,
                "lhs_design_id": lhs_design_id,
                "use_for_projection": False,
            },
            "optional_second_slice": {
                "enabled_if_first_slice_completes_under_hours": optional_second_slice_trigger_hours,
                "instance": optional_instance,
                "p_dep": optional_p_dep,
                "shot_budgets": shot_budgets,
                "richardson_orders": orders,
                "seed": mandatory_seed,
                "lhs_design_id": lhs_design_id,
            },
        },
        "manifest_fraction_estimate": {
            "method": "unweighted_manifest_fraction",
            "native_workload_units": native_units,
            "full_factorial_workload_units": full_units,
            "fraction_of_full_projection": simple_fraction,
            "estimated_native_hours": simple_hours,
            "used_for_decision": False,
            "limitation": "Does not account for H4 being a heavier six-qubit profiling instance.",
        },
        "analytical_weighted_estimate": {
            "method": "O0.75 instance/order work fractions",
            "mandatory_fraction_without_warmup": weighted_fraction,
            "warmup_fraction": warmup_fraction,
            "mandatory_fraction_with_warmup": weighted_fraction + warmup_fraction,
            "optional_second_slice_fraction": optional_fraction,
            "estimated_mandatory_hours_without_warmup": weighted_hours_without_warmup,
            "estimated_warmup_hours": warmup_hours,
            "estimated_mandatory_hours_with_warmup": weighted_hours_with_warmup,
            "estimated_mandatory_hours_with_safety": weighted_hours_with_safety,
            "estimated_optional_second_slice_hours": optional_hours,
            "estimated_optional_second_slice_hours_with_safety": optional_hours_with_safety,
            "estimated_mandatory_plus_optional_hours_with_safety": weighted_hours_with_safety
            + optional_hours_with_safety,
            "first_slice_predicted_under_second_slice_trigger": first_slice_predicted_under_second_slice_trigger,
            "used_for_decision": True,
        },
        "fallback_rule_if_needed": {
            "K_profiling": 10,
            "E_profiling": 30,
            "K_scaling_check": "compare K=10 versus K=20 at fixed E",
            "E_scaling_check": "compare E=30 versus E=60 or E=100 at fixed K",
            "native_E_fallback": "K=10 with E_COBYLA=300 if cap-linear extrapolation fails",
        },
        "source_artifacts": {
            "manifest_task_count_check": {
                "path": str((out / "manifest_task_count_check.json").relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(out / "manifest_task_count_check.json"),
            },
            "O0_75_analytical_cost_model": {
                "path": str((out / "O0_75_analytical_cost_model.json").relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(out / "O0_75_analytical_cost_model.json"),
            },
            "freeze_gate_runtime_projection": {
                "path": "results/phase_e/calibration/freeze_gates/phase_e_bias_variance_runtime_gate.json",
                "sha256": sha256_file(root / "results/phase_e/calibration/freeze_gates/phase_e_bias_variance_runtime_gate.json"),
            },
        },
        "notes": [
            "This artifact only decides the O1 profiling path. It is not O1 profiling and does not authorize official execution.",
            "Native O1 is considered feasible only because the conservative weighted estimate remains below the 2.5 h pre-launch threshold.",
        ],
    }
    write_json_atomic(json_path, report, overwrite=True)
    md_lines = [
        "# O1 Pre-Run Feasibility Estimate",
        "",
        f"Created UTC: `{report['created_at_utc']}`",
        f"Status: `{status}`",
        "",
        "Official execution remains blocked.",
        "",
        "## Decision",
        "",
        f"- Recommended next action: `{recommended_next_action}`",
        f"- Native O1 allowed: `{native_allowed}`",
        f"- Reduced fallback required now: `{not native_allowed}`",
        "",
        "## Estimates",
        "",
        f"- Unweighted manifest estimate: `{simple_hours:.3f}` h",
        f"- Weighted mandatory slice + warm-up: `{weighted_hours_with_warmup:.3f}` h",
        f"- Weighted mandatory slice + warm-up + safety factor: `{weighted_hours_with_safety:.3f}` h",
        f"- Optional second slice + safety factor: `{optional_hours_with_safety:.3f}` h",
        f"- Native fit threshold: `{native_fit_threshold_hours:.3f}` h",
        "",
        "The weighted estimate is used for the decision because the mandatory profiling slice is `H4`, a heavier selected instance than the unweighted factorial average.",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    write_deviation_log_entry(
        root,
        step="O1",
        entry_type="amendment",
        description="O1 pre-run feasibility estimate written; official execution remains blocked.",
        action_taken=f"Recommended {recommended_next_action}.",
        blocking=False,
        resolved=True,
    )
    write_pre_o1_status(root, force=True)
    return report


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _process_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return float(psutil.Process(os.getpid()).memory_info().rss / 1024**2)
    except Exception:  # pragma: no cover - platform diagnostic
        return None


def _environment_snapshot(root: Path) -> dict[str, Any]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        np.show_config()
    try:
        import psutil  # type: ignore

        physical_cores = psutil.cpu_count(logical=False)
        logical_cores = psutil.cpu_count(logical=True)
        vm = psutil.virtual_memory()
        background_cpu_percent = psutil.cpu_percent(interval=1.0)
        ram = {
            "total_gb": float(vm.total / 1024**3),
            "available_gb": float(vm.available / 1024**3),
        }
    except Exception as exc:  # pragma: no cover - platform diagnostic
        physical_cores = None
        logical_cores = os.cpu_count()
        background_cpu_percent = None
        ram = {"available": False, "error": repr(exc)}
    disk = shutil.disk_usage(root)
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy_version": np.__version__,
        "numpy_show_config": stdout.getvalue(),
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "ram": ram,
        "thread_settings": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
        "cpu_frequency": _cpu_freq_snapshot(),
        "background_cpu_percent": background_cpu_percent,
        "storage": {
            "path": str(root),
            "disk_total_gb": float(disk.total / 1024**3),
            "disk_free_gb": float(disk.free / 1024**3),
            "storage_medium": "unknown",
        },
    }


def _lhs_points(depth: int, count: int, seed: int) -> list[tuple[float, ...]]:
    from scipy.stats import qmc

    sampler = qmc.LatinHypercube(d=2 * int(depth), seed=int(seed))
    unit = sampler.random(n=int(count))
    periods = np.asarray(theta_periods(int(depth)), dtype=float)
    return [tuple(float(x) for x in row * periods) for row in unit]


def _timed_density_objective(
    graph: GraphInstance,
    theta: tuple[float, ...],
    *,
    p_dep: float,
    scale_factor: int = 1,
    shots: int | None = None,
    seed: int | None = None,
) -> tuple[float, float]:
    started = time.perf_counter()
    value = evaluate_custom_density_objective(
        graph,
        theta,
        p_dep=p_dep,
        scale_factor=int(scale_factor),
        shots=shots,
        seed=seed,
    ).objective_value
    return float(value), float(time.perf_counter() - started)


def _timed_richardson_value(
    graph: GraphInstance,
    theta: tuple[float, ...],
    *,
    p_dep: float,
    scale_factors: list[int],
    shots_total: int,
    seed: int,
) -> tuple[float, dict[str, float], int]:
    weights = richardson_weights(scale_factors)
    shot_allocation = allocate_shots(int(shots_total), weights, strategy="proportional_to_abs_weights")
    per_scale_values: list[float] = []
    per_scale_times: dict[str, float] = {}
    for scale, shots in zip(scale_factors, shot_allocation, strict=True):
        shot_seed = deterministic_seed(
            "phase_e_zne",
            graph.name,
            tuple(wrap_theta(theta)),
            p_dep,
            int(shots_total),
            int(seed),
            int(scale),
        )
        value, elapsed = _timed_density_objective(
            graph,
            theta,
            p_dep=p_dep,
            scale_factor=int(scale),
            shots=int(shots),
            seed=shot_seed,
        )
        per_scale_values.append(value)
        per_scale_times[str(scale)] = per_scale_times.get(str(scale), 0.0) + elapsed
    return float(sum(weight * value for weight, value in zip(weights, per_scale_values, strict=True))), per_scale_times, sum(shot_allocation)


def _merge_seconds(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = float(target.get(key, 0.0) + float(value))


def _run_lhs_profile_task(
    graph: GraphInstance,
    *,
    p_dep: float,
    shots_total: int,
    richardson_order: str,
    scale_factors: list[int],
    seed: int,
    lhs_design_id: int,
    n_lhs: int,
) -> dict[str, Any]:
    setup_started = time.perf_counter()
    lhs_seed = deterministic_seed("phase_e_lhs", graph.name, p_dep, int(shots_total), int(n_lhs), int(lhs_design_id))
    points = _lhs_points(depth=2, count=int(n_lhs), seed=lhs_seed)
    setup_time_s = time.perf_counter() - setup_started

    started = time.perf_counter()
    ideal_time_s = 0.0
    noisy_time_s = 0.0
    zne_per_scale_time_s: dict[str, float] = {str(scale): 0.0 for scale in scale_factors}
    noisy_errors: list[float] = []
    zne_errors: list[float] = []
    ideal_values: list[float] = []
    for index, theta in enumerate(points):
        ideal, elapsed = _timed_density_objective(graph, theta, p_dep=0.0)
        ideal_time_s += elapsed
        noisy, elapsed = _timed_density_objective(
            graph,
            theta,
            p_dep=p_dep,
            scale_factor=1,
            shots=int(shots_total),
            seed=deterministic_seed(
                "phase_e_lhs_noisy",
                graph.name,
                p_dep,
                int(shots_total),
                int(n_lhs),
                int(lhs_design_id),
                int(index),
            ),
        )
        noisy_time_s += elapsed
        zne, scale_times, _ = _timed_richardson_value(
            graph,
            theta,
            p_dep=p_dep,
            scale_factors=scale_factors,
            shots_total=int(shots_total),
            seed=deterministic_seed(
                "phase_e_lhs_zne",
                graph.name,
                p_dep,
                int(shots_total),
                int(n_lhs),
                int(lhs_design_id),
                int(index),
            ),
        )
        _merge_seconds(zne_per_scale_time_s, scale_times)
        ideal_values.append(ideal)
        noisy_errors.append((noisy - ideal) ** 2)
        zne_errors.append((zne - ideal) ** 2)
    elapsed_total = time.perf_counter() - started
    mse_noisy = float(np.mean(noisy_errors)) if noisy_errors else math.nan
    mse_zne = float(np.mean(zne_errors)) if zne_errors else math.nan
    return {
        "operation": "lhs_validation",
        "instance": graph.name,
        "p_dep": float(p_dep),
        "shots_total": int(shots_total),
        "richardson_order": richardson_order,
        "scale_factors": scale_factors,
        "seed": int(seed),
        "lhs_design_id": int(lhs_design_id),
        "N_LHS": int(n_lhs),
        "lhs_seed": int(lhs_seed),
        "setup_time_s": float(setup_time_s),
        "wall_time_s": float(setup_time_s + elapsed_total),
        "lhs_validation_time_s": float(elapsed_total),
        "objective_eval_time_s": float(ideal_time_s + noisy_time_s + sum(zne_per_scale_time_s.values())),
        "ideal_objective_time_s": float(ideal_time_s),
        "noisy_objective_time_s": float(noisy_time_s),
        "zne_per_scale_time_s": zne_per_scale_time_s,
        "MSE_noisy": mse_noisy,
        "MSE_ZNE_Richardson": mse_zne,
        "r_MSE_Richardson": float((mse_noisy - mse_zne) / mse_noisy) if mse_noisy > 0.0 else math.nan,
        "Var_L0": float(np.var(ideal_values)) if ideal_values else math.nan,
        "Range_L0": float(np.max(ideal_values) - np.min(ideal_values)) if ideal_values else math.nan,
        "memory_peak_mb": _process_rss_mb(),
    }


def _run_cobyla_profile_task(
    graph: GraphInstance,
    *,
    p_dep: float,
    shots_total: int,
    objective_family: str,
    richardson_order: str | None,
    scale_factors: list[int] | None,
    seed: int,
    k_starts: int,
    e_cobyla: int,
) -> dict[str, Any]:
    setup_started = time.perf_counter()
    starts = make_paired_initial_thetas(
        depth=2,
        starts=int(k_starts),
        seed=deterministic_seed("phase_e_o1_starts", graph.name, p_dep, int(shots_total), int(seed)),
    )
    setup_time_s = time.perf_counter() - setup_started
    config = COBYLAConfig(max_evaluations=int(e_cobyla))
    objective_eval_time_s = 0.0
    zne_per_scale_time_s: dict[str, float] = {str(scale): 0.0 for scale in scale_factors or []}
    results = []
    started = time.perf_counter()
    for start_id, theta0 in enumerate(starts):
        if objective_family == "noisy":

            def objective(theta: tuple[float, ...], start_id: int = start_id) -> float:
                nonlocal objective_eval_time_s
                value, elapsed = _timed_density_objective(
                    graph,
                    theta,
                    p_dep=p_dep,
                    scale_factor=1,
                    shots=int(shots_total),
                    seed=deterministic_seed(
                        "phase_e_o1_noisy_cobyla",
                        graph.name,
                        p_dep,
                        int(shots_total),
                        int(seed),
                        int(start_id),
                        tuple(round(x, 12) for x in theta),
                    ),
                )
                objective_eval_time_s += elapsed
                return value

        elif objective_family == "zne":
            if richardson_order is None or scale_factors is None:
                raise ValueError("ZNE profiling requires richardson_order and scale_factors.")

            def objective(theta: tuple[float, ...], start_id: int = start_id) -> float:
                nonlocal objective_eval_time_s
                value, scale_times, _ = _timed_richardson_value(
                    graph,
                    theta,
                    p_dep=p_dep,
                    scale_factors=scale_factors,
                    shots_total=int(shots_total),
                    seed=deterministic_seed(
                        "phase_e_o1_zne_cobyla",
                        graph.name,
                        p_dep,
                        int(shots_total),
                        richardson_order,
                        int(seed),
                        int(start_id),
                        tuple(round(x, 12) for x in theta),
                    ),
                )
                _merge_seconds(zne_per_scale_time_s, scale_times)
                objective_eval_time_s += sum(scale_times.values())
                return value

        else:
            raise ValueError(f"Unknown objective_family: {objective_family}")

        results.append(cobyla_minimize(objective, theta0, config))
    optimizer_wall_time_s = time.perf_counter() - started
    eval_counts = [int(result.num_evaluations) for result in results]
    finite_values = [float(result.best_value) for result in results if np.isfinite(float(result.best_value))]
    status_counts = {status: [result.status for result in results].count(status) for status in sorted({result.status for result in results})}
    return {
        "operation": f"{objective_family}_optimizer",
        "instance": graph.name,
        "p_dep": float(p_dep),
        "shots_total": int(shots_total),
        "objective_family": objective_family,
        "richardson_order": richardson_order,
        "scale_factors": [] if scale_factors is None else scale_factors,
        "seed": int(seed),
        "K": int(k_starts),
        "E_COBYLA": int(e_cobyla),
        "setup_time_s": float(setup_time_s),
        "wall_time_s": float(setup_time_s + optimizer_wall_time_s),
        "optimizer_wall_time_s": float(optimizer_wall_time_s),
        "objective_eval_time_s": float(objective_eval_time_s),
        "cobyla_overhead_s": float(max(0.0, optimizer_wall_time_s - objective_eval_time_s)),
        "zne_per_scale_time_s": zne_per_scale_time_s,
        "actual_cobyla_evaluations_min": int(min(eval_counts)) if eval_counts else 0,
        "actual_cobyla_evaluations_max": int(max(eval_counts)) if eval_counts else 0,
        "actual_cobyla_evaluations_mean": float(np.mean(eval_counts)) if eval_counts else math.nan,
        "actual_cobyla_evaluations_sum": int(sum(eval_counts)),
        "actual_evaluations_hit_requested_cap": bool(max(eval_counts) >= int(e_cobyla)) if eval_counts else False,
        "best_value_min": float(min(finite_values)) if finite_values else math.nan,
        "best_value_mean": float(np.mean(finite_values)) if finite_values else math.nan,
        "valid_start_count": len(finite_values),
        "invalid_start_count": int(k_starts) - len(finite_values),
        "optimizer_status_counts": status_counts,
        "memory_peak_mb": _process_rss_mb(),
    }


def _write_o1_progress(progress_path: Path, event: str, payload: dict[str, Any]) -> float:
    started = time.perf_counter()
    _append_jsonl(progress_path, {"timestamp_utc": _utc_now(), "event": event, **payload})
    return float(time.perf_counter() - started)


def run_o1_native_mandatory_slice(root: Path = PROJECT_ROOT, force: bool = False, run_optional_second_slice: bool = True) -> dict[str, Any]:
    out = _output_dir(root)
    json_path = out / "O1_runtime_report.json"
    md_path = out / "O1_runtime_report.md"
    csv_path = out / "O1_profile_cells.csv"
    progress_path = out / "O1_progress.jsonl"
    if any(path.exists() for path in (json_path, md_path, csv_path)) and not force:
        raise FileExistsError("Refusing to overwrite existing O1 runtime report artifacts.")
    if force and progress_path.exists():
        progress_path.unlink()

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "1"

    pre_status = _read_json(out / "pre_o1_status.json")
    if pre_status.get("status") != "pre_o1_ready_for_native_o1_profiling":
        raise RuntimeError(f"Native O1 is not authorized by pre_o1_status: {pre_status.get('status')}")
    prerun = _read_json(out / "O1_prerun_feasibility_estimate.json")
    if not bool(prerun.get("native_o1_allowed")):
        raise RuntimeError("O1 pre-run estimate did not allow native O1.")

    inputs = _common_inputs(root)
    config = inputs["config"]
    gate = inputs["freeze_gate"]["runtime_storage_projection"]
    graph = _graph_from_selected_instance(root, "H4")
    p_dep = 0.01
    seed = 42
    lhs_design_id = 0
    n_lhs = int(_selected_values(config, inputs["calibration"])["N_LHS"])
    k_starts = int(_selected_values(config, inputs["calibration"])["K"])
    e_cobyla = int(_selected_values(config, inputs["calibration"])["E_COBYLA"])
    shot_budgets = [int(value) for value in config["shot_budgets"]]
    orders = {order: [int(scale) for scale in config["zne"]["scale_factors"][order]] for order in ("cubic", "linear", "quadratic")}
    environment = _environment_snapshot(root)
    wall_started = time.perf_counter()
    start_freq = _cpu_freq_snapshot()
    _write_o1_progress(
        progress_path,
        "o1_started",
        {
            "instance": graph.name,
            "p_dep": p_dep,
            "shot_budgets": shot_budgets,
            "orders": list(orders),
            "K": k_starts,
            "E_COBYLA": e_cobyla,
            "N_LHS": n_lhs,
        },
    )

    rows: list[dict[str, Any]] = []
    checkpoint_time_s = 0.0

    warmup_started = time.perf_counter()
    warmup_theta = (0.37, 0.22, 0.81, 0.44)
    warmup_value, warmup_scale_times, _ = _timed_richardson_value(
        graph,
        warmup_theta,
        p_dep=p_dep,
        scale_factors=orders["cubic"],
        shots_total=max(shot_budgets),
        seed=deterministic_seed("phase_e_o1_warmup", graph.name, p_dep, max(shot_budgets), "cubic", seed),
    )
    warmup_row = {
        "operation": "warmup_zne_objective",
        "slice": "warmup",
        "use_for_projection": False,
        "instance": graph.name,
        "p_dep": p_dep,
        "shots_total": max(shot_budgets),
        "richardson_order": "cubic",
        "scale_factors": orders["cubic"],
        "seed": seed,
        "lhs_design_id": lhs_design_id,
        "wall_time_s": float(time.perf_counter() - warmup_started),
        "setup_time_s": 0.0,
        "objective_eval_time_s": float(sum(warmup_scale_times.values())),
        "zne_per_scale_time_s": warmup_scale_times,
        "objective_value": warmup_value,
        "memory_peak_mb": _process_rss_mb(),
        "checkpoint_time_s": 0.0,
    }
    rows.append(warmup_row)
    checkpoint_time_s += _write_o1_progress(progress_path, "warmup_complete", {"wall_time_s": time.perf_counter() - warmup_started})

    mandatory_started = time.perf_counter()
    native_timebox_seconds = 3.0 * 3600.0
    mandatory_stopped_by_timebox = False
    for shots_total in shot_budgets:
        if time.perf_counter() - mandatory_started >= native_timebox_seconds:
            mandatory_stopped_by_timebox = True
            checkpoint_time_s += _write_o1_progress(progress_path, "mandatory_timebox_reached_before_shot", {"shots_total": shots_total})
            break
        checkpoint_time_s += _write_o1_progress(progress_path, "shot_budget_started", {"shots_total": shots_total})
        noisy_row = _run_cobyla_profile_task(
            graph,
            p_dep=p_dep,
            shots_total=shots_total,
            objective_family="noisy",
            richardson_order=None,
            scale_factors=None,
            seed=seed,
            k_starts=k_starts,
            e_cobyla=e_cobyla,
        )
        noisy_row["slice"] = "mandatory"
        noisy_row["use_for_projection"] = True
        noisy_row["lhs_design_id"] = lhs_design_id
        noisy_row["checkpoint_time_s"] = _write_o1_progress(
            progress_path,
            "noisy_optimizer_complete",
            {"shots_total": shots_total, "wall_time_s": noisy_row["wall_time_s"]},
        )
        checkpoint_time_s += float(noisy_row["checkpoint_time_s"])
        rows.append(noisy_row)
        print(f"O1 noisy optimizer complete shots={shots_total} wall_s={noisy_row['wall_time_s']:.3f}", flush=True)

        for order, scales in orders.items():
            if time.perf_counter() - mandatory_started >= native_timebox_seconds:
                mandatory_stopped_by_timebox = True
                checkpoint_time_s += _write_o1_progress(
                    progress_path,
                    "mandatory_timebox_reached_before_order",
                    {"shots_total": shots_total, "richardson_order": order},
                )
                break
            zne_row = _run_cobyla_profile_task(
                graph,
                p_dep=p_dep,
                shots_total=shots_total,
                objective_family="zne",
                richardson_order=order,
                scale_factors=scales,
                seed=seed,
                k_starts=k_starts,
                e_cobyla=e_cobyla,
            )
            zne_row["slice"] = "mandatory"
            zne_row["use_for_projection"] = True
            zne_row["lhs_design_id"] = lhs_design_id
            zne_row["checkpoint_time_s"] = _write_o1_progress(
                progress_path,
                "zne_optimizer_complete",
                {"shots_total": shots_total, "richardson_order": order, "wall_time_s": zne_row["wall_time_s"]},
            )
            checkpoint_time_s += float(zne_row["checkpoint_time_s"])
            rows.append(zne_row)
            print(f"O1 zne optimizer complete shots={shots_total} order={order} wall_s={zne_row['wall_time_s']:.3f}", flush=True)

        if mandatory_stopped_by_timebox:
            break

        for order, scales in orders.items():
            if time.perf_counter() - mandatory_started >= native_timebox_seconds:
                mandatory_stopped_by_timebox = True
                checkpoint_time_s += _write_o1_progress(
                    progress_path,
                    "mandatory_timebox_reached_before_lhs",
                    {"shots_total": shots_total, "richardson_order": order},
                )
                break
            lhs_row = _run_lhs_profile_task(
                graph,
                p_dep=p_dep,
                shots_total=shots_total,
                richardson_order=order,
                scale_factors=scales,
                seed=seed,
                lhs_design_id=lhs_design_id,
                n_lhs=n_lhs,
            )
            lhs_row["slice"] = "mandatory"
            lhs_row["use_for_projection"] = True
            lhs_row["K"] = ""
            lhs_row["E_COBYLA"] = ""
            lhs_row["checkpoint_time_s"] = _write_o1_progress(
                progress_path,
                "lhs_validation_complete",
                {"shots_total": shots_total, "richardson_order": order, "wall_time_s": lhs_row["wall_time_s"]},
            )
            checkpoint_time_s += float(lhs_row["checkpoint_time_s"])
            rows.append(lhs_row)
            print(f"O1 lhs complete shots={shots_total} order={order} wall_s={lhs_row['wall_time_s']:.3f}", flush=True)
        if mandatory_stopped_by_timebox:
            break

    mandatory_wall_s = time.perf_counter() - mandatory_started
    optional_rows: list[dict[str, Any]] = []
    optional_allowed = bool(run_optional_second_slice and mandatory_wall_s / 3600.0 < 2.0)
    if optional_allowed:
        optional_graph = _graph_from_selected_instance(root, "H1")
        optional_p_dep = 0.003
        optional_started = time.perf_counter()
        checkpoint_time_s += _write_o1_progress(progress_path, "optional_second_slice_started", {"instance": optional_graph.name})
        # The second slice is intentionally objective-only: it cross-validates per-instance
        # density cost without duplicating the full H4 mandatory workload.
        theta = (0.37, 0.22, 0.81, 0.44)
        for shots_total in shot_budgets:
            for order, scales in orders.items():
                objective_started = time.perf_counter()
                value, scale_times, _ = _timed_richardson_value(
                    optional_graph,
                    theta,
                    p_dep=optional_p_dep,
                    scale_factors=scales,
                    shots_total=shots_total,
                    seed=deterministic_seed("phase_e_o1_optional_second_slice", optional_graph.name, optional_p_dep, shots_total, order, seed),
                )
                row = {
                    "operation": "optional_second_slice_zne_objective",
                    "slice": "optional_second_slice",
                    "use_for_projection": False,
                    "instance": optional_graph.name,
                    "p_dep": optional_p_dep,
                    "shots_total": shots_total,
                    "richardson_order": order,
                    "scale_factors": scales,
                    "seed": seed,
                    "lhs_design_id": lhs_design_id,
                    "wall_time_s": float(time.perf_counter() - objective_started),
                    "setup_time_s": 0.0,
                    "objective_eval_time_s": float(sum(scale_times.values())),
                    "zne_per_scale_time_s": scale_times,
                    "objective_value": value,
                    "memory_peak_mb": _process_rss_mb(),
                    "checkpoint_time_s": 0.0,
                }
                row["checkpoint_time_s"] = _write_o1_progress(
                    progress_path,
                    "optional_second_slice_cell_complete",
                    {"shots_total": shots_total, "richardson_order": order, "wall_time_s": row["wall_time_s"]},
                )
                checkpoint_time_s += float(row["checkpoint_time_s"])
                optional_rows.append(row)
                rows.append(row)
        checkpoint_time_s += _write_o1_progress(progress_path, "optional_second_slice_complete", {"wall_time_s": time.perf_counter() - optional_started})

    total_wall_s = time.perf_counter() - wall_started
    end_freq = _cpu_freq_snapshot()
    mandatory_rows = [row for row in rows if row.get("slice") == "mandatory" and row.get("use_for_projection")]
    mandatory_component_seconds = {
        "noisy_optimizer": float(sum(float(row.get("wall_time_s", 0.0)) for row in mandatory_rows if row.get("operation") == "noisy_optimizer")),
        "zne_optimizer": float(sum(float(row.get("wall_time_s", 0.0)) for row in mandatory_rows if row.get("operation") == "zne_optimizer")),
        "lhs_validation": float(sum(float(row.get("wall_time_s", 0.0)) for row in mandatory_rows if row.get("operation") == "lhs_validation")),
        "checkpoint": float(checkpoint_time_s),
    }
    expected_mandatory_operations = len(shot_budgets) + len(shot_budgets) * len(orders) * 2
    completed_mandatory_operations = len(mandatory_rows)
    mandatory_complete = bool(completed_mandatory_operations >= expected_mandatory_operations and not mandatory_stopped_by_timebox)
    mandatory_projection_seconds = sum(mandatory_component_seconds.values())
    estimated_mandatory_hours = float(prerun["analytical_weighted_estimate"]["estimated_mandatory_hours_without_warmup"])
    measured_mandatory_hours = mandatory_projection_seconds / 3600.0
    correction_factor = measured_mandatory_hours / estimated_mandatory_hours if mandatory_complete and estimated_mandatory_hours > 0.0 else math.nan
    frequency_start = start_freq.get("current_mhz") if start_freq.get("available") else None
    frequency_end = end_freq.get("current_mhz") if end_freq.get("available") else None
    if frequency_start and frequency_end and float(frequency_start) > 0.0:
        frequency_drop_ratio = 1.0 / (float(frequency_end) / float(frequency_start))
        thermal_safety_factor = max(1.05, float(frequency_drop_ratio) * 1.5 if frequency_drop_ratio > 1.0 else 1.05)
    else:
        frequency_drop_ratio = None
        thermal_safety_factor = 1.20
    corrected_projection_hours = (
        float(gate["primary_projected_wall_hours"]) * correction_factor * thermal_safety_factor
        if mandatory_complete and math.isfinite(correction_factor)
        else math.nan
    )
    original_projection_hours = float(gate["primary_projected_wall_hours"])

    def _fraction(seconds: float) -> float:
        return float(seconds / mandatory_projection_seconds) if mandatory_projection_seconds > 0.0 else math.nan

    phase_fraction_table = {
        "noisy_optimizer": _fraction(mandatory_component_seconds["noisy_optimizer"]),
        "zne_optimizer": _fraction(mandatory_component_seconds["zne_optimizer"]),
        "lhs_validation": _fraction(mandatory_component_seconds["lhs_validation"]),
        "checkpoint": _fraction(mandatory_component_seconds["checkpoint"]),
    }
    objective_eval_time = float(sum(float(row.get("objective_eval_time_s", 0.0)) for row in mandatory_rows))
    cobyla_overhead = float(sum(float(row.get("cobyla_overhead_s", 0.0)) for row in mandatory_rows))
    setup_time = float(sum(float(row.get("setup_time_s", 0.0)) for row in mandatory_rows))
    phase_fraction_table.update(
        {
            "objective_eval_time_nested_fraction": _fraction(objective_eval_time),
            "cobyla_overhead_nested_fraction": _fraction(cobyla_overhead),
            "setup_nested_fraction": _fraction(setup_time),
        }
    )
    all_eval_ratios = [
        float(row["actual_cobyla_evaluations_mean"]) / float(e_cobyla)
        for row in mandatory_rows
        if row.get("operation") in {"noisy_optimizer", "zne_optimizer"} and row.get("actual_cobyla_evaluations_mean") is not None
    ]
    cobyla_cap_binding = bool(all_eval_ratios and float(np.mean(all_eval_ratios)) > 0.90)
    cobyla_early_stopping_dominant = bool(all_eval_ratios and float(np.mean(all_eval_ratios)) < 0.50)
    conservative_single_sample_factor = 1.15
    if math.isfinite(corrected_projection_hours):
        corrected_projection_hours *= conservative_single_sample_factor
    projection_confidence = (
        "partial_timebox_low"
        if not mandatory_complete
        else "low_with_optional_objective_crosscheck"
        if optional_rows
        else "low_single_full_slice"
    )

    if not mandatory_complete:
        preliminary_case = "O1_partial_timebox_user_review_required"
    elif corrected_projection_hours <= float(gate["accepted_local_wall_hours"]):
        preliminary_case = "Case_A_full_scope_potentially_feasible_before_optimization"
    elif corrected_projection_hours <= original_projection_hours:
        preliminary_case = "Case_B_scope_or_optimization_still_required"
    else:
        preliminary_case = "Case_C_risk_increased_user_review_required"

    report = {
        "artifact": "O1_runtime_report",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "native_o1_runtime_report_complete" if mandatory_complete else "native_o1_runtime_report_partial_timebox",
        "official_execution_authorized": False,
        "profiling_scope": "native_controlled_thread_O1_mandatory_slice_with_optional_second_slice" if optional_rows else "native_controlled_thread_O1_mandatory_slice",
        "environment_snapshot": environment,
        "thread_settings": environment["thread_settings"],
        "pre_run_feasibility_estimate": {
            "path": str((out / "O1_prerun_feasibility_estimate.json").relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(out / "O1_prerun_feasibility_estimate.json"),
            "estimated_native_O1_hours_with_safety": prerun["analytical_weighted_estimate"]["estimated_mandatory_hours_with_safety"],
            "native_O1_allowed": prerun["native_o1_allowed"],
            "fallback_used": False,
            "extrapolation_rule": "native K=100 E_COBYLA=300 measured; no reduced K/E cap extrapolation used",
        },
        "profiling_cells": rows,
        "phase_fraction_table": phase_fraction_table,
        "mandatory_component_seconds": mandatory_component_seconds,
        "mandatory_completion": {
            "expected_operations": expected_mandatory_operations,
            "completed_operations": completed_mandatory_operations,
            "complete": mandatory_complete,
            "stopped_by_timebox": mandatory_stopped_by_timebox,
            "native_timebox_seconds": native_timebox_seconds,
        },
        "corrected_projections": {
            "original_full_projection_hours": original_projection_hours,
            "accepted_local_wall_hours": float(gate["accepted_local_wall_hours"]),
            "estimated_mandatory_hours_without_warmup": estimated_mandatory_hours,
            "measured_mandatory_hours": measured_mandatory_hours,
            "correction_factor": correction_factor,
            "thermal_safety_factor": thermal_safety_factor,
            "conservative_single_sample_factor": conservative_single_sample_factor,
            "corrected_full_projection_hours": corrected_projection_hours,
            "projection_confidence": projection_confidence,
            "preliminary_case": preliminary_case,
        },
        "thermal": {
            "cpu_frequency_start": start_freq,
            "cpu_frequency_end": end_freq,
            "measured_frequency_drop_ratio": frequency_drop_ratio,
            "thermal_safety_factor": thermal_safety_factor,
        },
        "cobyla_evaluation_behavior": {
            "mean_actual_eval_ratio_to_cap": float(np.mean(all_eval_ratios)) if all_eval_ratios else math.nan,
            "max_actual_eval_ratio_to_cap": float(np.max(all_eval_ratios)) if all_eval_ratios else math.nan,
            "cobyla_cap_binding": cobyla_cap_binding,
            "cobyla_early_stopping_dominant": cobyla_early_stopping_dominant,
        },
        "execution_blockers_status": {
            "storage_threshold": "pending_post_O1_review",
            "memory_cap": "pending_post_O1_review",
            "corrected_runtime_projection": "defined",
            "official_execution_authorized": False,
        },
        "next_action": "review_O1_runtime_report_and_make_preliminary_Case_A_B_C_scope_decision",
        "total_wall_time_s": float(total_wall_s),
    }
    _write_csv(csv_path, rows)
    write_json_atomic(json_path, report, overwrite=True)
    md_lines = [
        "# O1 Runtime Report",
        "",
        f"Created UTC: `{report['created_at_utc']}`",
        f"Status: `{report['status']}`",
        "",
        "Official execution remains blocked.",
        "",
        "## Runtime",
        "",
        f"- Mandatory measured hours: `{measured_mandatory_hours:.3f}`",
        f"- Correction factor vs pre-run weighted estimate: `{correction_factor:.3f}`",
        f"- Thermal safety factor: `{thermal_safety_factor:.3f}`",
        f"- Corrected full projection: `{corrected_projection_hours:.3f}` h",
        f"- Projection confidence: `{projection_confidence}`",
        f"- Preliminary case: `{preliminary_case}`",
        "",
        "## Phase Fractions",
        "",
    ]
    for name, fraction in phase_fraction_table.items():
        md_lines.append(f"- {name}: `{fraction:.6f}`")
    md_lines.extend(["", f"Next action: `{report['next_action']}`"])
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    _write_o1_progress(progress_path, "o1_complete", {"total_wall_time_s": total_wall_s, "corrected_full_projection_hours": corrected_projection_hours})
    write_deviation_log_entry(
        root,
        step="O1",
        entry_type="amendment",
        description="Native O1 runtime report written; official execution remains blocked.",
        action_taken=f"Recorded corrected projection {corrected_projection_hours:.3f} h.",
        blocking=False,
        resolved=True,
    )
    return report


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def write_o1_partial_timebox_anomaly_review(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    json_path = out / "O1_partial_timebox_anomaly_review.json"
    md_path = out / "O1_partial_timebox_anomaly_review.md"
    status_path = out / "post_o1_status.json"
    status_md_path = out / "post_o1_status.md"
    if any(path.exists() for path in (json_path, md_path, status_path, status_md_path)) and not force:
        raise FileExistsError("Refusing to overwrite existing O1 partial timebox review artifacts.")

    runtime_path = out / "O1_runtime_report.json"
    if not runtime_path.exists():
        raise FileNotFoundError(f"O1 runtime report not found: {runtime_path}")
    runtime = _read_json(runtime_path)
    projection = runtime["corrected_projections"]
    completion = runtime["mandatory_completion"]
    phase_fractions = runtime["phase_fraction_table"]
    eval_behavior = runtime["cobyla_evaluation_behavior"]

    original_projection_hours = float(projection["original_full_projection_hours"])
    accepted_local_wall_hours = float(projection["accepted_local_wall_hours"])
    estimated_mandatory_hours = float(projection["estimated_mandatory_hours_without_warmup"])
    measured_partial_hours = float(projection["measured_mandatory_hours"])
    thermal_safety_factor = float(projection["thermal_safety_factor"])
    single_sample_factor = float(projection["conservative_single_sample_factor"])
    upward_threshold_hours = 1.5 * original_projection_hours
    lower_bound_correction_factor = measured_partial_hours / estimated_mandatory_hours
    lower_bound_projection_hours = original_projection_hours * lower_bound_correction_factor
    lower_bound_projection_with_thermal_hours = lower_bound_projection_hours * thermal_safety_factor
    conservative_lower_bound_projection_hours = lower_bound_projection_with_thermal_hours * single_sample_factor
    required_speedup_to_240_lower_bound = conservative_lower_bound_projection_hours / accepted_local_wall_hours
    lower_bound_exceeds_upward_rule = bool(lower_bound_projection_with_thermal_hours > upward_threshold_hours)

    profiling_cells = runtime.get("profiling_cells", [])
    completed_by_shot: dict[str, list[str]] = {}
    for row in profiling_cells:
        if row.get("slice") != "mandatory" or not row.get("use_for_projection"):
            continue
        shot = str(row.get("shots_total"))
        operation = str(row.get("operation"))
        order = row.get("richardson_order")
        label = operation if order in (None, "", "None") else f"{operation}:{order}"
        completed_by_shot.setdefault(shot, []).append(label)

    triggers = {
        "O2_density_kernel_objective": bool(
            float(phase_fractions.get("objective_eval_time_nested_fraction", 0.0)) >= 0.50
            or float(phase_fractions.get("zne_optimizer", 0.0)) >= 0.50
        ),
        "O3_batching": bool(float(phase_fractions.get("zne_optimizer", 0.0)) >= 0.50),
        "O4_static_cache": bool(float(phase_fractions.get("setup_nested_fraction", 0.0)) >= 0.05),
        "O5_lhs_cache": bool(float(phase_fractions.get("lhs_validation", 0.0)) >= 0.05),
        "O6_threading": "defer_until_after_O2_O3_or_no_code_optimization_decision",
        "O7_worker_write_resume": bool(float(phase_fractions.get("checkpoint", 0.0)) >= 0.05),
    }

    review = {
        "artifact": "O1_partial_timebox_anomaly_review",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "upward_correction_lower_bound_triggered_user_review_required"
        if lower_bound_exceeds_upward_rule
        else "partial_timebox_review_user_review_required",
        "official_execution_authorized": False,
        "optimization_authorized": False,
        "source_runtime_report": {
            "path": str(runtime_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(runtime_path),
            "status": runtime["status"],
        },
        "timebox_outcome": {
            "native_timebox_hours": float(completion["native_timebox_seconds"]) / 3600.0,
            "measured_total_wall_hours": float(runtime["total_wall_time_s"]) / 3600.0,
            "measured_mandatory_partial_hours": measured_partial_hours,
            "completed_operations": int(completion["completed_operations"]),
            "expected_operations": int(completion["expected_operations"]),
            "mandatory_complete": bool(completion["complete"]),
            "stopped_by_timebox": bool(completion["stopped_by_timebox"]),
            "completed_by_shot": completed_by_shot,
        },
        "projection_lower_bound": {
            "original_projection_hours": original_projection_hours,
            "accepted_local_wall_hours": accepted_local_wall_hours,
            "estimated_full_mandatory_slice_hours_pre_o1": estimated_mandatory_hours,
            "measured_partial_mandatory_slice_hours": measured_partial_hours,
            "lower_bound_correction_factor": lower_bound_correction_factor,
            "upward_correction_threshold_hours": upward_threshold_hours,
            "lower_bound_full_projection_hours_no_thermal": lower_bound_projection_hours,
            "lower_bound_full_projection_hours_with_thermal": lower_bound_projection_with_thermal_hours,
            "conservative_lower_bound_full_projection_hours": conservative_lower_bound_projection_hours,
            "required_speedup_to_fit_240h_lower_bound": required_speedup_to_240_lower_bound,
            "lower_bound_exceeds_1_5x_original_projection": lower_bound_exceeds_upward_rule,
            "interpretation": (
                "The full corrected projection cannot be finalized from a partial O1 slice, "
                "but the partial slice alone already exceeds the pre-run full-slice estimate. "
                "Therefore the true correction factor is at least this lower bound."
            ),
        },
        "measured_phase_fractions": phase_fractions,
        "cobyla_evaluation_behavior": eval_behavior,
        "o_step_triggers": triggers,
        "case_status": {
            "Case_A": "blocked_by_partial_O1_and_lower_bound_far_above_240h",
            "Case_B": "requires_scope_amendment_and_cubic_robustness_budget_after_review",
            "Case_C": "plausible_without_major_optimization_or_scope_reduction",
            "preliminary_decision": "pause_before_optimization_and_user_review",
        },
        "next_action": "user_review_then_choose_O2_O3_child_plan_or_scope_reduction",
        "notes": [
            "O1 was profiling only; no official execution was authorized.",
            "Partial O1 outputs are technical timing data, not scientific endpoint analysis.",
            "COBYLA early stopping is dominant, so cap-linear extrapolation from E_COBYLA is not reliable.",
            "Setup and checkpoint overhead are not material under the measured profile.",
        ],
    }
    write_json_atomic(json_path, review, overwrite=True)
    md_lines = [
        "# O1 Partial Timebox Anomaly Review",
        "",
        f"Created UTC: `{review['created_at_utc']}`",
        f"Status: `{review['status']}`",
        "",
        "Official execution remains blocked. Optimization is not authorized until user review records proceed/amend/stop.",
        "",
        "## Timebox",
        "",
        f"- O1 runtime report status: `{runtime['status']}`",
        f"- Completed operations: `{completion['completed_operations']}/{completion['expected_operations']}`",
        f"- Total wall time: `{float(runtime['total_wall_time_s']) / 3600.0:.3f}` h",
        f"- Measured partial mandatory time: `{measured_partial_hours:.3f}` h",
        "",
        "## Projection Lower Bound",
        "",
        f"- Pre-O1 full mandatory-slice estimate: `{estimated_mandatory_hours:.3f}` h",
        f"- Lower-bound correction factor: `{lower_bound_correction_factor:.3f}x`",
        f"- 1.5x upward-rule threshold: `{upward_threshold_hours:.1f}` h",
        f"- Lower-bound full projection with thermal: `{lower_bound_projection_with_thermal_hours:.1f}` h",
        f"- Conservative lower-bound full projection: `{conservative_lower_bound_projection_hours:.1f}` h",
        f"- Required speedup to fit 240 h lower bound: `{required_speedup_to_240_lower_bound:.2f}x`",
        "",
        "## Phase Fractions",
        "",
    ]
    for name, fraction in phase_fractions.items():
        md_lines.append(f"- {name}: `{float(fraction):.6f}`")
    md_lines.extend(
        [
            "",
            "## Triggered Steps",
            "",
            f"- O2 density/objective optimization: `{triggers['O2_density_kernel_objective']}`",
            f"- O3 batching: `{triggers['O3_batching']}`",
            f"- O5 LHS/cache review: `{triggers['O5_lhs_cache']}`",
            f"- O4 static cache: `{triggers['O4_static_cache']}`",
            f"- O7 worker/write overhead: `{triggers['O7_worker_write_resume']}`",
            "",
            f"Next action: `{review['next_action']}`",
        ]
    )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    post_status = {
        "artifact": "post_o1_status",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "post_o1_user_review_required_before_optimization",
        "official_execution_authorized": False,
        "optimization_authorized": False,
        "runtime_report_status": runtime["status"],
        "review_artifact": str(json_path.relative_to(root)).replace("\\", "/"),
        "review_status": review["status"],
        "next_action": review["next_action"],
    }
    write_json_atomic(status_path, post_status, overwrite=True)
    status_md_path.write_text(
        "\n".join(
            [
                "# Post-O1 Status",
                "",
                f"Created UTC: `{post_status['created_at_utc']}`",
                f"Status: `{post_status['status']}`",
                "",
                "Official execution remains blocked.",
                "Optimization remains blocked until user review.",
                "",
                f"Next action: `{post_status['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_deviation_log_entry(
        root,
        step="O1",
        entry_type="anomaly",
        description="O1 native profiling exceeded the 3 h timebox and the partial slice lower-bound correction exceeded the 1.5x upward-correction threshold.",
        action_taken=f"Recorded {json_path.relative_to(root)} and blocked optimization pending user review.",
        blocking=True,
        resolved=False,
    )
    return review


def write_post_o1_user_review_and_child_plan(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    decision_json = out / "O1_user_review_decision.json"
    decision_md = out / "O1_user_review_decision.md"
    plan_json = out / "post_o1_o2_o3_o5_child_plan_v0.1-draft.json"
    plan_md = out / "post_o1_o2_o3_o5_child_plan_v0.1-draft.md"
    status_json = out / "post_o1_status.json"
    status_md = out / "post_o1_status.md"
    if any(path.exists() for path in (decision_json, decision_md, plan_json, plan_md)) and not force:
        raise FileExistsError("Refusing to overwrite existing post-O1 user-review/child-plan artifacts.")

    review_path = out / "O1_partial_timebox_anomaly_review.json"
    runtime_path = out / "O1_runtime_report.json"
    if not review_path.exists():
        raise FileNotFoundError(f"O1 anomaly review not found: {review_path}")
    review = _read_json(review_path)
    runtime = _read_json(runtime_path)
    lower_bound = review["projection_lower_bound"]
    phase_fractions = review["measured_phase_fractions"]

    decision = {
        "artifact": "O1_user_review_decision",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "trigger": "O1_partial_timebox_anomaly_review",
        "decision": "amend",
        "decision_source": "user approved assistant recommendation to continue with a bounded O2/O3/O5 child plan",
        "official_execution_authorized": False,
        "full_o1_rerun_authorized": False,
        "bounded_optimization_planning_authorized": True,
        "bounded_optimization_code_changes_authorized": False,
        "rationale": [
            "Native O1 exceeded the 3 h timebox and produced a partial low-confidence report.",
            "The lower-bound correction factor already exceeds the 1.5x upward-correction threshold.",
            "Measured fractions show objective/ZNE and LHS are material, while setup/checkpoint overhead is negligible.",
            "The next safe action is a narrow child plan, not official execution or broad optimization.",
        ],
        "source_review": {
            "path": str(review_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(review_path),
            "status": review["status"],
        },
    }
    write_json_atomic(decision_json, decision, overwrite=True)
    decision_md.write_text(
        "\n".join(
            [
                "# O1 User Review Decision",
                "",
                f"Created UTC: `{decision['created_at_utc']}`",
                "Decision: `amend`",
                "",
                "Official execution remains blocked.",
                "Full O1 rerun is not authorized.",
                "Bounded optimization planning is authorized; code optimization remains gated by the child plan.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    required_speedup = float(lower_bound["required_speedup_to_fit_240h_lower_bound"])
    child_plan = {
        "artifact": "post_o1_o2_o3_o5_child_plan",
        "created_at_utc": _utc_now(),
        "version": "post_o1_o2_o3_o5_child_plan_v0.1-draft",
        "status": "draft_ready_before_optimization_code_changes",
        "official_execution_authorized": False,
        "optimization_authorized": False,
        "scope": {
            "included_steps": ["O2_density_kernel_objective", "O3_batching", "O5_lhs_cache_review"],
            "excluded_steps": {
                "O4_static_cache": "not material under O1 setup fraction",
                "O7_worker_write_resume": "not material under O1 checkpoint/write fraction",
                "O6_threading": "defer until after O2/O3, or use only if no code optimization proceeds",
            },
            "no_full_o1_rerun": True,
            "no_official_execution": True,
            "no_scientific_endpoint_analysis": True,
        },
        "source_artifacts": {
            "O1_runtime_report": {
                "path": str(runtime_path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(runtime_path),
                "status": runtime["status"],
            },
            "O1_partial_timebox_anomaly_review": {
                "path": str(review_path.relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(review_path),
                "status": review["status"],
            },
            "O1_user_review_decision": {
                "path": str(decision_json.relative_to(root)).replace("\\", "/"),
                "sha256": sha256_file(decision_json),
                "decision": "amend",
            },
        },
        "measured_basis": {
            "phase_fractions": phase_fractions,
            "required_speedup_to_fit_240h_lower_bound": required_speedup,
            "conservative_lower_bound_full_projection_hours": float(lower_bound["conservative_lower_bound_full_projection_hours"]),
            "cobyla_early_stopping_dominant": bool(review["cobyla_evaluation_behavior"]["cobyla_early_stopping_dominant"]),
            "cobyla_cap_binding": bool(review["cobyla_evaluation_behavior"]["cobyla_cap_binding"]),
        },
        "hard_timebox_hours": {
            "total_child_plan_cap": 8.0,
            "O2_implementation_and_validation": 4.0,
            "O3_implementation_and_validation": 2.0,
            "O5_review_and_validation": 1.5,
            "post_child_decision_report": 0.5,
        },
        "success_and_stop_rules": {
            "minimum_to_continue_after_O2": ">=1.50x objective-evaluation speedup on fixed microbench or stop O2 and proceed to scope reduction",
            "minimum_to_continue_after_O3": ">=3.00x combined O2+O3 task-wall speedup on fixed microbench or stop optimization and proceed to scope reduction",
            "full_cubic_feasibility_threshold": f">={required_speedup:.2f}x lower-bound speedup plus equivalence pass; otherwise full cubic remains locally infeasible",
            "O5_acceptance": "material LHS reduction or >=1.03x wall-time speedup without seed/theta/row-order changes",
            "hard_stop": "if total child-plan effort reaches 8 h without >=3x credible speedup, stop optimization and write scope amendment",
        },
        "allowed_measurement_scope": {
            "microbench_only": True,
            "preferred_cells": [
                {
                    "instance": "H4",
                    "p_dep": 0.01,
                    "shots_total": 512,
                    "order": "cubic",
                    "reason": "hard O1 cell already measured; captures dominant ZNE/density cost",
                },
                {
                    "instance": "H4",
                    "p_dep": 0.01,
                    "shots_total": 512,
                    "order": "linear",
                    "reason": "lower-order comparison already measured",
                },
            ],
            "reduced_optimizer_probe": {
                "K_micro": 5,
                "E_COBYLA": 300,
                "purpose": "measure per-evaluation speedup without repeating full K=100 profiling",
                "rule": "do not extrapolate from E cap; record actual evaluations used",
            },
            "equivalence_smoke": {
                "classification_agreement_required": "100%",
                "tolerance_tier": "Tier 1 unless reduction-order change is explicitly documented, then Tier 2",
                "sign_of_d_R_must_match": True,
            },
        },
        "step_interfaces": {
            "O2_density_kernel_objective": {
                "triggered": bool(review["o_step_triggers"]["O2_density_kernel_objective"]),
                "input_dependencies": ["O1_runtime_report", "O0.75_cost_model", "equivalence_smoke_set"],
                "output_artifact": "O2_optimization_report.json/md",
                "allowed_work": "profile and optimize density/objective evaluation only; no scientific scope or seed changes",
                "rollback": "revert if equivalence fails or independent speedup is not credible",
            },
            "O3_batching": {
                "triggered": bool(review["o_step_triggers"]["O3_batching"]),
                "input_dependencies": ["O1_runtime_report", "accepted_O2_state_if_any"],
                "output_artifact": "O3_batching_report.json/md",
                "allowed_work": "batch/reuse deterministic per-scale work only when PRNG identity and outputs remain schema-compatible",
                "rollback": "revert failed O3; revert provisional O2 if composite rule fails and O2 lacks independent acceptance",
            },
            "O5_lhs_cache_review": {
                "triggered": bool(review["o_step_triggers"]["O5_lhs_cache"]),
                "input_dependencies": ["O1_runtime_report", "frozen_LHS_seed_table_hash"],
                "output_artifact": "O5_lhs_cache_report.json/md",
                "allowed_work": "cache deterministic ideal/static LHS values only; no noisy/ZNE sampled reuse without amendment",
                "rollback": "revert if seed, theta, row order, or schema identity changes",
            },
        },
        "next_action": "inspect_density_backend_hot_path_and_write_O2_microbench_plan",
    }
    write_json_atomic(plan_json, child_plan, overwrite=True)
    md_lines = [
        "# Post-O1 O2/O3/O5 Child Plan",
        "",
        f"Created UTC: `{child_plan['created_at_utc']}`",
        "Status: `draft_ready_before_optimization_code_changes`",
        "",
        "Official execution remains blocked. This plan authorizes only a bounded optimization planning path.",
        "",
        "## Scope",
        "",
        "- Include: `O2`, `O3`, `O5`.",
        "- Exclude for now: `O4`, `O7`.",
        "- Defer: `O6` until after O2/O3 or if code optimization is skipped.",
        "",
        "## Timebox",
        "",
        "- Total cap: `8.0 h`.",
        "- O2: `4.0 h` including validation.",
        "- O3: `2.0 h` including validation.",
        "- O5: `1.5 h` including validation.",
        "- Decision report: `0.5 h`.",
        "",
        "## Gates",
        "",
        "- Continue after O2 only with `>=1.50x` objective-evaluation speedup.",
        "- Continue after O3 only with `>=3.00x` combined task-wall speedup.",
        f"- Full cubic remains locally infeasible unless speedup reaches at least `{required_speedup:.2f}x` plus equivalence.",
        "",
        f"Next action: `{child_plan['next_action']}`",
    ]
    plan_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    post_status = {
        "artifact": "post_o1_status",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "post_o1_child_plan_ready_optimization_not_started",
        "official_execution_authorized": False,
        "optimization_authorized": False,
        "user_review_decision": "amend",
        "child_plan_artifact": str(plan_json.relative_to(root)).replace("\\", "/"),
        "next_action": child_plan["next_action"],
    }
    write_json_atomic(status_json, post_status, overwrite=True)
    status_md.write_text(
        "\n".join(
            [
                "# Post-O1 Status",
                "",
                f"Created UTC: `{post_status['created_at_utc']}`",
                f"Status: `{post_status['status']}`",
                "",
                "User review decision: `amend`.",
                "Official execution remains blocked.",
                "Optimization code changes have not started.",
                "",
                f"Next action: `{post_status['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_deviation_log_entry(
        root,
        step="O1",
        entry_type="user_review",
        description="User reviewed O1 partial-timebox anomaly and chose amend path.",
        action_taken=f"Recorded {decision_json.relative_to(root)} and {plan_json.relative_to(root)}.",
        blocking=False,
        resolved=True,
    )
    return {"user_review_decision": decision, "child_plan": child_plan, "post_o1_status": post_status}


def write_o2_density_hot_path_microbench_plan(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    review_json = out / "O2_density_hot_path_review.json"
    review_md = out / "O2_density_hot_path_review.md"
    plan_json = out / "O2_density_microbench_plan_v0.1-draft.json"
    plan_md = out / "O2_density_microbench_plan_v0.1-draft.md"
    status_json = out / "post_o1_status.json"
    status_md = out / "post_o1_status.md"
    if any(path.exists() for path in (review_json, review_md, plan_json, plan_md)) and not force:
        raise FileExistsError("Refusing to overwrite existing O2 hot-path/microbench artifacts.")

    child_plan_path = out / "post_o1_o2_o3_o5_child_plan_v0.1-draft.json"
    decision_path = out / "O1_user_review_decision.json"
    runtime_path = out / "O1_runtime_report.json"
    if not child_plan_path.exists():
        raise FileNotFoundError(f"Post-O1 child plan not found: {child_plan_path}")
    if not decision_path.exists():
        raise FileNotFoundError(f"O1 user review decision not found: {decision_path}")
    child_plan = _read_json(child_plan_path)
    decision = _read_json(decision_path)
    runtime = _read_json(runtime_path)
    if decision.get("decision") != "amend":
        raise RuntimeError("O2 planning requires the recorded O1 user-review decision to be amend.")
    if child_plan.get("official_execution_authorized") is not False:
        raise RuntimeError("O2 planning requires official execution to remain disabled.")

    source_files = [
        "src/phase_e_sim_backend.py",
        "src/phase_c_120_custom_kernel.py",
        "src/phase_e_cobyla_runtime_optimization.py",
        "src/phase_e_optimizers.py",
        "scripts/79_phase_e_cobyla_runtime_o0_o0_75.py",
    ]
    input_artifacts = [
        "results/phase_e/calibration/cobyla_runtime_optimization/O1_runtime_report.json",
        "results/phase_e/calibration/cobyla_runtime_optimization/O1_partial_timebox_anomaly_review.json",
        "results/phase_e/calibration/cobyla_runtime_optimization/O1_user_review_decision.json",
        "results/phase_e/calibration/cobyla_runtime_optimization/post_o1_o2_o3_o5_child_plan_v0.1-draft.json",
        "config/phase_e_config_v0.1-draft.json",
        "config/phase_e_instances_v0.1-draft.json",
    ]

    hot_path_review = {
        "artifact": "O2_density_hot_path_review",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "review_written_no_optimization_started",
        "official_execution_authorized": False,
        "optimization_code_changes_authorized": False,
        "source_hashes": _hashes_for_existing(root, source_files),
        "input_hashes": _hashes_for_existing(root, input_artifacts),
        "measured_o1_basis": {
            "runtime_status": runtime["status"],
            "phase_fractions": runtime["phase_fraction_table"],
            "cobyla_mean_actual_eval_ratio_to_cap": runtime["cobyla_evaluation_behavior"]["mean_actual_eval_ratio_to_cap"],
            "cobyla_cap_binding": runtime["cobyla_evaluation_behavior"]["cobyla_cap_binding"],
            "cobyla_early_stopping_dominant": runtime["cobyla_evaluation_behavior"]["cobyla_early_stopping_dominant"],
        },
        "call_graph": [
            {
                "step": "objective_entry",
                "function": "evaluate_custom_density_objective",
                "file": "src/phase_e_sim_backend.py:356",
                "role": "calls custom_density_probabilities, then converts probabilities to expected-cut objective and optional shot sample",
            },
            {
                "step": "probability_entry",
                "function": "custom_density_probabilities",
                "file": "src/phase_e_sim_backend.py:241",
                "role": "wraps theta, builds PhaseENoiseRates, obtains folded transpiled circuit, and calls density simulator",
            },
            {
                "step": "template_assignment",
                "function": "build_folded_transpiled_qaoa_depth",
                "file": "src/phase_e_sim_backend.py:166",
                "role": "reuses lru-cached folded/transpiled parameterized circuit and assigns theta values",
            },
            {
                "step": "density_loop",
                "function": "probabilities_from_transpiled_circuit_phase_e",
                "file": "src/phase_e_sim_backend.py:193",
                "role": "initializes rho and loops over circuit instructions, applying local unitary and depolarizing maps",
            },
            {
                "step": "unitary_update",
                "function": "_apply_local_unitary_density",
                "file": "src/phase_c_120_custom_kernel.py:126",
                "role": "left/right block updates on the density matrix for each one- or two-qubit gate",
            },
            {
                "step": "depolarizing_update",
                "function": "_apply_local_depolarizing",
                "file": "src/phase_c_120_custom_kernel.py:169",
                "role": "tensor reshape/transpose/einsum local depolarizing channel after noisy gates",
            },
            {
                "step": "zne_wrapper",
                "function": "richardson_zne_objective",
                "file": "src/phase_e_sim_backend.py:391",
                "role": "repeats the objective over Richardson scale factors and combines weighted values",
            },
        ],
        "measurement_priorities": [
            "separate template parameter assignment from density-loop time",
            "separate gate matrix construction, qarg extraction, unitary update, depolarizing update, and postprocess time",
            "record per-scale objective time for scale factors 1, 3, 5, 7",
            "record objective values and probabilities hash for equivalence before and after any later optimization",
            "keep OMP/MKL/OPENBLAS/NUMEXPR at 1 thread during O2 baseline",
        ],
        "preliminary_risk_assessment": {
            "transpile_template_cache": "already present through lru_cache; not the first O2 target unless microbench disproves this",
            "density_loop": "most likely target because O1 nested objective fraction is about 0.996",
            "depolarizing_channel": "likely material at p_dep > 0 because every supported gate applies local noise",
            "shot_sampling": "expected small but must be measured because official objectives use synthetic multinomial shots",
            "scientific_risk": "any accepted optimization must preserve endpoint classification, d_R sign, schema, seed identity, and probability/objective tolerances",
        },
        "next_action": "run_o2_baseline_microbench_no_optimization",
    }
    write_json_atomic(review_json, hot_path_review, overwrite=True)
    review_md.write_text(
        "\n".join(
            [
                "# O2 Density Hot-Path Review",
                "",
                f"Created UTC: `{hot_path_review['created_at_utc']}`",
                "Status: `review_written_no_optimization_started`",
                "",
                "O1 shows objective evaluation dominates runtime, so O2 should measure the density backend before any optimization.",
                "",
                "Primary path:",
                "",
                "- `evaluate_custom_density_objective` -> `custom_density_probabilities`.",
                "- `build_folded_transpiled_qaoa_depth` reuses the cached folded/transpiled parameterized template and assigns theta.",
                "- `probabilities_from_transpiled_circuit_phase_e` initializes rho and applies unitary plus depolarizing maps instruction by instruction.",
                "- `_apply_local_unitary_density` and `_apply_local_depolarizing` are the first functions to time at subphase level.",
                "- `richardson_zne_objective` multiplies this cost by the number of scale factors.",
                "",
                "No optimization or official execution is authorized by this artifact.",
                "",
                f"Next action: `{hot_path_review['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    microbench_plan = {
        "artifact": "O2_density_microbench_plan",
        "created_at_utc": _utc_now(),
        "version": "O2_density_microbench_plan_v0.1-draft",
        "status": "baseline_microbench_plan_ready_no_optimization_started",
        "official_execution_authorized": False,
        "optimization_code_changes_authorized": False,
        "parent_child_plan": {
            "path": str(child_plan_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(child_plan_path),
            "hard_total_child_cap_hours": child_plan["hard_timebox_hours"]["total_child_plan_cap"],
            "o2_cap_hours": child_plan["hard_timebox_hours"]["O2_implementation_and_validation"],
        },
        "baseline_measurement_scope": {
            "purpose": "establish subphase baseline for density/objective hot path before optimization",
            "not_scientific_endpoint_analysis": True,
            "cells": [
                {
                    "instance": "H4",
                    "p_dep": 0.01,
                    "shots_total": 512,
                    "richardson_order": "cubic",
                    "scale_factors": [1, 3, 5, 7],
                    "reason": "hard O1 cell with cubic timing already preserved",
                },
                {
                    "instance": "H4",
                    "p_dep": 0.01,
                    "shots_total": 512,
                    "richardson_order": "linear",
                    "scale_factors": [1, 3],
                    "reason": "lower-order comparator from the same O1 measured cell",
                },
            ],
            "theta_generation": {
                "depth": 2,
                "source": "make_paired_initial_thetas(depth=2, starts=3, seed=20260521)",
                "theta_count": 3,
            },
            "timing_protocol": {
                "thread_env": {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                },
                "warmup": "one untimed cubic objective on H4 before measured repeats",
                "measured_repeats_per_theta": 3,
                "max_baseline_microbench_wall_minutes": 20,
                "stop_rule": "stop and write anomaly if baseline microbench exceeds 20 minutes",
            },
            "subphases_to_record": [
                "theta_wrap_and_noise_rates",
                "template_assignment",
                "rho_initialization",
                "qargs_extraction",
                "gate_matrix_construction",
                "unitary_density_update",
                "depolarizing_update",
                "probability_normalization",
                "expected_cut_dot",
                "shot_multinomial_sampling",
            ],
        },
        "output_artifacts": {
            "json": "results/phase_e/calibration/cobyla_runtime_optimization/O2_density_microbench_baseline.json",
            "csv": "results/phase_e/calibration/cobyla_runtime_optimization/O2_density_microbench_baseline.csv",
            "markdown": "results/phase_e/calibration/cobyla_runtime_optimization/O2_density_microbench_baseline.md",
        },
        "acceptance_for_next_step": {
            "run_baseline_only_first": True,
            "optimization_may_start_only_after": [
                "baseline artifact exists",
                "equivalence keys and probability/objective hashes are recorded",
                "subphase fractions identify at most two plausible O2 edit targets",
                "no official execution or scientific endpoint analysis has occurred",
            ],
            "minimum_to_continue_after_O2": child_plan["success_and_stop_rules"]["minimum_to_continue_after_O2"],
            "if_no_single_subphase_exceeds_50_percent": "do not start broad O2 edits; write decision note and move to O3/O5 or scope reduction",
        },
        "prohibited_changes_during_baseline": [
            "no kernel optimization",
            "no change to QAOA circuit construction",
            "no seed, theta, shot, scale-factor, or instance changes",
            "no official task manifest execution",
            "no endpoint classification or scientific outcome inspection",
        ],
        "next_action": "run_o2_baseline_microbench_no_optimization",
    }
    write_json_atomic(plan_json, microbench_plan, overwrite=True)
    plan_md.write_text(
        "\n".join(
            [
                "# O2 Density Micro-Benchmark Plan",
                "",
                f"Created UTC: `{microbench_plan['created_at_utc']}`",
                "Status: `baseline_microbench_plan_ready_no_optimization_started`",
                "",
                "This is a baseline-only measurement plan. It does not authorize optimization code changes or official execution.",
                "",
                "Cells:",
                "",
                "- H4, `p_dep=0.01`, `shots_total=512`, cubic `[1,3,5,7]`.",
                "- H4, `p_dep=0.01`, `shots_total=512`, linear `[1,3]`.",
                "",
                "Timing:",
                "",
                "- Force `OMP/MKL/OPENBLAS/NUMEXPR=1`.",
                "- One warm-up cubic objective.",
                "- Three deterministic depth-2 theta points, three measured repeats per theta.",
                "- Stop and write anomaly if the baseline exceeds `20` minutes.",
                "",
                "Subphases to record include template assignment, rho initialization, qarg/gate construction, unitary update, depolarizing update, probability normalization, expected-cut dot, and shot sampling.",
                "",
                f"Next action: `{microbench_plan['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    post_status = {
        "artifact": "post_o1_status",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "o2_microbench_plan_ready_optimization_not_started",
        "official_execution_authorized": False,
        "optimization_authorized": False,
        "user_review_decision": "amend",
        "child_plan_artifact": str(child_plan_path.relative_to(root)).replace("\\", "/"),
        "o2_hot_path_review_artifact": str(review_json.relative_to(root)).replace("\\", "/"),
        "o2_microbench_plan_artifact": str(plan_json.relative_to(root)).replace("\\", "/"),
        "next_action": microbench_plan["next_action"],
    }
    write_json_atomic(status_json, post_status, overwrite=True)
    status_md.write_text(
        "\n".join(
            [
                "# Post-O1 Status",
                "",
                f"Created UTC: `{post_status['created_at_utc']}`",
                f"Status: `{post_status['status']}`",
                "",
                "User review decision: `amend`.",
                "Official execution remains blocked.",
                "Optimization code changes have not started.",
                "",
                f"O2 hot-path review: `{post_status['o2_hot_path_review_artifact']}`",
                f"O2 micro-benchmark plan: `{post_status['o2_microbench_plan_artifact']}`",
                f"Next action: `{post_status['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_deviation_log_entry(
        root,
        step="O2",
        entry_type="amendment",
        description="O2 density hot-path review and baseline micro-benchmark plan written.",
        action_taken=f"Recorded {review_json.relative_to(root)} and {plan_json.relative_to(root)}.",
        blocking=False,
        resolved=True,
    )
    return {"hot_path_review": hot_path_review, "microbench_plan": microbench_plan, "post_o1_status": post_status}


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _add_elapsed(totals: dict[str, float], key: str, elapsed: float) -> None:
    totals[key] = totals.get(key, 0.0) + float(elapsed)


def _instrumented_density_objective_for_o2(
    graph: GraphInstance,
    theta: tuple[float, ...],
    *,
    p_dep: float,
    scale_factor: int,
    shots: int,
    seed: int,
    p_1q_ratio: float = 0.1,
) -> dict[str, Any]:
    timings = {
        "theta_wrap_and_noise_rates": 0.0,
        "template_assignment": 0.0,
        "rho_initialization": 0.0,
        "qargs_extraction": 0.0,
        "gate_matrix_construction": 0.0,
        "unitary_density_update": 0.0,
        "depolarizing_update": 0.0,
        "probability_normalization": 0.0,
        "expected_cut_dot": 0.0,
        "shot_multinomial_sampling": 0.0,
    }
    wall_start = time.perf_counter()

    started = time.perf_counter()
    values = wrap_theta(theta)
    noise_rates = PhaseENoiseRates.from_p_dep(p_dep, p_1q_ratio=p_1q_ratio)
    _add_elapsed(timings, "theta_wrap_and_noise_rates", time.perf_counter() - started)

    started = time.perf_counter()
    circuit = build_folded_transpiled_qaoa_depth(graph, values, int(scale_factor), noise_rates)
    _add_elapsed(timings, "template_assignment", time.perf_counter() - started)

    n_qubits = circuit.num_qubits
    dim = 1 << n_qubits
    started = time.perf_counter()
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    gate_count = 0
    _add_elapsed(timings, "rho_initialization", time.perf_counter() - started)

    for instruction in circuit.data:
        name = instruction.operation.name
        if name in {"barrier", "measure", "save_probabilities"}:
            continue
        started = time.perf_counter()
        targets = _qargs_to_indices(circuit, instruction)
        _add_elapsed(timings, "qargs_extraction", time.perf_counter() - started)

        started = time.perf_counter()
        gate = _gate_matrix(name, list(instruction.operation.params))
        _add_elapsed(timings, "gate_matrix_construction", time.perf_counter() - started)

        started = time.perf_counter()
        rho = _apply_local_unitary_density(rho, gate, targets, n_qubits)
        _add_elapsed(timings, "unitary_density_update", time.perf_counter() - started)
        gate_count += 1

        started = time.perf_counter()
        if len(targets) == 1 and name in {"h", "rx", "rz"}:
            rho = _apply_local_depolarizing(rho, targets, n_qubits, noise_rates.p_1q)
        elif len(targets) == 2 and name == "cx":
            rho = _apply_local_depolarizing(rho, targets, n_qubits, noise_rates.p_2q)
        else:
            raise ValueError(f"No Phase E depolarizing rule for instruction {name} on {len(targets)} qubits.")
        _add_elapsed(timings, "depolarizing_update", time.perf_counter() - started)

    started = time.perf_counter()
    probabilities = np.real(np.diag(rho)).astype(float)
    min_prob = float(np.min(probabilities))
    if min_prob < -1e-10:
        raise RuntimeError(f"O2 microbench produced negative probability below tolerance: {min_prob}")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise RuntimeError("O2 microbench produced zero total probability.")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        probabilities = probabilities / total
    _add_elapsed(timings, "probability_normalization", time.perf_counter() - started)

    cut_values = np.zeros(dim, dtype=float)
    for index in range(dim):
        bits = [(index >> q) & 1 for q in range(graph.n_qubits)]
        cut_values[index] = sum(float(weight) for u, v, weight in graph.edges if bits[int(u)] != bits[int(v)])

    started = time.perf_counter()
    exact_expected_cut = float(np.dot(probabilities, cut_values))
    _add_elapsed(timings, "expected_cut_dot", time.perf_counter() - started)

    started = time.perf_counter()
    rng = np.random.default_rng(int(seed))
    counts = rng.multinomial(int(shots), probabilities)
    sampled_expected_cut = float(np.dot(counts / float(shots), cut_values))
    _add_elapsed(timings, "shot_multinomial_sampling", time.perf_counter() - started)

    wall_time = float(time.perf_counter() - wall_start)
    measured_sum = float(sum(timings.values()))
    return {
        "wall_time_s": wall_time,
        "measured_subphase_time_s": measured_sum,
        "unattributed_time_s": float(max(0.0, wall_time - measured_sum)),
        "objective_value": -float(sampled_expected_cut),
        "exact_objective_value": -float(exact_expected_cut),
        "probabilities_sha256": _array_sha256(probabilities),
        "probability_sum": float(np.sum(probabilities)),
        "transpiled_gate_count": int(gate_count),
        "transpiled_depth": int(circuit.depth() or 0),
        "scale_factor": int(scale_factor),
        "backend_version": PHASE_E_BACKEND_VERSION,
        "subphase_time_s": {key: float(value) for key, value in timings.items()},
    }


def run_o2_density_baseline_microbench(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    baseline_json = out / "O2_density_microbench_baseline.json"
    baseline_csv = out / "O2_density_microbench_baseline.csv"
    baseline_md = out / "O2_density_microbench_baseline.md"
    status_json = out / "post_o1_status.json"
    status_md = out / "post_o1_status.md"
    if any(path.exists() for path in (baseline_json, baseline_csv, baseline_md)) and not force:
        raise FileExistsError("Refusing to overwrite existing O2 baseline microbench artifacts.")

    plan_path = out / "O2_density_microbench_plan_v0.1-draft.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"O2 microbench plan not found: {plan_path}")
    plan = _read_json(plan_path)
    if plan.get("optimization_code_changes_authorized") is not False:
        raise RuntimeError("O2 baseline requires optimization code changes to remain unauthorized.")

    for name, value in plan["baseline_measurement_scope"]["timing_protocol"]["thread_env"].items():
        os.environ.setdefault(str(name), str(value))

    graph = _graph_from_selected_instance(root, "H4")
    theta_seed = 20260521
    theta_values = make_paired_initial_thetas(depth=2, starts=3, seed=theta_seed)
    repeats = int(plan["baseline_measurement_scope"]["timing_protocol"]["measured_repeats_per_theta"])
    stop_after_s = float(plan["baseline_measurement_scope"]["timing_protocol"]["max_baseline_microbench_wall_minutes"]) * 60.0
    benchmark_started = time.perf_counter()

    warmup_start = time.perf_counter()
    richardson_zne_objective(
        graph,
        theta_values[0],
        p_dep=0.01,
        scale_factors=[1, 3, 5, 7],
        shots_total=512,
        seed=deterministic_seed("phase_e_o2_microbench_warmup", graph.name, theta_seed),
    )
    warmup_wall_s = float(time.perf_counter() - warmup_start)

    rows: list[dict[str, Any]] = []
    subphase_totals: dict[str, float] = {}
    order_totals: dict[str, float] = {}
    scale_totals: dict[str, float] = {}
    for cell in plan["baseline_measurement_scope"]["cells"]:
        order = str(cell["richardson_order"])
        scale_factors = [int(scale) for scale in cell["scale_factors"]]
        weights = richardson_weights(scale_factors)
        shot_allocation = allocate_shots(int(cell["shots_total"]), weights, strategy="proportional_to_abs_weights")
        for theta_index, theta in enumerate(theta_values):
            for repeat_index in range(repeats):
                for scale_index, scale in enumerate(scale_factors):
                    if time.perf_counter() - benchmark_started > stop_after_s:
                        raise TimeoutError("O2 baseline microbench exceeded the 20-minute stop rule.")
                    shot_seed = deterministic_seed(
                        "phase_e_o2_microbench",
                        graph.name,
                        tuple(wrap_theta(theta)),
                        cell["p_dep"],
                        int(cell["shots_total"]),
                        theta_seed,
                        order,
                        repeat_index,
                        scale,
                    )
                    result = _instrumented_density_objective_for_o2(
                        graph,
                        theta,
                        p_dep=float(cell["p_dep"]),
                        scale_factor=scale,
                        shots=int(shot_allocation[scale_index]),
                        seed=shot_seed,
                    )
                    for key, value in result["subphase_time_s"].items():
                        _add_elapsed(subphase_totals, key, float(value))
                    _add_elapsed(order_totals, order, float(result["wall_time_s"]))
                    _add_elapsed(scale_totals, str(scale), float(result["wall_time_s"]))
                    row = {
                        "artifact": "O2_density_microbench_baseline",
                        "instance": graph.name,
                        "p_dep": float(cell["p_dep"]),
                        "shots_total": int(cell["shots_total"]),
                        "richardson_order": order,
                        "scale_factor": int(scale),
                        "allocated_shots": int(shot_allocation[scale_index]),
                        "theta_seed": int(theta_seed),
                        "theta_index": int(theta_index),
                        "repeat_index": int(repeat_index),
                        "wall_time_s": float(result["wall_time_s"]),
                        "measured_subphase_time_s": float(result["measured_subphase_time_s"]),
                        "unattributed_time_s": float(result["unattributed_time_s"]),
                        "objective_value": float(result["objective_value"]),
                        "exact_objective_value": float(result["exact_objective_value"]),
                        "probabilities_sha256": result["probabilities_sha256"],
                        "probability_sum": float(result["probability_sum"]),
                        "transpiled_gate_count": int(result["transpiled_gate_count"]),
                        "transpiled_depth": int(result["transpiled_depth"]),
                        "backend_version": result["backend_version"],
                    }
                    for key, value in result["subphase_time_s"].items():
                        row[f"subphase_{key}_s"] = float(value)
                    rows.append(row)

    total_wall_s = float(sum(float(row["wall_time_s"]) for row in rows))
    subphase_fractions = {key: float(value / total_wall_s) if total_wall_s > 0.0 else math.nan for key, value in subphase_totals.items()}
    top_subphases = sorted(
        [{"subphase": key, "time_s": float(value), "fraction_of_wall": float(subphase_fractions[key])} for key, value in subphase_totals.items()],
        key=lambda row: float(row["time_s"]),
        reverse=True,
    )
    likely_o2_targets = [row["subphase"] for row in top_subphases if float(row["fraction_of_wall"]) >= 0.10][:2]
    broad_o2_edit_allowed_by_measurement = bool(top_subphases and float(top_subphases[0]["fraction_of_wall"]) >= 0.50)
    next_action = (
        "write_o2_candidate_patch_plan_before_optimization"
        if broad_o2_edit_allowed_by_measurement
        else "write_o2_no_broad_edit_decision_and_move_to_o3_o5_or_scope_reduction"
    )

    baseline = {
        "artifact": "O2_density_microbench_baseline",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "o2_baseline_microbench_complete_optimization_not_started",
        "official_execution_authorized": False,
        "optimization_code_changes_authorized": False,
        "source_plan": {
            "path": str(plan_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(plan_path),
        },
        "thread_env": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "measurement_scope": {
            "instance": graph.name,
            "p_dep": 0.01,
            "shots_total": 512,
            "orders": ["cubic", "linear"],
            "theta_seed": theta_seed,
            "theta_count": len(theta_values),
            "repeats_per_theta": repeats,
            "density_objective_calls": len(rows),
            "warmup_wall_s": warmup_wall_s,
        },
        "summary": {
            "total_measured_rows": len(rows),
            "total_density_wall_time_s": total_wall_s,
            "benchmark_wall_time_s_including_warmup": float(time.perf_counter() - benchmark_started),
            "subphase_totals_s": {key: float(value) for key, value in subphase_totals.items()},
            "subphase_fractions_of_density_wall": subphase_fractions,
            "top_subphases": top_subphases,
            "order_wall_time_s": {key: float(value) for key, value in order_totals.items()},
            "scale_wall_time_s": {key: float(value) for key, value in scale_totals.items()},
            "likely_o2_targets": likely_o2_targets,
            "broad_o2_edit_allowed_by_measurement": broad_o2_edit_allowed_by_measurement,
        },
        "guards": {
            "not_scientific_endpoint_analysis": True,
            "no_kernel_optimization_applied": True,
            "no_official_task_manifest_execution": True,
            "next_step_still_requires_explicit_patch_plan": True,
        },
        "next_action": next_action,
    }
    write_json_atomic(baseline_json, baseline, overwrite=True)
    _write_csv(baseline_csv, rows)
    md_lines = [
        "# O2 Density Micro-Benchmark Baseline",
        "",
        f"Created UTC: `{baseline['created_at_utc']}`",
        "Status: `o2_baseline_microbench_complete_optimization_not_started`",
        "",
        f"Density objective calls: `{len(rows)}`",
        f"Total density wall time: `{total_wall_s:.6f}` s",
        f"Benchmark wall time including warm-up: `{baseline['summary']['benchmark_wall_time_s_including_warmup']:.6f}` s",
        f"Warm-up wall time: `{warmup_wall_s:.6f}` s",
        "",
        "Top subphases:",
        "",
    ]
    for row in top_subphases[:6]:
        md_lines.append(f"- `{row['subphase']}`: `{row['time_s']:.6f}` s (`{row['fraction_of_wall']:.3f}` of wall)")
    md_lines.extend(
        [
            "",
            f"Likely O2 targets: `{likely_o2_targets}`",
            f"Broad O2 edit allowed by measurement: `{broad_o2_edit_allowed_by_measurement}`",
            "",
            "No optimization code changes or official execution were performed.",
            "",
            f"Next action: `{next_action}`",
        ]
    )
    baseline_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    post_status = {
        "artifact": "post_o1_status",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "o2_baseline_microbench_complete_optimization_not_started",
        "official_execution_authorized": False,
        "optimization_authorized": False,
        "user_review_decision": "amend",
        "o2_microbench_baseline_artifact": str(baseline_json.relative_to(root)).replace("\\", "/"),
        "next_action": next_action,
    }
    write_json_atomic(status_json, post_status, overwrite=True)
    status_md.write_text(
        "\n".join(
            [
                "# Post-O1 Status",
                "",
                f"Created UTC: `{post_status['created_at_utc']}`",
                f"Status: `{post_status['status']}`",
                "",
                "User review decision: `amend`.",
                "Official execution remains blocked.",
                "Optimization code changes have not started.",
                "",
                f"O2 baseline: `{post_status['o2_microbench_baseline_artifact']}`",
                f"Next action: `{post_status['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_deviation_log_entry(
        root,
        step="O2",
        entry_type="amendment",
        description="O2 baseline density micro-benchmark completed without optimization code changes.",
        action_taken=f"Recorded {baseline_json.relative_to(root)}, {baseline_csv.relative_to(root)}, and {baseline_md.relative_to(root)}.",
        blocking=False,
        resolved=True,
    )
    return baseline


def write_o2_candidate_patch_plan(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    patch_json = out / "O2_candidate_patch_plan_v0.1-draft.json"
    patch_md = out / "O2_candidate_patch_plan_v0.1-draft.md"
    status_json = out / "post_o1_status.json"
    status_md = out / "post_o1_status.md"
    if any(path.exists() for path in (patch_json, patch_md)) and not force:
        raise FileExistsError("Refusing to overwrite existing O2 candidate patch plan artifacts.")

    baseline_path = out / "O2_density_microbench_baseline.json"
    child_plan_path = out / "post_o1_o2_o3_o5_child_plan_v0.1-draft.json"
    runtime_path = out / "O1_runtime_report.json"
    if not baseline_path.exists():
        raise FileNotFoundError(f"O2 baseline microbench not found: {baseline_path}")
    baseline = _read_json(baseline_path)
    child_plan = _read_json(child_plan_path)
    runtime = _read_json(runtime_path)
    if baseline.get("status") != "o2_baseline_microbench_complete_optimization_not_started":
        raise RuntimeError("O2 patch plan requires a completed baseline microbench.")
    summary = baseline["summary"]
    top_subphases = summary["top_subphases"]
    top_names = [str(row["subphase"]) for row in top_subphases[:2]]
    if "unitary_density_update" not in top_names:
        raise RuntimeError("O2 patch plan expected unitary_density_update to be a leading measured subphase.")
    objective_nested_fraction = float(runtime["phase_fraction_table"]["objective_eval_time_nested_fraction"])
    unitary_density_fraction = float(summary["subphase_fractions_of_density_wall"]["unitary_density_update"])
    depolarizing_density_fraction = float(summary["subphase_fractions_of_density_wall"]["depolarizing_update"])
    unitary_total_fraction = objective_nested_fraction * unitary_density_fraction
    target_overall_speedup = 1.50
    denominator = (1.0 / target_overall_speedup) - (1.0 - unitary_total_fraction)
    required_unitary_component_speedup = float(unitary_total_fraction / denominator) if denominator > 0.0 else math.inf

    patch_plan = {
        "artifact": "O2_candidate_patch_plan",
        "created_at_utc": _utc_now(),
        "version": "O2_candidate_patch_plan_v0.1-draft",
        "status": "candidate_patch_plan_ready_optimization_not_started",
        "official_execution_authorized": False,
        "optimization_code_changes_authorized": False,
        "source_baseline": {
            "path": str(baseline_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(baseline_path),
            "status": baseline["status"],
        },
        "measured_basis": {
            "total_density_wall_time_s": float(summary["total_density_wall_time_s"]),
            "benchmark_wall_time_s_including_warmup": float(summary["benchmark_wall_time_s_including_warmup"]),
            "top_subphases": top_subphases[:5],
            "likely_o2_targets": summary["likely_o2_targets"],
            "broad_o2_edit_allowed_by_measurement": bool(summary["broad_o2_edit_allowed_by_measurement"]),
            "o1_objective_eval_nested_fraction": objective_nested_fraction,
            "unitary_density_fraction": unitary_density_fraction,
            "depolarizing_density_fraction": depolarizing_density_fraction,
            "estimated_unitary_fraction_of_task_wall": unitary_total_fraction,
            "amdahl_target": {
                "overall_microbench_wall_speedup_required": target_overall_speedup,
                "required_unitary_component_speedup_if_O2_P1_only": required_unitary_component_speedup,
                "formula": "s_unitary = f_unitary_total / (1 / S_total - (1 - f_unitary_total))",
                "interpretation": "The >=1.50x O2 gate is on full fixed microbench wall time, not on unitary_density_update alone.",
            },
        },
        "remaining_o2_timebox_rule": {
            "parent_o2_cap_hours": child_plan["hard_timebox_hours"]["O2_implementation_and_validation"],
            "O2_P1_mini_timebox_hours": 4.0,
            "candidate_patch_plan_limit": "do not spend more than the remaining O2 cap after baseline and validation bookkeeping",
            "hard_stop": child_plan["success_and_stop_rules"]["minimum_to_continue_after_O2"],
            "early_stop_rule": "if O2-P1 does not compile and pass low-level 1q/2q equivalence within 4 h, stop O2-P1 and pivot to O3/O5 or scope reduction",
        },
        "patch_slices": [
            {
                "id": "O2-P1",
                "target": "unitary_density_update",
                "measured_fraction": float(summary["subphase_fractions_of_density_wall"]["unitary_density_update"]),
                "allowed_edit_scope": [
                    "src/phase_c_120_custom_kernel.py",
                    "src/phase_e_sim_backend.py only if needed to select the new equivalent low-level helper",
                ],
                "candidate_change": "replace or specialize the local density unitary update path while preserving the exact density-matrix channel semantics",
                "equivalence_tier": "Tier 2 allowed only if documented as floating-point reduction-order change; otherwise Tier 1",
                "validation_required": [
                    "low-level random density equivalence against _apply_local_unitary_density for 1q and 2q targets",
                    "O2 baseline microbench rerun with probability hashes/objectives recorded",
                    "Phase E automated tests",
                ],
                "acceptance": "continue only if fixed O2 microbench wall-time speedup is >=1.50x, or if O2-P1 clearly combines with O2-P2 under the same wall-time gate",
                "component_speedup_note": f"Given f_unitary_total={unitary_total_fraction:.6f}, O2-P1 alone needs about {required_unitary_component_speedup:.2f}x speedup in unitary_density_update to reach 1.50x overall wall-time speedup.",
                "mini_timebox_hours": 4.0,
                "rollback": "git revert O2-P1 if equivalence fails or speedup is not credible",
            },
            {
                "id": "O2-P2",
                "target": "depolarizing_update",
                "measured_fraction": float(summary["subphase_fractions_of_density_wall"]["depolarizing_update"]),
                "allowed_edit_scope": [
                    "src/phase_c_120_custom_kernel.py",
                    "src/phase_e_sim_backend.py only if needed to reuse cached layout metadata",
                ],
                "candidate_change": "cache or specialize depolarizing layout/permutation metadata without changing the depolarizing channel",
                "equivalence_tier": "Tier 2 allowed only for documented reduction-order effects; otherwise Tier 1",
                "validation_required": [
                    "low-level depolarizing equivalence against _apply_local_depolarizing for 1q and 2q targets",
                    "O2 baseline microbench rerun",
                    "Phase E automated tests",
                ],
                "acceptance": "attempt only if O2-P1 does not already settle the O2 decision and timebox remains",
                "rollback": "git revert O2-P2 independently if equivalence or speedup fails",
            },
        ],
        "explicitly_skipped_for_o2_patch": [
            {
                "target": "gate_matrix_construction",
                "reason": "measured fraction is about 1.6%, below the first O2 edit priority",
            },
            {
                "target": "template_assignment",
                "reason": "measured fraction is about 1.1%; template caching already exists",
            },
            {
                "target": "shot_multinomial_sampling",
                "reason": "measured fraction is below 0.1%",
            },
        ],
        "gates_before_any_optimization_code_change": [
            "user explicitly authorizes O2 optimization code changes after this patch plan",
            "work starts from the current baseline artifacts and source hashes",
            "no official execution and no endpoint classification analysis",
            "patch must be reverted if equivalence or speedup gates fail",
        ],
        "next_action": "await_user_authorization_for_o2_optimization_code_changes",
    }
    write_json_atomic(patch_json, patch_plan, overwrite=True)
    md_lines = [
        "# O2 Candidate Patch Plan",
        "",
        f"Created UTC: `{patch_plan['created_at_utc']}`",
        "Status: `candidate_patch_plan_ready_optimization_not_started`",
        "",
        "Measured O2 baseline localized the runtime:",
        "",
    ]
    for row in top_subphases[:5]:
        md_lines.append(f"- `{row['subphase']}`: `{row['fraction_of_wall']:.3f}` of density wall time")
    md_lines.extend(
        [
            "",
            "Amdahl gate:",
            "",
            f"- O1 objective-evaluation nested fraction: `{objective_nested_fraction:.6f}`.",
            f"- Estimated `unitary_density_update` fraction of task wall time: `{unitary_total_fraction:.6f}`.",
            "- The O2 success gate is `>=1.50x` on fixed micro-benchmark wall time, not on the unitary subcomponent alone.",
            f"- If O2-P1 is the only improvement, `unitary_density_update` needs about `{required_unitary_component_speedup:.2f}x` component speedup to reach that wall-time gate.",
            "",
            "Mini-timebox:",
            "",
            "- O2-P1 has a `4 h` maximum implementation + low-level equivalence window.",
            "- If it does not compile and pass 1q/2q equivalence inside that window, stop O2-P1 and pivot to O3/O5 or scope reduction.",
            "",
            "Patch slices:",
            "",
            "- `O2-P1`: target `unitary_density_update` first.",
            "- `O2-P2`: target `depolarizing_update` only if O2-P1 is insufficient and timebox remains.",
            "",
            "Skipped for first O2 patch: gate matrix construction, template assignment, and shot sampling, because measured fractions are small.",
            "",
            "This artifact still does not authorize optimization code changes or official execution.",
            "",
            f"Next action: `{patch_plan['next_action']}`",
        ]
    )
    patch_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    post_status = {
        "artifact": "post_o1_status",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "o2_candidate_patch_plan_ready_optimization_not_started",
        "official_execution_authorized": False,
        "optimization_authorized": False,
        "user_review_decision": "amend",
        "o2_microbench_baseline_artifact": str(baseline_path.relative_to(root)).replace("\\", "/"),
        "o2_candidate_patch_plan_artifact": str(patch_json.relative_to(root)).replace("\\", "/"),
        "next_action": patch_plan["next_action"],
    }
    write_json_atomic(status_json, post_status, overwrite=True)
    status_md.write_text(
        "\n".join(
            [
                "# Post-O1 Status",
                "",
                f"Created UTC: `{post_status['created_at_utc']}`",
                f"Status: `{post_status['status']}`",
                "",
                "User review decision: `amend`.",
                "Official execution remains blocked.",
                "Optimization code changes have not started.",
                "",
                f"O2 baseline: `{post_status['o2_microbench_baseline_artifact']}`",
                f"O2 candidate patch plan: `{post_status['o2_candidate_patch_plan_artifact']}`",
                f"Next action: `{post_status['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_deviation_log_entry(
        root,
        step="O2",
        entry_type="amendment",
        description="O2 candidate patch plan written from baseline micro-benchmark.",
        action_taken=f"Recorded {patch_json.relative_to(root)} and {patch_md.relative_to(root)}.",
        blocking=False,
        resolved=True,
    )
    return patch_plan


def _run_o2_p1_low_level_equivalence() -> dict[str, Any]:
    rng = np.random.default_rng(20260521)
    max_abs_diff = 0.0
    cases = 0
    for n_qubits in range(2, 7):
        dim = 1 << n_qubits
        target_sets: list[tuple[int, ...]] = [(q,) for q in range(n_qubits)]
        target_sets.extend((a, b) for a in range(n_qubits) for b in range(n_qubits) if a != b)
        for targets in target_sets:
            local_dim = 1 << len(targets)
            for _ in range(8):
                raw = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
                rho = raw @ raw.conj().T
                rho = rho / np.trace(rho)
                raw_gate = rng.normal(size=(local_dim, local_dim)) + 1j * rng.normal(size=(local_dim, local_dim))
                q_matrix, r_matrix = np.linalg.qr(raw_gate)
                phases = np.diag(r_matrix)
                phases = phases / np.abs(phases)
                gate = q_matrix * phases
                reference = _apply_local_unitary_density_reference(rho, gate, targets, n_qubits)
                candidate = _apply_local_unitary_density(rho, gate, targets, n_qubits)
                diff = float(np.max(np.abs(reference - candidate)))
                max_abs_diff = max(max_abs_diff, diff)
                cases += 1
                if diff > 1e-12:
                    return {
                        "status": "failed",
                        "cases": cases,
                        "max_abs_diff": max_abs_diff,
                        "failed_case": {"n_qubits": n_qubits, "targets": list(targets), "diff": diff},
                    }
    return {"status": "passed", "cases": cases, "max_abs_diff": max_abs_diff, "tolerance": 1e-12}


def write_o2_p1_result_report(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    result_json = out / "O2_P1_optimization_result.json"
    result_md = out / "O2_P1_optimization_result.md"
    equivalence_json = out / "O2_P1_low_level_equivalence.json"
    equivalence_md = out / "O2_P1_low_level_equivalence.md"
    after_json = out / "O2_P1_density_microbench_after_patch.json"
    after_csv = out / "O2_P1_density_microbench_after_patch.csv"
    after_md = out / "O2_P1_density_microbench_after_patch.md"
    status_json = out / "post_o1_status.json"
    status_md = out / "post_o1_status.md"
    if any(path.exists() for path in (result_json, result_md, equivalence_json, equivalence_md, after_json, after_csv, after_md)) and not force:
        raise FileExistsError("Refusing to overwrite existing O2-P1 result artifacts.")

    patch_plan_path = out / "O2_candidate_patch_plan_v0.1-draft.json"
    current_microbench_json = out / "O2_density_microbench_baseline.json"
    current_microbench_csv = out / "O2_density_microbench_baseline.csv"
    current_microbench_md = out / "O2_density_microbench_baseline.md"
    patch_plan = _read_json(patch_plan_path)
    after = _read_json(current_microbench_json)
    equivalence = _run_o2_p1_low_level_equivalence()
    write_json_atomic(equivalence_json, equivalence, overwrite=True)
    equivalence_md.write_text(
        "\n".join(
            [
                "# O2-P1 Low-Level Equivalence",
                "",
                f"Created UTC: `{_utc_now()}`",
                f"Status: `{equivalence['status']}`",
                f"Cases: `{equivalence['cases']}`",
                f"Max abs diff: `{equivalence['max_abs_diff']:.6e}`",
                "Tolerance: `1e-12`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if equivalence["status"] != "passed":
        raise RuntimeError("O2-P1 low-level equivalence failed; refusing to write accepted result.")

    shutil.copyfile(current_microbench_json, after_json)
    if current_microbench_csv.exists():
        shutil.copyfile(current_microbench_csv, after_csv)
    if current_microbench_md.exists():
        shutil.copyfile(current_microbench_md, after_md)

    before = patch_plan["measured_basis"]
    before_total = float(before["total_density_wall_time_s"])
    before_benchmark = float(before["benchmark_wall_time_s_including_warmup"])
    before_unitary = float(next(row["time_s"] for row in before["top_subphases"] if row["subphase"] == "unitary_density_update"))
    before_depolarizing = float(next(row["time_s"] for row in before["top_subphases"] if row["subphase"] == "depolarizing_update"))
    after_summary = after["summary"]
    after_total = float(after_summary["total_density_wall_time_s"])
    after_benchmark = float(after_summary["benchmark_wall_time_s_including_warmup"])
    after_unitary = float(after_summary["subphase_totals_s"]["unitary_density_update"])
    after_depolarizing = float(after_summary["subphase_totals_s"]["depolarizing_update"])

    density_wall_speedup = before_total / after_total
    benchmark_wall_speedup = before_benchmark / after_benchmark
    unitary_component_speedup = before_unitary / after_unitary
    depolarizing_observed_ratio = before_depolarizing / after_depolarizing
    pass_gate = bool(benchmark_wall_speedup >= 1.50 and unitary_component_speedup >= float(before["amdahl_target"]["required_unitary_component_speedup_if_O2_P1_only"]))
    status = "o2_p1_accepted_o3_o5_not_started" if pass_gate else "o2_p1_rejected_speedup_gate_failed"
    next_action = "decide_between_o2_p2_o3_o5_or_scope_reduction" if pass_gate else "revert_o2_p1_and_pivot_to_o3_o5_or_scope_reduction"

    result = {
        "artifact": "O2_P1_optimization_result",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": status,
        "official_execution_authorized": False,
        "o3_o5_started": False,
        "patch_scope": {
            "id": "O2-P1",
            "target": "unitary_density_update",
            "files_changed": ["src/phase_c_120_custom_kernel.py"],
            "change_summary": "vectorized local density unitary update across cached target blocks; reference loop retained for validation",
        },
        "source_patch_plan": {
            "path": str(patch_plan_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(patch_plan_path),
        },
        "equivalence": {
            "path": str(equivalence_json.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(equivalence_json),
            **equivalence,
        },
        "microbench_before": {
            "source": "O2_candidate_patch_plan_v0.1-draft measured_basis from prepatch baseline",
            "density_wall_time_s": before_total,
            "benchmark_wall_time_s_including_warmup": before_benchmark,
            "unitary_density_update_s": before_unitary,
            "depolarizing_update_s": before_depolarizing,
        },
        "microbench_after": {
            "path": str(after_json.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(after_json),
            "density_wall_time_s": after_total,
            "benchmark_wall_time_s_including_warmup": after_benchmark,
            "unitary_density_update_s": after_unitary,
            "depolarizing_update_s": after_depolarizing,
        },
        "speedups": {
            "density_wall_speedup": density_wall_speedup,
            "benchmark_wall_speedup_including_warmup": benchmark_wall_speedup,
            "unitary_component_speedup": unitary_component_speedup,
            "depolarizing_observed_ratio_not_targeted": depolarizing_observed_ratio,
        },
        "gate": {
            "required_fixed_microbench_wall_speedup": 1.50,
            "required_unitary_component_speedup_if_o2_p1_only": float(before["amdahl_target"]["required_unitary_component_speedup_if_O2_P1_only"]),
            "passed": pass_gate,
        },
        "guards": {
            "official_execution_performed": False,
            "endpoint_classification_analysis_performed": False,
            "o2_p2_started": False,
            "o3_o5_started": False,
        },
        "next_action": next_action,
    }
    write_json_atomic(result_json, result, overwrite=True)
    result_md.write_text(
        "\n".join(
            [
                "# O2-P1 Optimization Result",
                "",
                f"Created UTC: `{result['created_at_utc']}`",
                f"Status: `{result['status']}`",
                "",
                "Equivalence:",
                "",
                f"- Low-level cases: `{equivalence['cases']}`",
                f"- Max abs diff: `{equivalence['max_abs_diff']:.6e}`",
                "",
                "Speedups:",
                "",
                f"- Fixed micro-benchmark wall speedup: `{benchmark_wall_speedup:.3f}x`",
                f"- Density wall speedup: `{density_wall_speedup:.3f}x`",
                f"- `unitary_density_update` component speedup: `{unitary_component_speedup:.3f}x`",
                "",
                f"Gate passed: `{pass_gate}`",
                "",
                "No official execution, endpoint classification analysis, O2-P2, O3, or O5 was run.",
                "",
                f"Next action: `{next_action}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    post_status = {
        "artifact": "post_o1_status",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": status,
        "official_execution_authorized": False,
        "optimization_authorized": False,
        "user_review_decision": "amend",
        "o2_p1_result_artifact": str(result_json.relative_to(root)).replace("\\", "/"),
        "next_action": next_action,
    }
    write_json_atomic(status_json, post_status, overwrite=True)
    status_md.write_text(
        "\n".join(
            [
                "# Post-O1 Status",
                "",
                f"Created UTC: `{post_status['created_at_utc']}`",
                f"Status: `{post_status['status']}`",
                "",
                "User review decision: `amend`.",
                "Official execution remains blocked.",
                "O2-P2, O3, and O5 have not started.",
                "",
                f"O2-P1 result: `{post_status['o2_p1_result_artifact']}`",
                f"Next action: `{post_status['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_deviation_log_entry(
        root,
        step="O2",
        entry_type="amendment",
        description="O2-P1 unitary-density optimization result recorded.",
        action_taken=f"Recorded {result_json.relative_to(root)} and {equivalence_json.relative_to(root)}.",
        blocking=False,
        resolved=True,
    )
    return result


def write_o2_p2_candidate_patch_plan(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    plan_json = out / "O2_P2_candidate_patch_plan_v0.1-draft.json"
    plan_md = out / "O2_P2_candidate_patch_plan_v0.1-draft.md"
    status_json = out / "post_o1_status.json"
    status_md = out / "post_o1_status.md"
    if any(path.exists() for path in (plan_json, plan_md)) and not force:
        raise FileExistsError("Refusing to overwrite existing O2-P2 candidate patch plan artifacts.")

    o2_p1_path = out / "O2_P1_optimization_result.json"
    if not o2_p1_path.exists():
        raise FileNotFoundError(f"O2-P1 result not found: {o2_p1_path}")
    o2_p1 = _read_json(o2_p1_path)
    if o2_p1.get("status") != "o2_p1_accepted_o3_o5_not_started":
        raise RuntimeError("O2-P2 planning requires accepted O2-P1 and no O3/O5 start.")

    after = o2_p1["microbench_after"]
    depol_benchmark_fraction = float(after["depolarizing_update_s"]) / float(after["benchmark_wall_time_s_including_warmup"])
    depol_density_fraction = float(after["depolarizing_update_s"]) / float(after["density_wall_time_s"])
    target_wall_speedup = 1.10
    denominator = (1.0 / target_wall_speedup) - (1.0 - depol_benchmark_fraction)
    required_depol_component_speedup = float(depol_benchmark_fraction / denominator) if denominator > 0.0 else math.inf

    plan = {
        "artifact": "O2_P2_candidate_patch_plan",
        "created_at_utc": _utc_now(),
        "version": "O2_P2_candidate_patch_plan_v0.1-draft",
        "status": "o2_p2_candidate_patch_plan_ready_o3_o5_not_started",
        "official_execution_authorized": False,
        "o3_o5_started": False,
        "source_o2_p1_result": {
            "path": str(o2_p1_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(o2_p1_path),
            "status": o2_p1["status"],
        },
        "post_o2_p1_basis": {
            "benchmark_wall_time_s_including_warmup": float(after["benchmark_wall_time_s_including_warmup"]),
            "density_wall_time_s": float(after["density_wall_time_s"]),
            "depolarizing_update_s": float(after["depolarizing_update_s"]),
            "depolarizing_fraction_of_benchmark_wall": depol_benchmark_fraction,
            "depolarizing_fraction_of_density_wall": depol_density_fraction,
        },
        "amdahl_target": {
            "additional_fixed_microbench_wall_speedup_required": target_wall_speedup,
            "required_depolarizing_component_speedup_if_o2_p2_only": required_depol_component_speedup,
            "formula": "s_depol = f_depol_post_o2p1 / (1 / S_additional - (1 - f_depol_post_o2p1))",
            "interpretation": "The >=1.10x O2-P2 gate is additional wall-time speedup relative to the post-O2-P1 microbench, not component-only speedup.",
        },
        "timebox": {
            "ideal_hours": 2.0,
            "maximum_hours": 3.0,
            "early_stop_rule": "if O2-P2 does not compile and pass low-level 1q depolarizing equivalence inside the timebox, stop O2-P2 and pivot to O3/O5 or scope reduction",
        },
        "patch_slice": {
            "id": "O2-P2",
            "target": "depolarizing_update",
            "allowed_edit_scope": ["src/phase_c_120_custom_kernel.py"],
            "candidate_change": "cache depolarizing layout metadata and specialize construction of the mixed reduced state without changing the channel",
            "equivalence_tier": "Tier 1 expected; Tier 2 only if reduction-order change is explicitly documented",
            "validation_required": [
                "low-level random density equivalence against _apply_local_depolarizing_reference for 1q and 2q targets",
                "O2 microbench rerun relative to O2-P1 postpatch baseline",
                "Phase C custom-kernel validation",
                "Phase E automated tests",
            ],
            "acceptance": "accept only if additional fixed microbench wall-time speedup relative to post-O2-P1 is >=1.10x and equivalence passes",
            "rollback": "git revert O2-P2 if equivalence fails or additional wall-time speedup is below 1.10x",
        },
        "next_action": "implement_o2_p2_depolarizing_update",
    }
    write_json_atomic(plan_json, plan, overwrite=True)
    plan_md.write_text(
        "\n".join(
            [
                "# O2-P2 Candidate Patch Plan",
                "",
                f"Created UTC: `{plan['created_at_utc']}`",
                f"Status: `{plan['status']}`",
                "",
                "Post-O2-P1 basis:",
                "",
                f"- Benchmark wall time including warm-up: `{after['benchmark_wall_time_s_including_warmup']:.6f}` s",
                f"- Density wall time: `{after['density_wall_time_s']:.6f}` s",
                f"- `depolarizing_update`: `{after['depolarizing_update_s']:.6f}` s",
                f"- Depolarizing fraction of benchmark wall: `{depol_benchmark_fraction:.3f}`",
                f"- Depolarizing fraction of density wall: `{depol_density_fraction:.3f}`",
                "",
                "Amdahl gate:",
                "",
                "- O2-P2 success is `>=1.10x` additional fixed micro-benchmark wall speedup relative to post-O2-P1.",
                f"- If O2-P2 is the only improvement, `depolarizing_update` needs about `{required_depol_component_speedup:.2f}x` component speedup.",
                "",
                "Timebox:",
                "",
                "- Ideal: `2 h`.",
                "- Maximum: `3 h`.",
                "- Stop if low-level 1q/2q equivalence does not pass inside the timebox.",
                "",
                "No official execution, O3, or O5 is authorized by this artifact.",
                "",
                f"Next action: `{plan['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    post_status = {
        "artifact": "post_o1_status",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "o2_p2_candidate_patch_plan_ready_o3_o5_not_started",
        "official_execution_authorized": False,
        "optimization_authorized": False,
        "user_review_decision": "amend",
        "o2_p2_candidate_patch_plan_artifact": str(plan_json.relative_to(root)).replace("\\", "/"),
        "next_action": plan["next_action"],
    }
    write_json_atomic(status_json, post_status, overwrite=True)
    status_md.write_text(
        "\n".join(
            [
                "# Post-O1 Status",
                "",
                f"Created UTC: `{post_status['created_at_utc']}`",
                f"Status: `{post_status['status']}`",
                "",
                "Official execution remains blocked.",
                "O3 and O5 have not started.",
                "",
                f"O2-P2 candidate patch plan: `{post_status['o2_p2_candidate_patch_plan_artifact']}`",
                f"Next action: `{post_status['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_deviation_log_entry(
        root,
        step="O2",
        entry_type="amendment",
        description="O2-P2 candidate patch plan written from post-O2-P1 benchmark.",
        action_taken=f"Recorded {plan_json.relative_to(root)} and {plan_md.relative_to(root)}.",
        blocking=False,
        resolved=True,
    )
    return plan


def _run_o2_p2_low_level_equivalence() -> dict[str, Any]:
    rng = np.random.default_rng(20260521)
    max_abs_diff = 0.0
    cases = 0
    for n_qubits in range(2, 7):
        dim = 1 << n_qubits
        target_sets: list[tuple[int, ...]] = [(q,) for q in range(n_qubits)]
        target_sets.extend((a, b) for a in range(n_qubits) for b in range(n_qubits) if a != b)
        for targets in target_sets:
            for param in (0.001, 0.01, 0.02):
                for _ in range(5):
                    raw = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
                    rho = raw @ raw.conj().T
                    rho = rho / np.trace(rho)
                    reference = _apply_local_depolarizing_reference(rho, targets, n_qubits, param)
                    candidate = _apply_local_depolarizing(rho, targets, n_qubits, param)
                    diff = float(np.max(np.abs(reference - candidate)))
                    max_abs_diff = max(max_abs_diff, diff)
                    cases += 1
                    if diff > 1e-12:
                        return {
                            "status": "failed",
                            "cases": cases,
                            "max_abs_diff": max_abs_diff,
                            "failed_case": {"n_qubits": n_qubits, "targets": list(targets), "p_dep": param, "diff": diff},
                        }
    return {"status": "passed", "cases": cases, "max_abs_diff": max_abs_diff, "tolerance": 1e-12}


def write_o2_p2_result_report(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    result_json = out / "O2_P2_optimization_result.json"
    result_md = out / "O2_P2_optimization_result.md"
    equivalence_json = out / "O2_P2_low_level_equivalence_after_rollback.json"
    equivalence_md = out / "O2_P2_low_level_equivalence_after_rollback.md"
    status_json = out / "post_o1_status.json"
    status_md = out / "post_o1_status.md"
    if any(path.exists() for path in (result_json, result_md, equivalence_json, equivalence_md)) and not force:
        raise FileExistsError("Refusing to overwrite existing O2-P2 result artifacts.")

    plan_path = out / "O2_P2_candidate_patch_plan_v0.1-draft.json"
    o2_p1_path = out / "O2_P1_optimization_result.json"
    failed_path = out / "O2_P2_density_microbench_failed_patch.json"
    if not failed_path.exists():
        raise FileNotFoundError(f"O2-P2 failed-patch microbench not found: {failed_path}")
    plan = _read_json(plan_path)
    o2_p1 = _read_json(o2_p1_path)
    failed = _read_json(failed_path)
    equivalence = _run_o2_p2_low_level_equivalence()
    write_json_atomic(equivalence_json, equivalence, overwrite=True)
    equivalence_md.write_text(
        "\n".join(
            [
                "# O2-P2 Low-Level Equivalence After Rollback",
                "",
                f"Created UTC: `{_utc_now()}`",
                f"Status: `{equivalence['status']}`",
                f"Cases: `{equivalence['cases']}`",
                f"Max abs diff: `{equivalence['max_abs_diff']:.6e}`",
                "Tolerance: `1e-12`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    before = o2_p1["microbench_after"]
    after = failed["summary"]
    before_benchmark = float(before["benchmark_wall_time_s_including_warmup"])
    before_density = float(before["density_wall_time_s"])
    before_depol = float(before["depolarizing_update_s"])
    failed_benchmark = float(after["benchmark_wall_time_s_including_warmup"])
    failed_density = float(after["total_density_wall_time_s"])
    failed_depol = float(after["subphase_totals_s"]["depolarizing_update"])
    additional_benchmark_speedup = before_benchmark / failed_benchmark
    additional_density_speedup = before_density / failed_density
    depol_component_speedup = before_depol / failed_depol
    gate_passed = bool(additional_benchmark_speedup >= 1.10 and equivalence["status"] == "passed")

    result = {
        "artifact": "O2_P2_optimization_result",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "o2_p2_rejected_rolled_back_to_o2_p1_o3_o5_not_started",
        "official_execution_authorized": False,
        "o3_o5_started": False,
        "source_o2_p2_plan": {
            "path": str(plan_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(plan_path),
            "status": plan["status"],
        },
        "equivalence_after_rollback": {
            "path": str(equivalence_json.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(equivalence_json),
            **equivalence,
        },
        "microbench_post_o2_p1_baseline": {
            "source": str(o2_p1_path.relative_to(root)).replace("\\", "/"),
            "benchmark_wall_time_s_including_warmup": before_benchmark,
            "density_wall_time_s": before_density,
            "depolarizing_update_s": before_depol,
        },
        "microbench_failed_o2_p2_patch": {
            "path": str(failed_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(failed_path),
            "benchmark_wall_time_s_including_warmup": failed_benchmark,
            "density_wall_time_s": failed_density,
            "depolarizing_update_s": failed_depol,
        },
        "speedups": {
            "additional_fixed_microbench_wall_speedup": additional_benchmark_speedup,
            "additional_density_wall_speedup": additional_density_speedup,
            "depolarizing_component_speedup": depol_component_speedup,
        },
        "gate": {
            "required_additional_fixed_microbench_wall_speedup": 1.10,
            "required_depolarizing_component_speedup_if_o2_p2_only": plan["amdahl_target"]["required_depolarizing_component_speedup_if_o2_p2_only"],
            "passed": gate_passed,
        },
        "rollback": {
            "applied": True,
            "active_depolarizing_path": "_apply_local_depolarizing delegates to _apply_local_depolarizing_reference",
            "o2_p1_preserved": True,
        },
        "next_action": "skip_o2_p2_decide_o3_o5_or_scope_reduction",
    }
    write_json_atomic(result_json, result, overwrite=True)
    result_md.write_text(
        "\n".join(
            [
                "# O2-P2 Optimization Result",
                "",
                f"Created UTC: `{result['created_at_utc']}`",
                f"Status: `{result['status']}`",
                "",
                "O2-P2 was rejected because it slowed the fixed micro-benchmark.",
                "",
                f"- Additional fixed micro-benchmark wall speedup: `{additional_benchmark_speedup:.3f}x`",
                f"- Additional density wall speedup: `{additional_density_speedup:.3f}x`",
                f"- `depolarizing_update` component speedup: `{depol_component_speedup:.3f}x`",
                "",
                f"Gate passed: `{gate_passed}`",
                "",
                "Rollback was applied; O2-P1 remains accepted. No official execution, O3, or O5 was run.",
                "",
                f"Next action: `{result['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    post_status = {
        "artifact": "post_o1_status",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "o2_p2_rejected_rolled_back_to_o2_p1_o3_o5_not_started",
        "official_execution_authorized": False,
        "optimization_authorized": False,
        "user_review_decision": "amend",
        "o2_p2_result_artifact": str(result_json.relative_to(root)).replace("\\", "/"),
        "next_action": result["next_action"],
    }
    write_json_atomic(status_json, post_status, overwrite=True)
    status_md.write_text(
        "\n".join(
            [
                "# Post-O1 Status",
                "",
                f"Created UTC: `{post_status['created_at_utc']}`",
                f"Status: `{post_status['status']}`",
                "",
                "Official execution remains blocked.",
                "O2-P2 was rejected and rolled back. O2-P1 remains accepted.",
                "O3 and O5 have not started.",
                "",
                f"O2-P2 result: `{post_status['o2_p2_result_artifact']}`",
                f"Next action: `{post_status['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_deviation_log_entry(
        root,
        step="O2",
        entry_type="amendment",
        description="O2-P2 depolarizing optimization rejected and rolled back after speed gate failed.",
        action_taken=f"Recorded {result_json.relative_to(root)} and restored active depolarizing path to reference implementation.",
        blocking=False,
        resolved=True,
    )
    return result


def _lhs_point_reference_for_o3(
    graph: GraphInstance,
    theta: tuple[float, ...],
    *,
    p_dep: float,
    scale_factors: list[int],
    shots_total: int,
    noisy_seed: int,
    zne_seed: int,
) -> dict[str, Any]:
    ideal = evaluate_custom_density_objective(graph, theta, p_dep=0.0)
    noisy = evaluate_custom_density_objective(
        graph,
        theta,
        p_dep=p_dep,
        scale_factor=1,
        shots=int(shots_total),
        seed=int(noisy_seed),
    )
    zne = richardson_zne_objective(
        graph,
        theta,
        p_dep=p_dep,
        scale_factors=scale_factors,
        shots_total=int(shots_total),
        seed=int(zne_seed),
    )
    return {
        "ideal_objective": float(ideal.objective_value),
        "noisy_objective": float(noisy.objective_value),
        "zne_objective": float(zne.objective_value),
        "per_scale_values": [float(value) for value in zne.per_scale_values],
        "shot_allocation": [int(value) for value in (zne.shot_allocation or [])],
    }


def _lhs_point_o3_scale1_reuse(
    graph: GraphInstance,
    theta: tuple[float, ...],
    *,
    p_dep: float,
    scale_factors: list[int],
    shots_total: int,
    noisy_seed: int,
    zne_seed: int,
) -> dict[str, Any]:
    result = evaluate_lhs_point_with_scale1_probability_reuse(
        graph,
        theta,
        p_dep=p_dep,
        scale_factors=scale_factors,
        shots_total=int(shots_total),
        noisy_seed=int(noisy_seed),
        zne_seed=int(zne_seed),
    )
    return {
        "ideal_objective": float(result.ideal_objective),
        "noisy_objective": float(result.noisy_objective),
        "zne_objective": float(result.zne_objective),
        "per_scale_values": [float(value) for value in result.per_scale_values],
        "shot_allocation": [int(value) for value in result.shot_allocation],
    }


def run_o3_linear_lhs_reuse_microbench(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    result_json = out / "O3_linear_lhs_scale1_reuse_result.json"
    result_md = out / "O3_linear_lhs_scale1_reuse_result.md"
    result_csv = out / "O3_linear_lhs_scale1_reuse_microbench.csv"
    if any(path.exists() for path in (result_json, result_md, result_csv)) and not force:
        raise FileExistsError("Refusing to overwrite existing O3 linear LHS reuse artifacts.")

    authorization_path = out / "case_c_linear_first_o3_authorization.json"
    if not authorization_path.exists():
        raise FileNotFoundError(f"O3 authorization artifact not found: {authorization_path}")
    authorization = _read_json(authorization_path)
    if authorization.get("status") != "linear_first_scope_path_recorded_o3_bounded_authorized":
        raise RuntimeError(f"O3 is not authorized by current artifact status: {authorization.get('status')}")

    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(variable, "1")

    graph = _graph_from_selected_instance(root, "H4")
    p_dep = 0.01
    shots_total = 512
    scale_factors = [1, 3]
    lhs_count = 12
    design_id = 0
    repeats = 3
    lhs_seed = deterministic_seed("phase_e_o3_linear_lhs_microbench", graph.name, p_dep, shots_total, lhs_count, design_id)
    points = _lhs_points(depth=2, count=lhs_count, seed=lhs_seed)

    # Warm both paths once so the comparison is not dominated by template cache fill.
    warm_noisy_seed = deterministic_seed("phase_e_o3_lhs_noisy_warmup", graph.name, p_dep, shots_total)
    warm_zne_seed = deterministic_seed("phase_e_o3_lhs_zne_warmup", graph.name, p_dep, shots_total)
    _lhs_point_reference_for_o3(
        graph,
        points[0],
        p_dep=p_dep,
        scale_factors=scale_factors,
        shots_total=shots_total,
        noisy_seed=warm_noisy_seed,
        zne_seed=warm_zne_seed,
    )
    _lhs_point_o3_scale1_reuse(
        graph,
        points[0],
        p_dep=p_dep,
        scale_factors=scale_factors,
        shots_total=shots_total,
        noisy_seed=warm_noisy_seed,
        zne_seed=warm_zne_seed,
    )

    rows: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    reference_total_s = 0.0
    optimized_total_s = 0.0
    for repeat_index in range(repeats):
        for point_index, theta in enumerate(points):
            noisy_seed = deterministic_seed(
                "phase_e_lhs_noisy",
                graph.name,
                p_dep,
                shots_total,
                lhs_count,
                design_id,
                point_index,
            )
            zne_seed = deterministic_seed(
                "phase_e_lhs_zne",
                graph.name,
                p_dep,
                shots_total,
                lhs_count,
                design_id,
                point_index,
            )
            started = time.perf_counter()
            reference = _lhs_point_reference_for_o3(
                graph,
                theta,
                p_dep=p_dep,
                scale_factors=scale_factors,
                shots_total=shots_total,
                noisy_seed=noisy_seed,
                zne_seed=zne_seed,
            )
            reference_s = float(time.perf_counter() - started)

            started = time.perf_counter()
            optimized = _lhs_point_o3_scale1_reuse(
                graph,
                theta,
                p_dep=p_dep,
                scale_factors=scale_factors,
                shots_total=shots_total,
                noisy_seed=noisy_seed,
                zne_seed=zne_seed,
            )
            optimized_s = float(time.perf_counter() - started)

            diffs = {
                "ideal_abs_diff": abs(float(reference["ideal_objective"]) - float(optimized["ideal_objective"])),
                "noisy_abs_diff": abs(float(reference["noisy_objective"]) - float(optimized["noisy_objective"])),
                "zne_abs_diff": abs(float(reference["zne_objective"]) - float(optimized["zne_objective"])),
                "scale0_abs_diff": abs(float(reference["per_scale_values"][0]) - float(optimized["per_scale_values"][0])),
                "scale1_abs_diff": abs(float(reference["per_scale_values"][1]) - float(optimized["per_scale_values"][1])),
            }
            row_max_diff = max(float(value) for value in diffs.values())
            max_abs_diff = max(max_abs_diff, row_max_diff)
            reference_total_s += reference_s
            optimized_total_s += optimized_s
            rows.append(
                {
                    "artifact": "O3_linear_lhs_scale1_reuse_microbench",
                    "instance": graph.name,
                    "p_dep": p_dep,
                    "shots_total": shots_total,
                    "scale_factors": "|".join(str(scale) for scale in scale_factors),
                    "lhs_count": lhs_count,
                    "lhs_design_id": design_id,
                    "lhs_seed": lhs_seed,
                    "point_index": point_index,
                    "repeat_index": repeat_index,
                    "reference_wall_time_s": reference_s,
                    "optimized_wall_time_s": optimized_s,
                    "row_speedup": reference_s / optimized_s if optimized_s > 0.0 else math.inf,
                    **diffs,
                }
            )

    wall_speedup = float(reference_total_s / optimized_total_s) if optimized_total_s > 0.0 else math.inf
    equivalence_pass = bool(max_abs_diff <= 1e-12)
    speed_gate_pass = bool(wall_speedup >= float(authorization["o3_constraints"]["target_additional_wall_speedup"]))
    gate_pass = bool(equivalence_pass and speed_gate_pass)
    projected_linear_after_o3_hours = float(253.03672091286907 / wall_speedup) if wall_speedup > 0.0 else math.inf
    projected_linear_passes_240h = bool(projected_linear_after_o3_hours <= 240.0)
    status = "o3_linear_lhs_reuse_accepted_for_linear_only_projection" if gate_pass else "o3_linear_lhs_reuse_rejected_or_insufficient"

    report = {
        "artifact": "O3_linear_lhs_scale1_reuse_result",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": status,
        "stage_c_authorized": False,
        "stage_c_blocked": True,
        "official_execution_authorized": False,
        "scientific_outcomes_used": False,
        "optimization_scope": "linear_only_lhs_scale1_probability_reuse",
        "authorization": {
            "path": str(authorization_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(authorization_path),
            "status": authorization["status"],
        },
        "implementation": {
            "changed_files": [
                "src/phase_e_sim_backend.py",
                "src/phase_e_cobyla_runtime_optimization.py",
                "scripts/79_phase_e_cobyla_runtime_o0_o0_75.py"
            ],
            "change_summary": "Reuse the exact scale=1 probability vector across noisy and Richardson scale=1 LHS evaluations while preserving independent shot seeds.",
            "cobyla_sequence_changed": False,
            "seed_identity_changed": False,
            "shot_allocation_changed": False,
            "scale_factors_changed": False,
        },
        "microbench": {
            "instance": graph.name,
            "p_dep": p_dep,
            "shots_total": shots_total,
            "scale_factors": scale_factors,
            "lhs_count": lhs_count,
            "repeats": repeats,
            "case_count": len(rows),
            "reference_total_s": reference_total_s,
            "optimized_total_s": optimized_total_s,
            "wall_speedup": wall_speedup,
            "target_additional_wall_speedup": float(authorization["o3_constraints"]["target_additional_wall_speedup"]),
        },
        "equivalence": {
            "case_count": len(rows),
            "max_abs_diff": max_abs_diff,
            "tolerance": 1e-12,
            "passed": equivalence_pass,
        },
        "gate": {
            "speed_gate_pass": speed_gate_pass,
            "equivalence_pass": equivalence_pass,
            "passed": gate_pass,
        },
        "linear_only_projection_update": {
            "post_o2_p1_conservative_linear_hours": 253.03672091286907,
            "post_o3_conservative_linear_hours": projected_linear_after_o3_hours,
            "passes_240h": projected_linear_passes_240h,
            "accepted_local_wall_hours": 240.0,
        },
        "next_action": "write_linear_only_scope_amendment_or_reject_o3_and_choose_budget_scope_stop" if gate_pass else "reject_o3_and_choose_budget_scope_stop",
    }
    write_json_atomic(result_json, report, overwrite=True)
    _write_csv(result_csv, rows)
    result_md.write_text(
        "\n".join(
            [
                "# O3 Linear LHS Scale-1 Reuse Result",
                "",
                f"Created UTC: `{report['created_at_utc']}`",
                f"Status: `{status}`",
                "",
                "Official execution remains blocked.",
                "",
                "## Result",
                "",
                f"- Equivalence passed: `{equivalence_pass}`",
                f"- Max abs diff: `{max_abs_diff:.6e}`",
                f"- Wall speedup: `{wall_speedup:.3f}x`",
                f"- Gate passed: `{gate_pass}`",
                f"- Projected conservative linear-only hours after O3: `{projected_linear_after_o3_hours:.2f}`",
                f"- Linear-only passes 240 h: `{projected_linear_passes_240h}`",
                "",
                "No COBYLA sequence, seed identity, scale factors, shot allocation, or task manifest is changed.",
                "",
                f"Next action: `{report['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status_json = out / "post_o1_status.json"
    status_md = out / "post_o1_status.md"
    post_status = _read_json(status_json)
    post_status.update(
        {
            "created_at_utc": report["created_at_utc"],
            "status": status,
            "o3_result_artifact": str(result_json.relative_to(root)).replace("\\", "/"),
            "official_execution_authorized": False,
            "optimization_authorized": False,
            "next_action": report["next_action"],
        }
    )
    write_json_atomic(status_json, post_status, overwrite=True)
    status_md.write_text(
        "\n".join(
            [
                "# Post-O1 Status",
                "",
                f"Created UTC: `{report['created_at_utc']}`",
                f"Status: `{status}`",
                "",
                "Official execution remains blocked.",
                "O2-P1 remains accepted. O2-P2 remains rejected and rolled back.",
                "O3 linear LHS scale-1 reuse has been benchmarked.",
                "",
                f"O3 result: `{str(result_json.relative_to(root)).replace(chr(92), '/')}`",
                "",
                f"Next action: `{report['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_deviation_log_entry(
        root,
        step="O3",
        entry_type="amendment",
        description="Bounded O3 linear-only LHS scale-1 probability reuse microbenchmark completed.",
        action_taken=f"Recorded {result_json.relative_to(root)} with speedup {wall_speedup:.3f}x and gate_pass={gate_pass}.",
        blocking=False,
        resolved=True,
    )
    return report


def _format_p_dep_for_id(p_dep: float) -> str:
    return str(p_dep).replace(".", "p")


def run_linear_plus_robustness_chunked_manifest_dry_run(
    root: Path = PROJECT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    out = _output_dir(root)
    manifest_json = out / "linear_plus_robustness_chunked_manifest_dry_run.json"
    manifest_md = out / "linear_plus_robustness_chunked_manifest_dry_run.md"
    chunks_csv = out / "linear_plus_robustness_chunked_manifest_dry_run_chunks.csv"
    if any(path.exists() for path in (manifest_json, manifest_md, chunks_csv)) and not force:
        raise FileExistsError("Refusing to overwrite existing linear-plus-robustness manifest dry-run artifacts.")

    config = _read_json(root / "config" / "phase_e_config_v0.1-draft.json")
    calibration = _read_json(root / "results" / "phase_e" / "calibration" / "phase_e_calibration_status.json")
    selected = _selected_values(config, calibration)
    scope_path = out / "linear_plus_robustness_chunked_scope_amendment_v0.1.json"
    optimization_path = out / "chunked_execution_optimization_amendment_v0.1.json"
    scope = _read_json(scope_path)
    optimization = _read_json(optimization_path)

    seeds = [int(seed) for seed in calibration["observed_candidate_sets"]["seeds"]]
    if len(seeds) != int(config["depth2_design"]["seed_count"]):
        raise ValueError("Frozen seed list length does not match config depth2_design.seed_count.")

    all_noise_rates = [float(value) for value in config["noise_rates"]]
    all_shot_budgets = [int(value) for value in config["shot_budgets"]]
    scale_factors_by_order = {
        order: [int(scale) for scale in scales]
        for order, scales in config["zne"]["scale_factors"].items()
    }
    selected_instances = list(config["depth2_design"]["selected_instances"])
    macro_blocks = scope["macro_execution_order"]
    macro_by_id = {macro["macro_id"]: macro for macro in macro_blocks}

    chunk_rows: list[dict[str, Any]] = []
    macro_summaries: list[dict[str, Any]] = []

    totals = {
        "macro_blocks": 0,
        "endpoint_cell_seed_family_chunks": 0,
        "lhs_cell_design_family_chunks": 0,
        "reference_instance_chunks": 0,
        "noisy_optimizer_start_records": 0,
        "zne_optimizer_start_records": 0,
        "reference_optimizer_runs": 0,
        "endpoint_summary_records": 0,
        "lhs_point_scale_records": 0,
        "amended_family_cells": 0,
    }

    for macro in macro_blocks:
        macro_id = str(macro["macro_id"])
        family = str(macro["family"])
        instances = [str(item) for item in macro["instances"]]
        if family == "linear_full":
            order = "linear"
            noise_rates = all_noise_rates
            shot_budgets = all_shot_budgets
            produce_noisy_reference = True
            lhs_extra_scale_records_per_point = 1 + 1 + len(scale_factors_by_order[order])
        elif family == "quadratic_robustness":
            order = "quadratic"
            noise_rates = [0.003, 0.01]
            shot_budgets = [512, 4096]
            produce_noisy_reference = False
            lhs_extra_scale_records_per_point = len(scale_factors_by_order[order])
        elif family == "cubic_robustness":
            order = "cubic"
            noise_rates = [0.003, 0.01]
            shot_budgets = [512, 4096]
            produce_noisy_reference = False
            lhs_extra_scale_records_per_point = len(scale_factors_by_order[order])
        else:
            raise ValueError(f"Unexpected macro family: {family}")

        macro_counts = {
            "endpoint_cell_seed_family_chunks": 0,
            "lhs_cell_design_family_chunks": 0,
            "reference_instance_chunks": 0,
            "noisy_optimizer_start_records": 0,
            "zne_optimizer_start_records": 0,
            "reference_optimizer_runs": 0,
            "endpoint_summary_records": 0,
            "lhs_point_scale_records": 0,
            "amended_family_cells": len(instances) * len(noise_rates) * len(shot_budgets),
        }

        for instance in instances:
            for p_dep in noise_rates:
                for shots_total in shot_budgets:
                    for seed_index, seed in enumerate(seeds):
                        chunk_id = (
                            f"{macro_id}__endpoint__{instance}__p{_format_p_dep_for_id(p_dep)}"
                            f"__shots{shots_total}__seed{seed}"
                        )
                        noisy_records = int(selected["K"]) if produce_noisy_reference else 0
                        zne_records = int(selected["K"])
                        macro_counts["endpoint_cell_seed_family_chunks"] += 1
                        macro_counts["noisy_optimizer_start_records"] += noisy_records
                        macro_counts["zne_optimizer_start_records"] += zne_records
                        chunk_rows.append(
                            {
                                "manifest_version": "phase_e_linear_plus_qc_robustness_chunked_manifest_dry_run_v0_1",
                                "chunk_type": "endpoint_cell_seed_family",
                                "macro_id": macro_id,
                                "chunk_id": chunk_id,
                                "family": family,
                                "richardson_order": order,
                                "instance": instance,
                                "p_dep": p_dep,
                                "shots_total": shots_total,
                                "seed": seed,
                                "seed_index": seed_index,
                                "lhs_design_id": "",
                                "K": int(selected["K"]),
                                "E_COBYLA": int(selected["E_COBYLA"]),
                                "N_LHS": int(selected["N_LHS"]),
                                "scale_factors_json": json.dumps(scale_factors_by_order[order]),
                                "expected_noisy_optimizer_start_records": noisy_records,
                                "expected_zne_optimizer_start_records": zne_records,
                                "expected_reference_optimizer_runs": 0,
                                "expected_lhs_points": 0,
                                "expected_lhs_scale_records": 0,
                                "dependency": "" if produce_noisy_reference else "matching_linear_noisy_reference_endpoint_complete",
                                "expected_output_path": f"data/raw/phase_e/chunked_execution/{macro_id}/{chunk_id}.jsonl",
                                "checkpoint_key": f"{macro_id}/{chunk_id}",
                                "resume_key": f"{macro_id}:{chunk_id}",
                                "estimated_macro_wall_hours": float(macro["estimated_hours"]),
                                "estimated_chunk_overhead_hours": "pending_runner_measurement",
                            }
                        )

                    lhs_chunk_id = (
                        f"{macro_id}__lhs__{instance}__p{_format_p_dep_for_id(p_dep)}"
                        f"__shots{shots_total}__design0"
                    )
                    lhs_scale_records = int(selected["N_LHS"]) * lhs_extra_scale_records_per_point
                    macro_counts["lhs_cell_design_family_chunks"] += 1
                    macro_counts["lhs_point_scale_records"] += lhs_scale_records
                    chunk_rows.append(
                        {
                            "manifest_version": "phase_e_linear_plus_qc_robustness_chunked_manifest_dry_run_v0_1",
                            "chunk_type": "lhs_cell_design_family",
                            "macro_id": macro_id,
                            "chunk_id": lhs_chunk_id,
                            "family": family,
                            "richardson_order": order,
                            "instance": instance,
                            "p_dep": p_dep,
                            "shots_total": shots_total,
                            "seed": "",
                            "seed_index": "",
                            "lhs_design_id": 0,
                            "K": int(selected["K"]),
                            "E_COBYLA": int(selected["E_COBYLA"]),
                            "N_LHS": int(selected["N_LHS"]),
                            "scale_factors_json": json.dumps(scale_factors_by_order[order]),
                            "expected_noisy_optimizer_start_records": 0,
                            "expected_zne_optimizer_start_records": 0,
                            "expected_reference_optimizer_runs": 0,
                            "expected_lhs_points": int(selected["N_LHS"]),
                            "expected_lhs_scale_records": lhs_scale_records,
                            "dependency": "" if produce_noisy_reference else "matching_linear_lhs_noisy_ideal_complete",
                            "expected_output_path": f"data/raw/phase_e/chunked_execution/{macro_id}/{lhs_chunk_id}.jsonl",
                            "checkpoint_key": f"{macro_id}/{lhs_chunk_id}",
                            "resume_key": f"{macro_id}:{lhs_chunk_id}",
                            "estimated_macro_wall_hours": float(macro["estimated_hours"]),
                            "estimated_chunk_overhead_hours": "pending_runner_measurement",
                        }
                    )

        if produce_noisy_reference:
            for instance in instances:
                reference_chunk_id = f"{macro_id}__reference__{instance}"
                reference_runs = 16 * 2
                macro_counts["reference_instance_chunks"] += 1
                macro_counts["reference_optimizer_runs"] += reference_runs
                chunk_rows.append(
                    {
                        "manifest_version": "phase_e_linear_plus_qc_robustness_chunked_manifest_dry_run_v0_1",
                        "chunk_type": "reference_instance",
                        "macro_id": macro_id,
                        "chunk_id": reference_chunk_id,
                        "family": family,
                        "richardson_order": order,
                        "instance": instance,
                        "p_dep": "",
                        "shots_total": "",
                        "seed": "",
                        "seed_index": "",
                        "lhs_design_id": "",
                        "K": int(selected["K"]),
                        "E_COBYLA": int(selected["E_COBYLA"]),
                        "N_LHS": int(selected["N_LHS"]),
                        "scale_factors_json": json.dumps([]),
                        "expected_noisy_optimizer_start_records": 0,
                        "expected_zne_optimizer_start_records": 0,
                        "expected_reference_optimizer_runs": reference_runs,
                        "expected_lhs_points": 0,
                        "expected_lhs_scale_records": 0,
                        "dependency": "linear_family_reference_strategy",
                        "expected_output_path": f"data/raw/phase_e/chunked_execution/{macro_id}/{reference_chunk_id}.jsonl",
                        "checkpoint_key": f"{macro_id}/{reference_chunk_id}",
                        "resume_key": f"{macro_id}:{reference_chunk_id}",
                        "estimated_macro_wall_hours": float(macro["estimated_hours"]),
                        "estimated_chunk_overhead_hours": "pending_runner_measurement",
                    }
                )

        macro_counts["endpoint_summary_records"] = (
            macro_counts["noisy_optimizer_start_records"]
            + macro_counts["zne_optimizer_start_records"]
            + macro_counts["reference_optimizer_runs"]
        )
        for key, value in macro_counts.items():
            totals[key] += int(value)
        totals["macro_blocks"] += 1
        macro_summaries.append(
            {
                "macro_id": macro_id,
                "family": family,
                "richardson_order": order,
                "instances": instances,
                "p_dep": noise_rates,
                "shots_total": shot_budgets,
                "scale_factors": scale_factors_by_order[order],
                "estimated_wall_hours": float(macro["estimated_hours"]),
                "counts": macro_counts,
                "reuse_rule": "produces_noisy_reference" if produce_noisy_reference else "reuses_matching_linear_noisy_reference",
            }
        )

    expected_chunks = (
        totals["endpoint_cell_seed_family_chunks"]
        + totals["lhs_cell_design_family_chunks"]
        + totals["reference_instance_chunks"]
    )
    if expected_chunks != len(chunk_rows):
        raise AssertionError("Internal manifest chunk count mismatch.")

    expected_counts = {
        **totals,
        "total_chunks": len(chunk_rows),
        "seed_count": len(seeds),
        "selected_instance_count": len(selected_instances),
    }

    expected_counts_checks = {
        "macro_blocks_match": totals["macro_blocks"] == 9,
        "endpoint_chunks_match": totals["endpoint_cell_seed_family_chunks"] == 4140,
        "lhs_chunks_match": totals["lhs_cell_design_family_chunks"] == 138,
        "reference_chunks_match": totals["reference_instance_chunks"] == 6,
        "total_chunks_match": len(chunk_rows) == 4284,
        "noisy_optimizer_start_records_match": totals["noisy_optimizer_start_records"] == 270000,
        "zne_optimizer_start_records_match": totals["zne_optimizer_start_records"] == 414000,
        "reference_optimizer_runs_match": totals["reference_optimizer_runs"] == 192,
        "endpoint_summary_records_match": totals["endpoint_summary_records"] == 684192,
        "lhs_point_scale_records_match": totals["lhs_point_scale_records"] == 1081344,
    }
    checks_pass = all(expected_counts_checks.values())
    status = (
        "linear_plus_qc_robustness_chunked_manifest_dry_run_complete_o6_next"
        if checks_pass
        else "linear_plus_qc_robustness_chunked_manifest_dry_run_count_mismatch"
    )

    code_hash_files = [
        "src/phase_e_cobyla_runtime_optimization.py",
        "src/phase_e_sim_backend.py",
        "src/phase_e_optimizers.py",
        "scripts/79_phase_e_cobyla_runtime_o0_o0_75.py",
    ]
    input_hash_files = [
        "config/phase_e_config_v0.1-draft.json",
        "config/phase_e_instances_v0.1-draft.json",
        "config/phase_e_noise_model_v0.1-draft.json",
        "results/phase_e/calibration/phase_e_calibration_status.json",
        str(scope_path.relative_to(root)).replace("\\", "/"),
        str(optimization_path.relative_to(root)).replace("\\", "/"),
    ]
    projection = optimization["runtime_target"]
    manifest = {
        "artifact": "linear_plus_robustness_chunked_manifest_dry_run",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "manifest_version": "phase_e_linear_plus_qc_robustness_chunked_manifest_dry_run_v0_1",
        "status": status,
        "official_execution_authorized": False,
        "stage_c_authorized": False,
        "stage_c_blocked": True,
        "scientific_outcomes_used": False,
        "dry_run_only": True,
        "no_tasks_executed": True,
        "next_action": "run_o6_threading_layout_scouting_after_manifest_dry_run",
        "scope": {
            "primary_family": "linear_richardson_full_factorial",
            "quadratic": "fixed_robustness_subset_only",
            "cubic": "fixed_robustness_subset_only",
            "macro_block_count": len(macro_summaries),
        },
        "selected_values": {
            "E_COBYLA": int(selected["E_COBYLA"]),
            "K": int(selected["K"]),
            "N_LHS": int(selected["N_LHS"]),
            "reference_strategy": selected["reference_strategy"],
            "seeds": seeds,
            "lhs_design_ids": [0],
        },
        "projection_semantics": {
            "projected_hours_are_wall_hours_under_effective_shards": True,
            "conservative_effective_shards_already_assumed": float(projection["conservative_effective_shards_already_assumed"]),
            "do_not_double_count_o6_parallelism": True,
            "total_projected_wall_hours_excluding_overhead": float(projection["total_projected_hours"]),
            "serial_equivalent_hours_under_same_model": float(projection["serial_equivalent_hours_under_same_model"]),
            "accepted_local_planning_budget_hours": float(projection["accepted_local_planning_budget_hours"]),
            "margin_hours_excluding_overhead": float(projection["margin_hours"]),
            "chunk_checkpoint_overhead_hours": "pending_o6_and_runner_measurement",
            "overhead_gate_status": "pending",
        },
        "macro_blocks": macro_summaries,
        "expected_counts": expected_counts,
        "expected_count_checks": expected_counts_checks,
        "manifest_hash": _canonical_hash(chunk_rows),
        "chunk_manifest_csv": str(chunks_csv.relative_to(root)).replace("\\", "/"),
        "code_hashes": _hashes_for_existing(root, code_hash_files),
        "input_hashes": _hashes_for_existing(root, input_hash_files),
        "between_block_review_policy": optimization["between_block_review_policy"],
        "o6_first_scouting_rule": optimization["o6_first_scouting_rule"],
        "outputs": {
            "json": str(manifest_json.relative_to(root)).replace("\\", "/"),
            "md": str(manifest_md.relative_to(root)).replace("\\", "/"),
            "chunks_csv": str(chunks_csv.relative_to(root)).replace("\\", "/"),
        },
    }

    write_json_atomic(manifest_json, manifest, overwrite=True)
    _write_csv(
        chunks_csv,
        chunk_rows,
        fieldnames=[
            "manifest_version",
            "chunk_type",
            "macro_id",
            "chunk_id",
            "family",
            "richardson_order",
            "instance",
            "p_dep",
            "shots_total",
            "seed",
            "seed_index",
            "lhs_design_id",
            "K",
            "E_COBYLA",
            "N_LHS",
            "scale_factors_json",
            "expected_noisy_optimizer_start_records",
            "expected_zne_optimizer_start_records",
            "expected_reference_optimizer_runs",
            "expected_lhs_points",
            "expected_lhs_scale_records",
            "dependency",
            "expected_output_path",
            "checkpoint_key",
            "resume_key",
            "estimated_macro_wall_hours",
            "estimated_chunk_overhead_hours",
        ],
    )
    md_lines = [
        "# Linear Plus Robustness Chunked Manifest Dry-Run",
        "",
        f"Created UTC: `{manifest['created_at_utc']}`",
        f"Status: `{status}`",
        "",
        "Official execution remains blocked. No scientific tasks were executed.",
        "",
        "## Scope",
        "",
        "- Primary: linear Richardson full factorial.",
        "- Robustness: fixed quadratic and cubic subsets.",
        "- Macro-blocks: `9`.",
        "",
        "## Counts",
        "",
        f"- Total chunks: `{len(chunk_rows)}`",
        f"- Endpoint cell/seed/family chunks: `{totals['endpoint_cell_seed_family_chunks']}`",
        f"- LHS cell/design/family chunks: `{totals['lhs_cell_design_family_chunks']}`",
        f"- Reference instance chunks: `{totals['reference_instance_chunks']}`",
        f"- Noisy optimizer start records: `{totals['noisy_optimizer_start_records']}`",
        f"- ZNE optimizer start records: `{totals['zne_optimizer_start_records']}`",
        f"- Reference optimizer runs: `{totals['reference_optimizer_runs']}`",
        f"- Endpoint summary records: `{totals['endpoint_summary_records']}`",
        f"- LHS point-scale records: `{totals['lhs_point_scale_records']}`",
        "",
        "## Runtime Semantics",
        "",
        f"- Projected wall hours excluding overhead: `{float(projection['total_projected_hours']):.2f}`",
        f"- Effective shards already assumed: `{float(projection['conservative_effective_shards_already_assumed']):.1f}`",
        f"- Serial-equivalent hours under same model: `{float(projection['serial_equivalent_hours_under_same_model']):.2f}`",
        f"- Budget: `{float(projection['accepted_local_planning_budget_hours']):.2f}` h",
        f"- Margin excluding overhead: `{float(projection['margin_hours']):.2f}` h",
        "",
        "Chunk/checkpoint overhead is pending measurement; this dry-run does not pass the final overhead gate.",
        "",
        "## Next Action",
        "",
        "`run_o6_threading_layout_scouting_after_manifest_dry_run`",
    ]
    manifest_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    status_path = out / "post_o1_status.json"
    status_md_path = out / "post_o1_status.md"
    post_status = _read_json(status_path)
    post_status.update(
        {
            "created_at_utc": manifest["created_at_utc"],
            "status": status,
            "linear_plus_robustness_chunked_manifest_dry_run": str(manifest_json.relative_to(root)).replace("\\", "/"),
            "linear_plus_robustness_chunked_manifest_chunks_csv": str(chunks_csv.relative_to(root)).replace("\\", "/"),
            "official_execution_authorized": False,
            "stage_c_blocked": True,
            "scientific_outcomes_used": False,
            "next_action": manifest["next_action"],
        }
    )
    write_json_atomic(status_path, post_status, overwrite=True)
    status_md_path.write_text(
        "\n".join(
            [
                "# Post-O1 Status",
                "",
                f"Created UTC: `{manifest['created_at_utc']}`",
                f"Status: `{status}`",
                "",
                "Official execution remains blocked.",
                "Linear plus fixed quadratic/cubic robustness manifest dry-run has been generated.",
                "",
                f"Manifest dry-run: `{str(manifest_json.relative_to(root)).replace(chr(92), '/')}`",
                f"Chunk CSV: `{str(chunks_csv.relative_to(root)).replace(chr(92), '/')}`",
                "",
                f"Next action: `{manifest['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_deviation_log_entry(
        root,
        step="manifest",
        entry_type="amendment",
        description="Linear plus fixed quadratic/cubic robustness chunked manifest dry-run generated without scientific execution.",
        action_taken=(
            f"Recorded {manifest_json.relative_to(root)} and {chunks_csv.relative_to(root)} with "
            f"{len(chunk_rows)} dry-run chunks."
        ),
        blocking=False,
        resolved=True,
    )
    return manifest


def _o6_benchmark_tasks(root: Path) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    task_index = 0
    for instance in ("H1", "H4"):
        for order, scales in (("linear", [1, 3]), ("cubic", [1, 3, 5, 7])):
            for theta_index in range(12):
                tasks.append(
                    {
                        "task_index": task_index,
                        "task_id": f"o6_{instance}_{order}_{theta_index}",
                        "instance": instance,
                        "order": order,
                        "scale_factors": scales,
                        "theta_seed": deterministic_seed("phase_e_o6_theta", instance, order, theta_index),
                        "p_dep": 0.01,
                        "shots_total": 512,
                        "shot_seed": deterministic_seed("phase_e_o6_shots", instance, order, theta_index),
                    }
                )
                task_index += 1
    return tasks


def run_o6_layout_worker(
    root: Path = PROJECT_ROOT,
    *,
    layout_id: str,
    shard_count: int,
    shard_index: int,
) -> dict[str, Any]:
    tasks = [
        task
        for task in _o6_benchmark_tasks(root)
        if int(task["task_index"]) % int(shard_count) == int(shard_index)
    ]
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for task in tasks:
        graph = _graph_from_selected_instance(root, str(task["instance"]))
        theta = tuple(
            float(value)
            for value in np.random.default_rng(int(task["theta_seed"])).uniform(
                low=0.0,
                high=np.asarray(theta_periods(2), dtype=float),
            )
        )
        value = richardson_zne_objective(
            graph,
            theta,
            p_dep=float(task["p_dep"]),
            scale_factors=[int(scale) for scale in task["scale_factors"]],
            shots_total=int(task["shots_total"]),
            seed=int(task["shot_seed"]),
        ).objective_value
        rows.append(
            {
                "task_id": task["task_id"],
                "value": float(value),
            }
        )
    wall_s = float(time.perf_counter() - started)
    result_hash = _canonical_hash(sorted(rows, key=lambda row: str(row["task_id"])))
    return {
        "artifact": "O6_layout_worker_result",
        "layout_id": layout_id,
        "shard_count": int(shard_count),
        "shard_index": int(shard_index),
        "task_count": len(tasks),
        "wall_time_s": wall_s,
        "result_hash": result_hash,
        "rows": rows,
        "memory_peak_mb": _process_rss_mb(),
        "thread_settings": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    }


def _o6_candidate_layouts(physical_cores: int | None, logical_cores: int | None) -> list[dict[str, int]]:
    requested = [
        {"shards": 1, "threads_per_shard": 1},
        {"shards": 2, "threads_per_shard": 1},
        {"shards": 4, "threads_per_shard": 1},
        {"shards": 6, "threads_per_shard": 1},
        {"shards": 8, "threads_per_shard": 1},
        {"shards": 2, "threads_per_shard": 2},
        {"shards": 4, "threads_per_shard": 2},
        {"shards": 2, "threads_per_shard": 4},
    ]
    physical_cap = int(physical_cores or os.cpu_count() or 1)
    logical_cap = int(logical_cores or physical_cap)
    layouts: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for layout in requested:
        total_threads = int(layout["shards"]) * int(layout["threads_per_shard"])
        key = (int(layout["shards"]), int(layout["threads_per_shard"]))
        if total_threads <= max(logical_cap, 1) and key not in seen:
            layouts.append(
                {
                    "shards": int(layout["shards"]),
                    "threads_per_shard": int(layout["threads_per_shard"]),
                    "total_threads": total_threads,
                    "oversubscribes_physical_cores": bool(total_threads > physical_cap),
                }
            )
            seen.add(key)
    return layouts


def run_o6_threading_layout_scouting(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    result_json = out / "O6_threading_layout_scouting_result.json"
    result_md = out / "O6_threading_layout_scouting_result.md"
    result_csv = out / "O6_threading_layout_scouting_result.csv"
    if any(path.exists() for path in (result_json, result_md, result_csv)) and not force:
        raise FileExistsError("Refusing to overwrite existing O6 threading layout scouting artifacts.")

    manifest_path = out / "linear_plus_robustness_chunked_manifest_dry_run.json"
    if not manifest_path.exists():
        raise FileNotFoundError("O6 requires the linear-plus-robustness manifest dry-run artifact first.")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "linear_plus_qc_robustness_chunked_manifest_dry_run_complete_o6_next":
        raise ValueError("O6 requires a completed manifest dry-run with O6 as next action.")

    env_snapshot = _environment_snapshot(root)
    physical_cores = env_snapshot.get("physical_cores")
    logical_cores = env_snapshot.get("logical_cores")
    layouts = _o6_candidate_layouts(
        int(physical_cores) if physical_cores is not None else None,
        int(logical_cores) if logical_cores is not None else None,
    )
    script_path = root / "scripts" / "79_phase_e_cobyla_runtime_o0_o0_75.py"
    rows: list[dict[str, Any]] = []
    layout_reports: list[dict[str, Any]] = []
    baseline_hash: str | None = None
    baseline_wall_s: float | None = None
    aggregate_reference: dict[str, float] | None = None

    for layout in layouts:
        shards = int(layout["shards"])
        threads = int(layout["threads_per_shard"])
        layout_id = f"{shards}x{threads}"
        env = os.environ.copy()
        for thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[thread_var] = str(threads)
        started = time.perf_counter()
        processes = []
        for shard_index in range(shards):
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(script_path),
                        "--mode",
                        "o6_layout_worker",
                        "--o6-layout-id",
                        layout_id,
                        "--o6-shard-count",
                        str(shards),
                        "--o6-shard-index",
                        str(shard_index),
                    ],
                    cwd=root,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )
        worker_reports = []
        failures = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=600)
            if process.returncode != 0:
                failures.append({"returncode": process.returncode, "stdout": stdout, "stderr": stderr})
                continue
            try:
                wrapper = json.loads(stdout)
                worker_reports.append(wrapper["result"])
            except Exception as exc:
                failures.append({"returncode": process.returncode, "stdout": stdout, "stderr": stderr, "parse_error": repr(exc)})
        layout_wall_s = float(time.perf_counter() - started)
        aggregate_rows = []
        for worker in worker_reports:
            aggregate_rows.extend(worker["rows"])
        aggregate_rows = sorted(aggregate_rows, key=lambda row: str(row["task_id"]))
        aggregate_hash = _canonical_hash(aggregate_rows)
        task_count = len(aggregate_rows)
        max_memory_peak_mb = max(
            [float(worker["memory_peak_mb"]) for worker in worker_reports if worker.get("memory_peak_mb") is not None],
            default=math.nan,
        )
        if layout_id == "1x1":
            baseline_hash = aggregate_hash
            baseline_wall_s = layout_wall_s
            aggregate_reference = {str(row["task_id"]): float(row["value"]) for row in aggregate_rows}
        equivalence_pass = bool(baseline_hash is not None and aggregate_hash == baseline_hash and not failures)
        max_abs_diff = 0.0
        if aggregate_reference is not None and task_count == len(aggregate_reference):
            for row in aggregate_rows:
                max_abs_diff = max(max_abs_diff, abs(float(row["value"]) - aggregate_reference[str(row["task_id"])]))
        elif aggregate_reference is not None:
            max_abs_diff = math.inf
        speedup_vs_1x1 = float(baseline_wall_s / layout_wall_s) if baseline_wall_s else 1.0
        projected_wall_hours = float(manifest["projection_semantics"]["serial_equivalent_hours_under_same_model"]) / speedup_vs_1x1
        row = {
            "layout_id": layout_id,
            "shards": shards,
            "threads_per_shard": threads,
            "total_threads": int(layout["total_threads"]),
            "oversubscribes_physical_cores": bool(layout["oversubscribes_physical_cores"]),
            "worker_count": len(worker_reports),
            "task_count": task_count,
            "layout_wall_s": layout_wall_s,
            "baseline_wall_s": baseline_wall_s if baseline_wall_s is not None else layout_wall_s,
            "speedup_vs_1x1": speedup_vs_1x1,
            "projected_wall_hours_from_layout_speedup": projected_wall_hours,
            "equivalence_pass": equivalence_pass,
            "max_abs_diff_vs_1x1": max_abs_diff,
            "aggregate_hash": aggregate_hash,
            "max_memory_peak_mb": max_memory_peak_mb,
            "failure_count": len(failures),
        }
        rows.append(row)
        layout_reports.append(
            {
                **row,
                "thread_env": {key: env[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                "worker_reports": worker_reports,
                "failures": failures,
            }
        )

    valid_rows = [row for row in rows if bool(row["equivalence_pass"]) and int(row["failure_count"]) == 0]
    best = max(valid_rows, key=lambda row: float(row["speedup_vs_1x1"])) if valid_rows else None
    speed_gate_pass = bool(best and float(best["speedup_vs_1x1"]) >= 1.30)
    projection_margin_pass = bool(
        best
        and float(best["projected_wall_hours_from_layout_speedup"])
        <= float(manifest["projection_semantics"]["accepted_local_planning_budget_hours"])
    )
    accepted = bool(speed_gate_pass and projection_margin_pass)
    if accepted:
        status = "o6_threading_layout_accepted_projection_margin_pass"
    elif speed_gate_pass:
        status = "o6_threading_layout_speed_gate_pass_projection_margin_not_established"
    else:
        status = "o6_threading_layout_not_accepted"
    next_action = (
        "update_projection_with_o6_layout_then_freeze_or_run_o5_o4_o7_if_needed"
        if accepted
        else "run_representative_o6_chunk_benchmark_or_consider_o5_o4_o7_budget_scope_stop"
    )
    report = {
        "artifact": "O6_threading_layout_scouting_result",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": status,
        "official_execution_authorized": False,
        "stage_c_authorized": False,
        "stage_c_blocked": True,
        "scientific_outcomes_used": False,
        "dry_run_only": True,
        "manifest_artifact": str(manifest_path.relative_to(root)).replace("\\", "/"),
        "environment": {
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "platform": env_snapshot.get("platform"),
            "ram": env_snapshot.get("ram"),
        },
        "benchmark": {
            "task_count": len(_o6_benchmark_tasks(root)),
            "instances": ["H1", "H4"],
            "orders": ["linear", "cubic"],
            "p_dep": 0.01,
            "shots_total": 512,
            "official_execution": False,
        },
        "projection_semantics": {
            "input_serial_equivalent_hours": float(manifest["projection_semantics"]["serial_equivalent_hours_under_same_model"]),
            "input_wall_hours_at_6_effective_shards": float(manifest["projection_semantics"]["total_projected_wall_hours_excluding_overhead"]),
            "do_not_double_count_parallelism": True,
        },
        "layouts": layout_reports,
        "best_layout": best,
        "acceptance": {
            "speedup_threshold_vs_1x1": 1.30,
            "speed_gate_pass": speed_gate_pass,
            "projection_margin_pass": projection_margin_pass,
            "accepted_for_projection_freeze": accepted,
            "equivalence_required": True,
            "interpretation": "Short O6 scouting ranks layouts. Projection update is accepted only if the generalized layout projection stays within budget.",
        },
        "next_action": next_action,
        "outputs": {
            "json": str(result_json.relative_to(root)).replace("\\", "/"),
            "md": str(result_md.relative_to(root)).replace("\\", "/"),
            "csv": str(result_csv.relative_to(root)).replace("\\", "/"),
        },
    }
    write_json_atomic(result_json, report, overwrite=True)
    _write_csv(result_csv, rows)
    result_md.write_text(
        "\n".join(
            [
                "# O6 Threading/Layout Scouting Result",
                "",
                f"Created UTC: `{report['created_at_utc']}`",
                f"Status: `{status}`",
                "",
                "Official execution remains blocked. This is a non-official technical layout benchmark.",
                "",
                "## Best Layout",
                "",
                f"- Best layout: `{best['layout_id'] if best else 'none'}`",
                f"- Speedup vs 1x1: `{float(best['speedup_vs_1x1']) if best else math.nan:.3f}x`",
                f"- Illustrative wall hours if speedup generalized: `{float(best['projected_wall_hours_from_layout_speedup']) if best else math.nan:.2f}`",
                f"- Speed gate pass: `{speed_gate_pass}`",
                f"- Projection margin pass: `{projection_margin_pass}`",
                f"- Accepted for projection freeze: `{accepted}`",
                "",
                "## Note",
                "",
                "The input projection already assumes 6 effective shards. This benchmark updates the layout assumption and does not apply parallelism twice.",
                "",
                f"Next action: `{next_action}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status_path = out / "post_o1_status.json"
    status_md_path = out / "post_o1_status.md"
    post_status = _read_json(status_path)
    post_status.update(
        {
            "created_at_utc": report["created_at_utc"],
            "status": status,
            "o6_threading_layout_scouting_result": str(result_json.relative_to(root)).replace("\\", "/"),
            "official_execution_authorized": False,
            "stage_c_blocked": True,
            "scientific_outcomes_used": False,
            "next_action": next_action,
        }
    )
    write_json_atomic(status_path, post_status, overwrite=True)
    status_md_path.write_text(
        "\n".join(
            [
                "# Post-O1 Status",
                "",
                f"Created UTC: `{report['created_at_utc']}`",
                f"Status: `{status}`",
                "",
                "Official execution remains blocked.",
                "O6 threading/layout scouting has completed.",
                "",
                f"O6 result: `{str(result_json.relative_to(root)).replace(chr(92), '/')}`",
                "",
                f"Next action: `{next_action}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_deviation_log_entry(
        root,
        step="O6",
        entry_type="amendment",
        description="O6 threading/layout scouting completed without official execution.",
        action_taken=f"Recorded {result_json.relative_to(root)}; accepted={accepted}.",
        blocking=False,
        resolved=True,
    )
    return report


def _o6_representative_benchmark_tasks(root: Path) -> list[dict[str, Any]]:
    config = _read_json(root / "config" / "phase_e_config_v0.1-draft.json")
    scale_factors = {
        order: [int(scale) for scale in config["zne"]["scale_factors"][order]]
        for order in ("linear", "quadratic", "cubic")
    }
    return [
        {
            "task_index": 0,
            "task_id": "o6_rep_endpoint_linear_H1_p0p01_s512",
            "task_kind": "endpoint_linear",
            "instance": "H1",
            "richardson_order": "linear",
            "scale_factors": scale_factors["linear"],
            "p_dep": 0.01,
            "shots_total": 512,
            "seed": 6101,
            "K": 1,
            "E_COBYLA": 12,
        },
        {
            "task_index": 1,
            "task_id": "o6_rep_endpoint_linear_H4_p0p003_s4096",
            "task_kind": "endpoint_linear",
            "instance": "H4",
            "richardson_order": "linear",
            "scale_factors": scale_factors["linear"],
            "p_dep": 0.003,
            "shots_total": 4096,
            "seed": 6102,
            "K": 1,
            "E_COBYLA": 12,
        },
        {
            "task_index": 2,
            "task_id": "o6_rep_endpoint_quadratic_H2_p0p01_s512",
            "task_kind": "endpoint_zne",
            "instance": "H2",
            "richardson_order": "quadratic",
            "scale_factors": scale_factors["quadratic"],
            "p_dep": 0.01,
            "shots_total": 512,
            "seed": 6201,
            "K": 1,
            "E_COBYLA": 10,
        },
        {
            "task_index": 3,
            "task_id": "o6_rep_endpoint_quadratic_H11_p0p003_s4096",
            "task_kind": "endpoint_zne",
            "instance": "H11",
            "richardson_order": "quadratic",
            "scale_factors": scale_factors["quadratic"],
            "p_dep": 0.003,
            "shots_total": 4096,
            "seed": 6202,
            "K": 1,
            "E_COBYLA": 10,
        },
        {
            "task_index": 4,
            "task_id": "o6_rep_endpoint_cubic_H3_p0p01_s512",
            "task_kind": "endpoint_zne",
            "instance": "H3",
            "richardson_order": "cubic",
            "scale_factors": scale_factors["cubic"],
            "p_dep": 0.01,
            "shots_total": 512,
            "seed": 6301,
            "K": 1,
            "E_COBYLA": 8,
        },
        {
            "task_index": 5,
            "task_id": "o6_rep_endpoint_cubic_H12_p0p003_s4096",
            "task_kind": "endpoint_zne",
            "instance": "H12",
            "richardson_order": "cubic",
            "scale_factors": scale_factors["cubic"],
            "p_dep": 0.003,
            "shots_total": 4096,
            "seed": 6302,
            "K": 1,
            "E_COBYLA": 8,
        },
        {
            "task_index": 6,
            "task_id": "o6_rep_lhs_linear_H1_p0p01_s512",
            "task_kind": "lhs_linear",
            "instance": "H1",
            "richardson_order": "linear",
            "scale_factors": scale_factors["linear"],
            "p_dep": 0.01,
            "shots_total": 512,
            "lhs_design_id": 0,
            "N_LHS": 12,
        },
        {
            "task_index": 7,
            "task_id": "o6_rep_lhs_linear_H4_p0p003_s4096",
            "task_kind": "lhs_linear",
            "instance": "H4",
            "richardson_order": "linear",
            "scale_factors": scale_factors["linear"],
            "p_dep": 0.003,
            "shots_total": 4096,
            "lhs_design_id": 0,
            "N_LHS": 8,
        },
        {
            "task_index": 8,
            "task_id": "o6_rep_lhs_quadratic_H2_p0p01_s512",
            "task_kind": "lhs_zne",
            "instance": "H2",
            "richardson_order": "quadratic",
            "scale_factors": scale_factors["quadratic"],
            "p_dep": 0.01,
            "shots_total": 512,
            "lhs_design_id": 0,
            "N_LHS": 8,
        },
        {
            "task_index": 9,
            "task_id": "o6_rep_lhs_quadratic_H11_p0p003_s4096",
            "task_kind": "lhs_zne",
            "instance": "H11",
            "richardson_order": "quadratic",
            "scale_factors": scale_factors["quadratic"],
            "p_dep": 0.003,
            "shots_total": 4096,
            "lhs_design_id": 0,
            "N_LHS": 6,
        },
        {
            "task_index": 10,
            "task_id": "o6_rep_lhs_cubic_H3_p0p01_s512",
            "task_kind": "lhs_zne",
            "instance": "H3",
            "richardson_order": "cubic",
            "scale_factors": scale_factors["cubic"],
            "p_dep": 0.01,
            "shots_total": 512,
            "lhs_design_id": 0,
            "N_LHS": 6,
        },
        {
            "task_index": 11,
            "task_id": "o6_rep_lhs_cubic_H12_p0p003_s4096",
            "task_kind": "lhs_zne",
            "instance": "H12",
            "richardson_order": "cubic",
            "scale_factors": scale_factors["cubic"],
            "p_dep": 0.003,
            "shots_total": 4096,
            "lhs_design_id": 0,
            "N_LHS": 4,
        },
    ]


def _finite_digest(value: float) -> float | str:
    value = float(value)
    if math.isfinite(value):
        return float(f"{value:.12g}")
    return str(value)


def _run_o6_representative_task(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    graph = _graph_from_selected_instance(root, str(task["instance"]))
    task_kind = str(task["task_kind"])
    order = str(task["richardson_order"])
    scales = [int(scale) for scale in task["scale_factors"]]
    p_dep = float(task["p_dep"])
    shots_total = int(task["shots_total"])
    started = time.perf_counter()
    density_call_proxy = 0
    digest_payload: dict[str, Any] = {
        "task_id": str(task["task_id"]),
        "task_kind": task_kind,
        "instance": graph.name,
        "richardson_order": order,
        "p_dep": p_dep,
        "shots_total": shots_total,
        "scale_factors": scales,
        "results": [],
    }

    if task_kind in {"endpoint_linear", "endpoint_zne"}:
        starts = make_paired_initial_thetas(
            depth=2,
            starts=int(task["K"]),
            seed=deterministic_seed("phase_e_o6_representative_starts", graph.name, str(task["task_id"]), int(task["seed"])),
        )
        config = COBYLAConfig(max_evaluations=int(task["E_COBYLA"]))
        for start_id, theta0 in enumerate(starts):
            if task_kind == "endpoint_linear":

                def noisy_objective(theta: tuple[float, ...], start_id: int = start_id) -> float:
                    nonlocal density_call_proxy
                    density_call_proxy += 1
                    return float(
                        evaluate_custom_density_objective(
                            graph,
                            theta,
                            p_dep=p_dep,
                            scale_factor=1,
                            shots=shots_total,
                            seed=deterministic_seed(
                                "phase_e_o6_representative_noisy",
                                graph.name,
                                str(task["task_id"]),
                                int(start_id),
                                tuple(round(x, 12) for x in theta),
                            ),
                        ).objective_value
                    )

                noisy_result = cobyla_minimize(noisy_objective, theta0, config)
                digest_payload["results"].append(
                    {
                        "family": "noisy",
                        "start_id": int(start_id),
                        "status": noisy_result.status,
                        "evaluations": int(noisy_result.num_evaluations),
                        "best_value": _finite_digest(noisy_result.best_value),
                    }
                )

            def zne_objective(theta: tuple[float, ...], start_id: int = start_id) -> float:
                nonlocal density_call_proxy
                density_call_proxy += len(scales)
                return float(
                    richardson_zne_objective(
                        graph,
                        theta,
                        p_dep=p_dep,
                        scale_factors=scales,
                        shots_total=shots_total,
                        seed=deterministic_seed(
                            "phase_e_o6_representative_zne",
                            graph.name,
                            str(task["task_id"]),
                            order,
                            int(start_id),
                            tuple(round(x, 12) for x in theta),
                        ),
                    ).objective_value
                )

            zne_result = cobyla_minimize(zne_objective, theta0, config)
            digest_payload["results"].append(
                {
                    "family": "zne",
                    "start_id": int(start_id),
                    "status": zne_result.status,
                    "evaluations": int(zne_result.num_evaluations),
                    "best_value": _finite_digest(zne_result.best_value),
                }
            )

    elif task_kind in {"lhs_linear", "lhs_zne"}:
        lhs_count = int(task["N_LHS"])
        lhs_seed = deterministic_seed(
            "phase_e_o6_representative_lhs",
            graph.name,
            str(task["task_id"]),
            p_dep,
            shots_total,
            lhs_count,
            int(task["lhs_design_id"]),
        )
        points = _lhs_points(depth=2, count=lhs_count, seed=lhs_seed)
        for point_id, theta in enumerate(points):
            if task_kind == "lhs_linear":
                density_call_proxy += 1 + len(scales)
                result = evaluate_lhs_point_with_scale1_probability_reuse(
                    graph,
                    theta,
                    p_dep=p_dep,
                    scale_factors=scales,
                    shots_total=shots_total,
                    noisy_seed=deterministic_seed("phase_e_o6_representative_lhs_noisy", graph.name, str(task["task_id"]), point_id),
                    zne_seed=deterministic_seed("phase_e_o6_representative_lhs_zne", graph.name, str(task["task_id"]), point_id),
                )
                digest_payload["results"].append(
                    {
                        "point_id": int(point_id),
                        "ideal": _finite_digest(result.ideal_objective),
                        "noisy": _finite_digest(result.noisy_objective),
                        "zne": _finite_digest(result.zne_objective),
                    }
                )
            else:
                density_call_proxy += len(scales)
                result = richardson_zne_objective(
                    graph,
                    theta,
                    p_dep=p_dep,
                    scale_factors=scales,
                    shots_total=shots_total,
                    seed=deterministic_seed("phase_e_o6_representative_lhs_zne_only", graph.name, str(task["task_id"]), point_id),
                )
                digest_payload["results"].append(
                    {
                        "point_id": int(point_id),
                        "zne": _finite_digest(result.objective_value),
                    }
                )
    else:
        raise ValueError(f"Unknown O6 representative task kind: {task_kind}")

    wall_s = float(time.perf_counter() - started)
    return {
        "task_id": str(task["task_id"]),
        "task_kind": task_kind,
        "instance": graph.name,
        "richardson_order": order,
        "shots_total": shots_total,
        "p_dep": p_dep,
        "result_hash": _canonical_hash(digest_payload),
        "wall_time_s": wall_s,
        "density_call_proxy": int(density_call_proxy),
    }


def run_o6_representative_layout_worker(
    root: Path = PROJECT_ROOT,
    *,
    layout_id: str,
    shard_count: int,
    shard_index: int,
) -> dict[str, Any]:
    tasks = [
        task
        for task in _o6_representative_benchmark_tasks(root)
        if int(task["task_index"]) % int(shard_count) == int(shard_index)
    ]
    started = time.perf_counter()
    rows = [_run_o6_representative_task(root, task) for task in tasks]
    wall_s = float(time.perf_counter() - started)
    equivalence_rows = [
        {"task_id": row["task_id"], "result_hash": row["result_hash"]}
        for row in sorted(rows, key=lambda item: str(item["task_id"]))
    ]
    return {
        "artifact": "O6_representative_layout_worker_result",
        "layout_id": layout_id,
        "shard_count": int(shard_count),
        "shard_index": int(shard_index),
        "task_count": len(tasks),
        "wall_time_s": wall_s,
        "result_hash": _canonical_hash(equivalence_rows),
        "rows": rows,
        "memory_peak_mb": _process_rss_mb(),
        "thread_settings": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
        },
    }


def run_o6_representative_chunk_benchmark(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    result_json = out / "O6_representative_chunk_benchmark_result.json"
    result_md = out / "O6_representative_chunk_benchmark_result.md"
    result_csv = out / "O6_representative_chunk_benchmark_result.csv"
    if any(path.exists() for path in (result_json, result_md, result_csv)) and not force:
        raise FileExistsError("Refusing to overwrite existing O6 representative chunk benchmark artifacts.")

    manifest_path = out / "linear_plus_robustness_chunked_manifest_dry_run.json"
    scouting_path = out / "O6_threading_layout_scouting_result.json"
    if not manifest_path.exists() or not scouting_path.exists():
        raise FileNotFoundError("O6 representative benchmark requires manifest dry-run and short O6 scouting artifacts first.")
    manifest = _read_json(manifest_path)
    scouting = _read_json(scouting_path)

    env_snapshot = _environment_snapshot(root)
    physical_cores = env_snapshot.get("physical_cores")
    logical_cores = env_snapshot.get("logical_cores")
    layouts = _o6_candidate_layouts(
        int(physical_cores) if physical_cores is not None else None,
        int(logical_cores) if logical_cores is not None else None,
    )
    script_path = root / "scripts" / "79_phase_e_cobyla_runtime_o0_o0_75.py"
    rows: list[dict[str, Any]] = []
    layout_reports: list[dict[str, Any]] = []
    baseline_hash: str | None = None
    baseline_wall_s: float | None = None
    baseline_task_hashes: dict[str, str] | None = None
    tasks = _o6_representative_benchmark_tasks(root)
    serial_hours = float(manifest["projection_semantics"]["serial_equivalent_hours_under_same_model"])
    budget_hours = float(manifest["projection_semantics"]["accepted_local_planning_budget_hours"])

    for layout in layouts:
        shards = int(layout["shards"])
        threads = int(layout["threads_per_shard"])
        layout_id = f"{shards}x{threads}"
        env = os.environ.copy()
        for thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[thread_var] = str(threads)
        started = time.perf_counter()
        processes = []
        for shard_index in range(shards):
            processes.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        str(script_path),
                        "--mode",
                        "o6_representative_layout_worker",
                        "--o6-layout-id",
                        layout_id,
                        "--o6-shard-count",
                        str(shards),
                        "--o6-shard-index",
                        str(shard_index),
                    ],
                    cwd=root,
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            )
        worker_reports = []
        failures = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=900)
            if process.returncode != 0:
                failures.append({"returncode": process.returncode, "stdout": stdout, "stderr": stderr})
                continue
            try:
                wrapper = json.loads(stdout)
                worker_reports.append(wrapper["result"])
            except Exception as exc:
                failures.append({"returncode": process.returncode, "stdout": stdout, "stderr": stderr, "parse_error": repr(exc)})
        layout_wall_s = float(time.perf_counter() - started)
        aggregate_rows = []
        for worker in worker_reports:
            aggregate_rows.extend(worker["rows"])
        aggregate_rows = sorted(aggregate_rows, key=lambda row: str(row["task_id"]))
        equivalence_rows = [{"task_id": row["task_id"], "result_hash": row["result_hash"]} for row in aggregate_rows]
        aggregate_hash = _canonical_hash(equivalence_rows)
        task_hashes = {str(row["task_id"]): str(row["result_hash"]) for row in aggregate_rows}
        task_count = len(aggregate_rows)
        hash_mismatch_count = 0
        if baseline_task_hashes is not None:
            for task_id, result_hash in task_hashes.items():
                if baseline_task_hashes.get(task_id) != result_hash:
                    hash_mismatch_count += 1
            hash_mismatch_count += len(set(baseline_task_hashes) - set(task_hashes))
        max_memory_peak_mb = max(
            [float(worker["memory_peak_mb"]) for worker in worker_reports if worker.get("memory_peak_mb") is not None],
            default=math.nan,
        )
        if layout_id == "1x1":
            baseline_hash = aggregate_hash
            baseline_wall_s = layout_wall_s
            baseline_task_hashes = task_hashes
        equivalence_pass = bool(
            baseline_hash is not None
            and aggregate_hash == baseline_hash
            and task_count == len(tasks)
            and not failures
            and hash_mismatch_count == 0
        )
        speedup_vs_1x1 = float(baseline_wall_s / layout_wall_s) if baseline_wall_s else 1.0
        projected_wall_hours = float(serial_hours / speedup_vs_1x1)
        task_wall_sum_s = float(sum(float(row["wall_time_s"]) for row in aggregate_rows))
        row = {
            "layout_id": layout_id,
            "shards": shards,
            "threads_per_shard": threads,
            "total_threads": int(layout["total_threads"]),
            "oversubscribes_physical_cores": bool(layout["oversubscribes_physical_cores"]),
            "worker_count": len(worker_reports),
            "task_count": task_count,
            "layout_wall_s": layout_wall_s,
            "task_wall_sum_s": task_wall_sum_s,
            "baseline_wall_s": baseline_wall_s if baseline_wall_s is not None else layout_wall_s,
            "speedup_vs_1x1": speedup_vs_1x1,
            "projected_wall_hours_from_layout_speedup": projected_wall_hours,
            "equivalence_pass": equivalence_pass,
            "hash_mismatch_count": int(hash_mismatch_count),
            "aggregate_hash": aggregate_hash,
            "max_memory_peak_mb": max_memory_peak_mb,
            "failure_count": len(failures),
        }
        rows.append(row)
        layout_reports.append(
            {
                **row,
                "thread_env": {key: env[key] for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                "worker_reports": worker_reports,
                "failures": failures,
            }
        )

    valid_rows = [row for row in rows if bool(row["equivalence_pass"]) and int(row["failure_count"]) == 0]
    best = max(valid_rows, key=lambda row: float(row["speedup_vs_1x1"])) if valid_rows else None
    required_effective_speedup = float(serial_hours / budget_hours)
    speed_gate_pass = bool(best and float(best["speedup_vs_1x1"]) >= 1.30)
    projection_margin_pass = bool(best and float(best["projected_wall_hours_from_layout_speedup"]) <= budget_hours)
    accepted = bool(speed_gate_pass and projection_margin_pass)
    if accepted:
        status = "o6_representative_chunk_benchmark_accepted_projection_margin_pass"
    elif speed_gate_pass:
        status = "o6_representative_chunk_benchmark_speed_gate_pass_projection_margin_not_established"
    else:
        status = "o6_representative_chunk_benchmark_not_accepted"
    next_action = (
        "freeze_measured_o6_layout_then_run_remaining_gates"
        if accepted
        else "budget_scope_external_compute_or_measured_o5_o4_o7_decision_required_before_freeze"
    )
    report = {
        "artifact": "O6_representative_chunk_benchmark_result",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": status,
        "official_execution_authorized": False,
        "stage_c_authorized": False,
        "stage_c_blocked": True,
        "scientific_outcomes_used": False,
        "dry_run_only": True,
        "manifest_artifact": str(manifest_path.relative_to(root)).replace("\\", "/"),
        "short_o6_scouting_artifact": str(scouting_path.relative_to(root)).replace("\\", "/"),
        "environment": {
            "physical_cores": physical_cores,
            "logical_cores": logical_cores,
            "platform": env_snapshot.get("platform"),
            "ram": env_snapshot.get("ram"),
        },
        "benchmark": {
            "task_count": len(tasks),
            "task_ids": [str(task["task_id"]) for task in tasks],
            "task_kinds": sorted({str(task["task_kind"]) for task in tasks}),
            "instances": sorted({str(task["instance"]) for task in tasks}),
            "orders": sorted({str(task["richardson_order"]) for task in tasks}),
            "uses_cobyla": True,
            "uses_lhs": True,
            "records_scientific_values": False,
            "official_execution": False,
        },
        "projection_semantics": {
            "input_serial_equivalent_hours": serial_hours,
            "input_wall_hours_at_6_effective_shards": float(manifest["projection_semantics"]["total_projected_wall_hours_excluding_overhead"]),
            "accepted_local_planning_budget_hours": budget_hours,
            "required_effective_speedup_to_hit_budget_excluding_overhead": required_effective_speedup,
            "do_not_double_count_parallelism": True,
        },
        "short_o6_scouting_summary": {
            "status": scouting.get("status"),
            "best_layout": (scouting.get("best_layout") or {}).get("layout_id"),
            "best_speedup_vs_1x1": (scouting.get("best_layout") or {}).get("speedup_vs_1x1"),
            "accepted_for_projection_freeze": (scouting.get("acceptance") or {}).get("accepted_for_projection_freeze"),
        },
        "layouts": layout_reports,
        "best_layout": best,
        "acceptance": {
            "speedup_threshold_vs_1x1": 1.30,
            "required_effective_speedup_to_hit_budget_excluding_overhead": required_effective_speedup,
            "speed_gate_pass": speed_gate_pass,
            "projection_margin_pass": projection_margin_pass,
            "accepted_for_projection_freeze": accepted,
            "equivalence_required": True,
            "interpretation": "Representative non-official chunk benchmark updates the measured layout assumption; it does not apply parallelism twice.",
        },
        "next_action": next_action,
        "outputs": {
            "json": str(result_json.relative_to(root)).replace("\\", "/"),
            "md": str(result_md.relative_to(root)).replace("\\", "/"),
            "csv": str(result_csv.relative_to(root)).replace("\\", "/"),
        },
    }
    write_json_atomic(result_json, report, overwrite=True)
    _write_csv(result_csv, rows)
    result_md.write_text(
        "\n".join(
            [
                "# O6 Representative Chunk Benchmark Result",
                "",
                f"Created UTC: `{report['created_at_utc']}`",
                f"Status: `{status}`",
                "",
                "Official execution remains blocked. This benchmark used mini-chunks for technical timing and equivalence only.",
                "",
                "## Best Layout",
                "",
                f"- Best layout: `{best['layout_id'] if best else 'none'}`",
                f"- Speedup vs 1x1: `{float(best['speedup_vs_1x1']) if best else math.nan:.3f}x`",
                f"- Projected wall hours if generalized: `{float(best['projected_wall_hours_from_layout_speedup']) if best else math.nan:.2f}`",
                f"- Required effective speedup for 375 h excluding overhead: `{required_effective_speedup:.3f}x`",
                f"- Projection margin pass: `{projection_margin_pass}`",
                f"- Accepted for projection freeze: `{accepted}`",
                "",
                "## Guardrail",
                "",
                "No scientific values, endpoints, plots, or partial conclusions are recorded in this artifact.",
                "",
                f"Next action: `{next_action}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status_path = out / "post_o1_status.json"
    status_md_path = out / "post_o1_status.md"
    post_status = _read_json(status_path)
    post_status.update(
        {
            "created_at_utc": report["created_at_utc"],
            "status": status,
            "o6_representative_chunk_benchmark_result": str(result_json.relative_to(root)).replace("\\", "/"),
            "official_execution_authorized": False,
            "stage_c_blocked": True,
            "scientific_outcomes_used": False,
            "next_action": next_action,
        }
    )
    write_json_atomic(status_path, post_status, overwrite=True)
    status_md_path.write_text(
        "\n".join(
            [
                "# Post-O1 Status",
                "",
                f"Created UTC: `{report['created_at_utc']}`",
                f"Status: `{status}`",
                "",
                "Official execution remains blocked.",
                "O6 representative chunk benchmark has completed.",
                "",
                f"O6 representative result: `{str(result_json.relative_to(root)).replace(chr(92), '/')}`",
                "",
                f"Next action: `{next_action}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_deviation_log_entry(
        root,
        step="O6",
        entry_type="amendment",
        description="O6 representative chunk benchmark completed without official execution or scientific partial analysis.",
        action_taken=f"Recorded {result_json.relative_to(root)}; accepted={accepted}.",
        blocking=False,
        resolved=True,
    )
    return report


def _o5_dynamic_base_tasks(root: Path) -> list[dict[str, Any]]:
    config = _read_json(root / "config" / "phase_e_config_v0.1-draft.json")
    scale_factors = {
        order: [int(scale) for scale in config["zne"]["scale_factors"][order]]
        for order in ("linear", "quadratic", "cubic")
    }
    tasks: list[dict[str, Any]] = []
    for source_task in _o6_representative_benchmark_tasks(root):
        if str(source_task["task_kind"]) not in {"endpoint_linear", "endpoint_zne"}:
            continue
        task = dict(source_task)
        task["task_index"] = len(tasks)
        task["task_id"] = str(source_task["task_id"]).replace("o6_rep_", "o5_dyn_endpoint_")
        tasks.append(task)

    lhs_cells = [
        ("H4", 0.01, 512, 0, 12),
        ("H11", 0.003, 4096, 0, 8),
        ("H2", 0.01, 512, 0, 8),
        ("H12", 0.003, 4096, 0, 6),
    ]
    for instance, p_dep, shots_total, lhs_design_id, n_lhs in lhs_cells:
        tasks.append(
            {
                "task_index": len(tasks),
                "task_id": (
                    f"o5_dyn_lhs_cross_order_{instance}_p{_format_p_dep_for_id(p_dep)}"
                    f"_s{shots_total}_n{n_lhs}"
                ),
                "task_kind": "lhs_cross_order_cache",
                "instance": instance,
                "richardson_order": "cross_order",
                "scale_factors_by_order": scale_factors,
                "p_dep": float(p_dep),
                "shots_total": int(shots_total),
                "lhs_design_id": int(lhs_design_id),
                "N_LHS": int(n_lhs),
            }
        )
    return tasks


def _o5_dynamic_repeated_tasks(root: Path, repeats: int = 6) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for repeat_index in range(int(repeats)):
        for base_task in _o5_dynamic_base_tasks(root):
            task = dict(base_task)
            base_task_id = str(base_task["task_id"])
            task["base_task_id"] = base_task_id
            task["repeat_index"] = int(repeat_index)
            task["task_index"] = len(tasks)
            task["task_id"] = f"{base_task_id}__rep{repeat_index:02d}"
            if "seed" in task:
                task["seed"] = deterministic_seed("phase_e_o5_dynamic_repeat_seed", base_task_id, int(repeat_index), int(task["seed"]))
            tasks.append(task)
    return tasks


def _run_o5_dynamic_probe_task(root: Path, task: dict[str, Any]) -> dict[str, Any]:
    task_kind = str(task["task_kind"])
    if task_kind in {"endpoint_linear", "endpoint_zne"}:
        row = _run_o6_representative_task(root, task)
        row["benchmark_family"] = "endpoint_representative"
        row["repeat_index"] = int(task.get("repeat_index", 0))
        row["base_task_id"] = str(task.get("base_task_id", task["task_id"]))
        row["process_id"] = os.getpid()
        return row

    if task_kind != "lhs_cross_order_cache":
        raise ValueError(f"Unknown O5 dynamic task kind: {task_kind}")

    graph = _graph_from_selected_instance(root, str(task["instance"]))
    scales_by_order = {
        str(order): [int(scale) for scale in scales]
        for order, scales in dict(task["scale_factors_by_order"]).items()
    }
    p_dep = float(task["p_dep"])
    shots_total = int(task["shots_total"])
    lhs_count = int(task["N_LHS"])
    lhs_design_id = int(task["lhs_design_id"])
    lhs_seed = deterministic_seed("phase_e_lhs", graph.name, p_dep, shots_total, lhs_count, lhs_design_id)
    points = _lhs_points(depth=2, count=lhs_count, seed=lhs_seed)
    probability_scale_count = len({scale for scales in scales_by_order.values() for scale in scales})
    density_call_proxy = 0
    digest_payload: dict[str, Any] = {
        "task_id": str(task["task_id"]),
        "task_kind": task_kind,
        "instance": graph.name,
        "p_dep": p_dep,
        "shots_total": shots_total,
        "lhs_design_id": lhs_design_id,
        "N_LHS": lhs_count,
        "scale_factors_by_order": scales_by_order,
        "results": [],
    }
    started = time.perf_counter()
    for point_id, theta in enumerate(points):
        density_call_proxy += 1 + probability_scale_count
        result = evaluate_lhs_point_cross_order_probability_cache(
            graph,
            theta,
            p_dep=p_dep,
            scale_factors_by_order=scales_by_order,
            shots_total=shots_total,
            noisy_seed=deterministic_seed("phase_e_lhs_noisy", graph.name, p_dep, shots_total, lhs_count, lhs_design_id, int(point_id)),
            zne_seed=deterministic_seed("phase_e_lhs_zne", graph.name, p_dep, shots_total, lhs_count, lhs_design_id, int(point_id)),
        )
        digest_payload["results"].append(
            {
                "point_id": int(point_id),
                "ideal": _finite_digest(result.ideal_objective),
                "noisy": _finite_digest(result.noisy_objective),
                "zne_by_order": {
                    order: _finite_digest(value)
                    for order, value in sorted(result.zne_objectives_by_order.items())
                },
                "probability_scale_factors": result.probability_scale_factors,
            }
        )
    wall_s = float(time.perf_counter() - started)
    return {
        "task_id": str(task["task_id"]),
        "base_task_id": str(task.get("base_task_id", task["task_id"])),
        "repeat_index": int(task.get("repeat_index", 0)),
        "task_kind": task_kind,
        "benchmark_family": "lhs_cross_order_cache",
        "instance": graph.name,
        "richardson_order": "cross_order",
        "shots_total": shots_total,
        "p_dep": p_dep,
        "N_LHS": lhs_count,
        "result_hash": _canonical_hash(digest_payload),
        "wall_time_s": wall_s,
        "density_call_proxy": int(density_call_proxy),
        "process_id": os.getpid(),
    }


_O5_DYNAMIC_WORKER_ROOT: Path | None = None


def _o5_dynamic_worker_init(root_text: str) -> None:
    global _O5_DYNAMIC_WORKER_ROOT
    _O5_DYNAMIC_WORKER_ROOT = Path(root_text)


def _o5_dynamic_worker_run(task: dict[str, Any]) -> dict[str, Any]:
    if _O5_DYNAMIC_WORKER_ROOT is None:
        raise RuntimeError("O5 dynamic worker root was not initialized.")
    return _run_o5_dynamic_probe_task(_O5_DYNAMIC_WORKER_ROOT, task)


def run_o5_dynamic_worker_throughput_gate(
    root: Path = PROJECT_ROOT,
    *,
    worker_count: int = 8,
    repeats: int = 6,
    force: bool = False,
    artifact_stem: str | None = None,
    update_status: bool = False,
) -> dict[str, Any]:
    worker_count = int(worker_count)
    repeats = int(repeats)
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1.")
    if repeats < 1:
        raise ValueError("repeats must be >= 1.")

    out = _output_dir(root)
    if artifact_stem is None:
        artifact_stem = f"G_L1_o5_dynamic_worker{worker_count}_throughput_probe"
    result_json = out / f"{artifact_stem}.json"
    result_md = out / f"{artifact_stem}.md"
    result_csv = out / f"{artifact_stem}.csv"
    if any(path.exists() for path in (result_json, result_md, result_csv)) and not force:
        raise FileExistsError(f"Refusing to overwrite existing O5 dynamic worker artifacts for {artifact_stem}.")

    manifest_path = out / "linear_plus_robustness_chunked_manifest_dry_run.json"
    o5_probe_path = out / "O5_lhs_cross_order_probability_cache_probe.json"
    recheck_path = out / "post_o6_parallelism_recheck.json"
    if not manifest_path.exists() or not o5_probe_path.exists() or not recheck_path.exists():
        raise FileNotFoundError("O5 dynamic gate requires manifest dry-run, O5 probe, and post-O6 parallelism recheck artifacts.")
    manifest = _read_json(manifest_path)
    o5_probe = _read_json(o5_probe_path)
    recheck = _read_json(recheck_path)
    if not bool(o5_probe.get("equivalence", {}).get("passed")):
        raise RuntimeError("O5 dynamic gate requires the O5 cross-order cache equivalence probe to pass first.")

    for thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[thread_var] = "1"

    env_snapshot = _environment_snapshot(root)
    tasks = _o5_dynamic_repeated_tasks(root, repeats=repeats)
    serial_after_o5_hours = float(o5_probe["projection_sensitivity"]["serial_equivalent_hours_after_o5"])
    budget_hours = float(manifest["projection_semantics"]["accepted_local_planning_budget_hours"])
    required_speedup = float(serial_after_o5_hours / budget_hours)

    baseline_started = time.perf_counter()
    baseline_rows = [_run_o5_dynamic_probe_task(root, task) for task in tasks]
    baseline_wall_s = float(time.perf_counter() - baseline_started)

    dynamic_started = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=worker_count, initializer=_o5_dynamic_worker_init, initargs=(str(root),)) as pool:
        dynamic_rows = list(pool.imap_unordered(_o5_dynamic_worker_run, tasks, chunksize=1))
    dynamic_wall_s = float(time.perf_counter() - dynamic_started)

    baseline_hashes = {str(row["task_id"]): str(row["result_hash"]) for row in baseline_rows}
    dynamic_hashes = {str(row["task_id"]): str(row["result_hash"]) for row in dynamic_rows}
    hash_mismatch_ids = sorted(
        task_id
        for task_id in set(baseline_hashes) | set(dynamic_hashes)
        if baseline_hashes.get(task_id) != dynamic_hashes.get(task_id)
    )
    equivalence_rows = [
        {"task_id": task_id, "result_hash": result_hash}
        for task_id, result_hash in sorted(baseline_hashes.items())
    ]
    dynamic_equivalence_rows = [
        {"task_id": task_id, "result_hash": result_hash}
        for task_id, result_hash in sorted(dynamic_hashes.items())
    ]
    equivalence_pass = bool(
        len(baseline_rows) == len(tasks)
        and len(dynamic_rows) == len(tasks)
        and not hash_mismatch_ids
        and _canonical_hash(equivalence_rows) == _canonical_hash(dynamic_equivalence_rows)
    )
    dynamic_speedup = float(baseline_wall_s / dynamic_wall_s) if dynamic_wall_s > 0.0 else math.inf
    projected_wall_after_o5_dynamic_hours = float(serial_after_o5_hours / dynamic_speedup)
    projection_margin_pass = bool(projected_wall_after_o5_dynamic_hours <= budget_hours)
    accepted = bool(equivalence_pass and projection_margin_pass)
    status = (
        "o5_dynamic_worker_gate_passed_projection_margin"
        if accepted
        else "o5_dynamic_worker_gate_failed_projection_margin_stage_c_still_blocked"
        if equivalence_pass
        else "o5_dynamic_worker_gate_failed_equivalence"
    )
    next_action = (
        "freeze_o5_dynamic_runner_then_final_overhead_and_authorization_gate"
        if accepted
        else "external_compute_budget_scope_or_stop_decision_required_after_max_local_optimization_gate"
    )

    baseline_by_id = {str(row["task_id"]): row for row in baseline_rows}
    dynamic_by_id = {str(row["task_id"]): row for row in dynamic_rows}
    csv_rows: list[dict[str, Any]] = []
    for task_id in sorted(set(baseline_by_id) | set(dynamic_by_id)):
        baseline_row = baseline_by_id.get(task_id, {})
        dynamic_row = dynamic_by_id.get(task_id, {})
        row_source = baseline_row or dynamic_row
        csv_rows.append(
            {
                "task_id": task_id,
                "base_task_id": row_source.get("base_task_id"),
                "repeat_index": row_source.get("repeat_index"),
                "task_kind": row_source.get("task_kind"),
                "benchmark_family": row_source.get("benchmark_family"),
                "instance": row_source.get("instance"),
                "p_dep": row_source.get("p_dep"),
                "shots_total": row_source.get("shots_total"),
                "N_LHS": row_source.get("N_LHS", ""),
                "baseline_task_wall_s": baseline_row.get("wall_time_s", ""),
                "dynamic_task_wall_s": dynamic_row.get("wall_time_s", ""),
                "baseline_process_id": baseline_row.get("process_id", ""),
                "dynamic_process_id": dynamic_row.get("process_id", ""),
                "density_call_proxy": row_source.get("density_call_proxy", ""),
                "result_hash_match": baseline_hashes.get(task_id) == dynamic_hashes.get(task_id),
            }
        )

    unique_dynamic_processes = sorted({int(row["process_id"]) for row in dynamic_rows if row.get("process_id") is not None})
    dynamic_task_wall_sum_s = float(sum(float(row["wall_time_s"]) for row in dynamic_rows))
    baseline_task_wall_sum_s = float(sum(float(row["wall_time_s"]) for row in baseline_rows))
    report = {
        "artifact": artifact_stem,
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": status,
        "official_execution_authorized": False,
        "stage_c_authorized": False,
        "stage_c_blocked": True,
        "scientific_outcomes_used": False,
        "dry_run_only": True,
        "records_scientific_values": False,
        "source_artifacts": {
            "manifest": str(manifest_path.relative_to(root)).replace("\\", "/"),
            "o5_probe": str(o5_probe_path.relative_to(root)).replace("\\", "/"),
            "post_o6_parallelism_recheck": str(recheck_path.relative_to(root)).replace("\\", "/"),
        },
        "environment": {
            "physical_cores": env_snapshot.get("physical_cores"),
            "logical_cores": env_snapshot.get("logical_cores"),
            "platform": env_snapshot.get("platform"),
            "ram": env_snapshot.get("ram"),
            "thread_settings": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
                "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
                "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            },
        },
        "benchmark": {
            "base_task_count": len(_o5_dynamic_base_tasks(root)),
            "repeats": repeats,
            "total_task_count": len(tasks),
            "dynamic_worker_processes_requested": worker_count,
            "dynamic_worker_processes_observed": len(unique_dynamic_processes),
            "unique_dynamic_process_ids": unique_dynamic_processes,
            "task_kinds": sorted({str(task["task_kind"]) for task in tasks}),
            "uses_cobyla": True,
            "uses_lhs_cross_order_cache": True,
            "uses_dynamic_queue": True,
            "uses_persistent_pool": True,
        },
        "timing": {
            "baseline_1x1_wall_s": baseline_wall_s,
            "baseline_task_wall_sum_s": baseline_task_wall_sum_s,
            "dynamic_worker_wall_s": dynamic_wall_s,
            f"dynamic_{worker_count}x1_wall_s": dynamic_wall_s,
            "dynamic_task_wall_sum_s": dynamic_task_wall_sum_s,
            "dynamic_speedup_vs_1x1": dynamic_speedup,
            "parallel_task_slowdown_vs_baseline_task_sum": float(dynamic_task_wall_sum_s / baseline_task_wall_sum_s)
            if baseline_task_wall_sum_s > 0.0
            else math.nan,
        },
        "equivalence": {
            "o5_helper_probe_passed": bool(o5_probe.get("equivalence", {}).get("passed")),
            "dynamic_hash_equivalence_passed": equivalence_pass,
            "hash_mismatch_count": len(hash_mismatch_ids),
            "hash_mismatch_ids": hash_mismatch_ids,
            "baseline_aggregate_hash": _canonical_hash(equivalence_rows),
            "dynamic_aggregate_hash": _canonical_hash(dynamic_equivalence_rows),
        },
        "projection": {
            "serial_equivalent_hours_before_o5": float(o5_probe["projection_sensitivity"]["source_serial_equivalent_hours"]),
            "serial_equivalent_hours_after_o5": serial_after_o5_hours,
            "accepted_local_planning_budget_hours": budget_hours,
            "required_effective_speedup_after_o5_to_hit_budget_excluding_overhead": required_speedup,
            "measured_dynamic_speedup_vs_1x1": dynamic_speedup,
            "projected_wall_hours_after_o5_dynamic_speedup": projected_wall_after_o5_dynamic_hours,
            "projection_margin_pass": projection_margin_pass,
            "post_o6_repeat_probe_context": {
                "repeat_probe_8x1_static_projected_wall_hours": recheck["repeat_probe_non_official"]["layouts_tested"][0][
                    "projected_wall_hours_if_generalized_static"
                ],
                "repeat_probe_8x1_optimistic_no_coord_projected_wall_hours": recheck["repeat_probe_non_official"]["layouts_tested"][0][
                    "optimistic_no_imbalance_no_coord_projected_wall_hours"
                ],
                "repeat_probe_8x1_optimistic_with_coord_projected_wall_hours": recheck["repeat_probe_non_official"]["layouts_tested"][0][
                    "optimistic_no_imbalance_with_coord_projected_wall_hours"
                ],
            },
        },
        "acceptance": {
            "equivalence_required": True,
            "projection_margin_required": True,
            "accepted_for_projection_freeze": accepted,
            "interpretation": (
                "O5 cross-order LHS caching is valid, and this gate measures whether a dynamic persistent "
                f"{worker_count}-worker runner makes the current local machine clear the 375 h planning budget. It does not "
                "authorize official execution."
            ),
        },
        "code_hashes": _hashes_for_existing(
            root,
            [
                "src/phase_e_sim_backend.py",
                "src/phase_e_cobyla_runtime_optimization.py",
                "scripts/79_phase_e_cobyla_runtime_o0_o0_75.py",
            ],
        ),
        "next_action": next_action,
        "outputs": {
            "json": str(result_json.relative_to(root)).replace("\\", "/"),
            "md": str(result_md.relative_to(root)).replace("\\", "/"),
            "csv": str(result_csv.relative_to(root)).replace("\\", "/"),
        },
    }
    write_json_atomic(result_json, report, overwrite=True)
    _write_csv(result_csv, csv_rows)
    result_md.write_text(
        "\n".join(
            [
                f"# O5 Dynamic {worker_count}-Worker Throughput Gate Result",
                "",
                f"Created UTC: `{report['created_at_utc']}`",
                f"Status: `{status}`",
                "",
                "Official execution remains blocked. This is a non-official technical throughput gate.",
                "",
                "## Result",
                "",
                f"- O5 helper equivalence probe passed: `{bool(o5_probe.get('equivalence', {}).get('passed'))}`",
                f"- Dynamic hash equivalence passed: `{equivalence_pass}`",
                f"- Dynamic speedup vs 1x1: `{dynamic_speedup:.3f}x`",
                f"- Required speedup after O5 for 375 h: `{required_speedup:.3f}x`",
                f"- Projected wall after O5 + dynamic workers: `{projected_wall_after_o5_dynamic_hours:.2f}` h",
                f"- Projection margin pass: `{projection_margin_pass}`",
                f"- Accepted for projection freeze: `{accepted}`",
                "",
                "No endpoint classifications, d_R/r_MSE conclusions, plots, or partial scientific comparisons were recorded.",
                "",
                f"Next action: `{next_action}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    if update_status:
        status_path = out / "post_o1_status.json"
        status_md_path = out / "post_o1_status.md"
        post_status = _read_json(status_path)
        post_status.update(
            {
                "created_at_utc": report["created_at_utc"],
                "status": status,
                "o5_dynamic_8worker_throughput_gate_result": str(result_json.relative_to(root)).replace("\\", "/"),
                "official_execution_authorized": False,
                "stage_c_blocked": True,
                "scientific_outcomes_used": False,
                "next_action": next_action,
            }
        )
        write_json_atomic(status_path, post_status, overwrite=True)
        status_md_path.write_text(
            "\n".join(
                [
                    "# Post-O1 Status",
                    "",
                    f"Created UTC: `{report['created_at_utc']}`",
                    f"Status: `{status}`",
                    "",
                    "Official execution remains blocked.",
                    f"O5 cross-order cache plus dynamic {worker_count}-worker throughput gate has completed.",
                    "",
                    f"O5 dynamic gate: `{str(result_json.relative_to(root)).replace(chr(92), '/')}`",
                    "",
                    f"Next action: `{next_action}`",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        write_deviation_log_entry(
            root,
            step="O5/O6",
            entry_type="amendment",
            description=(
                f"O5 cross-order LHS cache helper and dynamic {worker_count}-worker throughput gate "
                "completed without official execution."
            ),
            action_taken=(
                f"Recorded {result_json.relative_to(root)} with dynamic_speedup={dynamic_speedup:.3f}x, "
                f"projected_wall={projected_wall_after_o5_dynamic_hours:.2f} h, accepted={accepted}."
            ),
            blocking=not accepted,
            resolved=accepted,
        )
    return report


def run_o5_dynamic_8worker_throughput_gate(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    return run_o5_dynamic_worker_throughput_gate(
        root,
        worker_count=8,
        repeats=6,
        force=force,
        artifact_stem="O5_dynamic_8worker_throughput_gate_result",
        update_status=True,
    )


def write_g_l1_local_revalidation_report(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    result_json = out / "G_L1_local_runner_revalidation_report.json"
    result_md = out / "G_L1_local_runner_revalidation_report.md"
    if any(path.exists() for path in (result_json, result_md)) and not force:
        raise FileExistsError("Refusing to overwrite existing G-L1 local revalidation report.")

    compileall_passed = bool(
        compileall.compile_dir(root / "src", quiet=1)
        and compileall.compile_dir(root / "scripts", quiet=1)
    )
    preflight_path = root / "results" / "phase_e" / "planning" / "phase_e_config_preflight.json"
    tests_path = root / "results" / "phase_e" / "planning" / "phase_e_automated_tests_summary.json"
    baseline_path = out / "O5_dynamic_8worker_throughput_gate_result.json"
    if not preflight_path.exists() or not tests_path.exists() or not baseline_path.exists():
        raise FileNotFoundError("G-L1 report requires preflight, automated tests, and pre-G-L1 O5 baseline artifacts.")

    preflight = _read_json(preflight_path)
    tests = _read_json(tests_path)
    baseline = _read_json(baseline_path)
    baseline_hash = str(baseline["equivalence"]["dynamic_aggregate_hash"])
    env_snapshot = _environment_snapshot(root)
    max_local_cores = int(env_snapshot.get("logical_cores") or os.cpu_count() or 1)
    required_worker_counts = sorted({1, 2, 8, max_local_cores})
    oversubscription_worker_count = max_local_cores + 1

    worker_reports: list[dict[str, Any]] = []
    for worker_count in [*required_worker_counts, oversubscription_worker_count]:
        probe_path = out / f"G_L1_o5_dynamic_worker{worker_count}_throughput_probe.json"
        if not probe_path.exists():
            worker_reports.append(
                {
                    "worker_count": int(worker_count),
                    "artifact": str(probe_path.relative_to(root)).replace("\\", "/"),
                    "exists": False,
                    "required": worker_count in required_worker_counts,
                    "equivalence_passed": False,
                    "aggregate_hash_matches_pre_G_L1": False,
                }
            )
            continue
        probe = _read_json(probe_path)
        dynamic_hash = str(probe["equivalence"]["dynamic_aggregate_hash"])
        worker_reports.append(
            {
                "worker_count": int(worker_count),
                "artifact": str(probe_path.relative_to(root)).replace("\\", "/"),
                "exists": True,
                "required": worker_count in required_worker_counts,
                "oversubscription_probe": worker_count == oversubscription_worker_count,
                "equivalence_passed": bool(probe["equivalence"]["dynamic_hash_equivalence_passed"]),
                "aggregate_hash": dynamic_hash,
                "aggregate_hash_matches_pre_G_L1": dynamic_hash == baseline_hash,
                "hash_mismatch_count": int(probe["equivalence"]["hash_mismatch_count"]),
                "dynamic_worker_processes_requested": int(probe["benchmark"]["dynamic_worker_processes_requested"]),
                "dynamic_worker_processes_observed": int(probe["benchmark"]["dynamic_worker_processes_observed"]),
                "total_task_count": int(probe["benchmark"]["total_task_count"]),
                "records_scientific_values": bool(probe.get("records_scientific_values")),
                "scientific_outcomes_used": bool(probe.get("scientific_outcomes_used")),
                "official_execution_authorized": bool(probe.get("official_execution_authorized")),
                "stage_c_blocked": bool(probe.get("stage_c_blocked")),
            }
        )

    required_worker_pass = all(
        row["exists"]
        and row["equivalence_passed"]
        and row["aggregate_hash_matches_pre_G_L1"]
        and int(row["hash_mismatch_count"]) == 0
        and not row["records_scientific_values"]
        and not row["scientific_outcomes_used"]
        and not row["official_execution_authorized"]
        and row["stage_c_blocked"]
        for row in worker_reports
        if row["required"]
    )
    oversubscription_report = next(
        (row for row in worker_reports if row.get("oversubscription_probe")),
        None,
    )
    oversubscription_pass = bool(
        oversubscription_report
        and oversubscription_report["exists"]
        and oversubscription_report["equivalence_passed"]
        and oversubscription_report["aggregate_hash_matches_pre_G_L1"]
    )
    tests_passed = (
        tests.get("status") == "passed"
        and int(tests.get("tests_run", -1)) == 20
        and int(tests.get("tests_passed", -1)) == 20
        and int(tests.get("tests_failed", -1)) == 0
    )
    preflight_passed = preflight.get("status") == "passed" and not preflight.get("errors")
    overall_passed = bool(compileall_passed and preflight_passed and tests_passed and required_worker_pass and oversubscription_pass)
    status = "g_l1_local_revalidation_passed" if overall_passed else "g_l1_local_revalidation_failed"
    next_action = "prepare_G0_local_archive_package" if overall_passed else "diagnose_G_L1_before_G0"

    report = {
        "artifact": "G_L1_local_runner_revalidation_report",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": status,
        "official_execution_authorized": False,
        "stage_c_authorized": False,
        "stage_c_blocked": True,
        "scientific_outcomes_used": False,
        "records_scientific_values": False,
        "dry_run_only": True,
        "checks": {
            "compileall_passed": compileall_passed,
            "phase_e_config_preflight_passed": preflight_passed,
            "phase_e_automated_tests_20_of_20_passed": tests_passed,
            "required_worker_count_hash_equivalence_passed": required_worker_pass,
            "oversubscription_worker_count_hash_equivalence_passed": oversubscription_pass,
        },
        "baseline": {
            "pre_G_L1_o5_dynamic_artifact": str(baseline_path.relative_to(root)).replace("\\", "/"),
            "pre_G_L1_o5_dynamic_aggregate_hash": baseline_hash,
        },
        "worker_validation": {
            "max_local_cores": max_local_cores,
            "required_worker_counts": required_worker_counts,
            "oversubscription_worker_count": oversubscription_worker_count,
            "reports": worker_reports,
        },
        "source_artifacts": {
            "preflight": str(preflight_path.relative_to(root)).replace("\\", "/"),
            "automated_tests": str(tests_path.relative_to(root)).replace("\\", "/"),
        },
        "code_hashes": _hashes_for_existing(
            root,
            [
                "src/phase_e_cobyla_runtime_optimization.py",
                "scripts/79_phase_e_cobyla_runtime_o0_o0_75.py",
                "src/phase_e_sim_backend.py",
            ],
        ),
        "next_action": next_action,
        "outputs": {
            "json": str(result_json.relative_to(root)).replace("\\", "/"),
            "md": str(result_md.relative_to(root)).replace("\\", "/"),
        },
    }
    write_json_atomic(result_json, report, overwrite=True)
    result_md.write_text(
        "\n".join(
            [
                "# G-L1 Local Runner Revalidation Report",
                "",
                f"Created UTC: `{report['created_at_utc']}`",
                f"Status: `{status}`",
                "",
                "Official execution remains blocked. This report validates only the non-scientific worker-count parameterization.",
                "",
                "## Checks",
                "",
                f"- compileall passed: `{compileall_passed}`",
                f"- Phase E config preflight passed: `{preflight_passed}`",
                f"- Phase E automated tests 20/20 passed: `{tests_passed}`",
                f"- Required worker-count hash equivalence passed: `{required_worker_pass}`",
                f"- Oversubscription hash equivalence passed: `{oversubscription_pass}`",
                "",
                "## Baseline",
                "",
                f"- Pre-G-L1 O5 aggregate hash: `{baseline_hash}`",
                "",
                "## Worker Counts",
                "",
                *[
                    (
                        f"- worker_count={row['worker_count']}: exists=`{row['exists']}`, "
                        f"equivalence=`{row['equivalence_passed']}`, "
                        f"hash_matches_pre_G_L1=`{row['aggregate_hash_matches_pre_G_L1']}`"
                    )
                    for row in worker_reports
                ],
                "",
                f"Next action: `{next_action}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status_path = out / "post_o1_status.json"
    status_md_path = out / "post_o1_status.md"
    post_status = _read_json(status_path)
    post_status.update(
        {
            "created_at_utc": report["created_at_utc"],
            "status": status,
            "g_l1_local_runner_revalidation_report": str(result_json.relative_to(root)).replace("\\", "/"),
            "official_execution_authorized": False,
            "stage_c_blocked": True,
            "scientific_outcomes_used": False,
            "next_action": next_action,
        }
    )
    write_json_atomic(status_path, post_status, overwrite=True)
    status_md_path.write_text(
        "\n".join(
            [
                "# Post-O1 Status",
                "",
                f"Created UTC: `{report['created_at_utc']}`",
                f"Status: `{status}`",
                "",
                "Official execution remains blocked.",
                "G-L1 local runner adaptation and revalidation completed.",
                "",
                f"G-L1 report: `{str(result_json.relative_to(root)).replace(chr(92), '/')}`",
                "",
                f"Next action: `{next_action}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_deviation_log_entry(
        root,
        step="G-L1",
        entry_type="amendment",
        description="Local worker-count parameterization revalidation completed without cloud execution or scientific partial analysis.",
        action_taken=f"Recorded {result_json.relative_to(root)}; passed={overall_passed}.",
        blocking=not overall_passed,
        resolved=overall_passed,
    )
    return report


def _g0_archive_source_files(root: Path, package_dir: Path) -> list[str]:
    files: set[str] = set()
    direct_files = [
        "README.md",
        "ENVIRONMENT.md",
        "requirements.txt",
        "requirements-lock.txt",
        "package_versions.txt",
        "computacion.md",
        "PHASE_E_COBYLA_ENDPOINT_PROTOCOL_v0.1-draft.md",
        "PHASE_E_COBYLA_CHUNKED_EXECUTION_OPTIMIZATION_AMENDMENT_v0.1-draft.md",
        "PHASE_E_GOOGLE_CLOUD_EXTERNAL_COMPUTE_PLAN_v0.1-draft.md",
        "config/phase_e_config_v0.1-draft.json",
        "config/phase_e_instances_v0.1-draft.json",
        "config/phase_e_noise_model_v0.1-draft.json",
        "results/phase_e/planning/phase_e_config_preflight.json",
        "results/phase_e/planning/phase_e_automated_tests_summary.json",
        "results/phase_e/planning/phase_e_automated_tests_report.md",
        "results/phase_e/planning/phase_e_backend_tiny_benchmark.json",
    ]
    for relative in direct_files:
        path = root / relative
        if path.is_file():
            files.add(relative)

    for base in (root / "src", root / "scripts"):
        if base.exists():
            for path in base.rglob("*.py"):
                if "__pycache__" not in path.parts:
                    files.add(str(path.relative_to(root)).replace("\\", "/"))

    runtime_dir = root / OUTPUT_DIR_RELATIVE
    if runtime_dir.exists():
        for path in runtime_dir.rglob("*"):
            if not path.is_file():
                continue
            if package_dir in path.parents:
                continue
            if path.suffix.lower() in {".zip", ".gz", ".7z"}:
                continue
            files.add(str(path.relative_to(root)).replace("\\", "/"))
    return sorted(files)


def _g0_secret_audit(root: Path, files: list[str]) -> dict[str, Any]:
    patterns = [
        ("google_api_key", r"AIza[0-9A-Za-z\-_]{35}"),
        ("openai_api_key", r"sk-[a-zA-Z0-9]{20,}"),
        ("private_key_header", r"-----BEGIN.*PRIVATE KEY-----"),
        ("google_application_credentials", r"GOOGLE_APPLICATION_CREDENTIALS"),
        ("client_secret", r"client_secret"),
        ("refresh_token", r"refresh_token"),
    ]
    compiled = [(name, re.compile(pattern)) for name, pattern in patterns]
    documentation_allowlist = {
        "PHASE_E_GOOGLE_CLOUD_EXTERNAL_COMPUTE_PLAN_v0.1-draft.md",
        str(OUTPUT_DIR_RELATIVE / "google_cloud_external_compute_plan_v0.1.json").replace("\\", "/"),
    }

    def _is_allowlisted_audit_pattern_definition(relative: str, line: str, pattern_name: str) -> bool:
        if relative != "src/phase_e_cobyla_runtime_optimization.py":
            return False
        return f'("{pattern_name}", r"' in line

    matches: list[dict[str, Any]] = []
    allowlisted_matches: list[dict[str, Any]] = []
    for relative in files:
        path = root / relative
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            matches.append({"path": relative, "line": None, "pattern": "read_error", "detail": repr(exc)})
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern_name, regex in compiled:
                if not regex.search(line):
                    continue
                record = {
                    "path": relative,
                    "line": int(line_number),
                    "pattern": pattern_name,
                    "text_preview": line.strip()[:160],
                }
                if relative in documentation_allowlist:
                    allowlisted_matches.append({**record, "allowlist_reason": "documents no-secret audit search patterns"})
                elif _is_allowlisted_audit_pattern_definition(relative, line, pattern_name):
                    allowlisted_matches.append({**record, "allowlist_reason": "defines G0 no-secret audit search pattern"})
                else:
                    matches.append(record)
    return {
        "status": "passed" if not matches else "failed",
        "patterns": [name for name, _pattern in patterns],
        "unresolved_match_count": len(matches),
        "allowlisted_documentation_match_count": len(allowlisted_matches),
        "unresolved_matches": matches,
        "allowlisted_documentation_matches": allowlisted_matches,
    }


def write_g0_local_archive_package(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    package_dir = ensure_dir(out / "G0_local_archive_package")
    report_json = package_dir / "G0_local_archive_package_report.json"
    report_md = package_dir / "G0_local_archive_package_report.md"
    manifest_json = package_dir / "G0_source_manifest.json"
    manifest_csv = package_dir / "G0_source_manifest.csv"
    audit_json = package_dir / "G0_no_secret_audit.json"
    audit_md = package_dir / "G0_no_secret_audit.md"
    archive_path = package_dir / "phase_e_g0_cloud_source.tar.gz"
    archive_sha256_path = package_dir / "phase_e_g0_cloud_source.sha256"
    if any(path.exists() for path in (report_json, report_md, manifest_json, manifest_csv, audit_json, audit_md, archive_path, archive_sha256_path)) and not force:
        raise FileExistsError("Refusing to overwrite existing G0 local archive package artifacts.")

    g_l1_report_path = out / "G_L1_local_runner_revalidation_report.json"
    dev_note_path = out / "DEV_GL1_001_pre_runner_tag_deviation.json"
    if not g_l1_report_path.exists() or not dev_note_path.exists():
        raise FileNotFoundError("G0 archive requires passed G-L1 revalidation report and DEV-GL1-001 deviation note.")
    g_l1_report = _read_json(g_l1_report_path)
    if g_l1_report.get("status") != "g_l1_local_revalidation_passed":
        raise RuntimeError("G0 archive requires passed G-L1 local revalidation.")

    files = _g0_archive_source_files(root, package_dir)
    records = [
        {
            "path": relative,
            "bytes": int((root / relative).stat().st_size),
            "sha256": sha256_file(root / relative),
        }
        for relative in files
    ]
    manifest = {
        "artifact": "G0_source_manifest",
        "created_at_utc": _utc_now(),
        "file_count": len(records),
        "source_root": str(root),
        "records": records,
        "manifest_hash": _canonical_hash(records),
    }
    write_json_atomic(manifest_json, manifest, overwrite=True)
    _write_csv(manifest_csv, records, fieldnames=["path", "bytes", "sha256"])

    audit = {
        "artifact": "G0_no_secret_audit",
        "created_at_utc": _utc_now(),
        "scope_file_count": len(records),
        **_g0_secret_audit(root, files),
    }
    write_json_atomic(audit_json, audit, overwrite=True)
    audit_md.write_text(
        "\n".join(
            [
                "# G0 No-Secret Audit",
                "",
                f"Created UTC: `{audit['created_at_utc']}`",
                f"Status: `{audit['status']}`",
                "",
                f"- Scope file count: `{audit['scope_file_count']}`",
                f"- Unresolved matches: `{audit['unresolved_match_count']}`",
                f"- Allowlisted documentation matches: `{audit['allowlisted_documentation_match_count']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    if audit["status"] != "passed":
        raise RuntimeError("G0 no-secret audit failed; archive was not created.")

    archive_members = [
        (root / relative, f"phase_e_g0_source/{relative}")
        for relative in files
    ] + [
        (manifest_json, "phase_e_g0_source/G0_source_manifest.json"),
        (manifest_csv, "phase_e_g0_source/G0_source_manifest.csv"),
        (audit_json, "phase_e_g0_source/G0_no_secret_audit.json"),
        (audit_md, "phase_e_g0_source/G0_no_secret_audit.md"),
    ]
    with tarfile.open(archive_path, "w:gz") as archive:
        for source, arcname in archive_members:
            archive.add(source, arcname=arcname)
    archive_sha256 = sha256_file(archive_path)
    archive_sha256_path.write_text(f"{archive_sha256}  {archive_path.name}\n", encoding="utf-8")

    report = {
        "artifact": "G0_local_archive_package",
        "created_at_utc": _utc_now(),
        "version": COBYLA_RUNTIME_OPTIMIZATION_VERSION,
        "status": "g0_local_archive_package_created",
        "official_execution_authorized": False,
        "stage_c_authorized": False,
        "stage_c_blocked": True,
        "scientific_outcomes_used": False,
        "records_scientific_values": False,
        "cloud_resources_created": False,
        "archive": {
            "path": str(archive_path.relative_to(root)).replace("\\", "/"),
            "sha256": archive_sha256,
            "bytes": int(archive_path.stat().st_size),
            "member_count": len(archive_members),
        },
        "hash_manifest": {
            "path": str(archive_sha256_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(archive_sha256_path),
        },
        "source_manifest": {
            "path": str(manifest_json.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(manifest_json),
            "manifest_hash": manifest["manifest_hash"],
            "file_count": len(records),
        },
        "no_secret_audit": {
            "path": str(audit_json.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(audit_json),
            "status": audit["status"],
            "unresolved_match_count": audit["unresolved_match_count"],
            "allowlisted_documentation_match_count": audit["allowlisted_documentation_match_count"],
        },
        "baseline_evidence": {
            "g_l1_revalidation_report": str(g_l1_report_path.relative_to(root)).replace("\\", "/"),
            "dev_gl1_001": str(dev_note_path.relative_to(root)).replace("\\", "/"),
            "pre_G_L1_o5_dynamic_aggregate_hash": g_l1_report["baseline"]["pre_G_L1_o5_dynamic_aggregate_hash"],
        },
        "next_action": "await_google_cloud_account_project_budget_inputs_before_G1",
    }
    write_json_atomic(report_json, report, overwrite=True)
    report_md.write_text(
        "\n".join(
            [
                "# G0 Local Archive Package",
                "",
                f"Created UTC: `{report['created_at_utc']}`",
                f"Status: `{report['status']}`",
                "",
                "No cloud resources were created. Official execution remains blocked.",
                "",
                f"- Source archive: `{report['archive']['path']}`",
                f"- Source archive sha256: `{archive_sha256}`",
                f"- Source manifest: `{report['source_manifest']['path']}`",
                f"- No-secret audit: `{report['no_secret_audit']['status']}`",
                f"- Files included: `{len(records)}`",
                "",
                f"Next action: `{report['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    status_path = out / "post_o1_status.json"
    status_md_path = out / "post_o1_status.md"
    post_status = _read_json(status_path)
    post_status.update(
        {
            "created_at_utc": report["created_at_utc"],
            "status": report["status"],
            "g0_local_archive_package_report": str(report_json.relative_to(root)).replace("\\", "/"),
            "g0_source_archive": str(archive_path.relative_to(root)).replace("\\", "/"),
            "g0_source_archive_sha256": archive_sha256,
            "official_execution_authorized": False,
            "stage_c_blocked": True,
            "scientific_outcomes_used": False,
            "next_action": report["next_action"],
        }
    )
    write_json_atomic(status_path, post_status, overwrite=True)
    status_md_path.write_text(
        "\n".join(
            [
                "# Post-O1 Status",
                "",
                f"Created UTC: `{report['created_at_utc']}`",
                f"Status: `{report['status']}`",
                "",
                "Official execution remains blocked.",
                "G0 local archive package has been created.",
                "",
                f"G0 report: `{str(report_json.relative_to(root)).replace(chr(92), '/')}`",
                "",
                f"Next action: `{report['next_action']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_deviation_log_entry(
        root,
        step="G0",
        entry_type="amendment",
        description="G0 local archive package created after G-L1 revalidation, without cloud resource creation.",
        action_taken=f"Recorded {report_json.relative_to(root)} and source archive {archive_path.relative_to(root)}.",
        blocking=False,
        resolved=True,
    )
    return report


def run_all_pre_o1(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    return {
        "O0": run_o0_bundle(root, force=force),
        "O0_5": run_o0_5_instrumentation_dry_run(root, force=force),
        "O0_75": write_o0_75_analytical_cost_model(root, force=force),
        "status": write_pre_o1_status(root, force=force),
    }
