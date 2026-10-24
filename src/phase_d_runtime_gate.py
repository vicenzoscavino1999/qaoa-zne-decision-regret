from __future__ import annotations

import csv
import ctypes
import json
import math
import shutil
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from .circuits import GraphInstance
from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .landscapes import parameter_grid
from .phase_c_120_custom_kernel import CUSTOM_KERNEL_VERSION, new_probability_cache
from .phase_c_120_operational_dry_run import _thermal_snapshot
from .phase_c_120_preflight import TARGET_GRID_SIZE


RUNTIME_GATE_VERSION = "phase_d_runtime_gate_v1"
DEFAULT_THETA_COUNT = 32
DEFAULT_NOISE = 0.005
DEFAULT_SCALE_FACTORS = (1, 3, 5, 7)
TARGET_LOCAL_HOURS = 72.0
AUTHORIZATION_REQUIRED_HOURS = 120.0
DISK_USAGE_LIMIT_FRACTION = 0.80
RAM_USAGE_LIMIT_FRACTION = 0.80
THERMAL_DROP_LIMIT_FRACTION = 0.20
PHASE_C_BLOCK_TASKS = 5403
PHASE_D_BLOCKS = 4
PHASE_D_TASKS = 21612
RAW_ARRAY_BYTES = TARGET_GRID_SIZE * TARGET_GRID_SIZE * 8
PHASE_D_EXPECTED_ARRAYS = PHASE_D_TASKS
STORAGE_OVERHEAD_FACTOR = 1.25


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _memory_status() -> dict[str, Any]:
    if hasattr(ctypes, "windll"):
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ok = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        if ok:
            return {
                "available": True,
                "total_physical_bytes": int(status.ullTotalPhys),
                "available_physical_bytes": int(status.ullAvailPhys),
                "memory_load_percent": int(status.dwMemoryLoad),
            }
    return {
        "available": False,
        "total_physical_bytes": None,
        "available_physical_bytes": None,
        "memory_load_percent": None,
    }


def _graph_from_final_payload(label: str, payload: dict[str, Any]) -> GraphInstance:
    edges = tuple((int(u), int(v), float(w)) for u, v, w in payload["edges"])
    return GraphInstance(name=label, edges=edges)


def _selected_theta_indices(grid_size: int, theta_count: int) -> list[int]:
    total = grid_size * grid_size
    if theta_count < 1 or theta_count > total:
        raise ValueError(f"theta_count must satisfy 1 <= theta_count <= {total}.")
    return sorted(set(int(index) for index in np.linspace(0, total - 1, theta_count, dtype=int)))


def _select_probe_instance(final_instances: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return max(
        final_instances.items(),
        key=lambda item: (
            int(item[1]["n_nodes"]),
            int(item[1]["n_edges"]),
            float(item[1]["density"]),
            item[0],
        ),
    )


def _run_timed_probe(root: Path, theta_count: int) -> dict[str, Any]:
    final_path = root / "results" / "phase_d" / "planning" / "phase_d_selected_instances_final.json"
    final_instances = _load_json(final_path)
    label, payload = _select_probe_instance(final_instances)
    graph = _graph_from_final_payload(label, payload)
    gammas, betas = parameter_grid(TARGET_GRID_SIZE)
    theta_indices = _selected_theta_indices(TARGET_GRID_SIZE, theta_count)
    cache = new_probability_cache()
    rows: list[dict[str, Any]] = []

    tracemalloc.start()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    for theta_index in theta_indices:
        i = theta_index // TARGET_GRID_SIZE
        j = theta_index % TARGET_GRID_SIZE
        gamma = float(gammas[i])
        beta = float(betas[j])
        for scale_factor in DEFAULT_SCALE_FACTORS:
            started = time.perf_counter()
            result = cache.get(graph, theta_index, gamma, beta, DEFAULT_NOISE, scale_factor)
            elapsed = time.perf_counter() - started
            rows.append({
                "selected_label": label,
                "candidate_id": payload["candidate_id"],
                "n_nodes": graph.n_qubits,
                "n_edges": len(graph.edges),
                "density": payload["density"],
                "theta_index": theta_index,
                "gamma_index": i,
                "beta_index": j,
                "p_noise": DEFAULT_NOISE,
                "scale_factor": scale_factor,
                "probability_seconds": elapsed,
                "probability_sum": float(np.sum(result.probabilities)),
                "probability_min": float(np.min(result.probabilities)),
                "probability_max": float(np.max(result.probabilities)),
                "transpiled_gate_count": result.transpiled_gate_count,
                "transpiled_depth": result.transpiled_depth,
            })
    wall_elapsed = time.perf_counter() - wall_started
    cpu_elapsed = time.process_time() - cpu_started
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cache_summary = cache.summary()
    probability_valid = all(
        math.isclose(float(row["probability_sum"]), 1.0, rel_tol=0.0, abs_tol=1e-9)
        and float(row["probability_min"]) >= -1e-12
        and math.isfinite(float(row["probability_seconds"]))
        for row in rows
    )
    return {
        "probe_instance_label": label,
        "probe_candidate_id": payload["candidate_id"],
        "probe_stratum_id": payload["stratum_id"],
        "probe_reason": "max n_nodes, then max n_edges, then max density among final H1-H12",
        "theta_count": len(theta_indices),
        "theta_indices": theta_indices,
        "scale_factors": list(DEFAULT_SCALE_FACTORS),
        "p_noise": DEFAULT_NOISE,
        "wall_elapsed_seconds": wall_elapsed,
        "cpu_elapsed_seconds": cpu_elapsed,
        "tracemalloc_current_bytes": current_bytes,
        "tracemalloc_peak_bytes": peak_bytes,
        "probability_cache": cache_summary,
        "seconds_per_probability_cache_miss": wall_elapsed / int(cache_summary["misses"]),
        "probabilities_valid": probability_valid,
        "measurements": rows,
        "measurement_summary": {
            "rows": len(rows),
            "mean_probability_seconds": float(np.mean([row["probability_seconds"] for row in rows])),
            "median_probability_seconds": float(np.median([row["probability_seconds"] for row in rows])),
            "max_probability_seconds": float(np.max([row["probability_seconds"] for row in rows])),
            "mean_transpiled_gate_count": float(np.mean([row["transpiled_gate_count"] for row in rows])),
            "mean_transpiled_depth": float(np.mean([row["transpiled_depth"] for row in rows])),
        },
    }


def _disk_report(path: Path, required_bytes: float) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    required_fraction_of_free = float(required_bytes / usage.free) if usage.free else math.inf
    projected_used_fraction = float((usage.used + required_bytes) / usage.total) if usage.total else math.inf
    passes_available_disk_gate = bool(required_fraction_of_free <= DISK_USAGE_LIMIT_FRACTION)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "required_bytes": int(required_bytes),
        "required_fraction_of_free": required_fraction_of_free,
        "projected_used_fraction": projected_used_fraction,
        "passes_80pct_available_disk_gate": passes_available_disk_gate,
        "passes_80pct_disk_gate": passes_available_disk_gate,
    }


def run_phase_d_runtime_gate(
    root: Path = PROJECT_ROOT,
    theta_count: int = DEFAULT_THETA_COUNT,
    force: bool = False,
) -> dict[str, Any]:
    planning_dir = ensure_dir(root / "results" / "phase_d" / "planning")
    summary_path = planning_dir / "phase_d_runtime_gate_summary.json"
    decision_path = planning_dir / "phase_d_runtime_gate_decision.json"
    measurements_path = planning_dir / "phase_d_runtime_probe_measurements.csv"
    interpretation_path = planning_dir / "phase_d_runtime_gate_interpretation.md"
    hashes_path = planning_dir / "phase_d_runtime_gate_output_hashes.json"
    if any(path.exists() for path in (summary_path, decision_path, measurements_path, interpretation_path, hashes_path)) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase D runtime gate outputs.")

    phase_c_manifest = _load_json(root / "results" / "phase_c_120_kernel_official" / "manifest_final.json")
    phase_c_estimate = _load_json(root / "results" / "phase_c_120_kernel_official" / "runtime_estimate.json")
    phase_c_soak = _load_json(root / "results" / "phase_c_120_operational_soak_test" / "operational_soak_test_summary.json")
    phase_c_actual_hours = float(phase_c_manifest["runtime"]["official_elapsed_hours"])
    phase_c_probe_seconds_per_miss = float(phase_c_estimate["seconds_per_probability_cache_miss"])

    probe = _run_timed_probe(root, theta_count=theta_count)
    phase_d_probe_seconds_per_miss = float(probe["seconds_per_probability_cache_miss"])
    complexity_ratio = phase_d_probe_seconds_per_miss / phase_c_probe_seconds_per_miss
    applied_complexity_factor = max(1.0, complexity_ratio)
    expected_hours_per_block = phase_c_actual_hours * applied_complexity_factor
    expected_total_hours = expected_hours_per_block * PHASE_D_BLOCKS
    planning_buffer_hours = 60.0
    runtime_pass = bool(expected_total_hours <= TARGET_LOCAL_HOURS)
    runtime_requires_authorization = bool(TARGET_LOCAL_HOURS < expected_total_hours <= AUTHORIZATION_REQUIRED_HOURS)
    runtime_stop = bool(expected_total_hours > AUTHORIZATION_REQUIRED_HOURS)

    phase_c_raw_bytes = _dir_size(root / "data" / "raw" / "phase_c_120_kernel_official")
    phase_c_results_bytes = _dir_size(root / "results" / "phase_c_120_kernel_official")
    phase_c_primary_footprint = phase_c_raw_bytes + phase_c_results_bytes
    raw_lower_bound_bytes = PHASE_D_EXPECTED_ARRAYS * RAW_ARRAY_BYTES
    estimated_primary_storage_bytes = max(
        raw_lower_bound_bytes,
        phase_c_primary_footprint * PHASE_D_BLOCKS,
    ) * STORAGE_OVERHEAD_FACTOR
    estimated_backup_storage_bytes = estimated_primary_storage_bytes
    primary_disk = _disk_report(root, estimated_primary_storage_bytes)
    backup_disk = _disk_report(Path("C:/"), estimated_backup_storage_bytes) if Path("C:/").exists() else {
        "passes_80pct_disk_gate": False,
        "note": "C:/ backup target unavailable",
    }

    memory = _memory_status()
    phase_c_peak_worker_bytes = int(phase_c_soak.get("memory", {}).get("peak_working_set_bytes") or 0)
    estimated_peak_workers_bytes = phase_c_peak_worker_bytes * 8 * applied_complexity_factor
    if memory["available"] and memory["total_physical_bytes"]:
        estimated_peak_memory_fraction = float(estimated_peak_workers_bytes / int(memory["total_physical_bytes"]))
        memory_pass = bool(estimated_peak_memory_fraction <= RAM_USAGE_LIMIT_FRACTION)
    else:
        estimated_peak_memory_fraction = None
        memory_pass = False

    thermal_before = _thermal_snapshot(root)
    phase_c_thermal = phase_c_soak.get("thermal", {})
    phase_c_throughput = phase_c_soak.get("throughput", {})
    throughput_drop = phase_c_throughput.get("drop_fraction")
    thermal_proxy_pass = bool(
        phase_c_soak.get("checks", {}).get("throughput_drop_fraction_le_20pct") is True
        and throughput_drop is not None
        and float(throughput_drop) <= THERMAL_DROP_LIMIT_FRACTION
    )
    thermal_pass = thermal_proxy_pass

    decision = {
        "created_at_utc": _utc_now(),
        "runtime_gate_version": RUNTIME_GATE_VERSION,
        "runtime_gate_pass": bool(runtime_pass and primary_disk["passes_80pct_disk_gate"] and backup_disk["passes_80pct_disk_gate"] and memory_pass and thermal_pass and probe["probabilities_valid"]),
        "runtime_pass": runtime_pass,
        "runtime_requires_explicit_authorization": runtime_requires_authorization,
        "runtime_stop_required": runtime_stop,
        "primary_disk_pass": bool(primary_disk["passes_80pct_disk_gate"]),
        "backup_disk_pass": bool(backup_disk["passes_80pct_disk_gate"]),
        "memory_pass": memory_pass,
        "thermal_pass": thermal_pass,
        "probe_probabilities_valid": bool(probe["probabilities_valid"]),
        "recommendation": "proceed_to_phase_d_runner_dry_run",
    }
    if not decision["runtime_gate_pass"]:
        decision["recommendation"] = "do_not_freeze_phase_d_v1_0_until_runtime_gate_issue_resolved"
    elif runtime_requires_authorization:
        decision["recommendation"] = "explicit_user_authorization_required_before_local_execution"

    summary = {
        "created_at_utc": _utc_now(),
        "runtime_gate_version": RUNTIME_GATE_VERSION,
        "status": "complete",
        "custom_kernel_version": CUSTOM_KERNEL_VERSION,
        "scope": "phase_d_runtime_disk_memory_thermal_gate_no_official_noisy_zne_outputs",
        "phase_c_actual_reference": {
            "official_elapsed_hours": phase_c_actual_hours,
            "tasks_total": PHASE_C_BLOCK_TASKS,
            "process_count": phase_c_manifest["summary"].get("official_process_count"),
            "runtime_estimate_sha256": sha256_file(root / "results" / "phase_c_120_kernel_official" / "runtime_estimate.json"),
            "manifest_final_sha256": sha256_file(root / "results" / "phase_c_120_kernel_official" / "manifest_final.json"),
        },
        "phase_d_probe": {key: value for key, value in probe.items() if key != "measurements"},
        "runtime_estimate": {
            "phase_c_probe_seconds_per_probability_cache_miss": phase_c_probe_seconds_per_miss,
            "phase_d_probe_seconds_per_probability_cache_miss": phase_d_probe_seconds_per_miss,
            "measured_complexity_ratio_vs_phase_c_probe": complexity_ratio,
            "applied_complexity_factor": applied_complexity_factor,
            "expected_hours_per_block": expected_hours_per_block,
            "expected_total_hours_H1_H12": expected_total_hours,
            "planning_buffer_hours": planning_buffer_hours,
            "target_local_hours": TARGET_LOCAL_HOURS,
            "authorization_required_upper_hours": AUTHORIZATION_REQUIRED_HOURS,
        },
        "storage_estimate": {
            "phase_c_raw_bytes": phase_c_raw_bytes,
            "phase_c_results_bytes": phase_c_results_bytes,
            "phase_c_primary_footprint_bytes": phase_c_primary_footprint,
            "phase_d_raw_lower_bound_bytes": raw_lower_bound_bytes,
            "storage_overhead_factor": STORAGE_OVERHEAD_FACTOR,
            "estimated_primary_storage_bytes": int(estimated_primary_storage_bytes),
            "estimated_backup_storage_bytes": int(estimated_backup_storage_bytes),
            "primary_disk": primary_disk,
            "backup_disk": backup_disk,
        },
        "memory_estimate": {
            "phase_c_soak_peak_worker_bytes": phase_c_peak_worker_bytes,
            "estimated_peak_8_worker_bytes": int(estimated_peak_workers_bytes),
            "estimated_peak_memory_fraction": estimated_peak_memory_fraction,
            "memory_status": memory,
        },
        "thermal_estimate": {
            "thermal_snapshot_before": thermal_before,
            "phase_c_soak_thermal": phase_c_thermal,
            "phase_c_soak_throughput_drop_fraction": throughput_drop,
            "thermal_basis": "temperature_sensor_if_available_else_phase_c_30min_soak_throughput_proxy",
        },
        "decision": decision,
        "limitations": [
            "The Phase D timed probe measures probability-cache miss throughput only and does not write official H1-H12 landscapes.",
            "Runtime projection uses the completed Phase C official elapsed time scaled by the hardest Phase D probe ratio.",
            "Temperature sensors may be unavailable on Windows; if unavailable, the prior Phase C 30-minute throughput stability test is used as the thermal proxy.",
        ],
    }
    _write_csv(measurements_path, probe["measurements"])
    write_json_atomic(decision_path, decision, overwrite=True)
    write_json_atomic(summary_path, summary, overwrite=True)
    interpretation = (
        "# Phase D Runtime Gate Interpretation\n\n"
        f"Created UTC: {summary['created_at_utc']}\n\n"
        f"Decision: `{decision['recommendation']}`\n\n"
        f"Expected total H1-H12 runtime: `{expected_total_hours:.2f}` hours.\n\n"
        f"Runtime gate pass: `{decision['runtime_gate_pass']}`.\n\n"
        "No official Phase D noisy/ZNE arrays were written by this gate.\n"
    )
    interpretation_path.write_text(interpretation, encoding="utf-8")
    output_hashes = {
        "phase_d_runtime_gate_summary_json": sha256_file(summary_path),
        "phase_d_runtime_gate_decision_json": sha256_file(decision_path),
        "phase_d_runtime_probe_measurements_csv": sha256_file(measurements_path),
        "phase_d_runtime_gate_interpretation_md": sha256_file(interpretation_path),
    }
    write_json_atomic(hashes_path, {
        "created_at_utc": _utc_now(),
        "runtime_gate_version": RUNTIME_GATE_VERSION,
        "hashes": output_hashes,
    }, overwrite=True)
    summary["output_hashes"] = output_hashes
    summary["output_hashes_file"] = hashes_path.relative_to(root).as_posix()
    return summary
