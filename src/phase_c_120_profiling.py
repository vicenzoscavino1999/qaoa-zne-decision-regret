from __future__ import annotations

import gc
import math
import os
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from .circuits import graph_from_config
from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, write_json_atomic
from .landscapes import generate_ideal_landscape, generate_noisy_landscape, generate_zne_landscape
from .phase_c import _selected_seeds, _simulator_options_for_profile
from .phase_c_120_preflight import PROTOCOL_VERSION, TARGET_GRID_SIZE, _write_csv, generate_task_inventory


PROFILE_VERSION = "phase_c_120_profile_v1"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_c_120_profiling"
DEFAULT_GRID_SIZES = (20, 40)
DEFAULT_SIMULATOR_PROFILE = "default_threads_1"
DEFAULT_INSTANCE_ORDER = ("G1_C4", "G2_weighted4", "G3_weighted6")
ENGINE_DECISION_TARGET_HOURS = {
    "full_run_gate": 72.0,
    "single_machine_preferred": 7.0 * 24.0,
    "single_machine_upper_soft": 14.0 * 24.0,
    "local_exit_review": 45.0 * 24.0,
}


def _rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]
    except Exception:
        return None
    return int(psutil.Process(os.getpid()).memory_info().rss)


def _finite_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


def _summarize_result(result: Any) -> dict[str, Any]:
    array = result[0] if isinstance(result, tuple) else result
    if not isinstance(array, np.ndarray):
        return {"array_shape": "", "array_finite": False, "array_min": None, "array_max": None}
    finite = bool(np.isfinite(array).all())
    return {
        "array_shape": "x".join(str(x) for x in array.shape),
        "array_finite": finite,
        "array_min": _finite_float(float(np.min(array))) if finite else None,
        "array_max": _finite_float(float(np.max(array))) if finite else None,
    }


def _measure(label: str, work: Callable[[], Any]) -> tuple[dict[str, Any], Any]:
    gc.collect()
    rss_before = _rss_bytes()
    tracemalloc.start()
    started = time.perf_counter()
    result = work()
    elapsed = time.perf_counter() - started
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = _rss_bytes()
    row = {
        "component": label,
        "seconds": elapsed,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "rss_delta_bytes": None if rss_before is None or rss_after is None else rss_after - rss_before,
        "tracemalloc_current_bytes": int(current),
        "tracemalloc_peak_bytes": int(peak),
        **_summarize_result(result),
    }
    return row, result


def _progress(prefix: str):
    def callback(done: int, total: int, tag: str) -> None:
        if done == 1 or done == total or done % max(1, total // 4) == 0:
            print(f"{prefix}:{tag}:{done}/{total}", flush=True)

    return callback


def _component_seconds(row: dict[str, Any]) -> float:
    return float(row.get("seconds", 0.0) or 0.0)


def _fit_linear_total_seconds(points: list[int], seconds: list[float], target_points: int) -> dict[str, Any]:
    if len(points) == 1:
        slope = seconds[0] / points[0]
        intercept = 0.0
        predicted = slope * target_points
        return {
            "model": "through_origin_single_point",
            "slope_seconds_per_point": slope,
            "intercept_seconds": intercept,
            "target_seconds": predicted,
            "r2": None,
            "max_relative_residual": None,
        }
    x = np.asarray(points, dtype=float)
    y = np.asarray(seconds, dtype=float)
    slope, intercept = np.polyfit(x, y, deg=1)
    predicted_y = slope * x + intercept
    residuals = y - predicted_y
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = None if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    rel_residuals = [
        abs(float(residual)) / max(abs(float(actual)), 1e-12)
        for residual, actual in zip(residuals, y, strict=True)
    ]
    return {
        "model": "linear_total_seconds_vs_grid_points",
        "slope_seconds_per_point": float(slope),
        "intercept_seconds": float(intercept),
        "target_seconds": float(slope * target_points + intercept),
        "r2": r2,
        "max_relative_residual": max(rel_residuals) if rel_residuals else None,
    }


def _runtime_from_instance_seconds(
    instance_target_seconds: dict[str, float],
    cfg: dict[str, Any],
    task_inventory: dict[str, Any],
) -> dict[str, Any]:
    seed_count = int(task_inventory["seed_count"])
    noise_values = [float(x) for x in cfg["noise"]["p_values"]]
    shot_budgets = [int(x) for x in cfg["shots"]["total_budget_primary"]]
    benchmark_shots = int(cfg["phase_c"]["benchmark_shots_total"])
    shot_scale_sum = sum(float(b) / float(benchmark_shots) for b in shot_budgets)
    multiplier = len(noise_values) * seed_count * shot_scale_sum
    serial_seconds = sum(instance_target_seconds.values()) * multiplier
    local_workers = max(1, int(os.cpu_count() or 1))
    local_ideal_hours = serial_seconds / 3600.0 / local_workers
    local_75pct_hours = serial_seconds / 3600.0 / (local_workers * 0.75)
    return {
        "instance_target_seconds_one_noise_seed_B1024": instance_target_seconds,
        "noise_levels": len(noise_values),
        "shot_budgets": shot_budgets,
        "shot_scaling_sum_vs_B1024": shot_scale_sum,
        "seed_count": seed_count,
        "serial_hours": serial_seconds / 3600.0,
        "local_workers": local_workers,
        "local_ideal_parallel_hours": local_ideal_hours,
        "local_75pct_parallel_hours": local_75pct_hours,
        "required_speedup_vs_72h_gate": local_75pct_hours / ENGINE_DECISION_TARGET_HOURS["full_run_gate"],
        "required_speedup_vs_7d_single_machine": local_75pct_hours / ENGINE_DECISION_TARGET_HOURS["single_machine_preferred"],
        "required_speedup_vs_14d_single_machine": local_75pct_hours / ENGINE_DECISION_TARGET_HOURS["single_machine_upper_soft"],
        "exceeds_45d_exit_review": local_75pct_hours > ENGINE_DECISION_TARGET_HOURS["local_exit_review"],
    }


def _decision_from_runtime(runtime: dict[str, Any]) -> str:
    hours = float(runtime["local_75pct_parallel_hours"])
    if hours <= ENGINE_DECISION_TARGET_HOURS["full_run_gate"]:
        return "aer_profile_passes_72h_gate"
    if hours <= ENGINE_DECISION_TARGET_HOURS["single_machine_preferred"]:
        return "aer_feasible_single_machine_but_above_72h_gate"
    if hours <= ENGINE_DECISION_TARGET_HOURS["single_machine_upper_soft"]:
        return "aer_possible_but_checkpointed_long_run_required"
    if hours <= ENGINE_DECISION_TARGET_HOURS["local_exit_review"]:
        return "custom_kernel_or_sharding_recommended"
    return "custom_kernel_sharding_or_exit_review_required"


def _profile_instance_grid(
    cfg: dict[str, Any],
    instance: str,
    grid_size: int,
    p_noise: float,
    shots_total: int,
    seed: int,
    simulator_profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    graph = graph_from_config(instance, cfg)
    simulator_options = _simulator_options_for_profile(simulator_profile)
    folding_backend = str(cfg["zne"]["folding_backend"])
    allocation = "proportional_to_abs_weights"
    transpile_cache: dict[tuple[object, ...], object] = {}
    rows: list[dict[str, Any]] = []
    grid_points = grid_size * grid_size

    row, result = _measure(
        "ideal",
        lambda: generate_ideal_landscape(graph, grid_size, progress=_progress(f"{instance}:grid{grid_size}")),
    )
    rows.append(row)
    del result

    row, result = _measure(
        "noisy",
        lambda: generate_noisy_landscape(
            graph,
            grid_size,
            p_noise,
            shots_total,
            seed,
            "phase_c_120_profile_noisy",
            progress=_progress(f"{instance}:grid{grid_size}"),
            transpile_cache=transpile_cache,
            simulator_options=simulator_options,
        ),
    )
    rows.append(row)
    del result

    for order, scales in cfg["zne"]["methods"].items():
        scale_factors = [int(x) for x in scales]
        row, result = _measure(
            f"zne_{order}",
            lambda order=str(order), scale_factors=scale_factors: generate_zne_landscape(
                graph,
                grid_size,
                p_noise,
                shots_total,
                seed,
                scale_factors,
                order,
                folding_backend=folding_backend,
                shot_allocation_strategy=allocation,
                progress=_progress(f"{instance}:grid{grid_size}"),
                transpile_cache=transpile_cache,
                simulator_options=simulator_options,
            ),
        )
        row["scale_factors"] = ",".join(str(x) for x in scale_factors)
        rows.append(row)
        del result

    for row in rows:
        row["created_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        row["protocol_version"] = PROTOCOL_VERSION
        row["profile_version"] = PROFILE_VERSION
        row["instance"] = instance
        row["grid_size"] = grid_size
        row["grid_points"] = grid_points
        row["p_noise"] = p_noise
        row["shots_total"] = shots_total
        row["seed"] = seed
        row["simulator_profile"] = simulator_profile
        row["seconds_per_point"] = float(row["seconds"]) / float(grid_points)
        row["points_per_second"] = float(grid_points) / max(float(row["seconds"]), 1e-12)
        row["transpile_cache_entries_after_component"] = len(transpile_cache)

    t10_seconds = sum(_component_seconds(row) for row in rows if str(row["component"]).startswith(("noisy", "zne_")))
    summary = {
        "instance": instance,
        "grid_size": grid_size,
        "grid_points": grid_points,
        "p_noise": p_noise,
        "shots_total": shots_total,
        "seed": seed,
        "simulator_profile": simulator_profile,
        "ideal_seconds": sum(_component_seconds(row) for row in rows if row["component"] == "ideal"),
        "t10_seconds": t10_seconds,
        "t10_seconds_per_point": t10_seconds / float(grid_points),
        "total_profile_seconds": sum(_component_seconds(row) for row in rows),
        "transpile_cache_entries": len(transpile_cache),
        "max_tracemalloc_peak_bytes": max(int(row["tracemalloc_peak_bytes"]) for row in rows),
        "max_rss_after_bytes": max((int(row["rss_after_bytes"]) for row in rows if row["rss_after_bytes"] is not None), default=None),
    }
    return rows, summary


def build_runtime_model(
    grid_summaries: list[dict[str, Any]],
    cfg: dict[str, Any],
    task_inventory: dict[str, Any],
) -> dict[str, Any]:
    target_points = TARGET_GRID_SIZE * TARGET_GRID_SIZE
    by_instance: dict[str, list[dict[str, Any]]] = {}
    for row in grid_summaries:
        by_instance.setdefault(str(row["instance"]), []).append(row)

    instance_models: dict[str, Any] = {}
    linear_target_seconds: dict[str, float] = {}
    last_point_target_seconds: dict[str, float] = {}
    max_point_target_seconds: dict[str, float] = {}
    for instance, rows in sorted(by_instance.items()):
        rows = sorted(rows, key=lambda row: int(row["grid_size"]))
        points = [int(row["grid_points"]) for row in rows]
        seconds = [float(row["t10_seconds"]) for row in rows]
        fit = _fit_linear_total_seconds(points, seconds, target_points)
        per_point_values = [float(row["t10_seconds_per_point"]) for row in rows]
        last = per_point_values[-1] * target_points
        max_point = max(per_point_values) * target_points
        linear_target_seconds[instance] = max(0.0, float(fit["target_seconds"]))
        last_point_target_seconds[instance] = last
        max_point_target_seconds[instance] = max_point
        instance_models[instance] = {
            "measured_grid_sizes": [int(row["grid_size"]) for row in rows],
            "measured_t10_seconds": seconds,
            "measured_t10_seconds_per_point": per_point_values,
            "linear_fit": fit,
            "target_seconds_linear_fit": linear_target_seconds[instance],
            "target_seconds_last_per_point": last,
            "target_seconds_max_per_point": max_point,
        }

    linear_runtime = _runtime_from_instance_seconds(linear_target_seconds, cfg, task_inventory)
    last_point_runtime = _runtime_from_instance_seconds(last_point_target_seconds, cfg, task_inventory)
    max_point_runtime = _runtime_from_instance_seconds(max_point_target_seconds, cfg, task_inventory)
    recommended = max_point_runtime
    return {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_version": PROTOCOL_VERSION,
        "profile_version": PROFILE_VERSION,
        "target_grid_size": TARGET_GRID_SIZE,
        "target_grid_points": target_points,
        "task_inventory_total": task_inventory["official_output_tasks"]["total"],
        "engine_decision_target_hours": ENGINE_DECISION_TARGET_HOURS,
        "instance_models": instance_models,
        "runtime_estimates": {
            "linear_fit": linear_runtime,
            "last_measured_per_point": last_point_runtime,
            "max_measured_per_point_recommended_conservative": max_point_runtime,
        },
        "recommended_runtime_basis": "max_measured_per_point_recommended_conservative",
        "phase1_decision": _decision_from_runtime(recommended),
    }


def run_phase_c_120_profiling(
    grid_sizes: tuple[int, ...] = DEFAULT_GRID_SIZES,
    instances: tuple[str, ...] = DEFAULT_INSTANCE_ORDER,
    simulator_profile: str = DEFAULT_SIMULATOR_PROFILE,
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_c_120_profiling")
    summary_path = output_dir / "profiling_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing profiling output: {summary_path}")

    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    seeds_cfg = read_yaml(root / "config" / "seeds.yaml")
    task_inventory = generate_task_inventory(cfg, seeds_cfg)
    seed = int(_selected_seeds(root, int(cfg["phase_c"]["seed_count"]))[0])
    p_noise = float(cfg["phase_c"]["benchmark_noise"])
    shots_total = int(cfg["phase_c"]["benchmark_shots_total"])

    component_rows: list[dict[str, Any]] = []
    grid_summaries: list[dict[str, Any]] = []
    started = time.perf_counter()
    for grid_size in grid_sizes:
        for instance in instances:
            rows, summary = _profile_instance_grid(
                cfg,
                str(instance),
                int(grid_size),
                p_noise,
                shots_total,
                seed,
                simulator_profile,
            )
            component_rows.extend(rows)
            grid_summaries.append(summary)
            _write_csv(output_dir / "component_timings_partial.csv", component_rows)
            _write_csv(output_dir / "grid_t10_timings_partial.csv", grid_summaries)

    model = build_runtime_model(grid_summaries, cfg, task_inventory)
    elapsed = time.perf_counter() - started
    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_version": PROTOCOL_VERSION,
        "profile_version": PROFILE_VERSION,
        "status": "phase_c_120_profiling_complete",
        "target_grid_size": TARGET_GRID_SIZE,
        "measured_grid_sizes": [int(x) for x in grid_sizes],
        "instances": list(instances),
        "simulator_profile": simulator_profile,
        "benchmark_noise": p_noise,
        "benchmark_shots_total": shots_total,
        "benchmark_seed": seed,
        "elapsed_seconds": elapsed,
        "n_component_rows": len(component_rows),
        "n_grid_summary_rows": len(grid_summaries),
        "phase1_decision": model["phase1_decision"],
        "recommended_runtime_basis": model["recommended_runtime_basis"],
    }

    _write_csv(output_dir / "component_timings.csv", component_rows)
    _write_csv(output_dir / "grid_t10_timings.csv", grid_summaries)
    write_json_atomic(output_dir / "runtime_model_120.json", model)
    write_json_atomic(summary_path, report)
    return report
