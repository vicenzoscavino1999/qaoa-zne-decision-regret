from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .circuits import graph_from_config
from .config import SUPPORTED_FOLDING_BACKEND
from .io_utils import PROJECT_ROOT, configure_logger, ensure_dir, read_yaml, write_json
from .landscapes import generate_noisy_landscape, generate_zne_landscape
from .manifest import build_manifest


OPTIMIZATION_CANDIDATES: tuple[dict[str, object], ...] = (
    {"label": "current_default", "simulator_options": {}},
    {"label": "default_threads_1", "simulator_options": {"max_parallel_threads": 1}},
    {"label": "density_matrix_threads_1", "simulator_options": {"method": "density_matrix", "max_parallel_threads": 1}},
    {"label": "gpu_automatic", "simulator_options": {"device": "GPU"}},
    {"label": "gpu_statevector", "simulator_options": {"method": "statevector", "device": "GPU"}},
    {"label": "gpu_density_matrix", "simulator_options": {"method": "density_matrix", "device": "GPU"}},
)
NUMERICAL_EQUIVALENCE_TOL = 1e-12


def _simulator_options_for_profile(profile: str) -> dict[str, object]:
    for candidate in OPTIMIZATION_CANDIDATES:
        if candidate["label"] == profile:
            return dict(candidate["simulator_options"])
    known = ", ".join(str(candidate["label"]) for candidate in OPTIMIZATION_CANDIDATES)
    raise ValueError(f"Unknown simulator profile: {profile}. Known profiles: {known}")


def _write_csv(path: Path, rows: list[dict[str, object]], overwrite: bool = False) -> None:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing CSV: {path}")
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _progress(logger, tag: str):
    def callback(done: int, total: int, inner_tag: str) -> None:
        logger.info("%s/%s progress %s/%s", tag, inner_tag, done, total)

    return callback


def _phase_c_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg["zne"]["folding_backend"] != SUPPORTED_FOLDING_BACKEND:
        raise ValueError(f"Unsupported folding backend: {cfg['zne']['folding_backend']}")
    phase_c = cfg.get("phase_c")
    if not isinstance(phase_c, dict):
        raise KeyError("Missing phase_c config.")
    if phase_c.get("decision_status") != "frozen_for_phase_c_preparation":
        raise RuntimeError("Phase C decisions are not frozen for preparation.")
    return phase_c


def _selected_seeds(root: Path, seed_count: int) -> list[int]:
    seeds_cfg = read_yaml(root / "config" / "seeds.yaml")
    candidates = [int(x) for x in seeds_cfg["official_candidates"]]
    if seed_count > len(candidates):
        raise ValueError(f"Requested {seed_count} seeds but only {len(candidates)} official candidates exist.")
    return candidates[:seed_count]


def _assert_phase_b_ready(root: Path, expected_seed_count: int) -> dict[str, Any]:
    summary_path = root / "results" / "phase_b_debug" / "phase_b_summary.json"
    analysis_path = root / "results" / "phase_b_debug" / "phase_b_analysis.json"
    if not summary_path.exists() or not analysis_path.exists():
        raise FileNotFoundError("Phase B summary/analysis must exist before Phase C.")
    summary = _read_json(summary_path)
    analysis = _read_json(analysis_path)
    if summary.get("phase_b_pass_preliminary") is not True:
        raise RuntimeError("Phase B did not pass preliminary checks.")
    if int(summary.get("recommended_seed_count")) != expected_seed_count:
        raise RuntimeError(
            f"Phase B recommended {summary.get('recommended_seed_count')} seeds, "
            f"but Phase C requested {expected_seed_count}."
        )
    return {"summary": summary, "analysis": analysis}


def _estimate_rows(
    instance_timings: list[dict[str, object]],
    cfg: dict[str, Any],
    phase_c: dict[str, Any],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, Any]]:
    grid_points = int(phase_c["grid_size"]) ** 2
    noise_values = [float(x) for x in cfg["noise"]["p_values"]]
    shot_budgets = [int(x) for x in cfg["shots"]["total_budget_primary"]]
    candidate_counts = [int(x) for x in cfg["seed_selection"]["candidate_seed_counts"]]
    selected_seed_count = int(phase_c["seed_count"])
    benchmark_shots = int(phase_c["benchmark_shots_total"])

    rows: list[dict[str, object]] = []
    totals_by_seed_count = {count: 0.0 for count in candidate_counts}
    conservative_totals_by_seed_count = {count: 0.0 for count in candidate_counts}
    shot_scale_sum = sum(b / benchmark_shots for b in shot_budgets)

    for timing in instance_timings:
        t10 = float(timing["t10_seconds_per_point"])
        for seed_count in candidate_counts:
            flat_hours = t10 * grid_points * len(noise_values) * len(shot_budgets) * seed_count / 3600.0
            conservative_hours = t10 * grid_points * len(noise_values) * shot_scale_sum * seed_count / 3600.0
            totals_by_seed_count[seed_count] += flat_hours
            conservative_totals_by_seed_count[seed_count] += conservative_hours
            rows.append({
                "instance": timing["instance"],
                "t10_seconds_per_point": t10,
                "grid_points": grid_points,
                "noise_levels": len(noise_values),
                "shot_budgets": len(shot_budgets),
                "seed_count": seed_count,
                "estimated_hours_protocol_flat_B": flat_hours,
                "estimated_hours_shot_scaled": conservative_hours,
            })

    thresholds = cfg["feasibility_gate"]["runtime_thresholds_hours"]
    selected_flat = totals_by_seed_count[selected_seed_count]
    selected_conservative = conservative_totals_by_seed_count[selected_seed_count]
    local_workers = max(1, int(os.cpu_count() or 1))
    full_threshold = float(thresholds["full_run"])
    reduce_seed_threshold = float(thresholds["reduce_seeds"])
    reduce_grid_threshold = float(thresholds["reduce_grid"])
    required_speedup = {
        "full_run_protocol_flat_B": selected_flat / full_threshold,
        "full_run_shot_scaled": selected_conservative / full_threshold,
        "reduce_seeds_shot_scaled": selected_conservative / reduce_seed_threshold,
        "reduce_grid_shot_scaled": selected_conservative / reduce_grid_threshold,
    }
    parallel_worker_counts = sorted({1, 2, 4, local_workers, 16, 32, 64, 128})
    parallel_rows: list[dict[str, object]] = []
    for workers in parallel_worker_counts:
        ideal_flat = selected_flat / workers
        ideal_shot_scaled = selected_conservative / workers
        efficiency_75_shot_scaled = selected_conservative / (workers * 0.75)
        parallel_rows.append({
            "workers": workers,
            "is_local_logical_worker_count": workers == local_workers,
            "ideal_hours_protocol_flat_B": ideal_flat,
            "ideal_hours_shot_scaled": ideal_shot_scaled,
            "efficiency_75pct_hours_shot_scaled": efficiency_75_shot_scaled,
            "ideal_full_run_allowed": ideal_shot_scaled <= full_threshold,
            "ideal_reduce_grid_allowed": ideal_shot_scaled <= reduce_grid_threshold,
        })
    allowed = selected_conservative <= float(thresholds["full_run"])
    if selected_conservative <= float(thresholds["full_run"]):
        decision = "full_run_allowed"
    elif selected_conservative <= float(thresholds["reduce_seeds"]):
        decision = "reduce_seeds_required_but_not_supported_by_phase_b"
    elif selected_conservative <= float(thresholds["reduce_grid"]):
        decision = "reduce_grid_required_but_not_core_phase_c"
    else:
        decision = "do_not_run_phase_c_optimize_more"

    summary = {
        "grid_points": grid_points,
        "noise_values": noise_values,
        "shot_budgets": shot_budgets,
        "benchmark_shots_total": benchmark_shots,
        "shot_scaling_assumption": "linear_runtime_scaling_from_benchmark_shots",
        "totals_by_seed_count_protocol_flat_B_hours": totals_by_seed_count,
        "totals_by_seed_count_shot_scaled_hours": conservative_totals_by_seed_count,
        "selected_seed_count": selected_seed_count,
        "selected_total_hours_protocol_flat_B": selected_flat,
        "selected_total_hours_shot_scaled": selected_conservative,
        "runtime_thresholds_hours": thresholds,
        "local_logical_workers": local_workers,
        "required_speedup": required_speedup,
        "minimum_ideal_workers": {
            "full_run_protocol_flat_B": math.ceil(required_speedup["full_run_protocol_flat_B"]),
            "full_run_shot_scaled": math.ceil(required_speedup["full_run_shot_scaled"]),
            "reduce_seeds_shot_scaled": math.ceil(required_speedup["reduce_seeds_shot_scaled"]),
            "reduce_grid_shot_scaled": math.ceil(required_speedup["reduce_grid_shot_scaled"]),
        },
        "local_ideal_parallel_hours_shot_scaled": selected_conservative / local_workers,
        "local_75pct_parallel_hours_shot_scaled": selected_conservative / (local_workers * 0.75),
        "parallelism_note": "Ideal worker estimates assume perfectly parallel independent Phase C cells with no scheduling or memory overhead.",
        "phase_c_allowed_by_gate": allowed,
        "decision": decision,
    }
    return rows, parallel_rows, summary


def run_phase_c_feasibility(
    config_path: str | Path | None = None,
    overwrite: bool = False,
    seed_count: int | None = None,
    simulator_profile: str = "current_default",
) -> dict[str, Any]:
    root = PROJECT_ROOT
    cfg = read_yaml(config_path or root / "config" / "config_official_pilot_v1.6.yaml")
    phase_c = _phase_c_config(cfg)
    selected_seed_count = int(seed_count or phase_c["seed_count"])
    if selected_seed_count != int(phase_c["seed_count"]):
        raise RuntimeError(f"Phase C seed_count is frozen at {phase_c['seed_count']}, got {selected_seed_count}.")
    phase_b = _assert_phase_b_ready(root, selected_seed_count)
    selected_seeds = _selected_seeds(root, selected_seed_count)

    result_dir = root / "results" / "phase_c"
    profile_suffix = "" if simulator_profile == "current_default" else f"_{simulator_profile}"
    report_path = result_dir / f"feasibility_gate{profile_suffix}.json"
    table_path = result_dir / f"feasibility_gate_table{profile_suffix}.csv"
    parallel_path = result_dir / f"feasibility_parallel_table{profile_suffix}.csv"
    timings_path = result_dir / f"feasibility_instance_timings{profile_suffix}.csv"
    for path in [report_path, table_path, parallel_path, timings_path]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing Phase C feasibility output: {path}")

    logger = configure_logger(root / "logs" / "phase_c_feasibility.log")
    logger.info("Starting Phase C feasibility gate")
    logger.info("seed_count=%s simulator_profile=%s instances=%s", selected_seed_count, simulator_profile, ",".join(phase_c["instances"]))

    benchmark_grid_size = int(phase_c["benchmark_grid_size"])
    benchmark_noise = float(phase_c["benchmark_noise"])
    benchmark_shots = int(phase_c["benchmark_shots_total"])
    seed = int(selected_seeds[0])
    folding_backend = str(cfg["zne"]["folding_backend"])
    allocation = "proportional_to_abs_weights"
    simulator_options = _simulator_options_for_profile(simulator_profile)

    instance_timings: list[dict[str, object]] = []
    for instance in phase_c["instances"]:
        graph = graph_from_config(str(instance), cfg)
        timings: dict[str, float] = {}
        transpile_cache: dict[tuple[object, ...], object] = {}
        n_points = benchmark_grid_size * benchmark_grid_size

        started = time.perf_counter()
        generate_noisy_landscape(
            graph,
            benchmark_grid_size,
            benchmark_noise,
            benchmark_shots,
            seed,
            "phase_c_gate_noisy",
            progress=_progress(logger, f"phase_c_gate_{graph.name}_noisy"),
            transpile_cache=transpile_cache,
            simulator_options=simulator_options,
        )
        timings["noisy_seconds"] = time.perf_counter() - started

        for order, scales in cfg["zne"]["methods"].items():
            started = time.perf_counter()
            generate_zne_landscape(
                graph,
                benchmark_grid_size,
                benchmark_noise,
                benchmark_shots,
                seed,
                [int(x) for x in scales],
                str(order),
                folding_backend=folding_backend,
                shot_allocation_strategy=allocation,
                progress=_progress(logger, f"phase_c_gate_{graph.name}_zne_{order}"),
                transpile_cache=transpile_cache,
                simulator_options=simulator_options,
            )
            timings[f"zne_{order}_seconds"] = time.perf_counter() - started

        t10_seconds_per_point = (
            timings["noisy_seconds"]
            + timings["zne_linear_seconds"]
            + timings["zne_quadratic_seconds"]
            + timings["zne_cubic_seconds"]
        ) / n_points
        instance_timings.append({
            "instance": graph.name,
            "benchmark_grid_size": benchmark_grid_size,
            "benchmark_points": n_points,
            "benchmark_noise": benchmark_noise,
            "benchmark_shots_total": benchmark_shots,
            "benchmark_seed": seed,
            "transpile_cache_entries": len(transpile_cache),
            "t10_seconds_per_point": t10_seconds_per_point,
            **timings,
        })
        logger.info("%s t10_seconds_per_point=%.6f", graph.name, t10_seconds_per_point)

    table_rows, parallel_rows, estimate_summary = _estimate_rows(instance_timings, cfg, phase_c)
    report = {
        "status": "phase_c_feasibility_complete",
        "experiment_id": cfg["experiment_id"],
        "phase_b_reference": {
            "tau_R": phase_b["analysis"].get("tau_R"),
            "recommended_seed_count": phase_b["summary"].get("recommended_seed_count"),
            "phase_b_pass_preliminary": phase_b["summary"].get("phase_b_pass_preliminary"),
        },
        "phase_c_config": {
            "decision_status": phase_c["decision_status"],
            "seed_count": selected_seed_count,
            "seeds": selected_seeds,
            "instances": phase_c["instances"],
            "grid_size": int(phase_c["grid_size"]),
            "benchmark_grid_size": benchmark_grid_size,
            "benchmark_noise": benchmark_noise,
            "benchmark_shots_total": benchmark_shots,
            "zne_methods": cfg["zne"]["methods"],
            "folding_backend": folding_backend,
            "allocation": allocation,
            "simulator_profile": simulator_profile,
            "simulator_options": simulator_options,
        },
        "instance_timings": instance_timings,
        "estimate": estimate_summary,
    }
    _write_csv(timings_path, instance_timings, overwrite=overwrite)
    _write_csv(table_path, table_rows, overwrite=overwrite)
    _write_csv(parallel_path, parallel_rows, overwrite=overwrite)
    write_json(report_path, report)
    build_manifest(root, root / "manifest_v1.6.json")
    logger.info(
        "Phase C feasibility completed; selected shot-scaled hours %.3f; decision=%s",
        estimate_summary["selected_total_hours_shot_scaled"],
        estimate_summary["decision"],
    )
    return report


def _run_phase_c_t10_workload(
    cfg: dict[str, Any],
    graph_name: str,
    size: int,
    p_noise: float,
    shots_total: int,
    seed: int,
    simulator_options: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    graph = graph_from_config(graph_name, cfg)
    folding_backend = str(cfg["zne"]["folding_backend"])
    allocation = "proportional_to_abs_weights"
    transpile_cache: dict[tuple[object, ...], object] = {}
    timings: dict[str, float] = {}
    arrays: dict[str, Any] = {}

    started = time.perf_counter()
    arrays["noisy"] = generate_noisy_landscape(
        graph,
        size,
        p_noise,
        shots_total,
        seed,
        "phase_c_optimization_noisy",
        transpile_cache=transpile_cache,
        simulator_options=simulator_options,
    )
    timings["noisy_seconds"] = time.perf_counter() - started

    for order, scales in cfg["zne"]["methods"].items():
        started = time.perf_counter()
        arrays[f"zne_{order}"], _ = generate_zne_landscape(
            graph,
            size,
            p_noise,
            shots_total,
            seed,
            [int(x) for x in scales],
            str(order),
            folding_backend=folding_backend,
            shot_allocation_strategy=allocation,
            transpile_cache=transpile_cache,
            simulator_options=simulator_options,
        )
        timings[f"zne_{order}_seconds"] = time.perf_counter() - started

    n_points = size * size
    t10_seconds_per_point = (
        timings["noisy_seconds"]
        + timings["zne_linear_seconds"]
        + timings["zne_quadratic_seconds"]
        + timings["zne_cubic_seconds"]
    ) / n_points
    timing_row: dict[str, object] = {
        "instance": graph.name,
        "benchmark_grid_size": size,
        "benchmark_points": n_points,
        "benchmark_noise": p_noise,
        "benchmark_shots_total": shots_total,
        "benchmark_seed": seed,
        "transpile_cache_entries": len(transpile_cache),
        "t10_seconds_per_point": t10_seconds_per_point,
        **timings,
    }
    return arrays, timing_row


def run_phase_c_optimization_benchmark(
    config_path: str | Path | None = None,
    overwrite: bool = False,
    seed_count: int | None = None,
) -> dict[str, Any]:
    root = PROJECT_ROOT
    cfg = read_yaml(config_path or root / "config" / "config_official_pilot_v1.6.yaml")
    phase_c = _phase_c_config(cfg)
    selected_seed_count = int(seed_count or phase_c["seed_count"])
    if selected_seed_count != int(phase_c["seed_count"]):
        raise RuntimeError(f"Phase C seed_count is frozen at {phase_c['seed_count']}, got {selected_seed_count}.")
    _assert_phase_b_ready(root, selected_seed_count)
    selected_seeds = _selected_seeds(root, selected_seed_count)

    result_dir = root / "results" / "phase_c"
    report_path = result_dir / "optimization_benchmark.json"
    candidates_path = result_dir / "optimization_candidates.csv"
    estimates_path = result_dir / "optimization_estimates.csv"
    for path in [report_path, candidates_path, estimates_path]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing Phase C optimization output: {path}")

    logger = configure_logger(root / "logs" / "phase_c_optimization.log")
    logger.info("Starting Phase C optimization benchmark")

    size = int(phase_c.get("optimization_benchmark_grid_size", 2))
    p_noise = float(phase_c["benchmark_noise"])
    shots_total = int(phase_c["benchmark_shots_total"])
    seed = int(selected_seeds[0])

    candidate_rows: list[dict[str, object]] = []
    candidate_timing_rows: dict[str, list[dict[str, object]]] = {}
    candidate_exact: dict[str, bool] = {}
    candidate_numeric: dict[str, bool] = {}
    candidate_max_abs_diff: dict[str, float] = {}

    for graph_name in phase_c["instances"]:
        logger.info("Optimization benchmark instance=%s", graph_name)
        baseline_options = dict(OPTIMIZATION_CANDIDATES[0]["simulator_options"])
        baseline_arrays, baseline_timing = _run_phase_c_t10_workload(
            cfg,
            str(graph_name),
            size,
            p_noise,
            shots_total,
            seed,
            baseline_options,
        )
        baseline_label = str(OPTIMIZATION_CANDIDATES[0]["label"])
        candidate_timing_rows.setdefault(baseline_label, []).append(baseline_timing)
        candidate_exact[baseline_label] = True
        candidate_numeric[baseline_label] = True
        candidate_max_abs_diff[baseline_label] = 0.0
        candidate_rows.append({
            "status": "ok",
            "candidate": baseline_label,
            "instance": graph_name,
            "simulator_options_json": json.dumps(baseline_options, sort_keys=True),
            "error": "",
            "exact_equal_to_current": True,
            "numerically_equivalent_to_current": True,
            "numerical_equivalence_tol": NUMERICAL_EQUIVALENCE_TOL,
            "max_abs_diff": 0.0,
            "t10_seconds_per_point": baseline_timing["t10_seconds_per_point"],
            "speedup_vs_current": 1.0,
        })

        for candidate in OPTIMIZATION_CANDIDATES[1:]:
            label = str(candidate["label"])
            options = dict(candidate["simulator_options"])
            try:
                arrays, timing = _run_phase_c_t10_workload(cfg, str(graph_name), size, p_noise, shots_total, seed, options)
            except Exception as exc:
                candidate_exact[label] = False
                candidate_numeric[label] = False
                candidate_max_abs_diff[label] = float("nan")
                candidate_rows.append({
                    "status": "error",
                    "candidate": label,
                    "instance": graph_name,
                    "simulator_options_json": json.dumps(options, sort_keys=True),
                    "error": repr(exc),
                    "exact_equal_to_current": False,
                    "numerically_equivalent_to_current": False,
                    "numerical_equivalence_tol": NUMERICAL_EQUIVALENCE_TOL,
                    "max_abs_diff": float("nan"),
                    "t10_seconds_per_point": float("nan"),
                    "speedup_vs_current": float("nan"),
                })
                logger.info("%s %s error=%s", graph_name, label, repr(exc))
                continue
            max_diff = 0.0
            exact = True
            for key, baseline in baseline_arrays.items():
                diff = float(np.max(np.abs(arrays[key] - baseline)))
                max_diff = max(max_diff, diff)
                exact = exact and bool(np.array_equal(arrays[key], baseline))
            candidate_timing_rows.setdefault(label, []).append(timing)
            candidate_exact[label] = candidate_exact.get(label, True) and exact
            numerically_equivalent = max_diff <= NUMERICAL_EQUIVALENCE_TOL
            candidate_numeric[label] = candidate_numeric.get(label, True) and numerically_equivalent
            candidate_max_abs_diff[label] = max(candidate_max_abs_diff.get(label, 0.0), max_diff)
            speedup = float(baseline_timing["t10_seconds_per_point"]) / float(timing["t10_seconds_per_point"])
            candidate_rows.append({
                "status": "ok",
                "candidate": label,
                "instance": graph_name,
                "simulator_options_json": json.dumps(options, sort_keys=True),
                "error": "",
                "exact_equal_to_current": exact,
                "numerically_equivalent_to_current": numerically_equivalent,
                "numerical_equivalence_tol": NUMERICAL_EQUIVALENCE_TOL,
                "max_abs_diff": max_diff,
                "t10_seconds_per_point": timing["t10_seconds_per_point"],
                "speedup_vs_current": speedup,
            })
            logger.info("%s %s exact=%s speedup=%.3f", graph_name, label, str(exact).lower(), speedup)

    estimate_rows: list[dict[str, object]] = []
    estimate_by_candidate: dict[str, Any] = {}
    for label, timing_rows in candidate_timing_rows.items():
        _, parallel_rows, estimate = _estimate_rows(timing_rows, cfg, phase_c)
        exact_all = candidate_exact.get(label, False)
        numeric_all = candidate_numeric.get(label, False)
        candidate_summary = {
            "exact_equal_to_current_all_instances": exact_all,
            "numerically_equivalent_to_current_all_instances": numeric_all,
            "numerical_equivalence_tol": NUMERICAL_EQUIVALENCE_TOL,
            "max_abs_diff_all_instances": candidate_max_abs_diff.get(label, float("nan")),
            "estimate": estimate,
            "parallel_rows": parallel_rows,
        }
        estimate_by_candidate[label] = candidate_summary
        estimate_rows.append({
            "candidate": label,
            "exact_equal_to_current_all_instances": exact_all,
            "numerically_equivalent_to_current_all_instances": numeric_all,
            "numerical_equivalence_tol": NUMERICAL_EQUIVALENCE_TOL,
            "max_abs_diff_all_instances": candidate_max_abs_diff.get(label, float("nan")),
            "selected_total_hours_shot_scaled": estimate["selected_total_hours_shot_scaled"],
            "local_ideal_parallel_hours_shot_scaled": estimate["local_ideal_parallel_hours_shot_scaled"],
            "required_speedup_full_run_shot_scaled": estimate["required_speedup"]["full_run_shot_scaled"],
            "minimum_ideal_workers_full_run_shot_scaled": estimate["minimum_ideal_workers"]["full_run_shot_scaled"],
            "phase_c_allowed_by_gate": estimate["phase_c_allowed_by_gate"],
            "decision": estimate["decision"],
        })

    eligible = [
        row
        for row in estimate_rows
        if row["exact_equal_to_current_all_instances"] is True and row["candidate"] != "current_default"
    ]
    best = min(eligible, key=lambda row: float(row["selected_total_hours_shot_scaled"])) if eligible else None
    numerically_eligible = [
        row
        for row in estimate_rows
        if row["numerically_equivalent_to_current_all_instances"] is True and row["candidate"] != "current_default"
    ]
    best_numeric = min(numerically_eligible, key=lambda row: float(row["selected_total_hours_shot_scaled"])) if numerically_eligible else None
    report = {
        "status": "phase_c_optimization_benchmark_complete",
        "benchmark_scope": {
            "grid_size": size,
            "noise": p_noise,
            "shots_total": shots_total,
            "seed": seed,
            "instances": phase_c["instances"],
            "note": "In-memory benchmark only; no official Phase C arrays are generated.",
        },
        "candidates": candidate_rows,
        "estimate_by_candidate": estimate_by_candidate,
        "best_exact_candidate": best,
        "best_numerically_equivalent_candidate": best_numeric,
    }
    _write_csv(candidates_path, candidate_rows, overwrite=overwrite)
    _write_csv(estimates_path, estimate_rows, overwrite=overwrite)
    write_json(report_path, report)
    build_manifest(root, root / "manifest_v1.6.json")
    return report


def run_phase_c_official(config_path: str | Path | None = None, seed_count: int | None = None) -> dict[str, Any]:
    root = PROJECT_ROOT
    cfg = read_yaml(config_path or root / "config" / "config_official_pilot_v1.6.yaml")
    phase_c = _phase_c_config(cfg)
    selected_seed_count = int(seed_count or phase_c["seed_count"])
    if selected_seed_count != int(phase_c["seed_count"]):
        raise RuntimeError(f"Phase C seed_count is frozen at {phase_c['seed_count']}, got {selected_seed_count}.")

    gate_path = root / "results" / "phase_c" / "feasibility_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError("Run scripts/03_run_official.py --feasibility-gate --accept-n-seeds 30 first.")
    gate = _read_json(gate_path)
    if gate["estimate"].get("phase_c_allowed_by_gate") is not True:
        raise RuntimeError("Phase C official run is blocked by the feasibility gate.")
    raise RuntimeError("Phase C official execution runner is not enabled until the passing gate is reviewed and output schema is frozen.")
