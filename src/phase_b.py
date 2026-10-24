from __future__ import annotations

import csv
import json
import math
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .bootstrap import bootstrap_median_ci
from .circuits import GraphInstance, graph_from_config
from .config import MetricsConfig, SUPPORTED_FOLDING_BACKEND
from .figures import save_phase_b_scatter
from .io_utils import PROJECT_ROOT, configure_logger, ensure_dir, read_yaml, save_array, write_json
from .landscapes import generate_ideal_landscape, generate_noisy_landscape, generate_zne_landscape
from .manifest import build_manifest
from .metrics import comparison_metrics, method_metrics
from .mitiq_wrapper import allocate_shots, richardson_weights


METRIC_KEYS = ["mse", "norm_regret", "spearman", "good_region_jaccard", "grad_align", "f_oob"]
COMPARISON_KEYS = ["noisy_mse", "zne_mse", "r_mse", "noisy_norm_regret", "zne_norm_regret", "d_R"]


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


def _phase_b_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if cfg["zne"]["folding_backend"] != SUPPORTED_FOLDING_BACKEND:
        raise ValueError(f"Unsupported folding backend: {cfg['zne']['folding_backend']}")
    return cfg["phase_b_debug"]


def _load_existing_array(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    array = np.load(path)
    if array.shape != expected_shape:
        raise ValueError(f"Unexpected shape for {path}: {array.shape}, expected {expected_shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite values found in {path}")
    return array


def _array_or_generate(
    path: Path,
    expected_shape: tuple[int, int],
    overwrite: bool,
    generate: Callable[[], np.ndarray],
) -> tuple[np.ndarray, bool]:
    if path.exists() and not overwrite:
        return _load_existing_array(path, expected_shape), True
    array = generate()
    if array.shape != expected_shape:
        raise ValueError(f"Generated array has shape {array.shape}, expected {expected_shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Generated array for {path} contains non-finite values.")
    save_array(path, array, overwrite=overwrite)
    return array, False


def _metadata_for_zne(scales: list[int], shots_total: int, folding_backend: str, allocation: str) -> dict[str, object]:
    weights = richardson_weights(scales)
    return {
        "scale_factors": scales,
        "folding_backend": folding_backend,
        "weights": weights,
        "shot_allocation": allocate_shots(shots_total, weights, strategy=allocation),
        "shot_allocation_strategy": allocation,
    }


def _append_method_metrics(
    rows: list[dict[str, object]],
    cfg: dict[str, Any],
    graph: GraphInstance,
    metrics_config: MetricsConfig,
    observed: np.ndarray,
    ideal: np.ndarray,
    noisy_reference: np.ndarray,
    p_noise: float,
    shots_total: int,
    seed: int,
    method: str,
    allocation: str,
    scope: str,
    shot_allocation: list[int] | None = None,
) -> None:
    metrics = method_metrics(observed, ideal, metrics_config=metrics_config, noisy_reference=noisy_reference)
    row: dict[str, object] = {
        "phase": "B",
        "experiment_id": cfg["experiment_id"],
        "instance": graph.name,
        "noise": p_noise,
        "shots_total": shots_total,
        "seed": seed,
        "method": method,
        "allocation": allocation,
        "scope": scope,
        "landscape_min": float(np.min(observed)),
        "landscape_max": float(np.max(observed)),
        "landscape_range": float(np.max(observed) - np.min(observed)),
        **metrics,
    }
    if shot_allocation is not None:
        row["shot_allocation"] = "|".join(str(x) for x in shot_allocation)
    rows.append(row)


def _append_comparison(
    rows: list[dict[str, object]],
    cfg: dict[str, Any],
    graph: GraphInstance,
    noisy: np.ndarray,
    observed: np.ndarray,
    ideal: np.ndarray,
    p_noise: float,
    shots_total: int,
    seed: int | str,
    comparison_type: str,
    zne_order: str,
    allocation: str,
    scope: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "phase": "B",
        "experiment_id": cfg["experiment_id"],
        "instance": graph.name,
        "noise": p_noise,
        "shots_total": shots_total,
        "seed": seed,
        "comparison_type": comparison_type,
        "zne_order": zne_order,
        "allocation": allocation,
        "scope": scope,
        **comparison_metrics(noisy, observed, ideal),
    }
    rows.append(row)
    return row


def _finite_rows(rows: list[dict[str, object]], keys: list[str]) -> bool:
    for row in rows:
        for key in keys:
            if key not in row:
                continue
            try:
                value = float(row[key])
            except (TypeError, ValueError):
                return False
            if not math.isfinite(value):
                return False
    return True


def _allocation_sensitivity_rows(comparison_rows: list[dict[str, object]], core_allocation: str) -> list[dict[str, object]]:
    core: dict[tuple[float, int, str], dict[str, object]] = {}
    sensitivity: list[dict[str, object]] = []
    for row in comparison_rows:
        if row.get("comparison_type") != "zne_vs_noisy":
            continue
        key = (float(row["noise"]), int(row["seed"]), str(row["zne_order"]))
        if row.get("allocation") == core_allocation:
            core[key] = row
        else:
            sensitivity.append(row)

    out: list[dict[str, object]] = []
    for row in sensitivity:
        key = (float(row["noise"]), int(row["seed"]), str(row["zne_order"]))
        reference = core.get(key)
        if reference is None:
            continue
        out.append({
            "noise": row["noise"],
            "seed": row["seed"],
            "zne_order": row["zne_order"],
            "sensitivity_allocation": row["allocation"],
            "core_allocation": core_allocation,
            "core_r_mse": reference["r_mse"],
            "sensitivity_r_mse": row["r_mse"],
            "delta_r_mse": float(row["r_mse"]) - float(reference["r_mse"]),
            "core_d_R": reference["d_R"],
            "sensitivity_d_R": row["d_R"],
            "delta_d_R": float(row["d_R"]) - float(reference["d_R"]),
        })
    return out


def _analyze_phase_b(
    cfg: dict[str, Any],
    benchmark: dict[str, Any],
    metric_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    allocation_rows: list[dict[str, object]],
    elapsed_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    phase_b = _phase_b_config(cfg)
    bootstrap_cfg = cfg["statistics"]["bootstrap"]
    bootstrap_seed = int(cfg["rng"]["bootstrap_seed"])
    resamples = int(bootstrap_cfg["resamples"])
    candidate_counts = [int(x) for x in cfg["seed_selection"]["candidate_seed_counts"]]
    core_allocation = str(phase_b["core_allocation"])

    baseline_d_r = [
        float(row["d_R"])
        for row in comparison_rows
        if row.get("comparison_type") == "noisy_vs_noisy_replica"
    ]
    sigma_baseline = float(np.std(baseline_d_r, ddof=1)) if len(baseline_d_r) > 1 else 0.0
    tau_r = max(0.005, 2.0 * sigma_baseline)

    groups: dict[tuple[float, str], list[float]] = defaultdict(list)
    for row in comparison_rows:
        if (
            row.get("comparison_type") == "zne_vs_noisy"
            and row.get("scope") == "core"
            and row.get("allocation") == core_allocation
        ):
            groups[(float(row["noise"]), str(row["zne_order"]))].append(float(row["d_R"]))

    seed_table: list[dict[str, object]] = []
    recommended_seed_count: int | None = None
    n_initial = max(1, len([int(x) for x in phase_b["seeds_initial"]]))
    estimated_core_hours = float(benchmark.get("estimated_core_phase_b_hours", float("nan")))
    for candidate in candidate_counts:
        widths = []
        group_stats: dict[str, dict[str, float | int]] = {}
        for index, (key, values) in enumerate(sorted(groups.items())):
            stats = bootstrap_median_ci(
                np.asarray(values, dtype=float),
                resamples=resamples,
                seed=bootstrap_seed + candidate * 100 + index,
                sample_size=candidate,
            )
            label = f"p={key[0]:g},zne={key[1]}"
            group_stats[label] = stats
            widths.append(float(stats["ci_width"]))
        max_width = max(widths) if widths else float("nan")
        passes = math.isfinite(max_width) and max_width < tau_r
        if passes and recommended_seed_count is None:
            recommended_seed_count = candidate
        seed_table.append({
            "N_seeds": candidate,
            "max_ci_width_median_d_R": max_width,
            "tau_R": tau_r,
            "runtime_estimate_debug_core_hours": estimated_core_hours * candidate / n_initial,
            "decision": "passes_tau_R" if passes else "does_not_pass_tau_R",
            "group_stats_json": json.dumps(group_stats, sort_keys=True),
        })

    comparison_finite = _finite_rows(comparison_rows, COMPARISON_KEYS)
    metrics_finite = _finite_rows(metric_rows, METRIC_KEYS)
    grad_align_values = [float(row["grad_align"]) for row in metric_rows if "grad_align" in row]
    metric_ranges = [float(row["landscape_range"]) for row in metric_rows if "landscape_range" in row]
    oob_values = [float(row["f_oob"]) for row in metric_rows if "f_oob" in row]
    breakdown_count = sum(bool(row.get("extrapolation_breakdown")) for row in metric_rows)
    sensitivity_complete = len(allocation_rows) > 0

    checks = {
        "all_metrics_finite": metrics_finite and comparison_finite,
        "grad_align_finite": all(math.isfinite(x) for x in grad_align_values),
        "grad_align_not_all_zero": any(abs(x) > 0.0 for x in grad_align_values),
        "landscape_ranges_positive": all(x > 0.0 for x in metric_ranges),
        "no_extrapolation_breakdown": breakdown_count == 0,
        "tau_R_calibrated": math.isfinite(tau_r) and tau_r >= 0.005,
        "seed_variance_estimated": len(seed_table) == len(candidate_counts),
        "allocation_sensitivity_compared": sensitivity_complete,
        "runtime_gate_passed": bool(benchmark.get("full_phase_b_allowed_by_gate", benchmark.get("full_phase_b_core_allowed_by_gate", False))),
    }
    checks["phase_b_pass_preliminary"] = all(checks.values()) and recommended_seed_count is not None

    analysis = {
        "status": "phase_b_analysis_complete",
        "elapsed_seconds": elapsed_seconds,
        "tau_R": tau_r,
        "sigma_baseline": sigma_baseline,
        "baseline_d_R_values": baseline_d_r,
        "recommended_seed_count": recommended_seed_count,
        "checks": checks,
        "max_f_oob": max(oob_values) if oob_values else float("nan"),
        "extrapolation_breakdown_count": breakdown_count,
        "n_metric_rows": len(metric_rows),
        "n_comparison_rows": len(comparison_rows),
        "n_allocation_sensitivity_rows": len(allocation_rows),
        "benchmark_reference": {
            "estimated_core_phase_b_hours": benchmark.get("estimated_core_phase_b_hours"),
            "estimated_full_phase_b_hours": benchmark.get("estimated_full_phase_b_hours"),
            "full_phase_b_allowed_by_gate": benchmark.get("full_phase_b_allowed_by_gate"),
        },
    }
    return analysis, seed_table


def run_phase_b_benchmark(config_path: str | Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    root = PROJECT_ROOT
    cfg = read_yaml(config_path or root / "config" / "config_official_pilot_v1.6.yaml")
    phase_b = _phase_b_config(cfg)
    logger = configure_logger(root / "logs" / "phase_b_benchmark.log")

    output_path = root / "results" / "phase_b_debug" / "runtime_benchmark.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing benchmark: {output_path}")

    graph = graph_from_config(phase_b["instance"], cfg)
    size = int(phase_b["benchmark_grid_size"])
    p_noise = float(phase_b["noise_values"][0])
    shots_total = int(phase_b["shots_total"])
    seed = int(phase_b["seeds_initial"][0])
    folding_backend = str(cfg["zne"]["folding_backend"])
    core_allocation = str(phase_b["core_allocation"])

    logger.info("Starting Phase B benchmark")
    logger.info("instance=%s grid=%s p=%s shots=%s seed=%s", graph.name, size, p_noise, shots_total, seed)

    timings: dict[str, float] = {}
    transpile_cache: dict[tuple[object, ...], object] = {}
    started = time.perf_counter()
    ideal = generate_ideal_landscape(graph, size, progress=_progress(logger, "benchmark_ideal"))
    timings["ideal_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    noisy = generate_noisy_landscape(
        graph,
        size,
        p_noise,
        shots_total,
        seed,
        "phase_b_benchmark_noisy",
        progress=_progress(logger, "benchmark_noisy"),
        transpile_cache=transpile_cache,
    )
    timings["noisy_seconds"] = time.perf_counter() - started

    zne_metadata: dict[str, Any] = {}
    zne_arrays: dict[str, np.ndarray] = {}
    for order, scales in phase_b["benchmark_orders"].items():
        started = time.perf_counter()
        zne_arrays[order], zne_metadata[order] = generate_zne_landscape(
            graph,
            size,
            p_noise,
            shots_total,
            seed,
            [int(x) for x in scales],
            str(order),
            folding_backend=folding_backend,
            shot_allocation_strategy=core_allocation,
            progress=_progress(logger, f"benchmark_zne_{order}"),
            transpile_cache=transpile_cache,
        )
        timings[f"zne_{order}_seconds"] = time.perf_counter() - started

    sensitivity_arrays: dict[str, np.ndarray] = {}
    sensitivity_metadata: dict[str, Any] = {}
    sensitivity_comparison_rows: list[dict[str, object]] = []
    for allocation in phase_b.get("sensitivity_allocations", []):
        allocation = str(allocation)
        if allocation == core_allocation:
            continue
        for order, scales in phase_b["zne_orders"].items():
            started = time.perf_counter()
            array, metadata = generate_zne_landscape(
                graph,
                size,
                p_noise,
                shots_total,
                seed,
                [int(x) for x in scales],
                str(order),
                folding_backend=folding_backend,
                shot_allocation_strategy=allocation,
                progress=_progress(logger, f"benchmark_sensitivity_{allocation}_{order}"),
                transpile_cache=transpile_cache,
            )
            key = f"{allocation}_{order}"
            sensitivity_arrays[key] = array
            sensitivity_metadata[key] = metadata
            timings[f"sensitivity_{key}_seconds"] = time.perf_counter() - started

    n_points = size * size
    t10_seconds_per_point = (
        timings["noisy_seconds"]
        + timings["zne_linear_seconds"]
        + timings["zne_quadratic_seconds"]
        + timings["zne_cubic_seconds"]
    ) / n_points
    full_phase_b_points = int(phase_b["grid_size"]) ** 2
    n_noise = len(phase_b["noise_values"])
    n_seeds = len(phase_b["seeds_initial"])
    n_sensitivity_noise = len(phase_b.get("sensitivity_noise_values", [phase_b["noise_values"][0]]))
    n_sensitivity_seeds = len(phase_b.get("sensitivity_seeds", [phase_b["seeds_initial"][0]]))
    n_sensitivity_allocations = len(phase_b.get("sensitivity_allocations", []))
    sensitivity_stage = str(phase_b.get("sensitivity_stage", "full_subset"))

    noisy_seconds_per_point = timings["noisy_seconds"] / n_points
    core_zne_seconds_per_point = (timings["zne_linear_seconds"] + timings["zne_cubic_seconds"]) / n_points
    core_seconds_per_point = noisy_seconds_per_point + core_zne_seconds_per_point
    estimated_core_hours = core_seconds_per_point * full_phase_b_points * n_noise * n_seeds / 3600.0
    estimated_baseline_hours = 0.0
    estimated_sensitivity_hours = 0.0
    if sensitivity_stage == "full_subset":
        estimated_sensitivity_hours = (
            core_zne_seconds_per_point
            * full_phase_b_points
            * n_sensitivity_noise
            * n_sensitivity_seeds
            * n_sensitivity_allocations
            / 3600.0
        )
    estimated_full_hours = estimated_core_hours + estimated_baseline_hours + estimated_sensitivity_hours

    benchmark_comparisons = {
        "noisy": comparison_metrics(noisy, noisy, ideal),
        **{f"zne_{order}": comparison_metrics(noisy, arr, ideal) for order, arr in zne_arrays.items()},
    }
    for key, array in sensitivity_arrays.items():
        allocation, order = key.split("_", 1)
        sensitivity_comparison_rows.append({
            "noise": p_noise,
            "seed": seed,
            "zne_order": order,
            "sensitivity_allocation": allocation,
            "core_allocation": core_allocation,
            "core_r_mse": benchmark_comparisons[f"zne_{order}"]["r_mse"],
            "sensitivity_r_mse": comparison_metrics(noisy, array, ideal)["r_mse"],
            "delta_r_mse": comparison_metrics(noisy, array, ideal)["r_mse"] - benchmark_comparisons[f"zne_{order}"]["r_mse"],
            "core_d_R": benchmark_comparisons[f"zne_{order}"]["d_R"],
            "sensitivity_d_R": comparison_metrics(noisy, array, ideal)["d_R"],
            "delta_d_R": comparison_metrics(noisy, array, ideal)["d_R"] - benchmark_comparisons[f"zne_{order}"]["d_R"],
        })
    threshold = float(phase_b["max_full_runtime_hours_without_confirmation"])
    report = {
        "status": "benchmark_complete",
        "config": {
            "instance": graph.name,
            "benchmark_grid_size": size,
            "phase_b_grid_size": int(phase_b["grid_size"]),
            "noise": p_noise,
            "shots_total": shots_total,
            "seed": seed,
            "folding_backend": folding_backend,
            "core_allocation": core_allocation,
            "sensitivity_allocations": phase_b.get("sensitivity_allocations", []),
            "sensitivity_stage": sensitivity_stage,
            "sensitivity_noise_values": phase_b.get("sensitivity_noise_values", [phase_b["noise_values"][0]]),
            "sensitivity_seeds": phase_b.get("sensitivity_seeds", [phase_b["seeds_initial"][0]]),
        },
        "timings": timings,
        "n_points": n_points,
        "transpile_cache_entries": len(transpile_cache),
        "t10_seconds_per_point": t10_seconds_per_point,
        "estimated_core_phase_b_hours": estimated_core_hours,
        "estimated_baseline_tau_R_hours": estimated_baseline_hours,
        "estimated_allocation_sensitivity_hours": estimated_sensitivity_hours,
        "estimated_full_phase_b_hours": estimated_full_hours,
        "tau_R_baseline_source": "independent_seed_pairs_from_core_noisy_landscapes",
        "max_full_runtime_hours_without_confirmation": threshold,
        "full_phase_b_core_allowed_by_gate": bool(estimated_core_hours <= threshold),
        "full_phase_b_allowed_by_gate": bool(estimated_full_hours <= threshold),
        "zne_metadata": zne_metadata,
        "sensitivity_metadata": sensitivity_metadata,
        "allocation_sensitivity": sensitivity_comparison_rows,
        "benchmark_metrics": benchmark_comparisons,
    }
    write_json(output_path, report)
    build_manifest(root, root / "manifest_v1.6.json")
    logger.info("Phase B benchmark completed; estimated full hours %.3f", estimated_full_hours)
    return report


def run_phase_b_full(config_path: str | Path | None = None, overwrite: bool = False) -> dict[str, Any]:
    root = PROJECT_ROOT
    cfg = read_yaml(config_path or root / "config" / "config_official_pilot_v1.6.yaml")
    phase_b = _phase_b_config(cfg)
    metrics_config = MetricsConfig.from_experiment_config(cfg)
    logger = configure_logger(root / "logs" / "phase_b_full.log")

    benchmark_path = root / "results" / "phase_b_debug" / "runtime_benchmark.json"
    if not benchmark_path.exists():
        raise FileNotFoundError("Run scripts/02_debug_pilot.py --benchmark-only before --full.")
    benchmark = _read_json(benchmark_path)
    if not benchmark.get("full_phase_b_allowed_by_gate", benchmark.get("full_phase_b_core_allowed_by_gate", False)):
        raise RuntimeError("Benchmark gate did not allow full Phase B run.")

    graph = graph_from_config(phase_b["instance"], cfg)
    size = int(phase_b["grid_size"])
    expected_shape = (size, size)
    shots_total = int(phase_b["shots_total"])
    folding_backend = str(cfg["zne"]["folding_backend"])
    core_allocation = str(phase_b["core_allocation"])
    noise_values = [float(x) for x in phase_b["noise_values"]]
    seeds = [int(x) for x in phase_b["seeds_initial"]]
    zne_orders = {str(k): [int(x) for x in v] for k, v in phase_b["zne_orders"].items()}
    sensitivity_allocations = [str(x) for x in phase_b.get("sensitivity_allocations", [])]
    sensitivity_noise_values = {float(x) for x in phase_b.get("sensitivity_noise_values", [noise_values[0]])}
    sensitivity_seeds = {int(x) for x in phase_b.get("sensitivity_seeds", [seeds[0]])}
    sensitivity_stage = str(phase_b.get("sensitivity_stage", "full_subset"))

    result_dir = root / "results" / "phase_b_debug"
    raw_root = root / "data" / "raw" / "phase_b_debug" / graph.name
    metrics_path = result_dir / "metrics_per_seed.csv"
    comparisons_path = result_dir / "comparisons_per_seed.csv"
    runtime_path = result_dir / "runtime_table.csv"
    allocation_path = result_dir / "allocation_sensitivity.csv"
    seed_table_path = result_dir / "seed_selection_table.csv"
    metadata_path = result_dir / "zne_metadata.json"
    analysis_path = result_dir / "phase_b_analysis.json"
    summary_path = result_dir / "phase_b_summary.json"
    scatter_path = result_dir / "phase_b_preliminary_scatter.png"
    for path in [
        metrics_path,
        comparisons_path,
        runtime_path,
        allocation_path,
        seed_table_path,
        metadata_path,
        analysis_path,
        summary_path,
        scatter_path,
    ]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing Phase B output: {path}")

    logger.info("Starting full Phase B run")
    ideal_path = raw_root / "C_ideal.npy"
    ideal, ideal_loaded = _array_or_generate(
        ideal_path,
        expected_shape,
        overwrite,
        lambda: generate_ideal_landscape(graph, size, progress=_progress(logger, "phase_b_ideal")),
    )
    logger.info("ideal_loaded_from_disk=%s", str(ideal_loaded).lower())

    metric_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    metadata_records: dict[str, object] = {}
    noisy_by_cell: dict[tuple[float, int], np.ndarray] = {}
    transpile_cache: dict[tuple[object, ...], object] = {}
    loaded_arrays = 1 if ideal_loaded else 0
    generated_arrays = 0 if ideal_loaded else 1
    started_all = time.perf_counter()

    for p_noise in noise_values:
        for seed in seeds:
            cell_started = time.perf_counter()
            base = raw_root / f"p_{p_noise:g}" / f"B_{shots_total}" / f"seed_{seed}"
            noisy_path = base / "C_noisy.npy"
            noisy, loaded = _array_or_generate(
                noisy_path,
                expected_shape,
                overwrite,
                lambda p=p_noise, s=seed: generate_noisy_landscape(
                    graph,
                    size,
                    p,
                    shots_total,
                    s,
                    "phase_b_noisy",
                    progress=_progress(logger, f"phase_b_noisy_p{p}_seed{s}"),
                    transpile_cache=transpile_cache,
                ),
            )
            loaded_arrays += int(loaded)
            generated_arrays += int(not loaded)
            noisy_by_cell[(p_noise, seed)] = noisy
            _append_method_metrics(
                metric_rows,
                cfg,
                graph,
                metrics_config,
                noisy,
                ideal,
                noisy,
                p_noise,
                shots_total,
                seed,
                "noisy",
                "none",
                "core",
            )

            allocation_plan = [(core_allocation, "core")]
            if sensitivity_stage == "full_subset" and p_noise in sensitivity_noise_values and seed in sensitivity_seeds:
                allocation_plan.extend((allocation, "sensitivity") for allocation in sensitivity_allocations if allocation != core_allocation)

            for allocation, scope in allocation_plan:
                for order, scales in zne_orders.items():
                    method = f"zne_{order}"
                    zne_path = base / f"C_{method}_{allocation}.npy"
                    generated_metadata: dict[str, object] | None = None

                    def generate() -> np.ndarray:
                        nonlocal generated_metadata
                        array, generated_metadata = generate_zne_landscape(
                            graph,
                            size,
                            p_noise,
                            shots_total,
                            seed,
                            scales,
                            order,
                            folding_backend=folding_backend,
                            shot_allocation_strategy=allocation,
                            progress=_progress(logger, f"phase_b_zne_{order}_{allocation}_p{p_noise}_seed{seed}"),
                            transpile_cache=transpile_cache,
                        )
                        return array

                    zne, loaded = _array_or_generate(zne_path, expected_shape, overwrite, generate)
                    loaded_arrays += int(loaded)
                    generated_arrays += int(not loaded)
                    metadata = generated_metadata or _metadata_for_zne(scales, shots_total, folding_backend, allocation)
                    metadata["loaded_from_disk"] = loaded
                    metadata_key = f"p_{p_noise:g}|B_{shots_total}|seed_{seed}|{order}|{allocation}"
                    metadata_records[metadata_key] = metadata
                    _append_method_metrics(
                        metric_rows,
                        cfg,
                        graph,
                        metrics_config,
                        zne,
                        ideal,
                        noisy,
                        p_noise,
                        shots_total,
                        seed,
                        method,
                        allocation,
                        scope,
                        shot_allocation=[int(x) for x in metadata["shot_allocation"]],
                    )
                    _append_comparison(
                        comparison_rows,
                        cfg,
                        graph,
                        noisy,
                        zne,
                        ideal,
                        p_noise,
                        shots_total,
                        seed,
                        "zne_vs_noisy",
                        order,
                        allocation,
                        scope,
                    )

            runtime_rows.append({
                "instance": graph.name,
                "noise": p_noise,
                "shots_total": shots_total,
                "seed": seed,
                "seconds": time.perf_counter() - cell_started,
            })

    for p_noise in noise_values:
        for seed_a, seed_b in combinations(seeds, 2):
            noisy_a = noisy_by_cell[(p_noise, seed_a)]
            noisy_b = noisy_by_cell[(p_noise, seed_b)]
            _append_comparison(
                comparison_rows,
                cfg,
                graph,
                noisy_a,
                noisy_b,
                ideal,
                p_noise,
                shots_total,
                f"{seed_a}_vs_{seed_b}",
                "noisy_vs_noisy_replica",
                "none",
                "none",
                "tau_R_baseline",
            )

    elapsed_seconds = time.perf_counter() - started_all
    allocation_rows = _allocation_sensitivity_rows(comparison_rows, core_allocation)
    if not allocation_rows:
        allocation_rows = list(benchmark.get("allocation_sensitivity", []))
    analysis, seed_table = _analyze_phase_b(cfg, benchmark, metric_rows, comparison_rows, allocation_rows, elapsed_seconds)

    _write_csv(metrics_path, metric_rows, overwrite=overwrite)
    _write_csv(comparisons_path, comparison_rows, overwrite=overwrite)
    _write_csv(runtime_path, runtime_rows, overwrite=overwrite)
    _write_csv(allocation_path, allocation_rows, overwrite=overwrite)
    _write_csv(seed_table_path, seed_table, overwrite=overwrite)
    write_json(metadata_path, metadata_records)
    write_json(analysis_path, analysis)
    save_phase_b_scatter(scatter_path, comparison_rows)
    summary = {
        "status": "phase_b_complete",
        "elapsed_seconds": elapsed_seconds,
        "generated_arrays": generated_arrays,
        "loaded_arrays": loaded_arrays,
        "transpile_cache_entries": len(transpile_cache),
        "n_metric_rows": len(metric_rows),
        "n_comparison_rows": len(comparison_rows),
        "n_allocation_sensitivity_rows": len(allocation_rows),
        "tau_R": analysis["tau_R"],
        "recommended_seed_count": analysis["recommended_seed_count"],
        "phase_b_pass_preliminary": analysis["checks"]["phase_b_pass_preliminary"],
        "config": {
            "instance": graph.name,
            "grid_size": size,
            "noise_values": noise_values,
            "shots_total": shots_total,
            "seeds": seeds,
            "zne_orders": zne_orders,
            "folding_backend": folding_backend,
            "core_allocation": core_allocation,
            "sensitivity_allocations": sensitivity_allocations,
            "sensitivity_stage": sensitivity_stage,
            "sensitivity_noise_values": sorted(sensitivity_noise_values),
            "sensitivity_seeds": sorted(sensitivity_seeds),
        },
    }
    write_json(summary_path, summary)
    build_manifest(root, root / "manifest_v1.6.json")
    logger.info("Full Phase B run completed")
    return summary
