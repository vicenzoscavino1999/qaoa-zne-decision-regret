from __future__ import annotations

import csv
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .circuits import GraphInstance
from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .mitiq_wrapper import deterministic_seed
from .phase_e_optimizers import SPSAConfig, make_paired_initial_thetas, spsa_minimize
from .phase_e_sim_backend import evaluate_custom_density_objective, richardson_zne_objective


PHASE_E_STAGE_C0_JITTER_VERSION = "phase_e_stage_c0_jitter_v0_1_draft"
DEFAULT_C0_JITTER_SMOKE_INSTANCES = ("H1",)
DEFAULT_C0_JITTER_SMOKE_NOISE_RATES = (0.003,)
DEFAULT_C0_JITTER_SMOKE_SHOTS = (512,)
DEFAULT_C0_JITTER_SMOKE_BASE_SEEDS = (42,)
DEFAULT_C0_JITTER_DUPLICATE_SEEDS = (101, 202, 303)
DEFAULT_C0_JITTER_T_BUDGET = 12
DEFAULT_C0_JITTER_STARTS = 1
DEFAULT_C0_JITTER_MAX_CASES = 2
DEFAULT_C0_JITTER_RANGE_REFERENCE_POINTS = 32
DEFAULT_C0_JITTER_SCALE_FACTORS = (1, 3)
JITTER_WARNING_FRACTION_OF_RANGE = 0.005
JITTER_BLOCK_FRACTION_OF_RANGE = 0.010


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _graph_from_instance(label: str, payload: dict[str, Any]) -> GraphInstance:
    edges = tuple((int(u), int(v), float(w)) for u, v, w in payload["edges"])
    return GraphInstance(name=label, edges=edges)


def _load_graphs(root: Path, labels: tuple[str, ...]) -> tuple[list[GraphInstance], dict[str, Any], str]:
    path = root / "config" / "phase_e_instances_v0.1-draft.json"
    payload = _read_json(path)
    graphs = [_graph_from_instance(label, payload["selected_instances"][label]) for label in labels]
    return graphs, payload, sha256_file(path)


def _objective_noisy(graph: GraphInstance, p_dep: float, shots: int, seed: int) -> Callable[[tuple[float, ...]], float]:
    def objective(theta: tuple[float, ...]) -> float:
        return evaluate_custom_density_objective(
            graph,
            theta,
            p_dep=p_dep,
            shots=shots,
            seed=deterministic_seed("phase_e_c0_jitter_noisy", graph.name, p_dep, shots, seed, tuple(round(x, 12) for x in theta)),
        ).objective_value

    return objective


def _objective_zne(graph: GraphInstance, p_dep: float, shots: int, seed: int) -> Callable[[tuple[float, ...]], float]:
    def objective(theta: tuple[float, ...]) -> float:
        return richardson_zne_objective(
            graph,
            theta,
            p_dep=p_dep,
            scale_factors=DEFAULT_C0_JITTER_SCALE_FACTORS,
            shots_total=shots,
            seed=deterministic_seed("phase_e_c0_jitter_zne", graph.name, p_dep, shots, seed, tuple(round(x, 12) for x in theta)),
        ).objective_value

    return objective


def _objective_ideal(graph: GraphInstance) -> Callable[[tuple[float, ...]], float]:
    def objective(theta: tuple[float, ...]) -> float:
        return evaluate_custom_density_objective(graph, theta, p_dep=0.0).objective_value

    return objective


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[phase-e-c0-jitter] {message}", flush=True)


def _ideal_reference_range(
    graph: GraphInstance,
    p_dep: float,
    shots: int,
    t_budget: int,
    reference_points: int,
) -> dict[str, float | int]:
    ideal = _objective_ideal(graph)
    points = make_paired_initial_thetas(
        depth=2,
        starts=int(reference_points),
        seed=deterministic_seed("phase_e_c0_jitter_range", graph.name, p_dep, shots, t_budget, reference_points),
    )
    values = [float(ideal(theta)) for theta in points]
    ideal_min = float(np.min(values))
    ideal_max = float(np.max(values))
    return {
        "ideal_reference_points": int(reference_points),
        "ideal_reference_min": ideal_min,
        "ideal_reference_max": ideal_max,
        "ideal_objective_range": float(ideal_max - ideal_min),
    }


def _case_id(graph: GraphInstance, p_dep: float, shots: int, base_seed: int, objective_family: str, t_budget: int) -> str:
    return f"{graph.name}|p={p_dep}|shots={shots}|base_seed={base_seed}|family={objective_family}|T={t_budget}"


def _run_duplicate_case(
    graph: GraphInstance,
    p_dep: float,
    shots: int,
    base_seed: int,
    objective_family: str,
    t_budget: int,
    starts_per_duplicate: int,
    duplicate_seeds: tuple[int, ...],
    range_reference_points: int,
    progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case = _case_id(graph, p_dep, shots, base_seed, objective_family, t_budget)
    ideal = _objective_ideal(graph)
    range_record = _ideal_reference_range(graph, p_dep, shots, t_budget, range_reference_points)
    objective_seed = deterministic_seed("phase_e_c0_jitter_fixed_objective", graph.name, p_dep, shots, base_seed, objective_family, t_budget)
    objective = _objective_noisy(graph, p_dep, shots, objective_seed) if objective_family == "noisy" else _objective_zne(graph, p_dep, shots, objective_seed)

    endpoint_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for duplicate_id, duplicate_seed in enumerate(duplicate_seeds):
        _progress(progress, f"duplicate {case} duplicate_id={duplicate_id} K={starts_per_duplicate}")
        starts = make_paired_initial_thetas(
            depth=2,
            starts=int(starts_per_duplicate),
            seed=deterministic_seed("phase_e_c0_jitter_starts", case, duplicate_seed, starts_per_duplicate),
        )
        started = time.perf_counter()
        results = []
        for start_id, theta0 in enumerate(starts):
            opt_seed = deterministic_seed("phase_e_c0_jitter_opt", case, duplicate_seed, start_id)
            results.append(
                spsa_minimize(
                    objective,
                    theta0,
                    SPSAConfig(max_evaluations=int(t_budget)),
                    seed=opt_seed,
                )
            )
        elapsed = time.perf_counter() - started
        best_start_id, best_result = min(enumerate(results), key=lambda item: item[1].best_value)
        ideal_value = float(ideal(best_result.best_theta))
        ideal_regret = float(ideal_value - float(range_record["ideal_reference_min"]))
        endpoint_rows.append(
            {
                "case_id": case,
                "instance": graph.name,
                "p_dep": p_dep,
                "shots_total": shots,
                "base_seed": base_seed,
                "objective_family": objective_family,
                "T_requested": t_budget,
                "duplicate_id": duplicate_id,
                "duplicate_seed": duplicate_seed,
                "starts_per_duplicate": starts_per_duplicate,
                "best_start_id": best_start_id,
                "best_observed_value": best_result.best_value,
                "ideal_value_at_best_theta": ideal_value,
                "ideal_regret_vs_reference_min": ideal_regret,
                "actual_num_evaluations_best": best_result.num_evaluations,
                "actual_num_evaluations_min": min(result.num_evaluations for result in results),
                "actual_num_evaluations_max": max(result.num_evaluations for result in results),
                "elapsed_seconds": elapsed,
                "theta_best_json": json.dumps(best_result.best_theta),
                **range_record,
                "jitter_semantics": "same deterministic objective seed per case; duplicate seeds change optimizer seed and start schedule",
            }
        )
        runtime_rows.append(
            {
                "case_id": case,
                "instance": graph.name,
                "p_dep": p_dep,
                "shots_total": shots,
                "base_seed": base_seed,
                "objective_family": objective_family,
                "T_requested": t_budget,
                "duplicate_id": duplicate_id,
                "starts_per_duplicate": starts_per_duplicate,
                "operation": "duplicate_endpoint_optimizer",
                "elapsed_seconds": elapsed,
            }
        )
    return endpoint_rows, runtime_rows


def _pairwise_jitter_rows(endpoint_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in endpoint_rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)

    out: list[dict[str, Any]] = []
    for case_id, rows in grouped.items():
        rows = sorted(rows, key=lambda item: int(item["duplicate_id"]))
        for left, right in itertools.combinations(rows, 2):
            ideal_range = float(left["ideal_objective_range"])
            abs_diff = abs(float(left["ideal_regret_vs_reference_min"]) - float(right["ideal_regret_vs_reference_min"]))
            out.append(
                {
                    "case_id": case_id,
                    "instance": left["instance"],
                    "p_dep": left["p_dep"],
                    "shots_total": left["shots_total"],
                    "base_seed": left["base_seed"],
                    "objective_family": left["objective_family"],
                    "T_requested": left["T_requested"],
                    "left_duplicate_id": left["duplicate_id"],
                    "right_duplicate_id": right["duplicate_id"],
                    "left_ideal_regret": left["ideal_regret_vs_reference_min"],
                    "right_ideal_regret": right["ideal_regret_vs_reference_min"],
                    "absolute_ideal_regret_difference": abs_diff,
                    "ideal_objective_range": ideal_range,
                    "jitter_fraction_of_ideal_range": abs_diff / ideal_range if ideal_range > 0.0 else math.nan,
                }
            )
    return out


def summarize_jitter_fractions(
    values: list[float],
    bootstrap_seed: int = 20260517,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    finite = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=float)
    if finite.size == 0:
        return {
            "count": 0,
            "p50": math.nan,
            "p90": math.nan,
            "p95": math.nan,
            "p95_bootstrap_ci_low": math.nan,
            "p95_bootstrap_ci_high": math.nan,
            "threshold_classification": "nonfinite_block",
        }

    rng = np.random.default_rng(int(bootstrap_seed))
    boot = []
    for _ in range(int(bootstrap_samples)):
        sample = rng.choice(finite, size=finite.size, replace=True)
        boot.append(float(np.percentile(sample, 95)))
    p95 = float(np.percentile(finite, 95))
    if p95 >= JITTER_BLOCK_FRACTION_OF_RANGE:
        threshold_classification = "block"
    elif p95 >= JITTER_WARNING_FRACTION_OF_RANGE:
        threshold_classification = "warning"
    else:
        threshold_classification = "pass"
    return {
        "count": int(finite.size),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p95": p95,
        "p95_bootstrap_ci_low": float(np.percentile(boot, 2.5)),
        "p95_bootstrap_ci_high": float(np.percentile(boot, 97.5)),
        "threshold_classification": threshold_classification,
    }


def _summary_rows(pairwise_rows: list[dict[str, Any]], bootstrap_samples: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add_row(scope: str, key: str, subset: list[dict[str, Any]]) -> None:
        stats = summarize_jitter_fractions(
            [float(row["jitter_fraction_of_ideal_range"]) for row in subset],
            bootstrap_seed=deterministic_seed("phase_e_c0_jitter_bootstrap", scope, key, len(subset)),
            bootstrap_samples=bootstrap_samples,
        )
        rows.append(
            {
                "scope": scope,
                "key": key,
                "pairwise_count": stats["count"],
                "jitter_fraction_p50": stats["p50"],
                "jitter_fraction_p90": stats["p90"],
                "jitter_fraction_p95": stats["p95"],
                "jitter_fraction_p95_bootstrap_ci_low": stats["p95_bootstrap_ci_low"],
                "jitter_fraction_p95_bootstrap_ci_high": stats["p95_bootstrap_ci_high"],
                "threshold_classification": stats["threshold_classification"],
            }
        )

    add_row("overall", "all", pairwise_rows)
    for family in sorted({str(row["objective_family"]) for row in pairwise_rows}):
        add_row("objective_family", family, [row for row in pairwise_rows if str(row["objective_family"]) == family])
    for instance in sorted({str(row["instance"]) for row in pairwise_rows}):
        add_row("instance", instance, [row for row in pairwise_rows if str(row["instance"]) == instance])
    return rows


def _build_cases(
    graphs: list[GraphInstance],
    noise_rates: tuple[float, ...],
    shots: tuple[int, ...],
    base_seeds: tuple[int, ...],
    include_zne: bool,
    max_cases: int | None,
) -> list[tuple[GraphInstance, float, int, int, str]]:
    families = ("noisy", "zne") if include_zne else ("noisy",)
    cases = [
        (graph, float(p_dep), int(shot_budget), int(base_seed), family)
        for graph in graphs
        for p_dep in noise_rates
        for shot_budget in shots
        for base_seed in base_seeds
        for family in families
    ]
    if max_cases is not None:
        return cases[: int(max_cases)]
    return cases


def run_phase_e_stage_c0_jitter(
    root: Path = PROJECT_ROOT,
    force: bool = False,
    instances: tuple[str, ...] = DEFAULT_C0_JITTER_SMOKE_INSTANCES,
    noise_rates: tuple[float, ...] = DEFAULT_C0_JITTER_SMOKE_NOISE_RATES,
    shots: tuple[int, ...] = DEFAULT_C0_JITTER_SMOKE_SHOTS,
    base_seeds: tuple[int, ...] = DEFAULT_C0_JITTER_SMOKE_BASE_SEEDS,
    duplicate_seeds: tuple[int, ...] = DEFAULT_C0_JITTER_DUPLICATE_SEEDS,
    t_budget: int = DEFAULT_C0_JITTER_T_BUDGET,
    starts_per_duplicate: int = DEFAULT_C0_JITTER_STARTS,
    include_zne: bool = True,
    max_cases: int | None = DEFAULT_C0_JITTER_MAX_CASES,
    range_reference_points: int = DEFAULT_C0_JITTER_RANGE_REFERENCE_POINTS,
    bootstrap_samples: int = 1000,
    progress: bool = False,
) -> dict[str, Any]:
    if int(t_budget) < 1:
        raise ValueError("t_budget must be positive.")
    if int(starts_per_duplicate) < 1:
        raise ValueError("starts_per_duplicate must be positive.")
    if len(duplicate_seeds) < 2:
        raise ValueError("At least two duplicate seeds are required to measure pairwise jitter.")
    if int(range_reference_points) < 2:
        raise ValueError("range_reference_points must be at least 2.")
    if int(bootstrap_samples) < 1:
        raise ValueError("bootstrap_samples must be positive.")

    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration" / "c0_jitter")
    paths = {
        "summary": output_dir / "phase_e_stage_c0_jitter_summary.json",
        "endpoints": output_dir / "duplicate_endpoint_jitter_endpoints.csv",
        "pairwise": output_dir / "duplicate_endpoint_jitter_pairwise.csv",
        "report": output_dir / "duplicate_endpoint_jitter_report.csv",
        "runtime": output_dir / "duplicate_endpoint_jitter_runtime.csv",
        "hashes": output_dir / "phase_e_stage_c0_jitter_hashes.json",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase E C0 jitter outputs.")

    started = time.perf_counter()
    graphs, instances_payload, instances_hash = _load_graphs(root, instances)
    cases = _build_cases(graphs, noise_rates, shots, base_seeds, include_zne, max_cases)

    endpoint_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for graph, p_dep, shot_budget, base_seed, family in cases:
        endpoints, runtimes = _run_duplicate_case(
            graph,
            p_dep,
            shot_budget,
            base_seed,
            family,
            int(t_budget),
            int(starts_per_duplicate),
            tuple(int(seed) for seed in duplicate_seeds),
            int(range_reference_points),
            progress,
        )
        endpoint_rows.extend(endpoints)
        runtime_rows.extend(runtimes)

    pairwise_rows = _pairwise_jitter_rows(endpoint_rows)
    report_rows = _summary_rows(pairwise_rows, int(bootstrap_samples))
    _write_csv(paths["endpoints"], endpoint_rows)
    _write_csv(paths["pairwise"], pairwise_rows)
    _write_csv(paths["report"], report_rows)
    _write_csv(paths["runtime"], runtime_rows)

    finite_values: list[float] = []
    for row in endpoint_rows + pairwise_rows + report_rows + runtime_rows:
        for value in row.values():
            if isinstance(value, float):
                finite_values.append(value)

    overall = next((row for row in report_rows if row["scope"] == "overall"), None)
    full_jitter_gate_complete = (
        len(cases) >= 30
        and len(duplicate_seeds) >= 3
        and int(starts_per_duplicate) >= 20
        and int(t_budget) >= 250
        and int(range_reference_points) >= DEFAULT_C0_JITTER_RANGE_REFERENCE_POINTS
    )
    threshold_classification = None if overall is None else str(overall["threshold_classification"])
    gate_status = threshold_classification if full_jitter_gate_complete else "smoke_complete_not_gate"
    summary = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_stage_c0_jitter_smoke_complete_not_gate" if not full_jitter_gate_complete else f"phase_e_stage_c0_jitter_gate_{threshold_classification}",
        "jitter_version": PHASE_E_STAGE_C0_JITTER_VERSION,
        "scope": "safe_smoke_jitter_not_full_c0_gate" if not full_jitter_gate_complete else "full_c0_duplicate_endpoint_jitter_gate",
        "official_execution_authorized": False,
        "full_jitter_gate_complete": full_jitter_gate_complete,
        "gate_status": gate_status,
        "threshold_classification": threshold_classification,
        "warning_threshold_fraction_of_ideal_range": JITTER_WARNING_FRACTION_OF_RANGE,
        "block_threshold_fraction_of_ideal_range": JITTER_BLOCK_FRACTION_OF_RANGE,
        "instances": list(instances),
        "phase_e_instances_hash": instances_hash,
        "selected_depth2_instances": instances_payload.get("selected_depth2_instances"),
        "noise_rates": list(noise_rates),
        "shot_budgets": list(shots),
        "base_seeds": list(base_seeds),
        "duplicate_seeds": list(duplicate_seeds),
        "include_zne": bool(include_zne),
        "max_cases": max_cases,
        "case_count": len(cases),
        "endpoint_rows": len(endpoint_rows),
        "pairwise_rows": len(pairwise_rows),
        "report_rows": len(report_rows),
        "T_requested": int(t_budget),
        "starts_per_duplicate": int(starts_per_duplicate),
        "range_reference_points": int(range_reference_points),
        "bootstrap_samples": int(bootstrap_samples),
        "overall_jitter_fraction_p95": None if overall is None else overall["jitter_fraction_p95"],
        "overall_jitter_fraction_p95_bootstrap_ci_low": None if overall is None else overall["jitter_fraction_p95_bootstrap_ci_low"],
        "overall_jitter_fraction_p95_bootstrap_ci_high": None if overall is None else overall["jitter_fraction_p95_bootstrap_ci_high"],
        "all_checked_numeric_finite": all(math.isfinite(value) for value in finite_values),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {key: str(path.relative_to(root)).replace("\\", "/") for key, path in paths.items()},
        "limitations": [
            "Smoke defaults are intentionally small and do not complete the duplicate endpoint jitter gate.",
            "Full C0 jitter gate requires at least 30 representative cases, 3 duplicate endpoint seeds per case, and at least the minimum protocol K=20 starts per duplicate.",
            "No final T, K, K_ref, or N_LHS values are selected.",
        ],
    }
    write_json_atomic(paths["summary"], summary, overwrite=True)
    hashes = {
        "created_at_utc": _utc_now(),
        "jitter_version": PHASE_E_STAGE_C0_JITTER_VERSION,
        "hashes": {key: sha256_file(path) for key, path in paths.items() if key != "hashes"},
    }
    write_json_atomic(paths["hashes"], hashes, overwrite=True)
    summary["output_hashes"] = hashes["hashes"]
    return summary
