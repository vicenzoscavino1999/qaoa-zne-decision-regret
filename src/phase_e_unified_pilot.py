from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import qmc

from .circuits import GraphInstance
from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .mitiq_wrapper import deterministic_seed
from .phase_e_optimizers import COBYLAConfig, cobyla_minimize, make_paired_initial_thetas
from .phase_e_sim_backend import (
    evaluate_custom_density_objective,
    richardson_zne_objective,
    theta_periods,
)


PHASE_E_UNIFIED_PILOT_VERSION = "phase_e_unified_depth2_pilot_v0_3_cobyla_draft"
DEFAULT_PILOT_NOISE_RATES = (0.003, 0.01)
DEFAULT_PILOT_SHOTS = (512, 4096)
DEFAULT_PILOT_ZNE_ORDER = "linear"
DEFAULT_PILOT_SCALE_FACTORS = (1, 3)
DEFAULT_TECHNICAL_E_COBYLA_CANDIDATES = (30,)
DEFAULT_TECHNICAL_K_CANDIDATES = (1,)
DEFAULT_TECHNICAL_LHS_CANDIDATES = (4,)
DEFAULT_TECHNICAL_SEEDS = (42,)
DEFAULT_TECHNICAL_LHS_DESIGNS = 1


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[phase-e-pilot] {message}", flush=True)


def _graph_from_instance(label: str, instance_payload: dict[str, Any]) -> GraphInstance:
    edges = tuple((int(u), int(v), float(w)) for u, v, w in instance_payload["edges"])
    return GraphInstance(name=label, edges=edges)


def _load_pilot_graphs(root: Path) -> tuple[list[GraphInstance], dict[str, Any], str]:
    path = root / "config" / "phase_e_instances_v0.1-draft.json"
    payload = _read_json(path)
    graphs = [_graph_from_instance(label, payload["selected_instances"][label]) for label in payload["pilot_depth2_instances"]]
    return graphs, payload, sha256_file(path)


def _objective_noisy(graph: GraphInstance, p_dep: float, shots: int, seed: int) -> Callable[[tuple[float, ...]], float]:
    def objective(theta: tuple[float, ...]) -> float:
        return evaluate_custom_density_objective(
            graph,
            theta,
            p_dep=p_dep,
            shots=shots,
            seed=deterministic_seed("phase_e_pilot_noisy", graph.name, p_dep, shots, seed, tuple(round(x, 12) for x in theta)),
        ).objective_value

    return objective


def _objective_zne(graph: GraphInstance, p_dep: float, shots: int, seed: int) -> Callable[[tuple[float, ...]], float]:
    def objective(theta: tuple[float, ...]) -> float:
        return richardson_zne_objective(
            graph,
            theta,
            p_dep=p_dep,
            scale_factors=DEFAULT_PILOT_SCALE_FACTORS,
            shots_total=shots,
            seed=deterministic_seed("phase_e_pilot_zne", graph.name, p_dep, shots, seed, tuple(round(x, 12) for x in theta)),
        ).objective_value

    return objective


def _objective_ideal(graph: GraphInstance) -> Callable[[tuple[float, ...]], float]:
    def objective(theta: tuple[float, ...]) -> float:
        return evaluate_custom_density_objective(graph, theta, p_dep=0.0).objective_value

    return objective


def _run_backend_measurements(
    graphs: list[GraphInstance],
    noise_rates: tuple[float, ...],
    shots: tuple[int, ...],
) -> list[dict[str, Any]]:
    theta = (0.37, 0.22, 0.81, 0.44)
    rows: list[dict[str, Any]] = []
    for graph in graphs:
        for p_dep in noise_rates:
            for shot_budget in shots:
                started = time.perf_counter()
                noisy = evaluate_custom_density_objective(
                    graph,
                    theta,
                    p_dep=p_dep,
                    shots=shot_budget,
                    seed=deterministic_seed("phase_e_pilot_backend", graph.name, p_dep, shot_budget, "noisy"),
                )
                rows.append(
                    {
                        "instance": graph.name,
                        "n_qubits": graph.n_qubits,
                        "p_dep": p_dep,
                        "shots_total": shot_budget,
                        "objective_family": "noisy",
                        "scale_factors": "1",
                        "elapsed_seconds": time.perf_counter() - started,
                        "objective_value": noisy.objective_value,
                        "transpiled_gate_count": noisy.metadata["transpiled_gate_count"],
                        "transpiled_depth": noisy.metadata["transpiled_depth"],
                    }
                )
                started = time.perf_counter()
                zne = richardson_zne_objective(
                    graph,
                    theta,
                    p_dep=p_dep,
                    scale_factors=DEFAULT_PILOT_SCALE_FACTORS,
                    shots_total=shot_budget,
                    seed=deterministic_seed("phase_e_pilot_backend", graph.name, p_dep, shot_budget, "zne"),
                )
                rows.append(
                    {
                        "instance": graph.name,
                        "n_qubits": graph.n_qubits,
                        "p_dep": p_dep,
                        "shots_total": shot_budget,
                        "objective_family": "zne_richardson_linear",
                        "scale_factors": "1|3",
                        "elapsed_seconds": time.perf_counter() - started,
                        "objective_value": zne.objective_value,
                        "variance_estimate": zne.variance_estimate,
                    }
                )
    return rows


def _lhs_points(depth: int, count: int, seed: int) -> list[tuple[float, ...]]:
    sampler = qmc.LatinHypercube(d=2 * depth, seed=int(seed))
    unit = sampler.random(n=int(count))
    periods = np.asarray(theta_periods(depth), dtype=float)
    return [tuple(float(x) for x in row * periods) for row in unit]


def _run_lhs_calibration(
    graphs: list[GraphInstance],
    noise_rates: tuple[float, ...],
    shots: tuple[int, ...],
    lhs_candidates: tuple[int, ...],
    lhs_designs: int,
    progress: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for graph in graphs:
        for p_dep in noise_rates:
            for shot_budget in shots:
                for n_lhs in lhs_candidates:
                    for design_id in range(int(lhs_designs)):
                        _progress(progress, f"LHS {graph.name} p={p_dep} shots={shot_budget} N={n_lhs} design={design_id + 1}/{lhs_designs}")
                        started = time.perf_counter()
                        lhs_seed = deterministic_seed("phase_e_lhs", graph.name, p_dep, shot_budget, n_lhs, design_id)
                        points = _lhs_points(depth=2, count=n_lhs, seed=lhs_seed)
                        noisy_errors = []
                        zne_errors = []
                        ideal_values = []
                        for index, theta in enumerate(points):
                            ideal = evaluate_custom_density_objective(graph, theta, p_dep=0.0).objective_value
                            noisy = evaluate_custom_density_objective(
                                graph,
                                theta,
                                p_dep=p_dep,
                                shots=shot_budget,
                                seed=deterministic_seed("phase_e_lhs_noisy", graph.name, p_dep, shot_budget, n_lhs, design_id, index),
                            ).objective_value
                            zne = richardson_zne_objective(
                                graph,
                                theta,
                                p_dep=p_dep,
                                scale_factors=DEFAULT_PILOT_SCALE_FACTORS,
                                shots_total=shot_budget,
                                seed=deterministic_seed("phase_e_lhs_zne", graph.name, p_dep, shot_budget, n_lhs, design_id, index),
                            ).objective_value
                            ideal_values.append(ideal)
                            noisy_errors.append((noisy - ideal) ** 2)
                            zne_errors.append((zne - ideal) ** 2)
                        mse_noisy = float(np.mean(noisy_errors))
                        mse_zne = float(np.mean(zne_errors))
                        rows.append(
                            {
                                "instance": graph.name,
                                "p_dep": p_dep,
                                "shots_total": shot_budget,
                                "N_LHS": n_lhs,
                                "lhs_design_id": design_id,
                                "lhs_seed": lhs_seed,
                                "MSE_noisy": mse_noisy,
                                "MSE_ZNE_Richardson": mse_zne,
                                "r_MSE_Richardson": float((mse_noisy - mse_zne) / mse_noisy) if mse_noisy > 0.0 else math.nan,
                                "r_MSE_sign": int(np.sign((mse_noisy - mse_zne) / mse_noisy)) if mse_noisy > 0.0 else 0,
                                "Var_L0": float(np.var(ideal_values)),
                                "Range_L0": float(np.max(ideal_values) - np.min(ideal_values)),
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        )
    return rows


def _run_optimizer_calibration(
    graphs: list[GraphInstance],
    noise_rates: tuple[float, ...],
    shots: tuple[int, ...],
    seeds: tuple[int, ...],
    e_cobyla_candidates: tuple[int, ...],
    k_candidates: tuple[int, ...],
    progress: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_k = max(k_candidates)

    def _finite_results(results: list[Any], k_value: int) -> list[Any]:
        return [result for result in results[:k_value] if np.isfinite(float(result.best_value))]

    for graph in graphs:
        ideal_objective = _objective_ideal(graph)
        for p_dep in noise_rates:
            for shot_budget in shots:
                for seed in seeds:
                    starts = make_paired_initial_thetas(depth=2, starts=max_k, seed=deterministic_seed("phase_e_pilot_starts", graph.name, p_dep, shot_budget, seed))
                    for e_budget in e_cobyla_candidates:
                        _progress(progress, f"COBYLA {graph.name} p={p_dep} shots={shot_budget} seed={seed} E={e_budget} maxK={max_k}")
                        noisy_results = []
                        zne_results = []
                        started = time.perf_counter()
                        for start_id, theta0 in enumerate(starts):
                            noisy_results.append(
                                cobyla_minimize(
                                    _objective_noisy(graph, p_dep, shot_budget, deterministic_seed("phase_e_pilot_noisy_cobyla", seed, start_id)),
                                    theta0,
                                    COBYLAConfig(max_evaluations=e_budget),
                                )
                            )
                            zne_results.append(
                                cobyla_minimize(
                                    _objective_zne(graph, p_dep, shot_budget, deterministic_seed("phase_e_pilot_zne_cobyla", seed, start_id)),
                                    theta0,
                                    COBYLAConfig(max_evaluations=e_budget),
                                )
                            )
                        elapsed = time.perf_counter() - started
                        noisy_eval_counts = [int(result.num_evaluations) for result in noisy_results]
                        zne_eval_counts = [int(result.num_evaluations) for result in zne_results]
                        for k_value in k_candidates:
                            noisy_valid = _finite_results(noisy_results, int(k_value))
                            zne_valid = _finite_results(zne_results, int(k_value))
                            noisy_best = min(noisy_valid, key=lambda item: item.best_value) if noisy_valid else None
                            zne_best = min(zne_valid, key=lambda item: item.best_value) if zne_valid else None
                            ideal_at_noisy = ideal_objective(noisy_best.best_theta) if noisy_best is not None else math.nan
                            ideal_at_zne = ideal_objective(zne_best.best_theta) if zne_best is not None else math.nan
                            within_cap = bool(
                                min(noisy_eval_counts + zne_eval_counts) <= int(e_budget)
                                and max(noisy_eval_counts + zne_eval_counts) <= int(e_budget)
                            )
                            hit_cap = bool(max(noisy_eval_counts + zne_eval_counts) >= int(e_budget))
                            rows.append(
                                {
                                    "instance": graph.name,
                                    "p_dep": p_dep,
                                    "shots_total": shot_budget,
                                    "seed": seed,
                                    "E_COBYLA": e_budget,
                                    "K": k_value,
                                    "observed_noisy_value": noisy_best.best_value if noisy_best is not None else math.nan,
                                    "observed_zne_value": zne_best.best_value if zne_best is not None else math.nan,
                                    "ideal_value_at_theta_noisy": ideal_at_noisy,
                                    "ideal_value_at_theta_zne": ideal_at_zne,
                                    "d_R": float(ideal_at_zne - ideal_at_noisy),
                                    "noisy_num_evaluations_best": noisy_best.num_evaluations if noisy_best is not None else math.nan,
                                    "zne_num_evaluations_best": zne_best.num_evaluations if zne_best is not None else math.nan,
                                    "noisy_num_evaluations_min": min(noisy_eval_counts),
                                    "noisy_num_evaluations_max": max(noisy_eval_counts),
                                    "noisy_num_evaluations_mean": float(np.mean(noisy_eval_counts)),
                                    "zne_num_evaluations_min": min(zne_eval_counts),
                                    "zne_num_evaluations_max": max(zne_eval_counts),
                                    "zne_num_evaluations_mean": float(np.mean(zne_eval_counts)),
                                    "actual_evaluations_within_requested_cap": within_cap,
                                    "actual_evaluations_hit_requested_cap": hit_cap,
                                    "actual_evaluations_equal_requested": bool(
                                        min(noisy_eval_counts + zne_eval_counts) == int(e_budget)
                                        and max(noisy_eval_counts + zne_eval_counts) == int(e_budget)
                                    ),
                                    "requested_max_evaluations": e_budget,
                                    "cobyla_budget_semantics": "max_evaluations_cap passed to scipy.optimize.minimize(method='COBYLA') as maxiter; actual function evaluations are recorded per start.",
                                    "valid_start_count_noisy": len(noisy_valid),
                                    "valid_start_count_zne": len(zne_valid),
                                    "invalid_start_count_noisy": int(k_value) - len(noisy_valid),
                                    "invalid_start_count_zne": int(k_value) - len(zne_valid),
                                    "optimizer_status_counts_noisy_json": json.dumps(
                                        {status: [result.status for result in noisy_results[:k_value]].count(status) for status in sorted({result.status for result in noisy_results[:k_value]})},
                                        sort_keys=True,
                                    ),
                                    "optimizer_status_counts_zne_json": json.dumps(
                                        {status: [result.status for result in zne_results[:k_value]].count(status) for status in sorted({result.status for result in zne_results[:k_value]})},
                                        sort_keys=True,
                                    ),
                                    "elapsed_seconds_for_max_k_pair": elapsed,
                                }
                            )
    return rows


def _decision(
    backend_rows: list[dict[str, Any]],
    lhs_rows: list[dict[str, Any]],
    optimizer_rows: list[dict[str, Any]],
    e_cobyla_candidates: tuple[int, ...],
    k_candidates: tuple[int, ...],
    lhs_candidates: tuple[int, ...],
    lhs_designs: int,
) -> dict[str, Any]:
    backend_noisy = [row["elapsed_seconds"] for row in backend_rows if row["objective_family"] == "noisy"]
    backend_zne = [row["elapsed_seconds"] for row in backend_rows if row["objective_family"] != "noisy"]
    opt_by_e: dict[int, list[float]] = {e: [] for e in e_cobyla_candidates}
    opt_by_k: dict[int, list[float]] = {k: [] for k in k_candidates}
    for row in optimizer_rows:
        opt_by_e[int(row["E_COBYLA"])].append(abs(float(row["d_R"])))
        opt_by_k[int(row["K"])].append(abs(float(row["d_R"])))
    lhs_by_n: dict[int, list[float]] = {n: [] for n in lhs_candidates}
    for row in lhs_rows:
        lhs_by_n[int(row["N_LHS"])].append(float(row["r_MSE_Richardson"]))
    lhs_sign_stability: dict[str, float] = {}
    for n_lhs, values in lhs_by_n.items():
        finite_values = [value for value in values if np.isfinite(value)]
        if finite_values:
            majority_sign = 1.0 if np.mean([value >= 0.0 for value in finite_values]) >= 0.5 else 0.0
            lhs_sign_stability[str(n_lhs)] = float(np.mean([(value >= 0.0) == bool(majority_sign) for value in finite_values]))
    eval_counts_by_e: dict[str, dict[str, float]] = {}
    for e_budget in e_cobyla_candidates:
        matching = [row for row in optimizer_rows if int(row["E_COBYLA"]) == int(e_budget)]
        mean_counts = [float(row["noisy_num_evaluations_mean"]) for row in matching] + [float(row["zne_num_evaluations_mean"]) for row in matching]
        min_counts = [float(row["noisy_num_evaluations_min"]) for row in matching] + [float(row["zne_num_evaluations_min"]) for row in matching]
        max_counts = [float(row["noisy_num_evaluations_max"]) for row in matching] + [float(row["zne_num_evaluations_max"]) for row in matching]
        hit_cap_rows = [
            row
            for row in matching
            if max(float(row["noisy_num_evaluations_max"]), float(row["zne_num_evaluations_max"])) >= float(e_budget)
        ]
        if mean_counts:
            eval_counts_by_e[str(e_budget)] = {
                "row_mean_min": float(np.min(mean_counts)),
                "row_mean": float(np.mean(mean_counts)),
                "row_mean_max": float(np.max(mean_counts)),
                "actual_start_min": float(np.min(min_counts)),
                "actual_start_max": float(np.max(max_counts)),
                "hit_cap_rows": int(len(hit_cap_rows)),
                "requested_max_evaluations": float(e_budget),
                "mean_gap_to_requested": float(e_budget - np.mean(mean_counts)),
            }
    sorted_e = sorted(e_cobyla_candidates)
    sorted_k = sorted(k_candidates)
    sorted_lhs = sorted(lhs_candidates)
    mean_abs_d_by_e = {str(k): float(np.mean(v)) for k, v in opt_by_e.items() if v}
    mean_abs_d_by_k = {str(k): float(np.mean(v)) for k, v in opt_by_k.items() if v}
    mean_r_by_lhs = {str(k): float(np.mean(v)) for k, v in lhs_by_n.items() if v}

    def _last_delta(values: dict[str, float], ordered: list[int]) -> float | None:
        if len(ordered) < 2:
            return None
        prev = values.get(str(ordered[-2]))
        current = values.get(str(ordered[-1]))
        if prev is None or current is None:
            return None
        return float(current - prev)

    return {
        "status": "technical_draft_pilot_complete_not_final_calibration",
        "selected_E_COBYLA_technical": max(e_cobyla_candidates),
        "selected_K_technical": max(k_candidates),
        "selected_N_LHS_technical": max(lhs_candidates),
        "expanded_calibration_started": bool(len(e_cobyla_candidates) > 1 or len(k_candidates) > 1 or len(lhs_candidates) > 1),
        "candidate_counts": {
            "E_COBYLA": len(e_cobyla_candidates),
            "K": len(k_candidates),
            "N_LHS": len(lhs_candidates),
            "LHS_designs": int(lhs_designs),
        },
        "technical_values_are_not_selected_final_values": True,
        "cobyla_budget_semantics": "E_COBYLA is passed to scipy.optimize.minimize(method='COBYLA') as maxiter; final calibration must record actual evaluation-count behavior before freeze.",
        "actual_cobyla_evaluations_by_E": eval_counts_by_e,
        "lhs_independent_designs": int(lhs_designs),
        "lhs_sign_stability_by_N_LHS": lhs_sign_stability,
        "final_calibration_required": True,
        "mean_noisy_objective_seconds": float(np.mean(backend_noisy)) if backend_noisy else None,
        "mean_zne_objective_seconds": float(np.mean(backend_zne)) if backend_zne else None,
        "mean_abs_d_R_by_E_COBYLA": mean_abs_d_by_e,
        "mean_abs_d_R_by_K": mean_abs_d_by_k,
        "mean_r_MSE_by_N_LHS": mean_r_by_lhs,
        "last_step_delta_abs_d_R_E_COBYLA": _last_delta(mean_abs_d_by_e, sorted_e),
        "last_step_delta_abs_d_R_K": _last_delta(mean_abs_d_by_k, sorted_k),
        "last_step_delta_r_MSE_N_LHS": _last_delta(mean_r_by_lhs, sorted_lhs),
        "limitations": [
            "Technical draft pilot uses reduced E_COBYLA/K/N_LHS/seed counts to validate pipeline and measure cost.",
            "It does not freeze production E_COBYLA, K, K_ref, N_LHS, tau_R_opt, tau_conc, or tau_trivial.",
            "A larger calibration pilot or explicit amendment is still required before final freeze.",
            "COBYLA E_COBYLA values in this technical pilot are max-evaluation caps, not proof that exactly E evaluations were consumed.",
        ],
    }


def _markdown(summary: dict[str, Any]) -> str:
    decision = summary["calibration_decision"]
    lines = [
        "# Phase E Unified Depth-2 Pilot",
        "",
        f"Created UTC: {summary['created_at_utc']}",
        f"Status: `{summary['status']}`",
        "",
        "This is a technical draft pilot with reduced calibration budgets. It validates the depth-2 pipeline and records preliminary runtime/diagnostic behavior; it is not an official Phase E run and does not freeze final calibration values.",
        "",
        "## Scope",
        "",
        f"- Instances: {', '.join(summary['pilot_instances'])}",
        f"- Noise rates: {summary['noise_rates']}",
        f"- Shot budgets: {summary['shot_budgets']}",
        f"- E_COBYLA candidates: {summary['e_cobyla_candidates']}",
        f"- K candidates: {summary['k_candidates']}",
        f"- N_LHS candidates: {summary['lhs_candidates']}",
        f"- LHS independent designs: {summary['lhs_independent_designs']}",
        "",
        "## Technical Decision",
        "",
        f"- Status: `{decision['status']}`",
        f"- Technical E_COBYLA: `{decision['selected_E_COBYLA_technical']}`",
        f"- Technical K: `{decision['selected_K_technical']}`",
        f"- Technical N_LHS: `{decision['selected_N_LHS_technical']}`",
        f"- Expanded calibration started: `{decision['expanded_calibration_started']}`",
        f"- Technical values are final: `{not decision['technical_values_are_not_selected_final_values']}`",
        f"- Final calibration required: `{decision['final_calibration_required']}`",
        f"- Mean noisy objective seconds: `{decision['mean_noisy_objective_seconds']}`",
        f"- Mean ZNE objective seconds: `{decision['mean_zne_objective_seconds']}`",
        f"- Last-step delta abs d_R by E_COBYLA: `{decision['last_step_delta_abs_d_R_E_COBYLA']}`",
        f"- Last-step delta abs d_R by K: `{decision['last_step_delta_abs_d_R_K']}`",
        f"- Last-step delta r_MSE by N_LHS: `{decision['last_step_delta_r_MSE_N_LHS']}`",
        "",
        "Next step: use these measurements to decide whether to expand the pilot to protocol-scale calibration or amend the computational budget before runtime/storage gates.",
        "",
    ]
    return "\n".join(lines)


def run_phase_e_unified_pilot(
    root: Path = PROJECT_ROOT,
    force: bool = False,
    e_cobyla_candidates: tuple[int, ...] = DEFAULT_TECHNICAL_E_COBYLA_CANDIDATES,
    k_candidates: tuple[int, ...] = DEFAULT_TECHNICAL_K_CANDIDATES,
    lhs_candidates: tuple[int, ...] = DEFAULT_TECHNICAL_LHS_CANDIDATES,
    seeds: tuple[int, ...] = DEFAULT_TECHNICAL_SEEDS,
    lhs_designs: int = DEFAULT_TECHNICAL_LHS_DESIGNS,
    calibration_stage: str = "technical_draft_pre_freeze_cobyla_budget",
    progress: bool = False,
) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration")
    paths = {
        "summary": output_dir / "phase_e_unified_pilot_summary.json",
        "cobyla": output_dir / "cobyla_budget_calibration.csv",
        "k": output_dir / "k_random_start_calibration.csv",
        "lhs": output_dir / "lhs_size_calibration.csv",
        "runtime": output_dir / "pilot_runtime_measurements.csv",
        "backend": output_dir / "backend_benchmark_measurements.csv",
        "cache": output_dir / "cache_benchmark_measurements.csv",
        "decision": output_dir / "calibration_decision.json",
        "report": output_dir / "phase_e_unified_pilot_report.md",
        "hashes": output_dir / "phase_e_unified_pilot_hashes.json",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase E unified pilot outputs.")

    started = time.perf_counter()
    graphs, instances_payload, instances_hash = _load_pilot_graphs(root)
    _progress(progress, f"starting stage={calibration_stage}")
    backend_rows = _run_backend_measurements(graphs, DEFAULT_PILOT_NOISE_RATES, DEFAULT_PILOT_SHOTS)
    lhs_rows = _run_lhs_calibration(graphs, DEFAULT_PILOT_NOISE_RATES, DEFAULT_PILOT_SHOTS, lhs_candidates, lhs_designs, progress=progress)
    optimizer_rows = _run_optimizer_calibration(graphs, DEFAULT_PILOT_NOISE_RATES, DEFAULT_PILOT_SHOTS, seeds, e_cobyla_candidates, k_candidates, progress=progress)
    runtime_rows = backend_rows + [
        {
            "operation": "lhs_cell",
            "instance": row["instance"],
            "p_dep": row["p_dep"],
            "shots_total": row["shots_total"],
            "N_LHS": row["N_LHS"],
            "lhs_design_id": row["lhs_design_id"],
            "elapsed_seconds": row["elapsed_seconds"],
        }
        for row in lhs_rows
    ]
    decision = _decision(backend_rows, lhs_rows, optimizer_rows, e_cobyla_candidates, k_candidates, lhs_candidates, lhs_designs)
    summary = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_unified_depth2_pilot_technical_draft_complete",
        "pilot_version": PHASE_E_UNIFIED_PILOT_VERSION,
        "scope": calibration_stage,
        "official_execution_authorized": False,
        "pilot_instances": [graph.name for graph in graphs],
        "selected_depth2_instances": instances_payload["selected_depth2_instances"],
        "phase_e_instances_hash": instances_hash,
        "noise_rates": list(DEFAULT_PILOT_NOISE_RATES),
        "shot_budgets": list(DEFAULT_PILOT_SHOTS),
        "zne_order": DEFAULT_PILOT_ZNE_ORDER,
        "scale_factors": list(DEFAULT_PILOT_SCALE_FACTORS),
        "seeds": list(seeds),
        "e_cobyla_candidates": list(e_cobyla_candidates),
        "k_candidates": list(k_candidates),
        "lhs_candidates": list(lhs_candidates),
        "lhs_independent_designs": int(lhs_designs),
        "backend_measurement_rows": len(backend_rows),
        "lhs_calibration_rows": len(lhs_rows),
        "optimizer_calibration_rows": len(optimizer_rows),
        "calibration_decision": decision,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {key: str(path.relative_to(root)).replace("\\", "/") for key, path in paths.items()},
    }
    _write_csv(paths["backend"], backend_rows)
    _write_csv(paths["cache"], [{"status": "not_measured_in_technical_draft", "reason": "cache layer not implemented for Phase E pilot v0.1"}])
    _write_csv(paths["lhs"], lhs_rows)
    _write_csv(paths["cobyla"], optimizer_rows)
    _write_csv(paths["k"], optimizer_rows)
    _write_csv(paths["runtime"], runtime_rows)
    write_json_atomic(paths["decision"], decision, overwrite=True)
    write_json_atomic(paths["summary"], summary, overwrite=True)
    paths["report"].write_text(_markdown(summary), encoding="utf-8")
    hashes = {
        "created_at_utc": _utc_now(),
        "pilot_version": PHASE_E_UNIFIED_PILOT_VERSION,
        "hashes": {key: sha256_file(path) for key, path in paths.items() if key != "hashes"},
    }
    write_json_atomic(paths["hashes"], hashes, overwrite=True)
    summary["output_hashes"] = hashes["hashes"]
    return summary
