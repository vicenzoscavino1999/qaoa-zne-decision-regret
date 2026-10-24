from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .circuits import GraphInstance
from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .mitiq_wrapper import deterministic_seed
from .phase_e_optimizers import (
    SPSAConfig,
    make_paired_initial_thetas,
    spsa_initialize_state,
    spsa_minimize,
    spsa_result_from_state,
    spsa_run_state,
)
from .phase_e_sim_backend import evaluate_custom_density_objective, richardson_zne_objective


PHASE_E_STAGE_C0_RESUME_VERSION = "phase_e_stage_c0_resume_v0_1_draft"
DEFAULT_C0_RESUME_SMOKE_INSTANCES = ("H1",)
DEFAULT_C0_RESUME_SMOKE_NOISE_RATES = (0.003,)
DEFAULT_C0_RESUME_SMOKE_SHOTS = (512,)
DEFAULT_C0_RESUME_SMOKE_SEEDS = (42,)
DEFAULT_C0_RESUME_T_BUDGET = 12
DEFAULT_C0_RESUME_SPLIT_BUDGET = 4
DEFAULT_C0_RESUME_STARTS = 1
DEFAULT_C0_RESUME_MAX_CASES = 2
DEFAULT_C0_RESUME_SCALE_FACTORS = (1, 3)


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


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
            seed=deterministic_seed("phase_e_c0_resume_noisy", graph.name, p_dep, shots, seed, tuple(round(x, 12) for x in theta)),
        ).objective_value

    return objective


def _objective_zne(graph: GraphInstance, p_dep: float, shots: int, seed: int) -> Callable[[tuple[float, ...]], float]:
    def objective(theta: tuple[float, ...]) -> float:
        return richardson_zne_objective(
            graph,
            theta,
            p_dep=p_dep,
            scale_factors=DEFAULT_C0_RESUME_SCALE_FACTORS,
            shots_total=shots,
            seed=deterministic_seed("phase_e_c0_resume_zne", graph.name, p_dep, shots, seed, tuple(round(x, 12) for x in theta)),
        ).objective_value

    return objective


def _objective_ideal(graph: GraphInstance) -> Callable[[tuple[float, ...]], float]:
    def objective(theta: tuple[float, ...]) -> float:
        return evaluate_custom_density_objective(graph, theta, p_dep=0.0).objective_value

    return objective


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[phase-e-c0-resume] {message}", flush=True)


def _case_id(graph: GraphInstance, p_dep: float, shots: int, seed: int, objective_family: str, t_budget: int) -> str:
    return f"{graph.name}|p={p_dep}|shots={shots}|seed={seed}|family={objective_family}|T={t_budget}"


def _endpoint_payload(result: Any, ideal_value_at_best_theta: float) -> dict[str, Any]:
    return {
        "status": result.status,
        "initial_theta": [float(x) for x in result.initial_theta],
        "final_theta": [float(x) for x in result.final_theta],
        "best_theta": [float(x) for x in result.best_theta],
        "final_value": float(result.final_value),
        "best_value": float(result.best_value),
        "ideal_value_at_best_theta": float(ideal_value_at_best_theta),
        "num_evaluations": int(result.num_evaluations),
        "runner_version": result.runner_version,
    }


def _build_cases(
    graphs: list[GraphInstance],
    noise_rates: tuple[float, ...],
    shots: tuple[int, ...],
    seeds: tuple[int, ...],
    include_zne: bool,
    max_cases: int | None,
) -> list[tuple[GraphInstance, float, int, int, str]]:
    families = ("noisy", "zne") if include_zne else ("noisy",)
    cases = [
        (graph, float(p_dep), int(shot_budget), int(seed), family)
        for graph in graphs
        for p_dep in noise_rates
        for shot_budget in shots
        for seed in seeds
        for family in families
    ]
    if max_cases is not None:
        return cases[: int(max_cases)]
    return cases


def _run_resume_case(
    graph: GraphInstance,
    p_dep: float,
    shots: int,
    seed: int,
    objective_family: str,
    t_budget: int,
    split_budget: int,
    starts_per_case: int,
    checkpoint_dir: Path,
    case_index: int,
    progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    case = _case_id(graph, p_dep, shots, seed, objective_family, t_budget)
    _progress(progress, f"resume case {case} K={starts_per_case} split={split_budget}")
    objective_seed = deterministic_seed("phase_e_c0_resume_fixed_objective", graph.name, p_dep, shots, seed, objective_family, t_budget)
    objective = _objective_noisy(graph, p_dep, shots, objective_seed) if objective_family == "noisy" else _objective_zne(graph, p_dep, shots, objective_seed)
    ideal = _objective_ideal(graph)
    starts = make_paired_initial_thetas(
        depth=2,
        starts=int(starts_per_case),
        seed=deterministic_seed("phase_e_c0_resume_starts", case, starts_per_case),
    )
    config = SPSAConfig(max_evaluations=int(t_budget))
    trajectory_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for start_id, theta0 in enumerate(starts):
        opt_seed = deterministic_seed("phase_e_c0_resume_opt", case, start_id)

        started = time.perf_counter()
        uninterrupted = spsa_minimize(objective, theta0, config, seed=opt_seed)
        uninterrupted_elapsed = time.perf_counter() - started
        uninterrupted_payload = _endpoint_payload(uninterrupted, ideal(uninterrupted.best_theta))
        uninterrupted_hash = _canonical_hash(uninterrupted_payload)

        started = time.perf_counter()
        state = spsa_initialize_state(objective, theta0, seed=opt_seed)
        partial = spsa_run_state(objective, state, config, stop_at_evaluations=int(split_budget))
        checkpoint_path = checkpoint_dir / f"resume_state_case{case_index:03d}_start{start_id:03d}.json"
        write_json_atomic(checkpoint_path, partial, overwrite=True)
        loaded = _read_json(checkpoint_path)
        resumed_state = spsa_run_state(objective, loaded, config, stop_at_evaluations=int(t_budget))
        resumed = spsa_result_from_state(resumed_state, optimizer="spsa")
        resumed_elapsed = time.perf_counter() - started
        resumed_payload = _endpoint_payload(resumed, ideal(resumed.best_theta))
        resumed_hash = _canonical_hash(resumed_payload)

        trajectory_rows.append(
            {
                "case_id": case,
                "instance": graph.name,
                "p_dep": p_dep,
                "shots_total": shots,
                "seed": seed,
                "objective_family": objective_family,
                "T_requested": t_budget,
                "split_budget": split_budget,
                "start_id": start_id,
                "uninterrupted_hash": uninterrupted_hash,
                "resumed_hash": resumed_hash,
                "hash_identical": uninterrupted_hash == resumed_hash,
                "uninterrupted_best_value": uninterrupted.best_value,
                "resumed_best_value": resumed.best_value,
                "uninterrupted_final_value": uninterrupted.final_value,
                "resumed_final_value": resumed.final_value,
                "uninterrupted_num_evaluations": uninterrupted.num_evaluations,
                "resumed_num_evaluations": resumed.num_evaluations,
                "partial_num_evaluations": partial["num_evaluations"],
                "partial_iteration": partial["iteration"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "uninterrupted_elapsed_seconds": uninterrupted_elapsed,
                "resume_path_elapsed_seconds": resumed_elapsed,
            }
        )
        runtime_rows.extend(
            [
                {
                    "case_id": case,
                    "instance": graph.name,
                    "objective_family": objective_family,
                    "start_id": start_id,
                    "T_requested": t_budget,
                    "split_budget": split_budget,
                    "operation": "uninterrupted_spsa",
                    "elapsed_seconds": uninterrupted_elapsed,
                },
                {
                    "case_id": case,
                    "instance": graph.name,
                    "objective_family": objective_family,
                    "start_id": start_id,
                    "T_requested": t_budget,
                    "split_budget": split_budget,
                    "operation": "checkpoint_write_read_resume_spsa",
                    "elapsed_seconds": resumed_elapsed,
                },
            ]
        )

    best_uninterrupted = min(trajectory_rows, key=lambda row: float(row["uninterrupted_best_value"]))
    best_resumed = min(trajectory_rows, key=lambda row: float(row["resumed_best_value"]))
    case_record = {
        "case_id": case,
        "instance": graph.name,
        "p_dep": p_dep,
        "shots_total": shots,
        "seed": seed,
        "objective_family": objective_family,
        "T_requested": t_budget,
        "split_budget": split_budget,
        "starts_per_case": starts_per_case,
        "trajectory_count": len(trajectory_rows),
        "all_trajectory_hashes_identical": all(bool(row["hash_identical"]) for row in trajectory_rows),
        "best_uninterrupted_start_id": best_uninterrupted["start_id"],
        "best_resumed_start_id": best_resumed["start_id"],
        "best_start_identical": best_uninterrupted["start_id"] == best_resumed["start_id"],
        "best_uninterrupted_value": best_uninterrupted["uninterrupted_best_value"],
        "best_resumed_value": best_resumed["resumed_best_value"],
        "best_hash_identical": best_uninterrupted["uninterrupted_hash"] == best_resumed["resumed_hash"],
    }
    return trajectory_rows, case_record, runtime_rows


def run_phase_e_stage_c0_resume(
    root: Path = PROJECT_ROOT,
    force: bool = False,
    instances: tuple[str, ...] = DEFAULT_C0_RESUME_SMOKE_INSTANCES,
    noise_rates: tuple[float, ...] = DEFAULT_C0_RESUME_SMOKE_NOISE_RATES,
    shots: tuple[int, ...] = DEFAULT_C0_RESUME_SMOKE_SHOTS,
    seeds: tuple[int, ...] = DEFAULT_C0_RESUME_SMOKE_SEEDS,
    t_budget: int = DEFAULT_C0_RESUME_T_BUDGET,
    split_budget: int = DEFAULT_C0_RESUME_SPLIT_BUDGET,
    starts_per_case: int = DEFAULT_C0_RESUME_STARTS,
    include_zne: bool = True,
    max_cases: int | None = DEFAULT_C0_RESUME_MAX_CASES,
    progress: bool = False,
) -> dict[str, Any]:
    if int(t_budget) < 4:
        raise ValueError("t_budget must allow at least one full SPSA iteration.")
    if int(split_budget) < 1 or int(split_budget) >= int(t_budget):
        raise ValueError("split_budget must be positive and lower than t_budget.")
    if int(starts_per_case) < 1:
        raise ValueError("starts_per_case must be positive.")

    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration" / "c0_resume")
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    paths = {
        "summary": output_dir / "phase_e_stage_c0_resume_summary.json",
        "trajectories": output_dir / "resume_integrity_trajectory_records.csv",
        "cases": output_dir / "resume_integrity_case_records.csv",
        "report": output_dir / "resume_integrity_report.json",
        "runtime": output_dir / "resume_integrity_runtime.csv",
        "hashes": output_dir / "phase_e_stage_c0_resume_hashes.json",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase E C0 resume outputs.")

    started = time.perf_counter()
    graphs, instances_payload, instances_hash = _load_graphs(root, instances)
    cases = _build_cases(graphs, noise_rates, shots, seeds, include_zne, max_cases)
    trajectory_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for case_index, (graph, p_dep, shot_budget, seed, family) in enumerate(cases):
        t_rows, c_row, r_rows = _run_resume_case(
            graph,
            p_dep,
            shot_budget,
            seed,
            family,
            int(t_budget),
            int(split_budget),
            int(starts_per_case),
            checkpoint_dir,
            case_index,
            progress,
        )
        trajectory_rows.extend(t_rows)
        case_rows.append(c_row)
        runtime_rows.extend(r_rows)

    _write_csv(paths["trajectories"], trajectory_rows)
    _write_csv(paths["cases"], case_rows)
    _write_csv(paths["runtime"], runtime_rows)
    all_trajectory_hashes_identical = all(bool(row["hash_identical"]) for row in trajectory_rows)
    all_case_best_hashes_identical = all(bool(row["best_hash_identical"]) for row in case_rows)
    report = {
        "created_at_utc": _utc_now(),
        "resume_version": PHASE_E_STAGE_C0_RESUME_VERSION,
        "trajectory_count": len(trajectory_rows),
        "case_count": len(case_rows),
        "all_trajectory_hashes_identical": all_trajectory_hashes_identical,
        "all_case_best_hashes_identical": all_case_best_hashes_identical,
        "status": "pass" if all_trajectory_hashes_identical and all_case_best_hashes_identical else "fail",
        "interpretation": "Compares canonical endpoint hashes from uninterrupted SPSA against checkpoint-write/read-resume SPSA.",
    }
    write_json_atomic(paths["report"], report, overwrite=True)

    finite_values: list[float] = []
    for row in trajectory_rows + case_rows + runtime_rows:
        for value in row.values():
            if isinstance(value, float):
                finite_values.append(value)

    full_resume_gate_complete = (
        len(case_rows) >= 2
        and int(t_budget) >= 250
        and int(starts_per_case) >= 20
        and all_trajectory_hashes_identical
        and all_case_best_hashes_identical
    )
    summary = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_stage_c0_resume_smoke_complete_not_gate" if not full_resume_gate_complete else "phase_e_stage_c0_resume_gate_passed",
        "resume_version": PHASE_E_STAGE_C0_RESUME_VERSION,
        "scope": "safe_smoke_resume_not_full_c0_gate" if not full_resume_gate_complete else "full_c0_resume_integrity_gate",
        "official_execution_authorized": False,
        "full_resume_gate_complete": full_resume_gate_complete,
        "resume_integrity_status": report["status"],
        "instances": list(instances),
        "phase_e_instances_hash": instances_hash,
        "selected_depth2_instances": instances_payload.get("selected_depth2_instances"),
        "noise_rates": list(noise_rates),
        "shot_budgets": list(shots),
        "seeds": list(seeds),
        "include_zne": bool(include_zne),
        "max_cases": max_cases,
        "case_count": len(case_rows),
        "trajectory_count": len(trajectory_rows),
        "T_requested": int(t_budget),
        "split_budget": int(split_budget),
        "starts_per_case": int(starts_per_case),
        "all_trajectory_hashes_identical": all_trajectory_hashes_identical,
        "all_case_best_hashes_identical": all_case_best_hashes_identical,
        "all_checked_numeric_finite": all(value == value and abs(value) != float("inf") for value in finite_values),
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {key: str(path.relative_to(root)).replace("\\", "/") for key, path in paths.items()},
        "checkpoint_dir": str(checkpoint_dir.relative_to(root)).replace("\\", "/"),
        "limitations": [
            "Smoke defaults are intentionally small and do not complete the full resume integrity gate.",
            "Full C0 resume gate requires protocol-scale minimum T and K on the fixed C0 subset.",
            "No final T, K, K_ref, or N_LHS values are selected.",
        ],
    }
    write_json_atomic(paths["summary"], summary, overwrite=True)
    checkpoint_hashes = {str(path.relative_to(root)).replace("\\", "/"): sha256_file(path) for path in sorted(checkpoint_dir.glob("*.json"))}
    hashes = {
        "created_at_utc": _utc_now(),
        "resume_version": PHASE_E_STAGE_C0_RESUME_VERSION,
        "hashes": {key: sha256_file(path) for key, path in paths.items() if key != "hashes"},
        "checkpoint_hashes": checkpoint_hashes,
    }
    write_json_atomic(paths["hashes"], hashes, overwrite=True)
    summary["output_hashes"] = hashes["hashes"]
    return summary
