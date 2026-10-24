from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .circuits import GraphInstance
from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .mitiq_wrapper import deterministic_seed
from .phase_e_optimizers import SPSAConfig, make_paired_initial_thetas, spsa_minimize, spsa_minimize_checkpoints
from .phase_e_sim_backend import evaluate_custom_density_objective, richardson_zne_objective


PHASE_E_STAGE_C0_BENCHMARK_VERSION = "phase_e_stage_c0_benchmark_v0_1_draft"
DEFAULT_C0_SMOKE_INSTANCES = ("H1",)
DEFAULT_C0_SMOKE_NOISE_RATES = (0.003,)
DEFAULT_C0_SMOKE_SHOTS = (512,)
DEFAULT_C0_SMOKE_SEEDS = (42,)
DEFAULT_C0_SMOKE_T_CHECKPOINTS = (6, 12)
DEFAULT_C0_SMOKE_K_PREFIXES = (1, 2)
DEFAULT_C0_SCALE_FACTORS = (1, 3)


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
            seed=deterministic_seed("phase_e_c0_noisy", graph.name, p_dep, shots, seed, tuple(round(x, 12) for x in theta)),
        ).objective_value

    return objective


def _objective_zne(graph: GraphInstance, p_dep: float, shots: int, seed: int) -> Callable[[tuple[float, ...]], float]:
    def objective(theta: tuple[float, ...]) -> float:
        return richardson_zne_objective(
            graph,
            theta,
            p_dep=p_dep,
            scale_factors=DEFAULT_C0_SCALE_FACTORS,
            shots_total=shots,
            seed=deterministic_seed("phase_e_c0_zne", graph.name, p_dep, shots, seed, tuple(round(x, 12) for x in theta)),
        ).objective_value

    return objective


def _objective_ideal(graph: GraphInstance) -> Callable[[tuple[float, ...]], float]:
    def objective(theta: tuple[float, ...]) -> float:
        return evaluate_custom_density_objective(graph, theta, p_dep=0.0).objective_value

    return objective


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[phase-e-c0] {message}", flush=True)


def _run_checkpointed_and_separate(
    graph: GraphInstance,
    p_dep: float,
    shots: int,
    seed: int,
    t_checkpoints: tuple[int, ...],
    k_max: int,
    include_zne: bool,
    run_separate_baseline: bool,
    progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_rows: list[dict[str, Any]] = []
    separate_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    starts = make_paired_initial_thetas(
        depth=2,
        starts=int(k_max),
        seed=deterministic_seed("phase_e_c0_starts", graph.name, p_dep, shots, seed, k_max),
    )
    ideal_objective = _objective_ideal(graph)
    families = ("noisy", "zne") if include_zne else ("noisy",)
    for family in families:
        for start_id, theta0 in enumerate(starts):
            objective_seed = deterministic_seed("phase_e_c0_objective_seed", graph.name, p_dep, shots, seed, family, start_id)
            objective = _objective_noisy(graph, p_dep, shots, objective_seed) if family == "noisy" else _objective_zne(graph, p_dep, shots, objective_seed)
            opt_seed = deterministic_seed("phase_e_c0_opt", graph.name, p_dep, shots, seed, family, start_id)
            _progress(progress, f"checkpointed {graph.name} p={p_dep} shots={shots} seed={seed} family={family} start={start_id}")
            started = time.perf_counter()
            checkpointed = spsa_minimize_checkpoints(
                objective,
                theta0,
                SPSAConfig(max_evaluations=max(t_checkpoints)),
                checkpoints=t_checkpoints,
                seed=opt_seed,
            )
            checkpoint_elapsed = time.perf_counter() - started
            runtime_rows.append(
                {
                    "operation": "checkpointed_spsa",
                    "instance": graph.name,
                    "p_dep": p_dep,
                    "shots_total": shots,
                    "seed": seed,
                    "objective_family": family,
                    "start_id": start_id,
                    "max_T": max(t_checkpoints),
                    "elapsed_seconds": checkpoint_elapsed,
                }
            )
            for checkpoint in checkpointed.checkpoints:
                ideal_at_best = ideal_objective(checkpoint.best_theta)
                checkpoint_rows.append(
                    {
                        "instance": graph.name,
                        "p_dep": p_dep,
                        "shots_total": shots,
                        "seed": seed,
                        "objective_family": family,
                        "start_id": start_id,
                        "T_requested": checkpoint.requested_max_evaluations,
                        "actual_num_evaluations": checkpoint.num_evaluations,
                        "iteration": checkpoint.iteration,
                        "best_observed_value": checkpoint.best_value,
                        "current_observed_value": checkpoint.current_value,
                        "ideal_value_at_best_theta": ideal_at_best,
                        "theta_best_json": json.dumps(checkpoint.best_theta),
                        "elapsed_seconds_for_max_T": checkpoint_elapsed,
                        "spsa_budget_semantics": "checkpointed max_evaluations cap; checkpoint endpoint is latest complete SPSA state at or below requested T",
                    }
                )

            if run_separate_baseline:
                for t_budget in t_checkpoints:
                    _progress(progress, f"separate {graph.name} p={p_dep} shots={shots} seed={seed} family={family} start={start_id} T={t_budget}")
                    started = time.perf_counter()
                    separate = spsa_minimize(
                        objective,
                        theta0,
                        SPSAConfig(max_evaluations=int(t_budget)),
                        seed=opt_seed,
                    )
                    elapsed = time.perf_counter() - started
                    runtime_rows.append(
                        {
                            "operation": "separate_spsa",
                            "instance": graph.name,
                            "p_dep": p_dep,
                            "shots_total": shots,
                            "seed": seed,
                            "objective_family": family,
                            "start_id": start_id,
                            "T": t_budget,
                            "elapsed_seconds": elapsed,
                        }
                    )
                    separate_rows.append(
                        {
                            "instance": graph.name,
                            "p_dep": p_dep,
                            "shots_total": shots,
                            "seed": seed,
                            "objective_family": family,
                            "start_id": start_id,
                            "T_requested": t_budget,
                            "actual_num_evaluations": separate.num_evaluations,
                            "best_observed_value": separate.best_value,
                            "ideal_value_at_best_theta": ideal_objective(separate.best_theta),
                            "elapsed_seconds": elapsed,
                        }
                    )
    return checkpoint_rows, separate_rows, runtime_rows


def _checkpoint_factor_rows(runtime_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], dict[str, float]] = {}
    for row in runtime_rows:
        key = (
            row.get("instance"),
            row.get("p_dep"),
            row.get("shots_total"),
            row.get("seed"),
            row.get("objective_family"),
            row.get("start_id"),
        )
        bucket = grouped.setdefault(key, {"checkpointed": 0.0, "separate": 0.0})
        if row["operation"] == "checkpointed_spsa":
            bucket["checkpointed"] += float(row["elapsed_seconds"])
        elif row["operation"] == "separate_spsa":
            bucket["separate"] += float(row["elapsed_seconds"])
    rows = []
    for key, value in grouped.items():
        checkpointed = value["checkpointed"]
        separate = value["separate"]
        factor = separate / checkpointed if checkpointed > 0.0 and separate > 0.0 else math.nan
        rows.append(
            {
                "instance": key[0],
                "p_dep": key[1],
                "shots_total": key[2],
                "seed": key[3],
                "objective_family": key[4],
                "start_id": key[5],
                "checkpointed_elapsed_seconds": checkpointed,
                "separate_elapsed_seconds": separate,
                "measured_checkpoint_factor": factor,
            }
        )
    return rows


def _nested_k_rows(checkpoint_rows: list[dict[str, Any]], k_prefixes: tuple[int, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in checkpoint_rows:
        key = (
            row["instance"],
            row["p_dep"],
            row["shots_total"],
            row["seed"],
            row["objective_family"],
            row["T_requested"],
        )
        grouped.setdefault(key, []).append(row)
    out = []
    for key, rows in grouped.items():
        sorted_rows = sorted(rows, key=lambda item: int(item["start_id"]))
        for k_prefix in k_prefixes:
            eligible = [row for row in sorted_rows if int(row["start_id"]) < int(k_prefix)]
            if not eligible:
                continue
            best = min(eligible, key=lambda item: float(item["best_observed_value"]))
            out.append(
                {
                    "instance": key[0],
                    "p_dep": key[1],
                    "shots_total": key[2],
                    "seed": key[3],
                    "objective_family": key[4],
                    "T_requested": key[5],
                    "K_prefix": k_prefix,
                    "eligible_starts": len(eligible),
                    "best_start_id": best["start_id"],
                    "best_observed_value": best["best_observed_value"],
                    "ideal_value_at_best_theta": best["ideal_value_at_best_theta"],
                }
            )
    return out


def _resume_integrity_smoke(checkpoint_rows: list[dict[str, Any]]) -> dict[str, Any]:
    uninterrupted_hash = _canonical_hash(checkpoint_rows)
    resumed_hash = _canonical_hash(json.loads(json.dumps(checkpoint_rows, sort_keys=True)))
    return {
        "status": "smoke_repeat_hash_pass" if uninterrupted_hash == resumed_hash else "failed",
        "full_crash_resume_gate_complete": False,
        "interpretation": "This is a deterministic serialization smoke check only; full C0 must still run crash-and-resume.",
        "uninterrupted_hash": uninterrupted_hash,
        "resumed_hash": resumed_hash,
    }


def _storage_projection(output_dir: Path) -> dict[str, Any]:
    files = [path for path in output_dir.glob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in files)
    return {
        "status": "smoke_projection_only",
        "output_dir": str(output_dir),
        "files_counted": len(files),
        "current_output_bytes": total_bytes,
        "full_storage_gate_complete": False,
        "interpretation": "Smoke output size only; full C0 must project checkpoint, log, hash, and backup bytes before Stage C.",
    }


def run_phase_e_stage_c0_benchmark(
    root: Path = PROJECT_ROOT,
    force: bool = False,
    instances: tuple[str, ...] = DEFAULT_C0_SMOKE_INSTANCES,
    noise_rates: tuple[float, ...] = DEFAULT_C0_SMOKE_NOISE_RATES,
    shots: tuple[int, ...] = DEFAULT_C0_SMOKE_SHOTS,
    seeds: tuple[int, ...] = DEFAULT_C0_SMOKE_SEEDS,
    t_checkpoints: tuple[int, ...] = DEFAULT_C0_SMOKE_T_CHECKPOINTS,
    k_prefixes: tuple[int, ...] = DEFAULT_C0_SMOKE_K_PREFIXES,
    include_zne: bool = True,
    run_separate_baseline: bool = True,
    progress: bool = False,
) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration" / "c0_benchmark")
    paths = {
        "summary": output_dir / "phase_e_stage_c0_benchmark_summary.json",
        "checkpointed": output_dir / "checkpointed_spsa_records.csv",
        "separate": output_dir / "separate_spsa_records.csv",
        "nested_k": output_dir / "nested_k_summary.csv",
        "checkpoint_factor": output_dir / "checkpoint_factor_measurements.csv",
        "runtime": output_dir / "runtime_profile_by_phase.csv",
        "storage": output_dir / "storage_projection.json",
        "resume": output_dir / "resume_integrity_report.json",
        "hashes": output_dir / "phase_e_stage_c0_benchmark_hashes.json",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase E Stage C0 benchmark outputs.")
    if max(k_prefixes) < 1:
        raise ValueError("k_prefixes must contain positive values.")
    if max(k_prefixes) <= 0:
        raise ValueError("k_prefixes must be positive.")
    k_max = max(k_prefixes)
    t_checkpoints = tuple(sorted({int(value) for value in t_checkpoints}))

    started = time.perf_counter()
    graphs, instances_payload, instances_hash = _load_graphs(root, instances)
    checkpoint_rows: list[dict[str, Any]] = []
    separate_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for graph in graphs:
        for p_dep in noise_rates:
            for shot_budget in shots:
                for seed in seeds:
                    c_rows, s_rows, r_rows = _run_checkpointed_and_separate(
                        graph,
                        p_dep=float(p_dep),
                        shots=int(shot_budget),
                        seed=int(seed),
                        t_checkpoints=t_checkpoints,
                        k_max=k_max,
                        include_zne=include_zne,
                        run_separate_baseline=run_separate_baseline,
                        progress=progress,
                    )
                    checkpoint_rows.extend(c_rows)
                    separate_rows.extend(s_rows)
                    runtime_rows.extend(r_rows)

    checkpoint_factor_rows = _checkpoint_factor_rows(runtime_rows)
    nested_rows = _nested_k_rows(checkpoint_rows, k_prefixes)
    _write_csv(paths["checkpointed"], checkpoint_rows)
    _write_csv(paths["separate"], separate_rows)
    _write_csv(paths["nested_k"], nested_rows)
    _write_csv(paths["checkpoint_factor"], checkpoint_factor_rows)
    _write_csv(paths["runtime"], runtime_rows)
    resume_report = _resume_integrity_smoke(checkpoint_rows)
    write_json_atomic(paths["resume"], resume_report, overwrite=True)
    storage = _storage_projection(output_dir)
    write_json_atomic(paths["storage"], storage, overwrite=True)

    finite_values = []
    for row in checkpoint_rows + separate_rows + nested_rows + checkpoint_factor_rows:
        for value in row.values():
            if isinstance(value, float):
                finite_values.append(value)
    measured_factors = [float(row["measured_checkpoint_factor"]) for row in checkpoint_factor_rows if math.isfinite(float(row["measured_checkpoint_factor"]))]
    summary = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_stage_c0_benchmark_smoke_complete_not_gate",
        "benchmark_version": PHASE_E_STAGE_C0_BENCHMARK_VERSION,
        "scope": "safe_smoke_benchmark_not_protocol_scale_stage_c",
        "official_execution_authorized": False,
        "full_c0_gate_complete": False,
        "instances": list(instances),
        "phase_e_instances_hash": instances_hash,
        "selected_depth2_instances": instances_payload.get("selected_depth2_instances"),
        "noise_rates": list(noise_rates),
        "shot_budgets": list(shots),
        "seeds": list(seeds),
        "t_checkpoints": list(t_checkpoints),
        "k_prefixes": list(k_prefixes),
        "include_zne": bool(include_zne),
        "run_separate_baseline": bool(run_separate_baseline),
        "checkpointed_rows": len(checkpoint_rows),
        "separate_rows": len(separate_rows),
        "nested_k_rows": len(nested_rows),
        "checkpoint_factor_rows": len(checkpoint_factor_rows),
        "mean_measured_checkpoint_factor": float(np.mean(measured_factors)) if measured_factors else None,
        "min_measured_checkpoint_factor": float(np.min(measured_factors)) if measured_factors else None,
        "max_measured_checkpoint_factor": float(np.max(measured_factors)) if measured_factors else None,
        "all_checked_numeric_finite": all(math.isfinite(value) for value in finite_values),
        "resume_integrity_smoke_status": resume_report["status"],
        "storage_projection_status": storage["status"],
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {key: str(path.relative_to(root)).replace("\\", "/") for key, path in paths.items()},
        "limitations": [
            "Smoke defaults are intentionally small and do not complete the full C0 gate.",
            "Resume report is a deterministic serialization smoke check, not the required crash-and-resume gate.",
            "Storage projection reports smoke artifact size only.",
            "No final T, K, K_ref, or N_LHS values are selected.",
        ],
    }
    write_json_atomic(paths["summary"], summary, overwrite=True)
    hashes = {
        "created_at_utc": _utc_now(),
        "benchmark_version": PHASE_E_STAGE_C0_BENCHMARK_VERSION,
        "hashes": {key: sha256_file(path) for key, path in paths.items() if key != "hashes"},
    }
    write_json_atomic(paths["hashes"], hashes, overwrite=True)
    summary["output_hashes"] = hashes["hashes"]
    return summary
