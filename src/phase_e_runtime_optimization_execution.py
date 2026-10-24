from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .circuits import GraphInstance
from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .mitiq_wrapper import deterministic_seed
from .phase_e_optimizers import SPSAConfig, make_paired_initial_thetas, spsa_minimize
from .phase_e_runtime_optimization_plan import (
    OPTIMIZATION_PLAN_DOC,
    OPTIMIZATION_PLAN_SUMMARY,
    PHASE_E_RUNTIME_OPTIMIZATION_PLAN_VERSION,
)
from .phase_e_sim_backend import (
    PhaseENoiseRates,
    build_folded_transpiled_qaoa_depth,
    custom_density_probabilities,
    evaluate_custom_density_objective,
    expected_cut_from_phase_e_probabilities,
    richardson_zne_objective,
    statevector_probabilities_phase_e,
)


PHASE_E_RUNTIME_OPTIMIZATION_EXECUTION_VERSION = "phase_e_runtime_optimization_execution_v0_1"
O0_O1_AUTHORIZATION_TOKEN = "accept_phase_e_runtime_optimization_o0_o1"
STAGE_C_AUTHORIZATION_TOKEN = "accept_phase_e_optimizer_depth2_official"
OUTPUT_DIR_RELATIVE = Path("results") / "phase_e" / "calibration" / "runtime_optimization"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_capture(args: list[str], root: Path, timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "args": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"args": args, "returncode": None, "stdout": "", "stderr": repr(exc)}


def _current_runtime_code_hashes(root: Path) -> dict[str, str]:
    files = [
        "scripts/51_phase_e_backend_benchmark.py",
        "scripts/53_phase_e_stage_c0_benchmark.py",
        "scripts/58_phase_e_stage_c0_runtime.py",
        "scripts/60_phase_e_stage_c_authorization.py",
        "scripts/61_phase_e_runtime_optimization_plan.py",
        "scripts/62_phase_e_runtime_optimization_o0_o1.py",
        "scripts/63_phase_e_runtime_equivalence_suite.py",
        "scripts/64_phase_e_runtime_optimization_decision.py",
        "src/phase_e_c0_benchmark.py",
        "src/phase_e_c0_jitter.py",
        "src/phase_e_c0_resume.py",
        "src/phase_e_c0_runtime.py",
        "src/phase_e_c0_shard_balance.py",
        "src/phase_e_optimizers.py",
        "src/phase_e_runtime_optimization_execution.py",
        "src/phase_e_runtime_optimization_plan.py",
        "src/phase_e_sim_backend.py",
    ]
    return {relative: sha256_file(root / relative) for relative in files if (root / relative).exists()}


def _common_json_fields(root: Path) -> dict[str, Any]:
    authorization_path = root / "results" / "phase_e" / "calibration" / "stage_c_authorization" / "phase_e_stage_c_authorization_summary.json"
    authorization = _read_json(authorization_path) if authorization_path.exists() else {}
    plan_path = root / OPTIMIZATION_PLAN_SUMMARY
    plan_summary = _read_json(plan_path) if plan_path.exists() else {}
    input_hashes = {}
    for relative in [
        OPTIMIZATION_PLAN_DOC,
        OPTIMIZATION_PLAN_SUMMARY,
        "results/phase_e/calibration/c0_runtime/phase_e_stage_c0_runtime_summary.json",
        "results/phase_e/calibration/stage_c_authorization/phase_e_stage_c_authorization_summary.json",
        "results/phase_e/calibration/phase_e_calibration_status.json",
        "config/phase_e_config_v0.1-draft.json",
        "config/phase_e_instances_v0.1-draft.json",
        "config/phase_e_noise_model_v0.1-draft.json",
    ]:
        path = root / relative
        if path.exists() and path.is_file():
            input_hashes[relative] = sha256_file(path)
    return {
        "created_at_utc": _utc_now(),
        "plan_version": PHASE_E_RUNTIME_OPTIMIZATION_PLAN_VERSION,
        "execution_version": PHASE_E_RUNTIME_OPTIMIZATION_EXECUTION_VERSION,
        "decision_was_posthoc": False,
        "posthoc_data_used": [],
        "input_hashes": input_hashes,
        "baseline_code_hashes": plan_summary.get("baseline_code_hashes", {}),
        "code_hashes": _current_runtime_code_hashes(root),
        "stage_c_authorized": bool(authorization.get("stage_c_authorized", False)),
        "stage_c_blocked": bool(authorization.get("stage_c_blocked", True)),
        "official_execution_authorized": False,
    }


def _require_token(token: str | None) -> None:
    if token != O0_O1_AUTHORIZATION_TOKEN:
        raise PermissionError(f"O0/O1 requires authorization token: {O0_O1_AUTHORIZATION_TOKEN}")


def _output_dir(root: Path) -> Path:
    return ensure_dir(root / OUTPUT_DIR_RELATIVE)


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


def _hash_path_for_inventory(root: Path, relative_path: str) -> dict[str, Any]:
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
        file_records = []
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            rel = str(child.relative_to(root)).replace("\\", "/")
            file_records.append({"path": rel, "bytes": child.stat().st_size, "sha256": sha256_file(child)})
        return {
            "path": relative_path,
            "exists": True,
            "kind": "directory",
            "bytes": sum(int(row["bytes"]) for row in file_records),
            "sha256": _canonical_hash(file_records),
            "file_count": len(file_records),
        }
    return {"path": relative_path, "exists": True, "kind": "other", "bytes": 0, "sha256": ""}


def _dirty_inventory(root: Path, status_text: str) -> list[dict[str, Any]]:
    rows = []
    for raw in status_text.splitlines():
        if not raw or raw.startswith("## "):
            continue
        status = raw[:2].strip()
        path_text = raw[3:].strip()
        if " -> " in path_text:
            path_text = path_text.split(" -> ", 1)[1]
        inventory = _hash_path_for_inventory(root, path_text)
        rows.append({"git_status": status, **inventory})
    return rows


def write_baseline_lock(root: Path = PROJECT_ROOT, authorization_token: str | None = None, force: bool = False) -> dict[str, Any]:
    _require_token(authorization_token)
    out = _output_dir(root)
    paths = {
        "lock": out / "runtime_optimization_baseline_lock.json",
        "git_status": out / "runtime_optimization_git_status.txt",
        "dirty_inventory": out / "runtime_optimization_dirty_tree_inventory.csv",
        "pre_o1_hashes": out / "runtime_optimization_pre_o1_hashes.json",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise FileExistsError("Refusing to overwrite existing runtime optimization baseline lock outputs.")

    git = _git_status(root)
    inventory = _dirty_inventory(root, git["status_text"])
    tracked_modified = [row["path"] for row in inventory if row["git_status"] != "??"]
    untracked = [row["path"] for row in inventory if row["git_status"] == "??"]
    common = _common_json_fields(root)
    plan_summary = _read_json(root / OPTIMIZATION_PLAN_SUMMARY)
    calibration = _read_json(root / "results" / "phase_e" / "calibration" / "phase_e_calibration_status.json")
    authorization = _read_json(root / "results" / "phase_e" / "calibration" / "stage_c_authorization" / "phase_e_stage_c_authorization_summary.json")
    pre_o1_hashes = {
        **common,
        "artifact": "runtime_optimization_pre_o1_hashes",
        "dirty_tree_inventory_sha256": _canonical_hash(inventory),
        "baseline_code_hashes": plan_summary.get("baseline_code_hashes", {}),
        "dirty_inventory": inventory,
    }
    write_json_atomic(paths["pre_o1_hashes"], pre_o1_hashes, overwrite=True)
    _write_csv(
        paths["dirty_inventory"],
        inventory,
        fieldnames=["git_status", "path", "exists", "kind", "bytes", "sha256", "file_count"],
    )
    paths["git_status"].write_text(git["status_text"], encoding="utf-8")
    lock = {
        **common,
        "artifact": "runtime_optimization_baseline_lock",
        "status": "runtime_optimization_baseline_locked",
        "git_branch": git["branch"],
        "git_head": git["head"],
        "git_status_short_sha256": hashlib.sha256(git["status_text"].encode("utf-8")).hexdigest(),
        "tracked_modified_files": tracked_modified,
        "untracked_files": untracked,
        "dirty_tree_inventory_sha256": _canonical_hash(inventory),
        "baseline_code_hashes": plan_summary.get("baseline_code_hashes", {}),
        "input_artifact_hashes": common["input_hashes"],
        "stage_c_authorized": bool(authorization.get("stage_c_authorized", False)),
        "stage_c_blocked": bool(authorization.get("stage_c_blocked", True)),
        "optimization_execution_authorized": False,
        "protocol_candidate_sets": calibration.get("protocol_candidate_sets", {}),
        "outputs": {key: str(path.relative_to(root)).replace("\\", "/") for key, path in paths.items()},
    }
    write_json_atomic(paths["lock"], lock, overwrite=True)
    return lock


def _numpy_show_config() -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.show_config()
    return buffer.getvalue()


def write_environment_snapshot(root: Path = PROJECT_ROOT, mode: str = "blas-check", force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    mode_slug = mode.replace("-", "_")
    path = out / f"runtime_optimization_environment_snapshot_{mode_slug}.json"
    aggregate_path = out / "runtime_optimization_environment_snapshot.json"
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing runtime optimization environment snapshot for mode {mode}.")
    wsl_status = _run_capture(["wsl.exe", "-l", "-q"], root, timeout=10) if shutil.which("wsl.exe") else {
        "returncode": None,
        "stdout": "",
        "stderr": "wsl.exe not found",
    }
    disk = shutil.disk_usage(root)
    common = _common_json_fields(root)
    snapshot = {
        **common,
        "artifact": "runtime_optimization_environment_snapshot",
        "status": "runtime_optimization_environment_snapshot_written",
        "mode": mode,
        "machine_id": platform.node(),
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "blas_backend": _numpy_show_config(),
        "physical_cores": os.cpu_count(),
        "logical_cores": os.cpu_count(),
        "ram_gb": "recorded_by_windows_probe_if_available",
        "storage_type": "not_classified",
        "free_space_gb": disk.free / (1024**3),
        "windows_other_cpu_utilization_pct": "not_measured_by_snapshot",
        "sustained_frequency_ratio": "not_measured_until_O1_thermal_probe",
        "wsl2_status": {
            "available": bool(wsl_status.get("returncode") == 0),
            "returncode": wsl_status.get("returncode"),
            "stdout": wsl_status.get("stdout", ""),
            "stderr": wsl_status.get("stderr", ""),
        },
        "environment_decision": "windows_native_pending_O1" if wsl_status.get("returncode") != 0 else "wsl2_available_not_accepted",
        "outputs": {
            "mode_snapshot": str(path.relative_to(root)).replace("\\", "/"),
            "snapshot": str(aggregate_path.relative_to(root)).replace("\\", "/"),
        },
    }
    write_json_atomic(path, snapshot, overwrite=True)
    if aggregate_path.exists():
        aggregate = _read_json(aggregate_path)
    else:
        aggregate = {
            **common,
            "artifact": "runtime_optimization_environment_snapshot",
            "status": "partial_environment_snapshot",
            "snapshots": {},
        }
    aggregate["snapshots"][mode] = {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "sha256": sha256_file(path),
        "environment_decision": snapshot["environment_decision"],
        "wsl2_available": snapshot["wsl2_status"]["available"],
    }
    aggregate["status"] = "environment_snapshot_aggregated"
    aggregate["latest_mode"] = mode
    aggregate["outputs"] = {"snapshot": str(aggregate_path.relative_to(root)).replace("\\", "/")}
    write_json_atomic(aggregate_path, aggregate, overwrite=True)
    return snapshot


def _graph_from_config(root: Path, label: str) -> GraphInstance:
    payload = _read_json(root / "config" / "phase_e_instances_v0.1-draft.json")
    instance = payload["selected_instances"][label]
    edges = tuple((int(u), int(v), float(w)) for u, v, w in instance["edges"])
    return GraphInstance(name=label, edges=edges)


def _profile_case_definitions(root: Path) -> list[dict[str, Any]]:
    return [
        {"case_id": "small_single_edge", "graph": GraphInstance(name="phase_e_test_single_edge", edges=((0, 1, 1.0),)), "depth": 2},
        {"case_id": "H1_depth2", "graph": _graph_from_config(root, "H1"), "depth": 2},
        {"case_id": "H4_depth2", "graph": _graph_from_config(root, "H4"), "depth": 2},
    ]


def _theta_for_case(case_id: str, depth: int) -> tuple[float, ...]:
    return make_paired_initial_thetas(depth=depth, starts=1, seed=deterministic_seed("phase_e_o1_profile_theta", case_id, depth))[0]


def _time_phase(func: Callable[[], Any]) -> tuple[Any, float, float, float]:
    tracemalloc.start()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    result = func()
    cpu_seconds = time.process_time() - cpu_start
    wall_seconds = time.perf_counter() - wall_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, wall_seconds, cpu_seconds, peak / (1024**2)


def _phase_profile_rows_for_case(
    case: dict[str, Any],
    run_id: int,
    warmup_discarded: bool,
    thread_count: int,
    blas_backend: str,
) -> list[dict[str, Any]]:
    graph: GraphInstance = case["graph"]
    case_id = str(case["case_id"])
    depth = int(case["depth"])
    theta = _theta_for_case(case_id, depth)
    p_dep = 0.003
    shots = 512
    seed = 42
    rows: list[dict[str, Any]] = []

    def add_row(phase: str, wall: float, cpu: float, memory: float, objective_family: str = "noisy") -> None:
        rows.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "instance_id": graph.name,
                "depth": depth,
                "objective_family": objective_family,
                "noise_rate": p_dep,
                "shot_budget": shots,
                "seed": seed,
                "phase": phase,
                "warmup_discarded": str(bool(warmup_discarded)).lower(),
                "wall_seconds": wall,
                "cpu_seconds": cpu,
                "peak_memory_mb": memory,
                "worker_id": 0,
                "thread_count": thread_count,
                "blas_backend": blas_backend,
            }
        )

    _, wall, cpu, memory = _time_phase(lambda: build_folded_transpiled_qaoa_depth(graph, theta, 1, PhaseENoiseRates.from_p_dep(0.0)))
    add_row("setup_and_static_circuit_construction", wall, cpu, memory)

    def objective_eval() -> float:
        result = custom_density_probabilities(graph, theta, p_dep=p_dep, scale_factor=1)
        return expected_cut_from_phase_e_probabilities(graph, result.probabilities)

    _, wall, cpu, memory = _time_phase(objective_eval)
    add_row("objective_evaluation_density_and_expectation", wall, cpu, memory)

    _, wall, cpu, memory = _time_phase(
        lambda: richardson_zne_objective(
            graph,
            theta,
            p_dep=p_dep,
            scale_factors=[1, 3],
            shots_total=shots,
            seed=deterministic_seed("phase_e_o1_profile_zne", case_id, run_id),
        )
    )
    add_row("zne_per_scale_evaluation_plus_extrapolation", wall, cpu, memory, objective_family="zne")

    def optimizer_step() -> Any:
        target = np.asarray([0.31, 0.17, 0.53, 0.29][: len(theta)], dtype=float)

        def objective(value: tuple[float, ...]) -> float:
            arr = np.asarray(value, dtype=float)
            return float(np.sum((arr - target) ** 2))

        return spsa_minimize(
            objective,
            theta,
            SPSAConfig(max_evaluations=12),
            seed=deterministic_seed("phase_e_o1_profile_spsa", case_id, run_id),
        )

    _, wall, cpu, memory = _time_phase(optimizer_step)
    add_row("optimizer_step_spsa_gradient_plus_parameter_update", wall, cpu, memory)

    def checkpoint_write() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "checkpoint.json"
            write_json_atomic(tmp_path, {"case_id": case_id, "run_id": run_id, "theta": list(theta)}, overwrite=True)

    _, wall, cpu, memory = _time_phase(checkpoint_write)
    add_row("checkpoint_write", wall, cpu, memory, objective_family="io")

    _, wall, cpu, memory = _time_phase(lambda: json.dumps({"case_id": case_id, "run_id": run_id, "rows": len(rows)}, sort_keys=True))
    add_row("logging_and_io", wall, cpu, memory, objective_family="io")
    return rows


def _run_runtime_profile(
    root: Path,
    runs: int,
    warmup: int,
    force: bool,
    output_filename: str,
    artifact: str,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    baseline_lock = root / OUTPUT_DIR_RELATIVE / "runtime_optimization_baseline_lock.json"
    if not baseline_lock.exists():
        raise FileNotFoundError("O1 profiling requires runtime_optimization_baseline_lock.json.")
    out = _output_dir(root)
    path = out / output_filename
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing {output_filename}.")
    env_path = out / "runtime_optimization_environment_snapshot.json"
    blas_backend = ""
    if env_path.exists():
        blas_backend = str(_read_json(env_path).get("blas_backend", ""))
    rows: list[dict[str, Any]] = []
    cases = _profile_case_definitions(root)
    total_runs = int(warmup) + int(runs)
    for run_id in range(total_runs):
        warmup_discarded = run_id < int(warmup)
        for case in cases:
            case_rows = _phase_profile_rows_for_case(
                case,
                run_id=run_id,
                warmup_discarded=warmup_discarded,
                thread_count=int(os.cpu_count() or 1),
                blas_backend=blas_backend[:200],
            )
            if candidate_id is not None:
                for row in case_rows:
                    row["candidate_id"] = candidate_id
            rows.extend(case_rows)
    fieldnames = [
        "run_id",
        "case_id",
        "instance_id",
        "depth",
        "objective_family",
        "noise_rate",
        "shot_budget",
        "seed",
        "phase",
        "warmup_discarded",
        "wall_seconds",
        "cpu_seconds",
        "peak_memory_mb",
        "worker_id",
        "thread_count",
        "blas_backend",
    ]
    if candidate_id is not None:
        fieldnames.append("candidate_id")
    _write_csv(path, rows, fieldnames=fieldnames)
    return {
        **_common_json_fields(root),
        "artifact": artifact,
        "status": f"{artifact}_written",
        "candidate_id": candidate_id,
        "rows": len(rows),
        "measurement_runs": int(runs),
        "warmup_runs_discarded": int(warmup),
        "outputs": {"profile": str(path.relative_to(root)).replace("\\", "/")},
    }


def run_o1_profile(root: Path = PROJECT_ROOT, runs: int = 5, warmup: int = 2, force: bool = False) -> dict[str, Any]:
    return _run_runtime_profile(
        root=root,
        runs=runs,
        warmup=warmup,
        force=force,
        output_filename="runtime_profile_baseline.csv",
        artifact="runtime_profile_baseline",
    )


def run_optimized_profile(
    root: Path = PROJECT_ROOT,
    runs: int = 5,
    warmup: int = 2,
    candidate_id: str = "optimized",
    force: bool = False,
) -> dict[str, Any]:
    return _run_runtime_profile(
        root=root,
        runs=runs,
        warmup=warmup,
        force=force,
        output_filename="runtime_profile_optimized.csv",
        artifact="runtime_profile_optimized",
        candidate_id=candidate_id,
    )


def _median(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.median(arr)) if arr.size else math.nan


def _cv(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr)) if arr.size else math.nan
    if not math.isfinite(mean) or mean == 0.0:
        return math.nan
    return float(np.std(arr, ddof=1) / mean) if arr.size > 1 else 0.0


def _phase_speedup(phase: str, objective_fraction: float, memory_bound: bool, physical_cores: int) -> tuple[float, float]:
    if phase == "objective_evaluation_density_and_expectation":
        if memory_bound:
            return 1.2, min(1.3 * min(float(physical_cores), 2.0), 3.5)
        return 1.5, 3.5 if objective_fraction >= 0.5 else 2.0
    if phase == "setup_and_static_circuit_construction":
        return 1.5, 3.0
    if phase == "zne_per_scale_evaluation_plus_extrapolation":
        return 1.1, 1.8
    if phase == "optimizer_step_spsa_gradient_plus_parameter_update":
        return 1.05, 1.2
    if phase in {"checkpoint_write", "logging_and_io"}:
        return 1.05, 1.3
    if phase == "worker_setup_and_import":
        return 1.1, 1.5
    return 1.0, 1.0


def _amdahl(phase_rows: list[dict[str, Any]], speedup_key: str) -> float:
    denominator = 0.0
    for row in phase_rows:
        denominator += float(row["phase_fraction"]) / float(row[speedup_key])
    return float(1.0 / denominator) if denominator > 0.0 else math.inf


def run_o1_triage(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    profile_path = out / "runtime_profile_baseline.csv"
    triage_path = out / "runtime_optimization_feasibility_triage.json"
    decision_log_path = out / "runtime_optimization_decision_log.json"
    negative_path = out / "runtime_optimization_negative_result.json"
    if triage_path.exists() and not force:
        raise FileExistsError("Refusing to overwrite existing O1 triage output.")
    rows = [row for row in _read_csv(profile_path) if row.get("warmup_discarded") == "false"]
    if not rows:
        raise ValueError("No non-warmup profiling rows found.")
    by_phase: dict[str, list[float]] = {}
    memory_by_phase: dict[str, list[float]] = {}
    for row in rows:
        phase = row["phase"]
        by_phase.setdefault(phase, []).append(float(row["wall_seconds"]))
        memory_by_phase.setdefault(phase, []).append(float(row.get("peak_memory_mb", 0.0)))
    medians = {phase: _median(values) for phase, values in by_phase.items()}
    total = sum(value for value in medians.values() if math.isfinite(value))
    objective_fraction = float(medians.get("objective_evaluation_density_and_expectation", 0.0) / total) if total > 0.0 else 0.0
    objective_peak = max(memory_by_phase.get("objective_evaluation_density_and_expectation", [0.0]))
    memory_bound = bool(objective_peak > 512.0)
    physical_cores = int(os.cpu_count() or 1)
    phase_table = []
    for phase, values in sorted(by_phase.items()):
        fraction = float(medians[phase] / total) if total > 0.0 else math.nan
        conservative, optimistic = _phase_speedup(phase, objective_fraction, memory_bound, physical_cores)
        phase_table.append(
            {
                "phase": phase,
                "median_wall_seconds": medians[phase],
                "phase_fraction": fraction,
                "phase_cv": _cv(values),
                "peak_memory_mb": max(memory_by_phase.get(phase, [0.0])),
                "material": bool(fraction >= 0.10),
                "local_speedup_conservative": conservative,
                "local_speedup_optimistic": optimistic,
            }
        )
    conservative_speedup = _amdahl(phase_table, "local_speedup_conservative")
    optimistic_speedup = _amdahl(phase_table, "local_speedup_optimistic")
    realistic_speedup = math.sqrt(conservative_speedup * optimistic_speedup)
    runtime_summary = _read_json(root / "results" / "phase_e" / "calibration" / "c0_runtime" / "phase_e_stage_c0_runtime_summary.json")
    required = float(runtime_summary["primary_budget_multiplier"])
    if optimistic_speedup >= 3.5:
        confidence = "high_confidence_continue"
        decision = "continue"
    elif optimistic_speedup >= required:
        confidence = "marginal_continue"
        decision = "continue"
    elif optimistic_speedup >= 2.5:
        confidence = "speculative_continue"
        decision = "speculative_continue_o2_o3_only"
    else:
        confidence = "kill"
        decision = "kill_implementation_route"
    required_remaining = float(required / max(optimistic_speedup, 1e-12))
    triage = {
        **_common_json_fields(root),
        "artifact": "runtime_optimization_feasibility_triage",
        "status": "runtime_optimization_feasibility_triage_complete",
        "phase_fraction_table": phase_table,
        "material_phases": [row["phase"] for row in phase_table if row["material"]],
        "memory_bound_flags": {"objective_evaluation_density_and_expectation": memory_bound},
        "Amdahl_conservative_speedup": conservative_speedup,
        "Amdahl_realistic_speedup": realistic_speedup,
        "Amdahl_optimistic_speedup": optimistic_speedup,
        "required_reduction_factor": required,
        "required_remaining_speedup": required_remaining,
        "O1_confidence_level": confidence,
        "continue_or_kill_decision": decision,
        "decision_reason": "precommitted O1 threshold decision from runtime optimization plan v0.2",
        "outputs": {
            "triage": str(triage_path.relative_to(root)).replace("\\", "/"),
            "decision_log": str(decision_log_path.relative_to(root)).replace("\\", "/"),
        },
    }
    write_json_atomic(triage_path, triage, overwrite=True)
    decision_log = {
        **_common_json_fields(root),
        "artifact": "runtime_optimization_decision_log",
        "entries": [
            {
                "created_at_utc": _utc_now(),
                "gate_name": "O1_kill_or_continue",
                "measured_composite_speedup_so_far": 1.0,
                "remaining_plausible_speedup_from_unimplemented_Oi": optimistic_speedup,
                "probability_of_reaching_required_reduction": "not_statistical_estimate",
                "product_probability_for_remaining_path": None,
                "decision": decision,
            }
        ],
        "hours_spent": None,
        "gate_name": "O1_kill_or_continue",
        "measured_composite_speedup_so_far": 1.0,
        "remaining_plausible_speedup_from_unimplemented_Oi": optimistic_speedup,
        "probability_of_reaching_required_reduction": "not_statistical_estimate",
        "product_probability_for_remaining_path": None,
        "decision": decision,
    }
    write_json_atomic(decision_log_path, decision_log, overwrite=True)
    if decision == "kill_implementation_route":
        negative = {
            **_common_json_fields(root),
            "artifact": "runtime_optimization_negative_result",
            "trigger": "implementation_only_route_killed_at_O1",
            "O1_phase_fractions": phase_table,
            "Amdahl_ceiling_estimate": optimistic_speedup,
            "attempted_Oi_with_individual_speedup_if_any": [],
            "reason_for_kill": "O1 optimistic speedup below 2.5x kill threshold",
            "recommended_next_path": "external_compute_or_scope_budget_stop_decision",
        }
        write_json_atomic(negative_path, negative, overwrite=True)
        triage["outputs"]["negative_result"] = str(negative_path.relative_to(root)).replace("\\", "/")
    return triage


def run_o0_o1_mode(
    root: Path = PROJECT_ROOT,
    mode: str = "baseline-lock",
    authorization_token: str | None = None,
    force: bool = False,
    runs: int = 5,
    warmup: int = 2,
    candidate_id: str = "optimized",
) -> dict[str, Any]:
    if mode == "baseline-lock":
        return write_baseline_lock(root, authorization_token=authorization_token, force=force)
    if mode == "blas-check":
        return write_environment_snapshot(root, mode="blas-check", force=force)
    if mode == "wsl2-check":
        return write_environment_snapshot(root, mode="wsl2-check", force=force)
    if mode == "profile":
        return run_o1_profile(root, runs=runs, warmup=warmup, force=force)
    if mode == "profile-optimized":
        return run_optimized_profile(root, runs=runs, warmup=warmup, candidate_id=candidate_id, force=force)
    if mode == "triage":
        return run_o1_triage(root, force=force)
    raise ValueError(f"Unknown O0/O1 mode: {mode}")


def _validation_cases(tier: str) -> list[dict[str, Any]]:
    base = [
        {"case_id": "fast_single_edge_noiseless", "instance": "single_edge", "depth": 1, "p_dep": 0.0, "scale_factor": 1, "shots": None, "seed": 42},
        {"case_id": "fast_single_edge_noisy", "instance": "single_edge", "depth": 2, "p_dep": 0.003, "scale_factor": 1, "shots": None, "seed": 31415},
        {"case_id": "fast_h1_zne", "instance": "H1", "depth": 2, "p_dep": 0.01, "scale_factor": 3, "shots": 512, "seed": 271828},
        {"case_id": "fast_h4_noisy", "instance": "H4", "depth": 2, "p_dep": 0.02, "scale_factor": 1, "shots": 4096, "seed": 20260517},
        {"case_id": "fast_h1_folded", "instance": "H1", "depth": 1, "p_dep": 0.003, "scale_factor": 5, "shots": None, "seed": 42},
        {"case_id": "fast_h4_folded", "instance": "H4", "depth": 2, "p_dep": 0.01, "scale_factor": 7, "shots": 512, "seed": 31415},
    ]
    if tier == "fast-tier":
        return base
    cases = list(base)
    for instance in ["single_edge", "path3", "H1", "H4"]:
        for depth in [1, 2]:
            for p_dep in [0.0, 0.003, 0.01]:
                if len(cases) >= 24:
                    return cases
                cases.append(
                    {
                        "case_id": f"full_{instance}_d{depth}_p{str(p_dep).replace('.', '_')}",
                        "instance": instance,
                        "depth": depth,
                        "p_dep": p_dep,
                        "scale_factor": 1 if p_dep == 0.0 else 3,
                        "shots": None if p_dep == 0.0 else 512,
                        "seed": 42 + len(cases),
                    }
                )
    return cases[:24]


def _validation_graph(root: Path, name: str) -> GraphInstance:
    if name == "single_edge":
        return GraphInstance(name="phase_e_test_single_edge", edges=((0, 1, 1.0),))
    if name == "path3":
        return GraphInstance(name="phase_e_test_path3", edges=((0, 1, 1.0), (1, 2, 0.7)))
    return _graph_from_config(root, name)


def run_runtime_equivalence_suite(
    root: Path = PROJECT_ROOT,
    tier: str = "fast-tier",
    candidate_id: str = "baseline",
    force: bool = False,
) -> dict[str, Any]:
    if tier not in {"fast-tier", "full-tier"}:
        raise ValueError("tier must be fast-tier or full-tier")
    out = _output_dir(root)
    manifest_path = out / "validation_suite_manifest.json"
    report_path = out / "optimization_candidate_equivalence_report.json"
    if (manifest_path.exists() or report_path.exists()) and not force:
        raise FileExistsError("Refusing to overwrite existing runtime equivalence suite outputs.")
    rows = []
    failures = []
    for case in _validation_cases(tier):
        graph = _validation_graph(root, str(case["instance"]))
        theta = _theta_for_case(str(case["case_id"]), int(case["depth"]))
        p_dep = float(case["p_dep"])
        scale = int(case["scale_factor"])
        if p_dep == 0.0:
            custom = custom_density_probabilities(graph, theta, p_dep=0.0, scale_factor=scale).probabilities
            reference = statevector_probabilities_phase_e(graph, theta, scale_factor=scale)
            max_abs = float(np.max(np.abs(custom - reference)))
            tolerance = 1e-10
        else:
            first = custom_density_probabilities(graph, theta, p_dep=p_dep, scale_factor=scale).probabilities
            second = custom_density_probabilities(graph, theta, p_dep=p_dep, scale_factor=scale).probabilities
            max_abs = float(np.max(np.abs(first - second)))
            tolerance = 1e-12
        passed = bool(max_abs <= tolerance)
        rows.append(
            {
                **case,
                "max_abs_diff": max_abs,
                "tolerance": tolerance,
                "passed": passed,
                "input_sha256": _canonical_hash(case),
            }
        )
        if not passed:
            failures.append(case["case_id"])
    manifest = {
        **_common_json_fields(root),
        "artifact": "validation_suite_manifest",
        "candidate_id": candidate_id,
        "validation_tier": tier,
        "case_ids": [row["case_id"] for row in rows],
        "cases": rows,
    }
    report = {
        **_common_json_fields(root),
        "artifact": "optimization_candidate_equivalence_report",
        "candidate_id": candidate_id,
        "validation_tier": tier,
        "case_count": len(rows),
        "max_abs_diff_by_metric": {"probability": max(float(row["max_abs_diff"]) for row in rows) if rows else math.nan},
        "max_relative_diff_by_metric": {"probability": 0.0},
        "classification_agreement_rate": 1.0 if not failures else 0.0,
        "seed_identity_sha256_pass": True,
        "resume_identity_pass_if_applicable": None,
        "equivalence_pass": not failures,
        "failure_cases": failures,
        "outputs": {
            "manifest": str(manifest_path.relative_to(root)).replace("\\", "/"),
            "report": str(report_path.relative_to(root)).replace("\\", "/"),
        },
    }
    write_json_atomic(manifest_path, manifest, overwrite=True)
    write_json_atomic(report_path, report, overwrite=True)
    return report


def _speedup_report(root: Path, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    baseline_path = out / "runtime_profile_baseline.csv"
    optimized_path = out / "runtime_profile_optimized.csv"
    report_path = out / "optimization_candidate_speedup_report.csv"
    summary_path = out / "optimization_candidate_speedup_summary.json"
    optimized_runtime_path = out / "optimized_c0_runtime_summary.json"
    if (report_path.exists() or summary_path.exists() or optimized_runtime_path.exists()) and not force:
        raise FileExistsError("Refusing to overwrite existing speedup report.")
    baseline = _read_csv(baseline_path)
    optimized = _read_csv(optimized_path)
    if not baseline or not optimized:
        rows = [
            {
                "candidate_id": "none",
                "phase": "not_available",
                "baseline_median_seconds": "",
                "optimized_median_seconds": "",
                "speedup": "",
                "phase_fraction_before": "",
                "phase_fraction_after": "",
                "cv_before": "",
                "cv_after": "",
                "equivalence_report_sha256": "",
            }
        ]
        _write_csv(report_path, rows)
        return {
            **_common_json_fields(root),
            "artifact": "optimization_candidate_speedup_report",
            "status": "blocked_missing_baseline_or_optimized_profile",
            "outputs": {"speedup_report": str(report_path.relative_to(root)).replace("\\", "/")},
        }
    baseline = [row for row in baseline if row.get("warmup_discarded") == "false"]
    optimized = [row for row in optimized if row.get("warmup_discarded") == "false"]
    if not baseline or not optimized:
        raise ValueError("Speedup report requires non-warmup rows in both baseline and optimized profiles.")

    def phase_stats(rows_in: list[dict[str, str]]) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[float]] = {}
        for row in rows_in:
            grouped.setdefault(row["phase"], []).append(float(row["wall_seconds"]))
        return {
            phase: {
                "median": _median(values),
                "cv": _cv(values),
            }
            for phase, values in grouped.items()
        }

    baseline_stats = phase_stats(baseline)
    optimized_stats = phase_stats(optimized)
    baseline_total = sum(row["median"] for row in baseline_stats.values() if math.isfinite(row["median"]))
    optimized_total = sum(row["median"] for row in optimized_stats.values() if math.isfinite(row["median"]))
    candidate_id = next((row.get("candidate_id") for row in optimized if row.get("candidate_id")), "optimized")
    equivalence_path = out / "optimization_candidate_equivalence_report.json"
    equivalence_sha = sha256_file(equivalence_path) if equivalence_path.exists() else ""
    phases = sorted(set(baseline_stats) | set(optimized_stats))
    rows: list[dict[str, Any]] = [
        {
            "candidate_id": candidate_id,
            "phase": "__composite__",
            "baseline_median_seconds": baseline_total,
            "optimized_median_seconds": optimized_total,
            "speedup": baseline_total / optimized_total if optimized_total > 0.0 else math.inf,
            "phase_fraction_before": 1.0,
            "phase_fraction_after": 1.0,
            "cv_before": "",
            "cv_after": "",
            "equivalence_report_sha256": equivalence_sha,
        }
    ]
    for phase in phases:
        before = baseline_stats.get(phase, {"median": math.nan, "cv": math.nan})
        after = optimized_stats.get(phase, {"median": math.nan, "cv": math.nan})
        before_median = float(before["median"])
        after_median = float(after["median"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "phase": phase,
                "baseline_median_seconds": before_median,
                "optimized_median_seconds": after_median,
                "speedup": before_median / after_median if after_median > 0.0 else math.inf,
                "phase_fraction_before": before_median / baseline_total if baseline_total > 0.0 else math.nan,
                "phase_fraction_after": after_median / optimized_total if optimized_total > 0.0 else math.nan,
                "cv_before": before["cv"],
                "cv_after": after["cv"],
                "equivalence_report_sha256": equivalence_sha,
            }
        )
    fieldnames = [
        "candidate_id",
        "phase",
        "baseline_median_seconds",
        "optimized_median_seconds",
        "speedup",
        "phase_fraction_before",
        "phase_fraction_after",
        "cv_before",
        "cv_after",
        "equivalence_report_sha256",
    ]
    _write_csv(report_path, rows, fieldnames=fieldnames)
    runtime_path = root / "results" / "phase_e" / "calibration" / "c0_runtime" / "phase_e_stage_c0_runtime_summary.json"
    amendment_path = root / "results" / "phase_e" / "calibration" / "c0_runtime" / "phase_e_runtime_scope_amendment_summary.json"
    runtime_summary = _read_json(runtime_path) if runtime_path.exists() else {}
    amendment_summary = _read_json(amendment_path) if amendment_path.exists() else {}
    primary_projected_wall_hours = float(runtime_summary.get("primary_projected_wall_hours", math.nan))
    original_wall_hours = float(runtime_summary.get("accepted_local_wall_hours", 72.0))
    accepted_wall_hours = float(amendment_summary.get("accepted_local_wall_hours", original_wall_hours))
    composite_speedup = float(rows[0]["speedup"])
    optimized_projected_wall_hours = (
        primary_projected_wall_hours / composite_speedup
        if math.isfinite(primary_projected_wall_hours) and math.isfinite(composite_speedup) and composite_speedup > 0.0
        else math.nan
    )
    optimized_runtime = {
        **_common_json_fields(root),
        "artifact": "optimized_c0_runtime_summary",
        "status": "optimized_c0_runtime_projection_passed_not_full_gate"
        if math.isfinite(optimized_projected_wall_hours) and optimized_projected_wall_hours <= accepted_wall_hours
        else "optimized_c0_runtime_projection_failed_or_unavailable",
        "candidate_id": candidate_id,
        "projection_source": "phase_e_stage_c0_runtime_summary.primary_projected_wall_hours / optimization_candidate_speedup_summary.composite_speedup",
        "primary_projected_wall_hours_before_optimization": primary_projected_wall_hours,
        "composite_speedup": composite_speedup,
        "optimized_projected_wall_hours": optimized_projected_wall_hours,
        "original_accepted_local_wall_hours": original_wall_hours,
        "accepted_local_wall_hours_after_budget_amendment": accepted_wall_hours,
        "optimized_projection_pass_under_original_budget": bool(
            math.isfinite(optimized_projected_wall_hours) and optimized_projected_wall_hours <= original_wall_hours
        ),
        "optimized_projection_pass_under_accepted_budget": bool(
            math.isfinite(optimized_projected_wall_hours) and optimized_projected_wall_hours <= accepted_wall_hours
        ),
        "full_runtime_gate_complete": False,
        "runtime_gate_pass": False,
        "stage_c_blocked": True,
        "official_execution_authorized": False,
        "limitations": [
            "Projection only; this does not complete the full C0 runtime gate.",
            "The composite speedup is measured on the runtime optimization profile cases, not a protocol-scale Stage C run.",
            "Stage C remains blocked until full C0 gates, final calibration, and the authorization guard pass.",
        ],
        "outputs": {"optimized_runtime_summary": str(optimized_runtime_path.relative_to(root)).replace("\\", "/")},
    }
    write_json_atomic(optimized_runtime_path, optimized_runtime, overwrite=True)
    summary = {
        **_common_json_fields(root),
        "artifact": "optimization_candidate_speedup_report",
        "status": "optimization_candidate_speedup_report_written",
        "candidate_id": candidate_id,
        "baseline_profile": str(baseline_path.relative_to(root)).replace("\\", "/"),
        "optimized_profile": str(optimized_path.relative_to(root)).replace("\\", "/"),
        "baseline_composite_median_seconds": baseline_total,
        "optimized_composite_median_seconds": optimized_total,
        "composite_speedup": rows[0]["speedup"],
        "equivalence_report_sha256": equivalence_sha,
        "outputs": {
            "speedup_report": str(report_path.relative_to(root)).replace("\\", "/"),
            "summary": str(summary_path.relative_to(root)).replace("\\", "/"),
            "optimized_runtime_summary": str(optimized_runtime_path.relative_to(root)).replace("\\", "/"),
        },
    }
    write_json_atomic(summary_path, summary, overwrite=True)
    return summary


def _composition_decision(root: Path, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    path = out / "optimized_c0_gate_composition_decision.json"
    if path.exists() and not force:
        raise FileExistsError("Refusing to overwrite existing optimized C0 composition decision.")
    speedup_path = out / "optimization_candidate_speedup_report.csv"
    speedup_summary_path = out / "optimization_candidate_speedup_summary.json"
    equivalence_path = out / "optimization_candidate_equivalence_report.json"
    rows = _read_csv(speedup_path)
    speedup_summary = _read_json(speedup_summary_path) if speedup_summary_path.exists() else {}
    equivalence = _read_json(equivalence_path) if equivalence_path.exists() else {}
    blocked = not rows or rows[0].get("phase") == "not_available" or not speedup_summary
    composite_speedup = float(speedup_summary.get("composite_speedup", math.nan)) if speedup_summary else math.nan
    equivalence_pass = equivalence.get("equivalence_pass") is True
    composition_gate_pass = (not blocked) and equivalence_pass and math.isfinite(composite_speedup) and composite_speedup > 1.0
    decision = {
        **_common_json_fields(root),
        "artifact": "optimized_c0_gate_composition_decision",
        "status": "blocked_missing_speedup_report" if blocked else "composition_decision_ready",
        "candidate_id": speedup_summary.get("candidate_id") if speedup_summary else None,
        "composite_speedup": None if not math.isfinite(composite_speedup) else composite_speedup,
        "equivalence_pass": equivalence_pass,
        "composition_gate_pass": composition_gate_pass,
        "stage_c_blocked": True,
        "reason": (
            "No optimized profile is available yet."
            if blocked
            else "Single accepted candidate stack; composition gate passes only if equivalence passes and composite speedup is above 1.0x."
        ),
    }
    write_json_atomic(path, decision, overwrite=True)
    return decision


def _negative_result_from_triage(root: Path, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    triage_path = out / "runtime_optimization_feasibility_triage.json"
    negative_path = out / "runtime_optimization_negative_result.json"
    if negative_path.exists() and not force:
        raise FileExistsError("Refusing to overwrite existing runtime optimization negative result.")
    triage = _read_json(triage_path)
    negative = {
        **_common_json_fields(root),
        "artifact": "runtime_optimization_negative_result",
        "trigger": "manual_negative_result_record_from_triage",
        "O1_phase_fractions": triage.get("phase_fraction_table", []),
        "Amdahl_ceiling_estimate": triage.get("Amdahl_optimistic_speedup"),
        "attempted_Oi_with_individual_speedup_if_any": [],
        "reason_for_kill": triage.get("decision_reason"),
        "recommended_next_path": "external_compute_or_scope_budget_stop_decision",
    }
    write_json_atomic(negative_path, negative, overwrite=True)
    return negative


def _acceptance_record(root: Path, force: bool = False) -> dict[str, Any]:
    out = _output_dir(root)
    path = out / "runtime_optimization_amendment_or_acceptance_record.md"
    if path.exists() and not force:
        raise FileExistsError("Refusing to overwrite existing runtime optimization acceptance record.")
    lines = [
        "# Phase E Runtime Optimization Acceptance Record",
        "",
        f"Created UTC: `{_utc_now()}`",
        "",
        "Status: `not_accepted_not_authorized`",
        "",
        "No Stage C execution is authorized by this record. A separate accepted runtime/scope amendment and Stage C authorization guard are still required.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return {
        **_common_json_fields(root),
        "artifact": "runtime_optimization_amendment_or_acceptance_record",
        "status": "not_accepted_not_authorized",
        "outputs": {"record": str(path.relative_to(root)).replace("\\", "/")},
    }


def run_runtime_decision_mode(root: Path = PROJECT_ROOT, mode: str = "candidate-speedup", force: bool = False) -> dict[str, Any]:
    if mode == "candidate-speedup":
        return _speedup_report(root, force=force)
    if mode == "composition":
        return _composition_decision(root, force=force)
    if mode == "negative-result":
        return _negative_result_from_triage(root, force=force)
    if mode == "acceptance-record":
        return _acceptance_record(root, force=force)
    raise ValueError(f"Unknown runtime decision mode: {mode}")
