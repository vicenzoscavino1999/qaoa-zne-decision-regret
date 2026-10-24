from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .circuits import graph_from_config
from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, save_array_atomic, sha256_file, write_json_atomic
from .landscapes import generate_ideal_landscape
from .phase_c import _selected_seeds
from .phase_c_120_custom_kernel import (
    CUSTOM_KERNEL_VERSION,
    KernelProbabilityCache,
    generate_custom_kernel_noisy_landscape,
    generate_custom_kernel_zne_landscape,
    new_probability_cache,
)
from .phase_c_120_preflight import PROTOCOL_VERSION, TARGET_GRID_SIZE


KERNEL_RUNNER_VERSION = "phase_c_120_kernel_runner_v1"
OFFICIAL_RUN_NAME = "phase_c_120_kernel_official"
SELF_TEST_RUN_NAME = "phase_c_120_kernel_runner_selftest"
DRY_RUN_NAME = "phase_c_120_operational_dry_run"
CORE_ALLOCATION = "proportional_to_abs_weights"
SELF_TEST_GRID_SIZE = 3
SELF_TEST_INSTANCES = ("G1_C4", "G3_weighted6")
SELF_TEST_NOISE_VALUES = (0.005,)
SELF_TEST_SHOT_BUDGETS = (64,)
SELF_TEST_SEEDS = (42, 1337)
DRY_RUN_GRID_SIZE = 8
DRY_RUN_INSTANCES = ("G1_C4", "G2_weighted4", "G3_weighted6")
DRY_RUN_NOISE_VALUES = (0.001, 0.005, 0.01)
DRY_RUN_SHOT_BUDGETS = (128,)
DRY_RUN_SEEDS = (42,)
DRY_RUN_ZNE_METHODS = {"linear": (1, 3)}
OFFICIAL_ACCEPT_TOKEN = "accept_phase_c_120_kernel_official"


@dataclass(frozen=True)
class KernelRunnerTask:
    index: int
    group_index: int
    group_key: str
    task_id: str
    kind: str
    instance: str
    grid_size: int
    output_path: str
    p_noise: float | None = None
    shots_total: int | None = None
    seed: int | None = None
    zne_order: str | None = None
    scale_factors: tuple[int, ...] = ()

    @property
    def done_path(self) -> Path:
        return Path(self.output_path).with_suffix(".done.json")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _clean_token(value: object) -> str:
    return str(value).replace(".", "p").replace("/", "_").replace("\\", "_")


def _task_id(parts: list[object]) -> str:
    return "_".join(_clean_token(part) for part in parts)


def _safe_reset(path: Path, root: Path) -> None:
    resolved = path.resolve()
    allowed = {
        (root / "data" / "raw" / SELF_TEST_RUN_NAME).resolve(),
        (root / "results" / SELF_TEST_RUN_NAME).resolve(),
        (root / "data" / "raw" / DRY_RUN_NAME).resolve(),
        (root / "results" / DRY_RUN_NAME).resolve(),
    }
    if resolved not in allowed:
        raise RuntimeError(f"Refusing to remove non-self-test path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _build_task(
    index: int,
    group_index: int,
    group_key: str,
    run_name: str,
    root: Path,
    kind: str,
    instance: str,
    grid_size: int,
    p_noise: float | None = None,
    shots_total: int | None = None,
    seed: int | None = None,
    zne_order: str | None = None,
    scale_factors: tuple[int, ...] = (),
) -> KernelRunnerTask:
    raw_root = root / "data" / "raw" / run_name / instance / f"grid_{grid_size}"
    if kind == "ideal":
        output_path = raw_root / "C_ideal.npy"
        task_id = _task_id([f"{index:06d}", instance, f"grid{grid_size}", "ideal"])
    else:
        if p_noise is None or shots_total is None or seed is None:
            raise ValueError(f"{kind} task requires p_noise, shots_total, and seed.")
        base = raw_root / f"p_{p_noise:g}" / f"B_{shots_total}" / f"seed_{seed}"
        if kind == "noisy":
            output_path = base / "C_noisy_custom_kernel.npy"
            task_id = _task_id([f"{index:06d}", instance, f"p{p_noise:g}", f"B{shots_total}", f"seed{seed}", "noisy"])
        elif kind == "zne":
            if zne_order is None or not scale_factors:
                raise ValueError("ZNE task requires zne_order and scale_factors.")
            output_path = base / f"C_zne_{zne_order}_{CORE_ALLOCATION}_custom_kernel.npy"
            task_id = _task_id([f"{index:06d}", instance, f"p{p_noise:g}", f"B{shots_total}", f"seed{seed}", f"zne{zne_order}", CORE_ALLOCATION])
        else:
            raise ValueError(f"Unknown runner task kind: {kind}")
    return KernelRunnerTask(
        index=index,
        group_index=group_index,
        group_key=group_key,
        task_id=task_id,
        kind=kind,
        instance=instance,
        grid_size=grid_size,
        output_path=str(output_path),
        p_noise=p_noise,
        shots_total=shots_total,
        seed=seed,
        zne_order=zne_order,
        scale_factors=scale_factors,
    )


def _plan_payload(
    run_name: str,
    mode: str,
    grid_size: int,
    instances: tuple[str, ...],
    noise_values: tuple[float, ...],
    shot_budgets: tuple[int, ...],
    seeds: tuple[int, ...],
    zne_methods: dict[str, tuple[int, ...]],
    tasks: list[KernelRunnerTask],
    root: Path,
) -> dict[str, Any]:
    unique_scales = sorted({scale for scales in zne_methods.values() for scale in scales})
    noise_group_count = len(instances) * len(noise_values)
    grid_points = grid_size * grid_size
    probability_cache_misses = noise_group_count * grid_points * len(unique_scales)
    scale_requests_per_condition = 1 + sum(len(scales) for scales in zne_methods.values())
    probability_cache_requests = noise_group_count * grid_points * len(shot_budgets) * len(seeds) * scale_requests_per_condition
    task_rows = []
    for task in tasks:
        row = asdict(task)
        row["scale_factors"] = list(task.scale_factors)
        row["output_path"] = Path(task.output_path).relative_to(root).as_posix()
        row["done_path"] = task.done_path.relative_to(root).as_posix()
        task_rows.append(row)
    return {
        "runner_version": KERNEL_RUNNER_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "custom_kernel_version": CUSTOM_KERNEL_VERSION,
        "run_name": run_name,
        "mode": mode,
        "grid_size": grid_size,
        "instances": list(instances),
        "noise_values": list(noise_values),
        "shot_budgets": list(shot_budgets),
        "seeds": list(seeds),
        "zne_methods": {key: list(value) for key, value in sorted(zne_methods.items())},
        "unique_scale_factors": unique_scales,
        "tasks_total": len(tasks),
        "groups_total": len({task.group_key for task in tasks}),
        "probability_cache_model": {
            "noise_group_count": noise_group_count,
            "grid_points": grid_points,
            "scale_requests_per_condition": scale_requests_per_condition,
            "expected_misses": probability_cache_misses,
            "expected_requests": probability_cache_requests,
            "expected_hits": probability_cache_requests - probability_cache_misses,
            "expected_reuse_factor": float(probability_cache_requests / probability_cache_misses) if probability_cache_misses else 0.0,
        },
        "output_storage_bytes_float64": len(tasks) * grid_points * 8,
        "tasks": task_rows,
    }


def build_kernel_runner_plan(mode: str = "official", root: Path = PROJECT_ROOT) -> dict[str, Any]:
    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    if mode == "official":
        run_name = OFFICIAL_RUN_NAME
        grid_size = TARGET_GRID_SIZE
        instances = tuple(str(value) for value in cfg["phase_c"]["instances"])
        noise_values = tuple(float(value) for value in cfg["noise"]["p_values"])
        shot_budgets = tuple(int(value) for value in cfg["shots"]["total_budget_primary"])
        seeds = tuple(_selected_seeds(root, int(cfg["phase_c"]["seed_count"])))
        zne_methods = {str(key): tuple(int(scale) for scale in scales) for key, scales in cfg["zne"]["methods"].items()}
        plan_mode = "official_120x120_plan_not_executed_by_default"
    elif mode == "self_test":
        run_name = SELF_TEST_RUN_NAME
        grid_size = SELF_TEST_GRID_SIZE
        instances = SELF_TEST_INSTANCES
        noise_values = SELF_TEST_NOISE_VALUES
        shot_budgets = SELF_TEST_SHOT_BUDGETS
        seeds = SELF_TEST_SEEDS
        zne_methods = {str(key): tuple(int(scale) for scale in scales) for key, scales in cfg["zne"]["methods"].items()}
        plan_mode = "runner_self_test"
    elif mode == "dry_run":
        run_name = DRY_RUN_NAME
        grid_size = DRY_RUN_GRID_SIZE
        instances = DRY_RUN_INSTANCES
        noise_values = DRY_RUN_NOISE_VALUES
        shot_budgets = DRY_RUN_SHOT_BUDGETS
        seeds = DRY_RUN_SEEDS
        zne_methods = DRY_RUN_ZNE_METHODS
        plan_mode = "operational_dry_run_not_official_phase_c_data"
    else:
        raise ValueError(f"Unknown kernel runner mode: {mode}")

    tasks: list[KernelRunnerTask] = []
    index = 0
    for instance_index, instance in enumerate(instances):
        instance_noise_group_offset = instance_index * len(noise_values)
        group_key = f"{instance}:ideal"
        tasks.append(_build_task(index, instance_noise_group_offset, group_key, run_name, root, "ideal", instance, grid_size))
        index += 1
        for noise_index, p_noise in enumerate(noise_values):
            group_index = instance_noise_group_offset + noise_index
            group_key = f"{instance}:p{p_noise:g}"
            for shots_total in shot_budgets:
                for seed in seeds:
                    tasks.append(_build_task(index, group_index, group_key, run_name, root, "noisy", instance, grid_size, p_noise, shots_total, seed))
                    index += 1
                    for zne_order, scale_factors in zne_methods.items():
                        tasks.append(
                            _build_task(
                                index,
                                group_index,
                                group_key,
                                run_name,
                                root,
                                "zne",
                                instance,
                                grid_size,
                                p_noise,
                                shots_total,
                                seed,
                                zne_order,
                                scale_factors,
                            )
                        )
                        index += 1

    return _plan_payload(run_name, plan_mode, grid_size, instances, noise_values, shot_budgets, seeds, zne_methods, tasks, root)


def _result_dir_for_mode(mode: str, root: Path) -> Path:
    if mode == "official":
        return root / "results" / OFFICIAL_RUN_NAME
    if mode == "self_test":
        return root / "results" / SELF_TEST_RUN_NAME
    if mode == "dry_run":
        return root / "results" / DRY_RUN_NAME
    raise ValueError(f"Unknown kernel runner mode: {mode}")


def _raw_dir_for_mode(mode: str, root: Path) -> Path:
    if mode == "official":
        return root / "data" / "raw" / OFFICIAL_RUN_NAME
    if mode == "self_test":
        return root / "data" / "raw" / SELF_TEST_RUN_NAME
    if mode == "dry_run":
        return root / "data" / "raw" / DRY_RUN_NAME
    raise ValueError(f"Unknown kernel runner mode: {mode}")


def _write_plan(plan: dict[str, Any], result_dir: Path) -> None:
    write_json_atomic(result_dir / "plan.json", plan, overwrite=True)


def _runtime_estimate_from_pilot(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    pilot_summary_path = root / "results" / "phase_c_120_kernel_pilot" / "pilot_summary.json"
    pilot_summary = json.loads(pilot_summary_path.read_text(encoding="utf-8")) if pilot_summary_path.is_file() else {}
    timed_probe_path = root / "results" / "phase_c_120_timed_probe" / "timed_probe_summary.json"
    timed_probe = json.loads(timed_probe_path.read_text(encoding="utf-8")) if timed_probe_path.is_file() else {}
    cache_model = plan["probability_cache_model"]
    official_misses = int(cache_model["expected_misses"])
    pilot_misses = int(pilot_summary.get("cache", {}).get("misses", 0) or 0)
    pilot_elapsed = float(pilot_summary.get("elapsed_seconds", math.nan))
    local_workers = max(1, int(os.cpu_count() or 1))
    timed_probe_seconds_per_miss = float(timed_probe.get("seconds_per_probability_cache_miss", math.nan))
    timed_probe_is_valid = (
        timed_probe.get("status") == "phase_c_120_timed_probe_passed"
        and int(timed_probe.get("official_probability_cache_misses", 0)) == official_misses
        and math.isfinite(timed_probe_seconds_per_miss)
        and timed_probe_seconds_per_miss > 0.0
    )
    if timed_probe_is_valid:
        basis = "phase_c_120_timed_probe_seconds_per_probability_cache_miss"
        seconds_per_probability_miss = timed_probe_seconds_per_miss
        serial_seconds = seconds_per_probability_miss * official_misses
    elif pilot_misses > 0 and math.isfinite(pilot_elapsed):
        basis = "phase_c_120_kernel_pilot_elapsed_per_probability_cache_miss"
        seconds_per_probability_miss = pilot_elapsed / pilot_misses
        serial_seconds = seconds_per_probability_miss * official_misses
    else:
        basis = "unavailable"
        seconds_per_probability_miss = math.nan
        serial_seconds = math.nan
    return {
        "created_at_utc": _utc_now(),
        "runner_version": KERNEL_RUNNER_VERSION,
        "custom_kernel_version": CUSTOM_KERNEL_VERSION,
        "basis": basis,
        "pilot_summary": "results/phase_c_120_kernel_pilot/pilot_summary.json",
        "pilot_elapsed_seconds": pilot_elapsed,
        "pilot_probability_cache_misses": pilot_misses,
        "timed_probe_summary": "results/phase_c_120_timed_probe/timed_probe_summary.json" if timed_probe_is_valid else None,
        "timed_probe_elapsed_seconds": timed_probe.get("wall_elapsed_seconds"),
        "timed_probe_probability_cache_misses": timed_probe.get("probability_cache", {}).get("misses"),
        "official_probability_cache_misses": official_misses,
        "seconds_per_probability_cache_miss": seconds_per_probability_miss,
        "serial_hours": serial_seconds / 3600.0 if math.isfinite(serial_seconds) else math.nan,
        "local_workers": local_workers,
        "local_75pct_parallel_hours": serial_seconds / 3600.0 / (local_workers * 0.75) if math.isfinite(serial_seconds) else math.nan,
        "passes_72h_gate": bool(math.isfinite(serial_seconds) and serial_seconds / 3600.0 / (local_workers * 0.75) <= 72.0),
        "tasks_total": plan["tasks_total"],
        "groups_total": plan["groups_total"],
        "probability_cache_model": cache_model,
        "decision": "official_kernel_runner_plan_ready_not_executed",
        "limitations": [
            "This estimate is empirical from a 4 x 4 kernel pilot and should be refined with a timed official-style shard before launching the full run.",
            "The official 120 x 120 run remains blocked until the user explicitly accepts execution.",
        ],
    }


def write_official_kernel_plan(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    result_dir = ensure_dir(root / "results" / OFFICIAL_RUN_NAME)
    plan = build_kernel_runner_plan("official", root)
    _write_plan(plan, result_dir)
    estimate = _runtime_estimate_from_pilot(plan, root)
    write_json_atomic(result_dir / "runtime_estimate.json", estimate, overwrite=True)
    return {"plan": plan, "runtime_estimate": estimate}


def _task_from_plan_row(row: dict[str, Any], root: Path) -> KernelRunnerTask:
    return KernelRunnerTask(
        index=int(row["index"]),
        group_index=int(row["group_index"]),
        group_key=str(row["group_key"]),
        task_id=str(row["task_id"]),
        kind=str(row["kind"]),
        instance=str(row["instance"]),
        grid_size=int(row["grid_size"]),
        output_path=str(root / row["output_path"]),
        p_noise=None if row.get("p_noise") is None else float(row["p_noise"]),
        shots_total=None if row.get("shots_total") is None else int(row["shots_total"]),
        seed=None if row.get("seed") is None else int(row["seed"]),
        zne_order=None if row.get("zne_order") is None else str(row["zne_order"]),
        scale_factors=tuple(int(scale) for scale in row.get("scale_factors", [])),
    )


def _load_array_checked(task: KernelRunnerTask) -> np.ndarray:
    array = np.load(task.output_path, allow_pickle=False)
    if array.shape != (task.grid_size, task.grid_size):
        raise RuntimeError(f"Wrong array shape for {task.task_id}: {array.shape}")
    if not bool(np.isfinite(array).all()):
        raise RuntimeError(f"Non-finite array for {task.task_id}")
    return array


def _is_complete(task: KernelRunnerTask) -> bool:
    array_path = Path(task.output_path)
    done_path = task.done_path
    if not array_path.exists():
        if done_path.exists():
            raise RuntimeError(f"Done metadata exists without array: {done_path}")
        return False
    if not done_path.exists():
        raise RuntimeError(f"Array exists without done metadata: {array_path}")
    array = _load_array_checked(task)
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("array_sha256") != sha256_file(array_path):
        raise RuntimeError(f"Hash mismatch for {array_path}")
    if done.get("task_id") != task.task_id:
        raise RuntimeError(f"Task id mismatch for {done_path}")
    if done.get("shape") != list(array.shape):
        raise RuntimeError(f"Shape metadata mismatch for {done_path}")
    if done.get("runner_version") != KERNEL_RUNNER_VERSION:
        raise RuntimeError(f"Runner version mismatch for {done_path}")
    return True


def _write_done(task: KernelRunnerTask, metadata: dict[str, Any], elapsed_seconds: float) -> None:
    array = _load_array_checked(task)
    payload = {
        "runner_version": KERNEL_RUNNER_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "custom_kernel_version": CUSTOM_KERNEL_VERSION,
        "status": "complete",
        "task_id": task.task_id,
        "task_index": task.index,
        "group_index": task.group_index,
        "group_key": task.group_key,
        "kind": task.kind,
        "instance": task.instance,
        "grid_size": task.grid_size,
        "shape": list(array.shape),
        "p_noise": task.p_noise,
        "shots_total": task.shots_total,
        "seed": task.seed,
        "zne_order": task.zne_order,
        "scale_factors": list(task.scale_factors),
        "allocation": CORE_ALLOCATION if task.kind == "zne" else "none",
        "array_sha256": sha256_file(task.output_path),
        "array_min": float(np.min(array)),
        "array_max": float(np.max(array)),
        "array_mean": float(np.mean(array)),
        "array_path": Path(task.output_path).relative_to(PROJECT_ROOT).as_posix(),
        "elapsed_seconds": elapsed_seconds,
        "written_at_utc": _utc_now(),
        "metadata": metadata,
    }
    write_json_atomic(task.done_path, payload, overwrite=True)


def _execute_task(task: KernelRunnerTask, cfg: dict[str, Any], cache: KernelProbabilityCache) -> tuple[bool, float]:
    if _is_complete(task):
        return False, 0.0
    graph = graph_from_config(task.instance, cfg)
    started = time.perf_counter()
    if task.kind == "ideal":
        array = generate_ideal_landscape(graph, task.grid_size)
        metadata: dict[str, Any] = {"engine": "qiskit_statevector_ideal", "method": "ideal"}
    elif task.kind == "noisy":
        if task.p_noise is None or task.shots_total is None or task.seed is None:
            raise RuntimeError(f"Incomplete noisy task: {task.task_id}")
        array, metadata = generate_custom_kernel_noisy_landscape(
            graph,
            task.grid_size,
            task.p_noise,
            task.shots_total,
            task.seed,
            probability_cache=cache,
        )
    elif task.kind == "zne":
        if task.p_noise is None or task.shots_total is None or task.seed is None or task.zne_order is None:
            raise RuntimeError(f"Incomplete ZNE task: {task.task_id}")
        array, metadata = generate_custom_kernel_zne_landscape(
            graph,
            task.grid_size,
            task.p_noise,
            task.shots_total,
            task.seed,
            list(task.scale_factors),
            task.zne_order,
            shot_allocation_strategy=CORE_ALLOCATION,
            probability_cache=cache,
        )
    else:
        raise RuntimeError(f"Unknown runner task kind: {task.kind}")
    elapsed = time.perf_counter() - started
    save_array_atomic(task.output_path, array, overwrite=False)
    _write_done(task, metadata, elapsed)
    return True, elapsed


def _shard_tasks(tasks: list[KernelRunnerTask], shard_index: int, shard_count: int) -> list[KernelRunnerTask]:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must satisfy 0 <= index < shard_count, got {shard_index}.")
    return [task for task in tasks if task.group_index % shard_count == shard_index]


def _progress_payload(
    plan: dict[str, Any],
    shard_index: int,
    shard_count: int,
    shard_tasks: list[KernelRunnerTask],
    generated: int,
    skipped: int,
    status: str,
    cache_totals: dict[str, int | float],
    started_at_utc: str,
) -> dict[str, Any]:
    completed = sum(1 for task in shard_tasks if Path(task.output_path).exists() and task.done_path.exists())
    return {
        "runner_version": KERNEL_RUNNER_VERSION,
        "run_name": plan["run_name"],
        "mode": plan["mode"],
        "grid_size": plan["grid_size"],
        "shard_index": shard_index,
        "shard_count": shard_count,
        "status": status,
        "tasks_in_shard": len(shard_tasks),
        "tasks_completed_in_shard": completed,
        "tasks_pending_in_shard": len(shard_tasks) - completed,
        "generated_this_run": generated,
        "skipped_this_run": skipped,
        "cache_totals": cache_totals,
        "started_at_utc": started_at_utc,
        "updated_at_utc": _utc_now(),
    }


def run_kernel_runner_shard(
    mode: str,
    shard_index: int = 0,
    shard_count: int = 1,
    stop_after_tasks: int | None = None,
    force_reset_self_test: bool = False,
    accept_official_token: str | None = None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    if mode == "official" and accept_official_token != OFFICIAL_ACCEPT_TOKEN:
        raise RuntimeError("Official 120 x 120 execution requires the explicit accept token.")
    if mode in {"self_test", "dry_run"} and force_reset_self_test:
        _safe_reset(_raw_dir_for_mode(mode, root), root)
        _safe_reset(_result_dir_for_mode(mode, root), root)

    result_dir = ensure_dir(_result_dir_for_mode(mode, root))
    plan = build_kernel_runner_plan(mode, root)
    _write_plan(plan, result_dir)
    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    tasks = [_task_from_plan_row(row, root) for row in plan["tasks"]]
    shard_tasks = _shard_tasks(tasks, shard_index, shard_count)
    if not shard_tasks:
        raise RuntimeError(f"Shard {shard_index}/{shard_count} has no tasks.")

    progress_path = result_dir / f"progress_shard_{shard_index:03d}_of_{shard_count:03d}.json"
    started_at_utc = _utc_now()
    generated = 0
    skipped = 0
    processed = 0
    cache_totals = {"entries": 0, "hits": 0, "misses": 0, "requests": 0, "reuse_factor": 0.0}
    active_cache: KernelProbabilityCache | None = None
    current_group: str | None = None

    for task in shard_tasks:
        if current_group != task.group_key:
            if active_cache is not None:
                summary = active_cache.summary()
                cache_totals["entries"] = int(cache_totals["entries"]) + int(summary["entries"])
                cache_totals["hits"] = int(cache_totals["hits"]) + int(summary["hits"])
                cache_totals["misses"] = int(cache_totals["misses"]) + int(summary["misses"])
                cache_totals["requests"] = int(cache_totals["requests"]) + int(summary["requests"])
            active_cache = new_probability_cache()
            current_group = task.group_key
        was_generated, _elapsed = _execute_task(task, cfg, active_cache)
        generated += int(was_generated)
        skipped += int(not was_generated)
        processed += 1
        status = "running"
        if stop_after_tasks is not None and processed >= stop_after_tasks:
            status = "interrupted_for_test"
        write_json_atomic(
            progress_path,
            _progress_payload(plan, shard_index, shard_count, shard_tasks, generated, skipped, status, cache_totals, started_at_utc),
            overwrite=True,
        )
        if status == "interrupted_for_test":
            break

    if active_cache is not None:
        summary = active_cache.summary()
        cache_totals["entries"] = int(cache_totals["entries"]) + int(summary["entries"])
        cache_totals["hits"] = int(cache_totals["hits"]) + int(summary["hits"])
        cache_totals["misses"] = int(cache_totals["misses"]) + int(summary["misses"])
        cache_totals["requests"] = int(cache_totals["requests"]) + int(summary["requests"])
    cache_totals["reuse_factor"] = float(cache_totals["requests"] / cache_totals["misses"]) if int(cache_totals["misses"]) else 0.0
    final_status = "interrupted_for_test" if stop_after_tasks is not None and processed >= stop_after_tasks else "complete"
    payload = _progress_payload(plan, shard_index, shard_count, shard_tasks, generated, skipped, final_status, cache_totals, started_at_utc)
    write_json_atomic(progress_path, payload, overwrite=True)
    return payload


def summarize_kernel_runner(run_name: str, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    result_dir = root / "results" / run_name
    plan_path = result_dir / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Runner plan not found: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tasks = [_task_from_plan_row(row, root) for row in plan["tasks"]]
    completed = 0
    corrupt: list[str] = []
    for task in tasks:
        try:
            if _is_complete(task):
                completed += 1
        except Exception:
            corrupt.append(task.task_id)
    progress_files = sorted(result_dir.glob("progress_shard_*.json"))
    progress_rows = [json.loads(path.read_text(encoding="utf-8")) for path in progress_files]
    cache_hits = sum(int(row.get("cache_totals", {}).get("hits", 0)) for row in progress_rows)
    cache_misses = sum(int(row.get("cache_totals", {}).get("misses", 0)) for row in progress_rows)
    cache_requests = sum(int(row.get("cache_totals", {}).get("requests", 0)) for row in progress_rows)
    status = "complete" if completed == len(tasks) and not corrupt else "incomplete"
    summary = {
        "created_at_utc": _utc_now(),
        "runner_version": KERNEL_RUNNER_VERSION,
        "run_name": run_name,
        "status": status,
        "tasks_total": len(tasks),
        "tasks_completed": completed,
        "tasks_pending": len(tasks) - completed,
        "corrupt_tasks": corrupt,
        "progress_files": [path.name for path in progress_files],
        "cache_totals": {
            "hits": cache_hits,
            "misses": cache_misses,
            "requests": cache_requests,
            "reuse_factor": float(cache_requests / cache_misses) if cache_misses else 0.0,
        },
    }
    write_json_atomic(result_dir / "runner_summary.json", summary, overwrite=True)
    return summary


def run_self_test_all_shards(shard_count: int = 2, force_reset: bool = True, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    outputs = []
    for shard_index in range(shard_count):
        outputs.append(
            run_kernel_runner_shard(
                "self_test",
                shard_index=shard_index,
                shard_count=shard_count,
                force_reset_self_test=force_reset and shard_index == 0,
                root=root,
            )
        )
    summary = summarize_kernel_runner(SELF_TEST_RUN_NAME, root)
    summary["shards"] = outputs
    write_json_atomic(root / "results" / SELF_TEST_RUN_NAME / "runner_summary.json", summary, overwrite=True)
    return summary
