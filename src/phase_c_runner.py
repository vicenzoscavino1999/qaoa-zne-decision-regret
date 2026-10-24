from __future__ import annotations

import ctypes
import json
import math
import os
import shutil
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .circuits import graph_from_config
from .io_utils import (
    PROJECT_ROOT,
    configure_logger,
    ensure_dir,
    read_yaml,
    save_array_atomic,
    sha256_file,
    write_json_atomic,
)
from .landscapes import generate_ideal_landscape, generate_noisy_landscape, generate_zne_landscape
from .phase_c import _assert_phase_b_ready, _phase_c_config, _selected_seeds, _simulator_options_for_profile


RUNNER_VERSION = "phase_c_checkpoint_runner_v1"
DEFAULT_LOCAL_GRID_SIZE = 60
DEFAULT_SIMULATOR_PROFILE = "default_threads_1"
CORE_ALLOCATION = "proportional_to_abs_weights"
ALLOWED_LOCAL_GRID_SIZES = {36, 38, 40, 41, 60}


@dataclass(frozen=True)
class PhaseCTask:
    index: int
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


@dataclass(frozen=True)
class RunnerPlan:
    run_name: str
    mode: str
    grid_size: int
    simulator_profile: str
    simulator_options: dict[str, object]
    instances: tuple[str, ...]
    noise_values: tuple[float, ...]
    shot_budgets: tuple[int, ...]
    seeds: tuple[int, ...]
    zne_methods: dict[str, tuple[int, ...]]
    tasks: tuple[PhaseCTask, ...]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _path_id(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _task_id(parts: list[object]) -> str:
    cleaned = [str(part).replace(".", "p").replace("/", "_").replace("\\", "_") for part in parts]
    return "_".join(cleaned)


def _task_to_json(task: PhaseCTask, root: Path) -> dict[str, Any]:
    data = asdict(task)
    data["scale_factors"] = list(task.scale_factors)
    data["output_path"] = Path(task.output_path).relative_to(root).as_posix()
    data["done_path"] = task.done_path.relative_to(root).as_posix()
    return data


def _plan_json(plan: RunnerPlan, root: Path) -> dict[str, Any]:
    return {
        "runner_version": RUNNER_VERSION,
        "run_name": plan.run_name,
        "mode": plan.mode,
        "grid_size": plan.grid_size,
        "simulator_profile": plan.simulator_profile,
        "simulator_options": plan.simulator_options,
        "instances": list(plan.instances),
        "noise_values": list(plan.noise_values),
        "shot_budgets": list(plan.shot_budgets),
        "seeds": list(plan.seeds),
        "zne_methods": {key: list(value) for key, value in sorted(plan.zne_methods.items())},
        "tasks_total": len(plan.tasks),
        "tasks": [_task_to_json(task, root) for task in plan.tasks],
    }


def _stable_hash(data: dict[str, Any]) -> str:
    raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def _safe_remove_tree(path: Path, root: Path) -> None:
    path = path.resolve()
    root = root.resolve()
    allowed = {
        (root / "data" / "raw" / "phase_c_runner_selftest").resolve(),
        (root / "results" / "phase_c_runner_selftest").resolve(),
    }
    if path not in allowed:
        raise RuntimeError(f"Refusing to remove non-self-test path: {path}")
    if path.exists():
        shutil.rmtree(path)


def reset_self_test_outputs(root: Path | None = None) -> None:
    root = Path(root or PROJECT_ROOT)
    _safe_remove_tree(root / "data" / "raw" / "phase_c_runner_selftest", root)
    _safe_remove_tree(root / "results" / "phase_c_runner_selftest", root)


def _validate_grid_size(grid_size: int, self_test: bool) -> None:
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}.")
    if not self_test and grid_size not in ALLOWED_LOCAL_GRID_SIZES:
        allowed = ", ".join(str(value) for value in sorted(ALLOWED_LOCAL_GRID_SIZES))
        raise ValueError(f"Unsupported Phase C local grid_size={grid_size}. Allowed documented sizes: {allowed}.")


def _validate_shard(shard_index: int, shard_count: int) -> None:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1.")
    if shard_index < 0 or shard_index >= shard_count:
        raise ValueError(f"shard_index must satisfy 0 <= index < {shard_count}, got {shard_index}.")


def _validate_simulator_profile(simulator_profile: str, self_test: bool) -> None:
    if simulator_profile != DEFAULT_SIMULATOR_PROFILE and not self_test:
        raise RuntimeError(
            "Local reduced Phase C is locked to default_threads_1 unless this code is explicitly extended "
            "and revalidated. GPU and density_matrix profiles are not accepted for core output."
        )


def _build_task(
    index: int,
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
) -> PhaseCTask:
    raw_root = root / "data" / "raw" / run_name / instance / f"grid_{grid_size}"
    if kind == "ideal":
        output_path = raw_root / "C_ideal.npy"
        task_id = _task_id([f"{index:06d}", instance, f"grid{grid_size}", "ideal"])
    else:
        if p_noise is None or shots_total is None or seed is None:
            raise ValueError(f"Non-ideal task requires p_noise/shots_total/seed: {kind}")
        base = raw_root / f"p_{p_noise:g}" / f"B_{shots_total}" / f"seed_{seed}"
        if kind == "noisy":
            output_path = base / "C_noisy.npy"
            task_id = _task_id([f"{index:06d}", instance, f"p{_path_id(p_noise)}", f"B{shots_total}", f"seed{seed}", "noisy"])
        elif kind == "zne":
            if zne_order is None or not scale_factors:
                raise ValueError("ZNE task requires zne_order and scale_factors.")
            output_path = base / f"C_zne_{zne_order}_{CORE_ALLOCATION}.npy"
            task_id = _task_id([
                f"{index:06d}",
                instance,
                f"p{_path_id(p_noise)}",
                f"B{shots_total}",
                f"seed{seed}",
                f"zne{zne_order}",
                CORE_ALLOCATION,
            ])
        else:
            raise ValueError(f"Unknown Phase C task kind: {kind}")

    return PhaseCTask(
        index=index,
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


def build_runner_plan(
    config_path: str | Path | None = None,
    grid_size: int = DEFAULT_LOCAL_GRID_SIZE,
    simulator_profile: str = DEFAULT_SIMULATOR_PROFILE,
    self_test: bool = False,
) -> RunnerPlan:
    root = PROJECT_ROOT
    _validate_grid_size(grid_size, self_test=self_test)
    _validate_simulator_profile(simulator_profile, self_test=self_test)
    cfg = read_yaml(config_path or root / "config" / "config_official_pilot_v1.6.yaml")
    simulator_options = _simulator_options_for_profile(simulator_profile)

    if self_test:
        run_name = "phase_c_runner_selftest"
        mode = "self_test"
        instances = ("G1_C4",)
        noise_values = (0.005,)
        shot_budgets = (64,)
        seeds = (42,)
        zne_methods = {"linear": (1, 3)}
    else:
        phase_c = _phase_c_config(cfg)
        if int(phase_c["seed_count"]) != 30:
            raise RuntimeError("Phase C local runner is only validated for the frozen seed_count=30.")
        _assert_phase_b_ready(root, 30)
        if grid_size >= int(phase_c["grid_size"]):
            raise ValueError(
                f"Checkpoint runner is only enabled for documented local reductions below "
                f"the frozen official grid_size={phase_c['grid_size']}."
            )
        run_name = f"phase_c_local_grid{grid_size}"
        mode = "local_reduced"
        instances = tuple(str(value) for value in phase_c["instances"])
        noise_values = tuple(float(value) for value in cfg["noise"]["p_values"])
        shot_budgets = tuple(int(value) for value in cfg["shots"]["total_budget_primary"])
        seeds = tuple(_selected_seeds(root, 30))
        zne_methods = {str(key): tuple(int(value) for value in values) for key, values in cfg["zne"]["methods"].items()}

    index = 0
    tasks: list[PhaseCTask] = []
    for instance in instances:
        tasks.append(_build_task(index, run_name, root, "ideal", instance, grid_size))
        index += 1
        for p_noise in noise_values:
            for shots_total in shot_budgets:
                for seed in seeds:
                    tasks.append(_build_task(index, run_name, root, "noisy", instance, grid_size, p_noise, shots_total, seed))
                    index += 1
                    for zne_order, scale_factors in zne_methods.items():
                        tasks.append(
                            _build_task(
                                index,
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

    return RunnerPlan(
        run_name=run_name,
        mode=mode,
        grid_size=grid_size,
        simulator_profile=simulator_profile,
        simulator_options=simulator_options,
        instances=instances,
        noise_values=noise_values,
        shot_budgets=shot_budgets,
        seeds=seeds,
        zne_methods=zne_methods,
        tasks=tuple(tasks),
    )


def _write_or_validate_plan(plan: RunnerPlan, result_dir: Path, root: Path) -> str:
    plan_path = result_dir / "plan.json"
    data = _plan_json(plan, root)
    plan_hash = _stable_hash(data)
    payload = {**data, "plan_sha256": plan_hash, "written_at_utc": _utc_now()}
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing.get("plan_sha256") != plan_hash:
            raise RuntimeError(f"Existing plan does not match current plan: {plan_path}")
    else:
        write_json_atomic(plan_path, payload, overwrite=False)
    return plan_hash


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def _run_lock(lock_path: Path, force_unlock: bool = False) -> Iterator[None]:
    ensure_dir(lock_path.parent)
    if lock_path.exists():
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            pid = int(lock.get("pid", -1))
        except Exception:
            pid = -1
        stale = pid > 0 and not _process_exists(pid)
        if force_unlock or stale:
            lock_path.unlink()
        else:
            raise RuntimeError(f"Runner lock already exists: {lock_path}")

    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "created_at_utc": _utc_now()}, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        yield
    finally:
        if lock_path.exists():
            try:
                lock = json.loads(lock_path.read_text(encoding="utf-8"))
            except Exception:
                lock = {}
            if int(lock.get("pid", os.getpid())) == os.getpid():
                lock_path.unlink()


def _load_completed_array(task: PhaseCTask, plan: RunnerPlan, repair_missing_metadata: bool) -> bool:
    array_path = Path(task.output_path)
    done_path = task.done_path
    if not array_path.exists():
        if done_path.exists():
            raise RuntimeError(f"Metadata exists without array: {done_path}")
        return False

    array = np.load(array_path, allow_pickle=False)
    expected_shape = (task.grid_size, task.grid_size)
    if array.shape != expected_shape:
        raise RuntimeError(f"Checkpoint has wrong shape: {array_path} shape={array.shape} expected={expected_shape}")
    if not bool(np.isfinite(array).all()):
        raise RuntimeError(f"Checkpoint contains non-finite values: {array_path}")
    actual_hash = sha256_file(array_path)

    if done_path.exists():
        done = json.loads(done_path.read_text(encoding="utf-8"))
        if done.get("array_sha256") != actual_hash:
            raise RuntimeError(f"Checkpoint hash mismatch: {array_path}")
        if done.get("task_id") != task.task_id:
            raise RuntimeError(f"Checkpoint task_id mismatch: {done_path}")
        if done.get("simulator_profile") != plan.simulator_profile:
            raise RuntimeError(f"Checkpoint simulator profile mismatch: {done_path}")
        if done.get("runner_version") != RUNNER_VERSION:
            raise RuntimeError(f"Checkpoint runner version mismatch: {done_path}")
        return True

    if not repair_missing_metadata:
        raise RuntimeError(f"Array exists without done metadata: {array_path}")
    _write_done_metadata(task, plan, actual_hash, elapsed_seconds=None, zne_metadata=None, repaired=True)
    return True


def _write_done_metadata(
    task: PhaseCTask,
    plan: RunnerPlan,
    array_sha256: str,
    elapsed_seconds: float | None,
    zne_metadata: dict[str, object] | None,
    repaired: bool,
) -> None:
    array = np.load(task.output_path, allow_pickle=False)
    payload = {
        "runner_version": RUNNER_VERSION,
        "status": "complete",
        "task_id": task.task_id,
        "task_index": task.index,
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
        "simulator_profile": plan.simulator_profile,
        "simulator_options": plan.simulator_options,
        "array_sha256": array_sha256,
        "array_min": float(np.min(array)),
        "array_max": float(np.max(array)),
        "array_mean": float(np.mean(array)),
        "array_path": Path(task.output_path).relative_to(PROJECT_ROOT).as_posix(),
        "elapsed_seconds": elapsed_seconds,
        "metadata_repaired": repaired,
        "written_at_utc": _utc_now(),
        "zne_metadata": zne_metadata,
    }
    write_json_atomic(task.done_path, payload, overwrite=True)


def _progress_payload(
    plan: RunnerPlan,
    plan_hash: str,
    shard_index: int,
    shard_count: int,
    shard_tasks: list[PhaseCTask],
    status: str,
    generated_this_run: int,
    skipped_this_run: int,
    started_at_utc: str,
) -> dict[str, Any]:
    completed = 0
    pending = 0
    for task in shard_tasks:
        if Path(task.output_path).exists() and task.done_path.exists():
            completed += 1
        else:
            pending += 1
    return {
        "runner_version": RUNNER_VERSION,
        "run_name": plan.run_name,
        "mode": plan.mode,
        "grid_size": plan.grid_size,
        "simulator_profile": plan.simulator_profile,
        "plan_sha256": plan_hash,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "status": status,
        "tasks_in_shard": len(shard_tasks),
        "tasks_completed_in_shard": completed,
        "tasks_pending_in_shard": pending,
        "generated_this_run": generated_this_run,
        "skipped_this_run": skipped_this_run,
        "started_at_utc": started_at_utc,
        "updated_at_utc": _utc_now(),
    }


def _execute_task(task: PhaseCTask, cfg: dict[str, Any], plan: RunnerPlan, logger) -> None:
    graph = graph_from_config(task.instance, cfg)
    started = time.perf_counter()
    zne_metadata: dict[str, object] | None = None
    if task.kind == "ideal":
        array = generate_ideal_landscape(graph, task.grid_size)
    elif task.kind == "noisy":
        if task.p_noise is None or task.shots_total is None or task.seed is None:
            raise RuntimeError(f"Incomplete noisy task: {task.task_id}")
        array = generate_noisy_landscape(
            graph,
            task.grid_size,
            task.p_noise,
            task.shots_total,
            task.seed,
            "phase_c_checkpoint_noisy",
            simulator_options=plan.simulator_options,
        )
    elif task.kind == "zne":
        if task.p_noise is None or task.shots_total is None or task.seed is None or task.zne_order is None:
            raise RuntimeError(f"Incomplete ZNE task: {task.task_id}")
        array, zne_metadata = generate_zne_landscape(
            graph,
            task.grid_size,
            task.p_noise,
            task.shots_total,
            task.seed,
            list(task.scale_factors),
            task.zne_order,
            folding_backend=str(cfg["zne"]["folding_backend"]),
            shot_allocation_strategy=CORE_ALLOCATION,
            simulator_options=plan.simulator_options,
        )
    else:
        raise RuntimeError(f"Unknown task kind: {task.kind}")

    if array.shape != (task.grid_size, task.grid_size):
        raise RuntimeError(f"Generated array has wrong shape for {task.task_id}: {array.shape}")
    if not bool(np.isfinite(array).all()):
        raise RuntimeError(f"Generated array has non-finite values for {task.task_id}")

    output_path = Path(task.output_path)
    save_array_atomic(output_path, array, overwrite=False)
    array_sha256 = sha256_file(output_path)
    elapsed = time.perf_counter() - started
    _write_done_metadata(task, plan, array_sha256, elapsed_seconds=elapsed, zne_metadata=zne_metadata, repaired=False)
    logger.info("completed task=%s seconds=%.3f", task.task_id, elapsed)


def run_phase_c_checkpointed(
    config_path: str | Path | None = None,
    grid_size: int = DEFAULT_LOCAL_GRID_SIZE,
    simulator_profile: str = DEFAULT_SIMULATOR_PROFILE,
    shard_index: int = 0,
    shard_count: int = 1,
    stop_after_tasks: int | None = None,
    force_unlock: bool = False,
    repair_missing_metadata: bool = True,
    self_test: bool = False,
) -> dict[str, Any]:
    root = PROJECT_ROOT
    _validate_shard(shard_index, shard_count)
    plan = build_runner_plan(config_path=config_path, grid_size=grid_size, simulator_profile=simulator_profile, self_test=self_test)
    result_dir = root / "results" / plan.run_name
    ensure_dir(result_dir)
    plan_hash = _write_or_validate_plan(plan, result_dir, root)
    shard_tasks = [task for task in plan.tasks if task.index % shard_count == shard_index]
    if not shard_tasks:
        raise RuntimeError(f"Shard {shard_index}/{shard_count} has no tasks.")

    cfg = read_yaml(config_path or root / "config" / "config_official_pilot_v1.6.yaml")
    logger = configure_logger(root / "logs" / f"{plan.run_name}_shard_{shard_index:03d}_of_{shard_count:03d}.log")
    progress_path = result_dir / f"progress_shard_{shard_index:03d}_of_{shard_count:03d}.json"
    lock_path = result_dir / f"shard_{shard_index:03d}_of_{shard_count:03d}.lock"
    started_at_utc = _utc_now()
    generated_this_run = 0
    skipped_this_run = 0
    processed_this_run = 0

    with _run_lock(lock_path, force_unlock=force_unlock):
        write_json_atomic(
            progress_path,
            _progress_payload(plan, plan_hash, shard_index, shard_count, shard_tasks, "running", 0, 0, started_at_utc),
            overwrite=True,
        )
        for task in shard_tasks:
            if _load_completed_array(task, plan, repair_missing_metadata=repair_missing_metadata):
                skipped_this_run += 1
            else:
                logger.info("starting task=%s", task.task_id)
                _execute_task(task, cfg, plan, logger)
                generated_this_run += 1
            processed_this_run += 1

            status = "running"
            if stop_after_tasks is not None and processed_this_run >= stop_after_tasks:
                status = "interrupted_for_test"
            write_json_atomic(
                progress_path,
                _progress_payload(
                    plan,
                    plan_hash,
                    shard_index,
                    shard_count,
                    shard_tasks,
                    status,
                    generated_this_run,
                    skipped_this_run,
                    started_at_utc,
                ),
                overwrite=True,
            )
            if status == "interrupted_for_test":
                break

        final_status = "interrupted_for_test" if stop_after_tasks is not None and processed_this_run >= stop_after_tasks else "complete"
        write_json_atomic(
            progress_path,
            _progress_payload(
                plan,
                plan_hash,
                shard_index,
                shard_count,
                shard_tasks,
                final_status,
                generated_this_run,
                skipped_this_run,
                started_at_utc,
            ),
            overwrite=True,
        )

    return json.loads(progress_path.read_text(encoding="utf-8"))


def summarize_runner_status(run_name: str, root: Path | None = None) -> dict[str, Any]:
    root = Path(root or PROJECT_ROOT)
    result_dir = root / "results" / run_name
    plan_path = result_dir / "plan.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"Runner plan not found: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    tasks = plan.get("tasks", [])
    completed = 0
    corrupt: list[str] = []
    for task in tasks:
        array_path = root / task["output_path"]
        done_path = root / task["done_path"]
        if not array_path.exists() or not done_path.exists():
            continue
        try:
            done = json.loads(done_path.read_text(encoding="utf-8"))
            if done.get("array_sha256") != sha256_file(array_path):
                corrupt.append(task["task_id"])
                continue
            array = np.load(array_path, allow_pickle=False)
            expected_shape = (int(task["grid_size"]), int(task["grid_size"]))
            if array.shape != expected_shape or not bool(np.isfinite(array).all()):
                corrupt.append(task["task_id"])
                continue
        except Exception:
            corrupt.append(task["task_id"])
            continue
        completed += 1

    total = len(tasks)
    locks = sorted(path.name for path in result_dir.glob("*.lock"))
    temp_files = sorted(path.relative_to(root).as_posix() for path in root.rglob("*.tmp.*") if run_name in path.as_posix())
    percent = 100.0 * completed / total if total else math.nan
    return {
        "runner_version": RUNNER_VERSION,
        "run_name": run_name,
        "status": "complete" if completed == total and not corrupt else "incomplete",
        "tasks_total": total,
        "tasks_completed": completed,
        "tasks_pending": total - completed,
        "percent_complete": percent,
        "corrupt_tasks": corrupt,
        "active_or_stale_locks": locks,
        "temporary_files": temp_files,
        "updated_at_utc": _utc_now(),
    }
