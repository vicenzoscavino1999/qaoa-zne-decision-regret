from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .circuits import graph_from_config
from .config import MetricsConfig
from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, save_array_atomic, sha256_file, write_json_atomic
from .landscapes import generate_ideal_landscape
from .metrics import comparison_metrics, method_metrics
from .phase_c_120_custom_kernel import (
    CUSTOM_KERNEL_VERSION,
    KernelProbabilityCache,
    generate_custom_kernel_noisy_landscape,
    generate_custom_kernel_zne_landscape,
    new_probability_cache,
)
from .phase_c_120_preflight import PROTOCOL_VERSION


KERNEL_PILOT_VERSION = "phase_c_120_kernel_pilot_v1"
RUN_NAME = "phase_c_120_kernel_pilot"
GRID_SIZE = 4
INSTANCES = ("G1_C4", "G2_weighted4", "G3_weighted6")
NOISE_VALUES = (0.005, 0.01)
SHOT_BUDGETS = (512, 1024)
SEED_COUNT = 2
CORE_ALLOCATION = "proportional_to_abs_weights"


@dataclass(frozen=True)
class KernelPilotTask:
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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _clean_token(value: object) -> str:
    return str(value).replace(".", "p").replace("/", "_").replace("\\", "_")


def _task_id(parts: list[object]) -> str:
    return "_".join(_clean_token(part) for part in parts)


def _safe_reset(path: Path, root: Path) -> None:
    resolved = path.resolve()
    allowed = {
        (root / "data" / "raw" / RUN_NAME).resolve(),
        (root / "results" / RUN_NAME).resolve(),
    }
    if resolved not in allowed:
        raise RuntimeError(f"Refusing to remove non-pilot path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _build_task(
    index: int,
    root: Path,
    kind: str,
    instance: str,
    p_noise: float | None = None,
    shots_total: int | None = None,
    seed: int | None = None,
    zne_order: str | None = None,
    scale_factors: tuple[int, ...] = (),
) -> KernelPilotTask:
    raw_root = root / "data" / "raw" / RUN_NAME / instance / f"grid_{GRID_SIZE}"
    if kind == "ideal":
        output_path = raw_root / "C_ideal.npy"
        task_id = _task_id([f"{index:06d}", instance, "grid4", "ideal"])
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
            raise ValueError(f"Unknown pilot task kind: {kind}")
    return KernelPilotTask(
        index=index,
        task_id=task_id,
        kind=kind,
        instance=instance,
        grid_size=GRID_SIZE,
        output_path=str(output_path),
        p_noise=p_noise,
        shots_total=shots_total,
        seed=seed,
        zne_order=zne_order,
        scale_factors=scale_factors,
    )


def build_kernel_pilot_plan(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    seeds_cfg = read_yaml(root / "config" / "seeds.yaml")
    selected_seeds = tuple(int(seed) for seed in seeds_cfg["official_candidates"][:SEED_COUNT])
    zne_methods = {str(order): tuple(int(scale) for scale in scales) for order, scales in cfg["zne"]["methods"].items()}
    tasks: list[KernelPilotTask] = []
    index = 0
    for instance in INSTANCES:
        tasks.append(_build_task(index, root, "ideal", instance))
        index += 1
        for p_noise in NOISE_VALUES:
            for shots_total in SHOT_BUDGETS:
                for seed in selected_seeds:
                    tasks.append(_build_task(index, root, "noisy", instance, p_noise, shots_total, seed))
                    index += 1
                    for zne_order, scale_factors in zne_methods.items():
                        tasks.append(_build_task(index, root, "zne", instance, p_noise, shots_total, seed, zne_order, scale_factors))
                        index += 1
    return {
        "pilot_version": KERNEL_PILOT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "custom_kernel_version": CUSTOM_KERNEL_VERSION,
        "run_name": RUN_NAME,
        "mode": "custom_kernel_end_to_end_pilot",
        "grid_size": GRID_SIZE,
        "instances": list(INSTANCES),
        "noise_values": list(NOISE_VALUES),
        "shot_budgets": list(SHOT_BUDGETS),
        "seeds": list(selected_seeds),
        "zne_methods": {order: list(scales) for order, scales in zne_methods.items()},
        "tasks_total": len(tasks),
        "tasks": [{**asdict(task), "scale_factors": list(task.scale_factors), "done_path": task.done_path.relative_to(root).as_posix(), "output_path": Path(task.output_path).relative_to(root).as_posix()} for task in tasks],
    }


def _load_task(task: KernelPilotTask) -> np.ndarray:
    array = np.load(task.output_path, allow_pickle=False)
    if array.shape != (task.grid_size, task.grid_size):
        raise RuntimeError(f"Pilot task has wrong shape: {task.task_id} {array.shape}")
    if not bool(np.isfinite(array).all()):
        raise RuntimeError(f"Pilot task has non-finite values: {task.task_id}")
    return array


def _is_task_complete(task: KernelPilotTask) -> bool:
    array_path = Path(task.output_path)
    done_path = task.done_path
    if not array_path.exists():
        if done_path.exists():
            raise RuntimeError(f"Pilot done metadata exists without array: {done_path}")
        return False
    if not done_path.exists():
        raise RuntimeError(f"Pilot array exists without done metadata: {array_path}")
    array = _load_task(task)
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if done.get("array_sha256") != sha256_file(array_path):
        raise RuntimeError(f"Pilot hash mismatch: {array_path}")
    if done.get("task_id") != task.task_id:
        raise RuntimeError(f"Pilot task id mismatch: {done_path}")
    if done.get("shape") != list(array.shape):
        raise RuntimeError(f"Pilot shape metadata mismatch: {done_path}")
    return True


def _write_done(task: KernelPilotTask, metadata: dict[str, Any], elapsed_seconds: float) -> None:
    array = _load_task(task)
    payload = {
        "pilot_version": KERNEL_PILOT_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "custom_kernel_version": CUSTOM_KERNEL_VERSION,
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
        "array_sha256": sha256_file(task.output_path),
        "array_min": float(np.min(array)),
        "array_max": float(np.max(array)),
        "array_mean": float(np.mean(array)),
        "array_path": Path(task.output_path).relative_to(PROJECT_ROOT).as_posix(),
        "elapsed_seconds": elapsed_seconds,
        "written_at_utc": _utc_now(),
        "metadata": metadata,
    }
    write_json_atomic(task.done_path, payload)


def _task_from_json(row: dict[str, Any], root: Path) -> KernelPilotTask:
    return KernelPilotTask(
        index=int(row["index"]),
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


def _execute_task(task: KernelPilotTask, cfg: dict[str, Any], cache: KernelProbabilityCache) -> tuple[bool, float]:
    if _is_task_complete(task):
        return False, 0.0
    graph = graph_from_config(task.instance, cfg)
    started = time.perf_counter()
    if task.kind == "ideal":
        array = generate_ideal_landscape(graph, task.grid_size)
        metadata: dict[str, Any] = {"engine": "qiskit_statevector_ideal", "method": "ideal"}
    elif task.kind == "noisy":
        if task.p_noise is None or task.shots_total is None or task.seed is None:
            raise RuntimeError(f"Incomplete noisy pilot task: {task.task_id}")
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
            raise RuntimeError(f"Incomplete ZNE pilot task: {task.task_id}")
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
        raise RuntimeError(f"Unknown pilot task kind: {task.kind}")
    elapsed = time.perf_counter() - started
    save_array_atomic(task.output_path, array, overwrite=False)
    _write_done(task, metadata, elapsed)
    return True, elapsed


def _write_progress(result_dir: Path, plan: dict[str, Any], generated: int, skipped: int, status: str, cache: KernelProbabilityCache) -> None:
    payload = {
        "pilot_version": KERNEL_PILOT_VERSION,
        "run_name": RUN_NAME,
        "status": status,
        "tasks_total": int(plan["tasks_total"]),
        "generated_this_run": generated,
        "skipped_this_run": skipped,
        "cache": cache.summary(),
        "updated_at_utc": _utc_now(),
    }
    write_json_atomic(result_dir / "progress.json", payload)


def _write_metrics(root: Path, plan: dict[str, Any], result_dir: Path) -> dict[str, int]:
    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    metrics_config = MetricsConfig.from_experiment_config(cfg)
    tasks = [_task_from_json(row, root) for row in plan["tasks"]]
    by_key = {(task.instance, task.kind, task.p_noise, task.shots_total, task.seed, task.zne_order): task for task in tasks}
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for instance in INSTANCES:
        ideal = _load_task(by_key[(instance, "ideal", None, None, None, None)])
        for p_noise in NOISE_VALUES:
            for shots_total in SHOT_BUDGETS:
                for seed in plan["seeds"]:
                    noisy_task = by_key[(instance, "noisy", p_noise, shots_total, seed, None)]
                    noisy = _load_task(noisy_task)
                    noisy_metrics = method_metrics(noisy, ideal, metrics_config)
                    metric_rows.append({
                        "instance": instance,
                        "noise": p_noise,
                        "shots_total": shots_total,
                        "seed": seed,
                        "method": "noisy",
                        "allocation": "none",
                        **noisy_metrics,
                    })
                    for zne_order in sorted(plan["zne_methods"]):
                        zne_task = by_key[(instance, "zne", p_noise, shots_total, seed, zne_order)]
                        zne = _load_task(zne_task)
                        zne_metrics = method_metrics(zne, ideal, metrics_config, noisy_reference=noisy)
                        metric_rows.append({
                            "instance": instance,
                            "noise": p_noise,
                            "shots_total": shots_total,
                            "seed": seed,
                            "method": f"zne_{zne_order}",
                            "allocation": CORE_ALLOCATION,
                            **zne_metrics,
                        })
                        comparison_rows.append({
                            "instance": instance,
                            "noise": p_noise,
                            "shots_total": shots_total,
                            "seed": seed,
                            "zne_order": zne_order,
                            "allocation": CORE_ALLOCATION,
                            "comparison_type": "zne_vs_noisy",
                            **comparison_metrics(noisy, zne, ideal),
                        })

    _write_csv(result_dir / "metrics.csv", metric_rows)
    _write_csv(result_dir / "comparisons.csv", comparison_rows)
    return {"metric_rows": len(metric_rows), "comparison_rows": len(comparison_rows)}


def _determinism_check(cfg: dict[str, Any]) -> bool:
    graph = graph_from_config("G1_C4", cfg)
    left, _ = generate_custom_kernel_noisy_landscape(graph, 3, 0.005, 128, 42, probability_cache=new_probability_cache())
    right, _ = generate_custom_kernel_noisy_landscape(graph, 3, 0.005, 128, 42, probability_cache=new_probability_cache())
    return bool(np.array_equal(left, right))


def run_kernel_pilot(overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    result_dir = root / "results" / RUN_NAME
    raw_dir = root / "data" / "raw" / RUN_NAME
    if overwrite:
        _safe_reset(raw_dir, root)
        _safe_reset(result_dir, root)
    ensure_dir(result_dir)

    plan = build_kernel_pilot_plan(root)
    write_json_atomic(result_dir / "plan.json", plan, overwrite=True)
    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    tasks = [_task_from_json(row, root) for row in plan["tasks"]]
    cache = new_probability_cache()
    generated = 0
    skipped = 0
    elapsed_by_task: list[float] = []
    started = time.perf_counter()

    for task in tasks:
        was_generated, elapsed = _execute_task(task, cfg, cache)
        if was_generated:
            generated += 1
            elapsed_by_task.append(elapsed)
        else:
            skipped += 1
        _write_progress(result_dir, plan, generated, skipped, "running", cache)

    resume_skips = sum(1 for task in tasks if _is_task_complete(task))
    row_counts = _write_metrics(root, plan, result_dir)
    deterministic_sampling_passed = _determinism_check(cfg)
    expected_probability_misses = len(INSTANCES) * len(NOISE_VALUES) * (GRID_SIZE * GRID_SIZE) * len({scale for scales in plan["zne_methods"].values() for scale in scales})
    cache_summary = cache.summary()
    output_arrays = [Path(task.output_path) for task in tasks]
    done_files = [task.done_path for task in tasks]
    report = {
        "created_at_utc": _utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "pilot_version": KERNEL_PILOT_VERSION,
        "custom_kernel_version": CUSTOM_KERNEL_VERSION,
        "status": "phase_c_120_kernel_pilot_passed",
        "scope": "end_to_end_custom_kernel_pilot_with_probability_reuse_sampling_metrics_hashes_and_resume_checks",
        "run_name": RUN_NAME,
        "grid_size": GRID_SIZE,
        "instances": list(INSTANCES),
        "noise_values": list(NOISE_VALUES),
        "shot_budgets": list(SHOT_BUDGETS),
        "seeds": list(plan["seeds"]),
        "zne_methods": plan["zne_methods"],
        "tasks_total": len(tasks),
        "tasks_generated": generated,
        "tasks_skipped_initial": skipped,
        "tasks_complete_after_run": resume_skips,
        "output_arrays": len(output_arrays),
        "done_files": len(done_files),
        "metrics_rows": row_counts["metric_rows"],
        "comparison_rows": row_counts["comparison_rows"],
        "cache": cache_summary,
        "expected_probability_cache_misses": expected_probability_misses,
        "probability_cache_misses_match_expected": int(cache_summary["misses"]) == expected_probability_misses,
        "probability_reuse_validated": int(cache_summary["hits"]) > int(cache_summary["misses"]),
        "deterministic_sampling_passed": deterministic_sampling_passed,
        "arrays_finite": all(bool(np.isfinite(np.load(path, allow_pickle=False)).all()) for path in output_arrays),
        "elapsed_seconds": time.perf_counter() - started,
        "mean_generated_task_seconds": float(np.mean(elapsed_by_task)) if elapsed_by_task else 0.0,
        "decision": "kernel_runner_pilot_ready_for_official_runner_integration",
        "limitations": [
            "This is a 4 x 4 operational pilot, not an official Phase C result.",
            "The pilot validates probability reuse and deterministic NumPy finite-shot sampling, not bit-identical Aer sampling.",
            "A full 120 x 120 runner still needs sharding/monitoring policy before official execution.",
        ],
    }
    if not (
        report["tasks_complete_after_run"] == report["tasks_total"]
        and report["output_arrays"] == report["tasks_total"]
        and report["done_files"] == report["tasks_total"]
        and report["probability_cache_misses_match_expected"]
        and report["probability_reuse_validated"]
        and report["deterministic_sampling_passed"]
        and report["arrays_finite"]
    ):
        report["status"] = "phase_c_120_kernel_pilot_failed"
        report["decision"] = "do_not_integrate_kernel_runner_until_investigated"
    write_json_atomic(result_dir / "pilot_summary.json", report)
    _write_progress(result_dir, plan, generated, skipped, report["status"], cache)
    return report
