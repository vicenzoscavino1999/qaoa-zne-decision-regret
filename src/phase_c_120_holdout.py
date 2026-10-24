from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
from .phase_c_120_kernel_runner import CORE_ALLOCATION
from .phase_c_120_robustness import _benjamini_hochberg, _bootstrap_pair_stats, _classify_rows, _summary_from_classification
from .phase_c_120_preflight import TARGET_GRID_SIZE


HOLDOUT_VERSION = "phase_c_120_holdout_v1"
HOLDOUT_RUN_NAME = "phase_c_120_holdout_replication"
HOLDOUT_SEED_SOURCE = 20260513
HOLDOUT_SEED_COUNT = 10
RESAMPLES = 10000


@dataclass(frozen=True)
class HoldoutCell:
    instance: str
    noise: float
    shots_total: int
    zne_order: str

    def id(self) -> str:
        return f"{self.instance}|p={self.noise:g}|B={self.shots_total}|zne={self.zne_order}"

    def condition_id(self) -> str:
        return f"{self.instance}|p={self.noise:g}|B={self.shots_total}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool = False) -> None:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float:
    return float(str(value).strip())


def _as_int(value: Any) -> int:
    return int(str(value).strip())


def _csv_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _holdout_dirs(root: Path) -> tuple[Path, Path]:
    return ensure_dir(root / "data" / "raw" / HOLDOUT_RUN_NAME), ensure_dir(root / "results" / HOLDOUT_RUN_NAME)


def _safe_reset(path: Path, root: Path) -> None:
    resolved = path.resolve()
    allowed = {
        (root / "data" / "raw" / HOLDOUT_RUN_NAME).resolve(),
        (root / "results" / HOLDOUT_RUN_NAME).resolve(),
    }
    if resolved not in allowed:
        raise RuntimeError(f"Refusing to remove non-holdout path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _load_targets(root: Path) -> list[HoldoutCell]:
    path = root / "results" / "phase_c_120_analysis" / "paper_table_discordant_fdr.csv"
    rows = _read_csv(path)
    cells = [
        HoldoutCell(
            instance=str(row["instance"]),
            noise=_as_float(row["noise"]),
            shots_total=_as_int(row["shots_total"]),
            zne_order=str(row["zne_order"]),
        )
        for row in rows
    ]
    return sorted(cells, key=lambda c: (c.instance, c.noise, c.shots_total, c.zne_order))


def _official_seeds(root: Path) -> set[int]:
    path = root / "results" / "phase_c_120_kernel_official" / "manifest_final.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {int(seed) for seed in manifest.get("engine", {}).get("seeds", [])}


def _holdout_seeds(root: Path) -> list[int]:
    official = _official_seeds(root)
    generated = np.random.SeedSequence(HOLDOUT_SEED_SOURCE).generate_state(HOLDOUT_SEED_COUNT * 3, dtype=np.uint32)
    out: list[int] = []
    for value in generated:
        seed = int(value)
        if seed in official or seed in out:
            continue
        out.append(seed)
        if len(out) == HOLDOUT_SEED_COUNT:
            return out
    raise RuntimeError("Could not generate enough holdout seeds.")


def _raw_base(raw_dir: Path, cell: HoldoutCell, seed: int) -> Path:
    return raw_dir / cell.instance / f"grid_{TARGET_GRID_SIZE}" / f"p_{cell.noise:g}" / f"B_{cell.shots_total}" / f"seed_{seed}"


def _ideal_path(root: Path, instance: str) -> Path:
    return root / "data" / "raw" / "phase_c_120_kernel_official" / instance / f"grid_{TARGET_GRID_SIZE}" / "C_ideal.npy"


def _load_or_generate_ideal(root: Path, cfg: dict[str, Any], instance: str) -> np.ndarray:
    path = _ideal_path(root, instance)
    if path.is_file():
        return np.load(path, allow_pickle=False)
    return generate_ideal_landscape(graph_from_config(instance, cfg), TARGET_GRID_SIZE)


def _save_done(array_path: Path, payload: dict[str, Any]) -> None:
    array = np.load(array_path, allow_pickle=False)
    done = {
        **payload,
        "status": "complete",
        "shape": list(array.shape),
        "array_sha256": sha256_file(array_path),
        "array_min": float(np.min(array)),
        "array_max": float(np.max(array)),
        "array_mean": float(np.mean(array)),
        "written_at_utc": _utc_now(),
    }
    write_json_atomic(array_path.with_suffix(".done.json"), done, overwrite=True)


def _generate_condition(
    root: Path,
    raw_dir: Path,
    cfg: dict[str, Any],
    condition: tuple[str, float, int],
    seed: int,
    zne_orders: list[str],
    cache: KernelProbabilityCache,
) -> dict[str, Any]:
    instance, noise, shots = condition
    graph = graph_from_config(instance, cfg)
    base = raw_dir / instance / f"grid_{TARGET_GRID_SIZE}" / f"p_{noise:g}" / f"B_{shots}" / f"seed_{seed}"
    noisy_path = base / "C_noisy_custom_kernel.npy"
    generated = 0
    if not noisy_path.is_file():
        array, metadata = generate_custom_kernel_noisy_landscape(
            graph,
            TARGET_GRID_SIZE,
            noise,
            shots,
            seed,
            probability_cache=cache,
        )
        save_array_atomic(noisy_path, array, overwrite=False)
        _save_done(noisy_path, {
            "holdout_version": HOLDOUT_VERSION,
            "custom_kernel_version": CUSTOM_KERNEL_VERSION,
            "kind": "noisy",
            "instance": instance,
            "noise": noise,
            "shots_total": shots,
            "seed": seed,
            "metadata": metadata,
        })
        generated += 1
    for zne_order in zne_orders:
        scales = tuple(int(scale) for scale in cfg["zne"]["methods"][zne_order])
        zne_path = base / f"C_zne_{zne_order}_{CORE_ALLOCATION}_custom_kernel.npy"
        if zne_path.is_file():
            continue
        array, metadata = generate_custom_kernel_zne_landscape(
            graph,
            TARGET_GRID_SIZE,
            noise,
            shots,
            seed,
            list(scales),
            zne_order,
            shot_allocation_strategy=CORE_ALLOCATION,
            probability_cache=cache,
        )
        save_array_atomic(zne_path, array, overwrite=False)
        _save_done(zne_path, {
            "holdout_version": HOLDOUT_VERSION,
            "custom_kernel_version": CUSTOM_KERNEL_VERSION,
            "kind": "zne",
            "instance": instance,
            "noise": noise,
            "shots_total": shots,
            "seed": seed,
            "zne_order": zne_order,
            "scale_factors": list(scales),
            "allocation": CORE_ALLOCATION,
            "metadata": metadata,
        })
        generated += 1
    return {"condition": "|".join(str(x) for x in condition), "seed": seed, "generated": generated}


def _generate_group_worker(args: tuple[str, str, list[tuple[str, float, int]], dict[str, list[str]], list[int]]) -> list[dict[str, Any]]:
    root_s, raw_dir_s, conditions, condition_to_orders, seeds = args
    root = Path(root_s)
    raw_dir = Path(raw_dir_s)
    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    group = (conditions[0][0], conditions[0][1])
    cache = new_probability_cache()
    rows: list[dict[str, Any]] = []
    for condition in sorted(conditions, key=lambda item: item[2]):
        key = "|".join(str(x) for x in condition)
        for seed in seeds:
            row = _generate_condition(root, raw_dir, cfg, condition, seed, condition_to_orders[key], cache)
            row["cache_group"] = f"{group[0]}|p={group[1]:g}"
            rows.append(row)
    cache_summary = cache.summary()
    for row in rows:
        row["cache_group_requests"] = cache_summary["requests"]
        row["cache_group_misses"] = cache_summary["misses"]
        row["cache_group_hits"] = cache_summary["hits"]
    return rows


def _compute_holdout_metrics(
    root: Path,
    raw_dir: Path,
    cfg: dict[str, Any],
    targets: list[HoldoutCell],
    seeds: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics_cfg = MetricsConfig.from_experiment_config(cfg)
    ideal_cache: dict[str, np.ndarray] = {}
    noisy_cache: dict[tuple[str, float, int, int], np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    for seed in seeds:
        for condition in sorted({(c.instance, c.noise, c.shots_total) for c in targets}):
            instance, noise, shots = condition
            if instance not in ideal_cache:
                ideal_cache[instance] = _load_or_generate_ideal(root, cfg, instance)
            ideal = ideal_cache[instance]
            base = raw_dir / instance / f"grid_{TARGET_GRID_SIZE}" / f"p_{noise:g}" / f"B_{shots}" / f"seed_{seed}"
            noisy = np.load(base / "C_noisy_custom_kernel.npy", allow_pickle=False)
            noisy_cache[(instance, noise, shots, seed)] = noisy
            noisy_metrics = method_metrics(noisy, ideal, metrics_cfg)
            metric_rows.append({
                "phase": "C120_HOLDOUT",
                "run_name": HOLDOUT_RUN_NAME,
                "instance": instance,
                "noise": noise,
                "shots_total": shots,
                "seed": seed,
                "method": "noisy",
                "zne_order": "",
                "allocation": "none",
                "array_path": (base / "C_noisy_custom_kernel.npy").relative_to(root).as_posix(),
                "array_sha256": sha256_file(base / "C_noisy_custom_kernel.npy"),
                **noisy_metrics,
            })
        for cell in targets:
            ideal = ideal_cache[cell.instance]
            noisy = noisy_cache[(cell.instance, cell.noise, cell.shots_total, seed)]
            base = _raw_base(raw_dir, cell, seed)
            zne_path = base / f"C_zne_{cell.zne_order}_{CORE_ALLOCATION}_custom_kernel.npy"
            zne = np.load(zne_path, allow_pickle=False)
            zne_metrics = method_metrics(zne, ideal, metrics_cfg, noisy_reference=noisy)
            comp = comparison_metrics(noisy, zne, ideal)
            metric_rows.append({
                "phase": "C120_HOLDOUT",
                "run_name": HOLDOUT_RUN_NAME,
                "instance": cell.instance,
                "noise": cell.noise,
                "shots_total": cell.shots_total,
                "seed": seed,
                "method": f"zne_{cell.zne_order}",
                "zne_order": cell.zne_order,
                "allocation": CORE_ALLOCATION,
                "scale_factors": "|".join(str(x) for x in cfg["zne"]["methods"][cell.zne_order]),
                "array_path": zne_path.relative_to(root).as_posix(),
                "array_sha256": sha256_file(zne_path),
                **zne_metrics,
            })
            comparison_rows.append({
                "phase": "C120_HOLDOUT",
                "run_name": HOLDOUT_RUN_NAME,
                "comparison_type": "zne_vs_noisy",
                "cell_id": cell.id(),
                "instance": cell.instance,
                "noise": cell.noise,
                "shots_total": cell.shots_total,
                "seed": seed,
                "zne_order": cell.zne_order,
                "allocation": CORE_ALLOCATION,
                "scale_factors": "|".join(str(x) for x in cfg["zne"]["methods"][cell.zne_order]),
                "noisy_array_path": (base / "C_noisy_custom_kernel.npy").relative_to(root).as_posix(),
                "noisy_array_sha256": sha256_file(base / "C_noisy_custom_kernel.npy"),
                "zne_array_path": zne_path.relative_to(root).as_posix(),
                "zne_array_sha256": sha256_file(zne_path),
                **comp,
            })
    bootstrap_rows = _bootstrap_holdout(cfg, targets, comparison_rows, metric_rows)
    return metric_rows, comparison_rows, bootstrap_rows


def _bootstrap_holdout(
    cfg: dict[str, Any],
    targets: list[HoldoutCell],
    comparison_rows: list[dict[str, Any]],
    metric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {cell.id(): [] for cell in targets}
    for row in comparison_rows:
        grouped[str(row["cell_id"])].append(row)
    metrics_by_key = {
        (row["instance"], float(row["noise"]), int(row["shots_total"]), int(row["seed"]), row["method"]): row
        for row in metric_rows
    }
    tau_r = float(cfg["phase_c"]["tau_R"])
    rows: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(targets, start=1):
        source = sorted(grouped[cell.id()], key=lambda row: int(row["seed"]))
        r_values = np.asarray([float(row["r_mse"]) for row in source], dtype=float)
        d_values = np.asarray([float(row["d_R"]) for row in source], dtype=float)
        stats = _bootstrap_pair_stats(
            r_values,
            d_values,
            RESAMPLES,
            seed=HOLDOUT_SEED_SOURCE + 1000 + cell_index,
            include_bca=False,
        )
        zne_metric_rows = [
            metrics_by_key[(cell.instance, cell.noise, cell.shots_total, int(row["seed"]), f"zne_{cell.zne_order}")]
            for row in source
        ]
        below_count = sum(_csv_bool(row["below_detection_floor"]) for row in zne_metric_rows)
        breakdown_count = sum(_csv_bool(row["extrapolation_breakdown"]) for row in zne_metric_rows)
        rows.append({
            "cell_index": cell_index,
            "cell_id": cell.id(),
            "instance": cell.instance,
            "noise": cell.noise,
            "shots_total": cell.shots_total,
            "zne_order": cell.zne_order,
            "allocation": CORE_ALLOCATION,
            "tau_R": tau_r,
            "below_detection_floor_count": below_count,
            "below_detection_floor_rate": float(below_count / len(source)),
            "excluded_below_detection_floor": bool(below_count > 0),
            "extrapolation_breakdown_count": breakdown_count,
            "extrapolation_breakdown_rate": float(breakdown_count / len(source)),
            "excluded_extrapolation_breakdown": bool(breakdown_count > 0),
            **stats,
        })
    _benjamini_hochberg(rows, "p_conjunction", "q_conjunction_bh", "fdr_conjunction_pass", 0.10)
    return _classify_rows(rows, cfg, q_threshold=0.10)


def _combined_target_bootstrap(
    root: Path,
    cfg: dict[str, Any],
    targets: list[HoldoutCell],
    holdout_comparisons: list[dict[str, Any]],
    holdout_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    official_comparisons = _read_csv(root / "results" / "phase_c_120_analysis" / "comparisons_per_seed.csv")
    official_metrics = _read_csv(root / "results" / "phase_c_120_analysis" / "metrics_per_seed.csv")
    target_ids = {cell.id() for cell in targets}
    official_target_rows = []
    for row in official_comparisons:
        cell_id = f"{row['instance']}|p={float(row['noise']):g}|B={int(row['shots_total'])}|zne={row['zne_order']}"
        if cell_id in target_ids:
            official_target_rows.append({**row, "cell_id": cell_id})
    combined_comparisons: list[dict[str, Any]] = [*official_target_rows, *holdout_comparisons]
    combined_metrics: list[dict[str, Any]] = [*official_metrics, *holdout_metrics]
    return _bootstrap_holdout(cfg, targets, combined_comparisons, combined_metrics)


def _support_summary(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    positive_direction = [
        row for row in rows
        if float(row["median_r_mse"]) > 0.0 and float(row["median_d_R"]) > 0.0
    ]
    discordant_fdr = [row for row in rows if _csv_bool(row["discordant_fdr"])]
    discordant_pre = [row for row in rows if _csv_bool(row["discordant_pre_fdr"])]
    return {
        "label": label,
        "cells": len(rows),
        "positive_direction_count": len(positive_direction),
        "discordant_pre_fdr_count": len(discordant_pre),
        "discordant_fdr_count": len(discordant_fdr),
        "median_r_mse_min": min(float(row["median_r_mse"]) for row in rows),
        "median_d_R_min": min(float(row["median_d_R"]) for row in rows),
        "supportive_direction_majority": bool(len(positive_direction) >= math.ceil(len(rows) / 2)),
        "supportive_all_direction": bool(len(positive_direction) == len(rows)),
    }


def run_phase_c_120_holdout(
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    raw_dir, result_dir = _holdout_dirs(root)
    if overwrite:
        _safe_reset(raw_dir, root)
        _safe_reset(result_dir, root)
        raw_dir, result_dir = _holdout_dirs(root)
    elif any(result_dir.iterdir()) or any(raw_dir.iterdir()):
        raise FileExistsError("Holdout outputs already exist. Use --force to overwrite.")

    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    targets = _load_targets(root)
    seeds = _holdout_seeds(root)
    unique_conditions = sorted({(cell.instance, cell.noise, cell.shots_total) for cell in targets})
    plan = {
        "version": HOLDOUT_VERSION,
        "created_at_utc": _utc_now(),
        "run_name": HOLDOUT_RUN_NAME,
        "grid_size": TARGET_GRID_SIZE,
        "seed_source": f"SeedSequence({HOLDOUT_SEED_SOURCE})",
        "holdout_seed_count": len(seeds),
        "holdout_seeds": seeds,
        "target_cells": [cell.__dict__ | {"cell_id": cell.id()} for cell in targets],
        "unique_noisy_conditions": [
            {"instance": instance, "noise": noise, "shots_total": shots}
            for instance, noise, shots in unique_conditions
        ],
        "landscape_tasks_expected": len(seeds) * (len(unique_conditions) + len(targets)),
        "scope_limitation": "targeted_holdout_not_full_grid_replication",
    }
    write_json_atomic(result_dir / "holdout_seed_inventory.json", plan, overwrite=True)
    write_json_atomic(result_dir / "holdout_runner_plan.json", plan, overwrite=True)

    started = time.perf_counter()
    condition_to_orders: dict[tuple[str, float, int], list[str]] = {}
    for condition in unique_conditions:
        condition_to_orders[condition] = sorted({cell.zne_order for cell in targets if (cell.instance, cell.noise, cell.shots_total) == condition})

    progress_rows: list[dict[str, Any]] = []
    generated_total = 0
    grouped_conditions: dict[tuple[str, float], list[tuple[str, float, int]]] = {}
    for condition in unique_conditions:
        grouped_conditions.setdefault((condition[0], condition[1]), []).append(condition)
    condition_to_orders_tokenized = {
        "|".join(str(x) for x in condition): orders
        for condition, orders in condition_to_orders.items()
    }
    worker_count = min(len(grouped_conditions), max(1, int(os.cpu_count() or 1)))
    worker_args = [
        (str(root), str(raw_dir), grouped_conditions[group], condition_to_orders_tokenized, seeds)
        for group in sorted(grouped_conditions)
    ]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(_generate_group_worker, args) for args in worker_args]
        for future in as_completed(futures):
            rows = future.result()
            generated_total += sum(int(row["generated"]) for row in rows)
            progress_rows.extend(rows)
    progress_rows = sorted(progress_rows, key=lambda row: (str(row["cache_group"]), str(row["condition"]), int(row["seed"])))
    metric_rows, comparison_rows, holdout_bootstrap = _compute_holdout_metrics(root, raw_dir, cfg, targets, seeds)
    combined_bootstrap = _combined_target_bootstrap(root, cfg, targets, comparison_rows, metric_rows)
    holdout_summary = _support_summary(holdout_bootstrap, "holdout_only")
    combined_summary = _support_summary(combined_bootstrap, "primary_plus_holdout_targets")
    elapsed = time.perf_counter() - started

    _write_csv(result_dir / "holdout_generation_progress.csv", progress_rows, overwrite=True)
    _write_csv(result_dir / "holdout_metrics_per_seed.csv", metric_rows, overwrite=True)
    _write_csv(result_dir / "holdout_comparisons_per_seed.csv", comparison_rows, overwrite=True)
    _write_csv(result_dir / "holdout_replication_by_cell.csv", holdout_bootstrap, overwrite=True)
    _write_csv(result_dir / "holdout_combined_primary_plus_holdout_by_cell.csv", combined_bootstrap, overwrite=True)

    status = "supportive" if holdout_summary["supportive_direction_majority"] and combined_summary["supportive_direction_majority"] else "mixed_or_not_supportive"
    summary = {
        "version": HOLDOUT_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_c_120_holdout_complete",
        "interpretation": status,
        "wall_clock_seconds": elapsed,
        "generated_arrays": generated_total,
        "landscape_tasks_expected": plan["landscape_tasks_expected"],
        "holdout_only": holdout_summary,
        "primary_plus_holdout_targets": combined_summary,
        "scope_limitation": "Targeted holdout is not a full-grid replication and does not redefine primary GO/NO-GO.",
        "outputs": {
            "holdout_seed_inventory": "results/phase_c_120_holdout_replication/holdout_seed_inventory.json",
            "holdout_runner_plan": "results/phase_c_120_holdout_replication/holdout_runner_plan.json",
            "holdout_metrics_per_seed": "results/phase_c_120_holdout_replication/holdout_metrics_per_seed.csv",
            "holdout_comparisons_per_seed": "results/phase_c_120_holdout_replication/holdout_comparisons_per_seed.csv",
            "holdout_replication_by_cell": "results/phase_c_120_holdout_replication/holdout_replication_by_cell.csv",
            "holdout_combined_primary_plus_holdout_by_cell": "results/phase_c_120_holdout_replication/holdout_combined_primary_plus_holdout_by_cell.csv",
        },
    }
    write_json_atomic(result_dir / "holdout_replication_summary.json", summary, overwrite=True)
    return summary
