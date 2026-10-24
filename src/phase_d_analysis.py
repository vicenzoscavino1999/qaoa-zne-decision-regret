from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from .config import MetricsConfig
from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, sha256_file, write_json_atomic
from .metrics import comparison_metrics, method_metrics
from .phase_c_120_analysis import (
    _add_landscape_stats,
    _as_float,
    _as_int,
    _benjamini_hochberg,
    _bootstrap_pair_stats,
    _csv_bool,
    _read_csv,
    _write_csv,
)
from .phase_c_120_preflight import PROTOCOL_VERSION
from .phase_d_completion import BLOCK_REPORT_NAME, FINAL_MANIFEST_NAME, GLOBAL_RESULT_DIR
from .phase_d_kernel_runner import CORE_ALLOCATION, OFFICIAL_RUN_NAME, PHASE_D_BLOCK_IDS, PHASE_D_KERNEL_RUNNER_VERSION, TARGET_GRID_SIZE


PHASE_D_ANALYSIS_VERSION = "phase_d_analysis_v1"
VALIDATION_REPORT = "analysis_input_validation.json"
INVENTORY_CSV = "analysis_input_inventory.csv"
AGGREGATION_SUMMARY = "aggregation_summary.json"
METRICS_CSV = "metrics_per_seed.csv"
COMPARISONS_CSV = "comparisons_per_seed.csv"
ANALYSIS_MANIFEST = "analysis_manifest.json"
BOOTSTRAP_BY_CELL_CSV = "bootstrap_by_cell.csv"
CLASSIFICATION_MATRIX_CSV = "classification_matrix.csv"
BLOCK_GO_SUMMARY = "block_go_summary.json"
GLOBAL_SUMMARY = "fdr_global_540_summary.json"
INSTANCE_ENDPOINT_CSV = "instance_primary_endpoint.csv"
INSTANCE_ENDPOINT_JSON = "instance_primary_endpoint.json"
FDR_SENSITIVITY_CSV = "fdr_sensitivity_q05_q10_holm_by.csv"
TAU_SENSITIVITY_CSV = "tau_r_sensitivity_0_5x_1_0x_1_5x.csv"
TAU_NULL_JSON = "tau_r_phase_d_null_diagnostic.json"
INDEPENDENT_REPLICATION_JSON = "phase_d_independent_replication_summary.json"
PRIMARY_Q = 0.05
CONTINUITY_Q = 0.10


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _result_dir(scope: str, root: Path) -> Path:
    if scope in PHASE_D_BLOCK_IDS:
        return ensure_dir(root / "results" / "phase_d" / scope)
    if scope == "global":
        return ensure_dir(root / "results" / "phase_d" / GLOBAL_RESULT_DIR)
    raise ValueError(f"scope must be one of {PHASE_D_BLOCK_IDS} or 'global', got {scope!r}.")


def _manifest_path(scope: str, root: Path) -> Path:
    return _result_dir(scope, root) / FINAL_MANIFEST_NAME


def _completion_report_path(scope: str, root: Path) -> Path:
    return _result_dir(scope, root) / BLOCK_REPORT_NAME


def _condition_key(task: dict[str, Any]) -> tuple[Any, ...]:
    return (task.get("instance"), task.get("p_noise"), task.get("shots_total"), task.get("seed"))


def _zne_key(task: dict[str, Any]) -> tuple[Any, ...]:
    return (*_condition_key(task), task.get("zne_order"))


def _load_array(root: Path, task: dict[str, Any], mmap: bool = False) -> np.ndarray:
    kwargs: dict[str, Any] = {"allow_pickle": False}
    if mmap:
        kwargs["mmap_mode"] = "r"
    return np.load(root / str(task["output_path"]), **kwargs)


def _current_analysis_hashes(root: Path) -> dict[str, str | None]:
    rels = [
        "src/phase_d_analysis.py",
        "scripts/36_phase_d_analysis.py",
        "src/phase_d_completion.py",
        "scripts/35_phase_d_completion.py",
        "src/metrics.py",
        "src/bootstrap.py",
        "config/config_official_pilot_v1.6.yaml",
        "richardson_weights_v1.6.json",
    ]
    return {rel: sha256_file(root / rel) if (root / rel).is_file() else None for rel in rels}


def _expected_counts(scope: str) -> dict[str, int]:
    if scope == "global":
        return {
            "tasks": 21612,
            "ideal": 12,
            "noisy": 5400,
            "zne": 16200,
            "metrics": 21600,
            "comparisons": 16200,
            "cells": 540,
            "instances": 12,
        }
    return {
        "tasks": 5403,
        "ideal": 3,
        "noisy": 1350,
        "zne": 4050,
        "metrics": 5400,
        "comparisons": 4050,
        "cells": 135,
        "instances": 3,
    }


def _validate_inventory(root: Path, manifest: dict[str, Any], cfg: dict[str, Any], scope: str, check_hashes: bool) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tasks = list(manifest.get("tasks", []))
    expected = _expected_counts(scope)
    task_ids = [str(task.get("task_id")) for task in tasks]
    kind_counts = dict(sorted(Counter(str(task.get("kind")) for task in tasks).items()))
    expected_instances = sorted({str(task.get("instance")) for task in tasks if task.get("instance")})
    expected_noise = [float(value) for value in cfg["noise"]["p_values"]]
    expected_shots = [int(value) for value in cfg["shots"]["total_budget_primary"]]
    expected_seeds = sorted({int(task["seed"]) for task in tasks if task.get("seed") is not None})
    expected_orders = sorted(str(order) for order in cfg["zne"]["methods"])
    ideal_by_instance = {str(task["instance"]): task for task in tasks if task.get("kind") == "ideal"}
    noisy_by_key = {_condition_key(task): task for task in tasks if task.get("kind") == "noisy"}
    zne_by_key = {_zne_key(task): task for task in tasks if task.get("kind") == "zne"}

    missing_ideal = sorted(instance for instance in expected_instances if instance not in ideal_by_instance)
    missing_noisy: list[str] = []
    missing_zne: list[str] = []
    for instance in expected_instances:
        for p_noise in expected_noise:
            for shots in expected_shots:
                for seed in expected_seeds:
                    key = (instance, p_noise, shots, seed)
                    if key not in noisy_by_key:
                        missing_noisy.append("|".join(str(x) for x in key))
                    for order in expected_orders:
                        zkey = (*key, order)
                        if zkey not in zne_by_key:
                            missing_zne.append("|".join(str(x) for x in zkey))

    duplicate_task_ids = sorted(key for key, count in Counter(task_ids).items() if count > 1)
    duplicate_noisy_conditions = sorted("|".join(str(x) for x in key) for key, count in Counter(_condition_key(task) for task in tasks if task.get("kind") == "noisy").items() if count > 1)
    duplicate_zne_conditions = sorted("|".join(str(x) for x in key) for key, count in Counter(_zne_key(task) for task in tasks if task.get("kind") == "zne").items() if count > 1)

    inventory_rows: list[dict[str, Any]] = []
    file_failures: dict[str, list[str]] = {}
    for task in tasks:
        failures: list[str] = []
        array_path = root / str(task["output_path"])
        done_path = root / str(task["done_path"])
        if not array_path.is_file():
            failures.append("missing_array")
        if not done_path.is_file():
            failures.append("missing_done")
        shape: tuple[int, ...] | None = None
        dtype: str | None = None
        if array_path.is_file():
            array = _load_array(root, task, mmap=True)
            shape = tuple(int(x) for x in array.shape)
            dtype = str(array.dtype)
            if shape != (TARGET_GRID_SIZE, TARGET_GRID_SIZE):
                failures.append("shape_not_120x120")
            if dtype != "float64":
                failures.append("dtype_not_float64")
            if check_hashes and task.get("array_sha256") != sha256_file(array_path):
                failures.append("array_hash_mismatch")
        if done_path.is_file() and check_hashes and task.get("done_sha256") != sha256_file(done_path):
            failures.append("done_hash_mismatch")
        if failures:
            file_failures[str(task.get("task_id"))] = failures
        inventory_rows.append({
            "task_id": task.get("task_id"),
            "kind": task.get("kind"),
            "block_id": task.get("block_id"),
            "instance": task.get("instance"),
            "grid_size": task.get("grid_size"),
            "noise": task.get("p_noise"),
            "shots_total": task.get("shots_total"),
            "seed": task.get("seed"),
            "zne_order": task.get("zne_order"),
            "scale_factors": "|".join(str(x) for x in task.get("scale_factors", [])),
            "array_path": task.get("output_path"),
            "done_path": task.get("done_path"),
            "shape": "" if shape is None else "x".join(str(x) for x in shape),
            "dtype": dtype or "",
            "array_sha256": task.get("array_sha256"),
            "done_sha256": task.get("done_sha256"),
        })

    checks = {
        "manifest_status_complete": str(manifest.get("status", "")).endswith("post_merge_complete"),
        "grid_size_120": int(manifest.get("summary", {}).get("grid_size", -1)) == TARGET_GRID_SIZE,
        "task_count_expected": len(tasks) == expected["tasks"],
        "kind_counts_expected": kind_counts == {"ideal": expected["ideal"], "noisy": expected["noisy"], "zne": expected["zne"]},
        "instance_count_expected": len(expected_instances) == expected["instances"],
        "missing_ideal": missing_ideal,
        "missing_noisy_count": len(missing_noisy),
        "missing_zne_count": len(missing_zne),
        "duplicate_task_ids": duplicate_task_ids,
        "duplicate_noisy_conditions": duplicate_noisy_conditions,
        "duplicate_zne_conditions": duplicate_zne_conditions,
        "file_failure_count": len(file_failures),
        "check_hashes": check_hashes,
    }
    return {
        "checks": checks,
        "file_failures": file_failures,
        "expected": {
            "instances": expected_instances,
            "noise_values": expected_noise,
            "shot_budgets": expected_shots,
            "seeds": expected_seeds,
            "zne_orders": expected_orders,
            "counts": expected,
        },
        "kind_counts": kind_counts,
    }, inventory_rows


def validate_phase_d_analysis_inputs(scope: str, overwrite: bool = False, check_hashes: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    result_dir = _result_dir(scope, root)
    report_path = result_dir / VALIDATION_REPORT
    inventory_path = result_dir / INVENTORY_CSV
    if not overwrite:
        for path in (report_path, inventory_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing Phase D analysis validation output: {path}")
    manifest = _load_json(_manifest_path(scope, root))
    completion_report = _load_json(_completion_report_path(scope, root))
    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    inventory_report, inventory_rows = _validate_inventory(root, manifest, cfg, scope, check_hashes)
    checks = inventory_report["checks"]
    readiness_pass = (
        str(completion_report.get("status", "")).endswith("post_merge_complete")
        and checks["manifest_status_complete"] is True
        and checks["grid_size_120"] is True
        and checks["task_count_expected"] is True
        and checks["kind_counts_expected"] is True
        and checks["instance_count_expected"] is True
        and checks["missing_ideal"] == []
        and checks["missing_noisy_count"] == 0
        and checks["missing_zne_count"] == 0
        and checks["duplicate_task_ids"] == []
        and checks["duplicate_noisy_conditions"] == []
        and checks["duplicate_zne_conditions"] == []
        and checks["file_failure_count"] == 0
    )
    report = {
        "analysis_version": PHASE_D_ANALYSIS_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_d_analysis_inputs_validated" if readiness_pass else "phase_d_analysis_inputs_blocked",
        "decision": "ready_to_run_phase_d_metric_aggregation" if readiness_pass else "block_phase_d_analysis_until_input_issues_are_fixed",
        "scope": scope,
        "completion_report_status": completion_report.get("status"),
        "manifest_path": _manifest_path(scope, root).relative_to(root).as_posix(),
        "checks": checks,
        "expected": inventory_report["expected"],
        "kind_counts": inventory_report["kind_counts"],
        "file_failures": inventory_report["file_failures"],
        "analysis_hashes_current": _current_analysis_hashes(root),
        "outputs": {"inventory_csv": inventory_path.relative_to(root).as_posix()},
    }
    _write_csv(inventory_path, inventory_rows, overwrite=True)
    write_json_atomic(report_path, report, overwrite=True)
    return report


def aggregate_phase_d_metrics(scope: str, overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    result_dir = _result_dir(scope, root)
    metrics_path = result_dir / METRICS_CSV
    comparisons_path = result_dir / COMPARISONS_CSV
    summary_path = result_dir / AGGREGATION_SUMMARY
    manifest_path = result_dir / ANALYSIS_MANIFEST
    if not overwrite:
        for path in (metrics_path, comparisons_path, summary_path, manifest_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing Phase D analysis output: {path}")
    validation = validate_phase_d_analysis_inputs(scope, overwrite=True, check_hashes=False, root=root)
    if validation.get("status") != "phase_d_analysis_inputs_validated":
        raise RuntimeError("Phase D analysis input validation did not pass.")

    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    metrics_config = MetricsConfig.from_experiment_config(cfg)
    manifest = _load_json(_manifest_path(scope, root))
    tasks = list(manifest["tasks"])
    ideals = {str(task["instance"]): task for task in tasks if task["kind"] == "ideal"}
    noisy_tasks = sorted([task for task in tasks if task["kind"] == "noisy"], key=lambda row: (row["instance"], row["p_noise"], row["shots_total"], row["seed"]))
    zne_by_key = {_zne_key(task): task for task in tasks if task["kind"] == "zne"}
    zne_orders = sorted(str(order) for order in cfg["zne"]["methods"])
    experiment_id = str(cfg["experiment_id"])
    ideal_cache: dict[str, np.ndarray] = {}
    metrics_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for noisy_task in noisy_tasks:
        instance = str(noisy_task["instance"])
        if instance not in ideal_cache:
            ideal_cache[instance] = _load_array(root, ideals[instance])
        ideal = ideal_cache[instance]
        noisy = _load_array(root, noisy_task)
        base_row = {
            "phase": "D120",
            "experiment_id": experiment_id,
            "run_name": OFFICIAL_RUN_NAME,
            "scope": scope,
            "block_id": noisy_task.get("block_id"),
            "grid_size": TARGET_GRID_SIZE,
            "instance": instance,
            "noise": noisy_task["p_noise"],
            "shots_total": noisy_task["shots_total"],
            "seed": noisy_task["seed"],
            "method": "noisy",
            "zne_order": "",
            "allocation": "none",
            "scale_factors": "",
            "array_path": noisy_task["output_path"],
            "array_sha256": noisy_task["array_sha256"],
        }
        _add_landscape_stats(base_row, noisy)
        base_row.update(method_metrics(noisy, ideal, metrics_config=metrics_config, noisy_reference=noisy))
        metrics_rows.append(base_row)
        for order in zne_orders:
            zne_task = zne_by_key[(*_condition_key(noisy_task), order)]
            zne = _load_array(root, zne_task)
            method_row = {
                "phase": "D120",
                "experiment_id": experiment_id,
                "run_name": OFFICIAL_RUN_NAME,
                "scope": scope,
                "block_id": noisy_task.get("block_id"),
                "grid_size": TARGET_GRID_SIZE,
                "instance": instance,
                "noise": noisy_task["p_noise"],
                "shots_total": noisy_task["shots_total"],
                "seed": noisy_task["seed"],
                "method": f"zne_{order}",
                "zne_order": order,
                "allocation": CORE_ALLOCATION,
                "scale_factors": "|".join(str(x) for x in zne_task.get("scale_factors", [])),
                "array_path": zne_task["output_path"],
                "array_sha256": zne_task["array_sha256"],
            }
            _add_landscape_stats(method_row, zne)
            method_row.update(method_metrics(zne, ideal, metrics_config=metrics_config, noisy_reference=noisy))
            metrics_rows.append(method_row)
            comparison_rows.append({
                "phase": "D120",
                "experiment_id": experiment_id,
                "run_name": OFFICIAL_RUN_NAME,
                "scope": scope,
                "block_id": noisy_task.get("block_id"),
                "grid_size": TARGET_GRID_SIZE,
                "instance": instance,
                "noise": noisy_task["p_noise"],
                "shots_total": noisy_task["shots_total"],
                "seed": noisy_task["seed"],
                "comparison_type": "zne_vs_noisy",
                "zne_order": order,
                "allocation": CORE_ALLOCATION,
                "scale_factors": "|".join(str(x) for x in zne_task.get("scale_factors", [])),
                "noisy_array_path": noisy_task["output_path"],
                "zne_array_path": zne_task["output_path"],
                "noisy_array_sha256": noisy_task["array_sha256"],
                "zne_array_sha256": zne_task["array_sha256"],
                **comparison_metrics(noisy, zne, ideal),
            })

    expected = _expected_counts(scope)
    _write_csv(metrics_path, metrics_rows, overwrite=True)
    _write_csv(comparisons_path, comparison_rows, overwrite=True)
    summary = {
        "analysis_version": PHASE_D_ANALYSIS_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_d_metric_aggregation_complete",
        "scope": scope,
        "metrics_rows": len(metrics_rows),
        "comparison_rows": len(comparison_rows),
        "expected_metrics_rows": expected["metrics"],
        "expected_comparison_rows": expected["comparisons"],
        "outputs": {
            "metrics_per_seed": metrics_path.relative_to(root).as_posix(),
            "comparisons_per_seed": comparisons_path.relative_to(root).as_posix(),
            "analysis_manifest": manifest_path.relative_to(root).as_posix(),
        },
        "input_validation": validation,
    }
    analysis_manifest = {
        "analysis_version": PHASE_D_ANALYSIS_VERSION,
        "created_at_utc": summary["created_at_utc"],
        "status": summary["status"],
        "scope": scope,
        "source_manifest_final": _manifest_path(scope, root).relative_to(root).as_posix(),
        "metrics_per_seed_sha256": sha256_file(metrics_path),
        "comparisons_per_seed_sha256": sha256_file(comparisons_path),
        "analysis_hashes_current": _current_analysis_hashes(root),
    }
    write_json_atomic(summary_path, summary, overwrite=True)
    write_json_atomic(manifest_path, analysis_manifest, overwrite=True)
    return summary


def _holm_adjust(rows: list[dict[str, Any]], p_key: str, q_key: str, pass_key: str, alpha: float) -> None:
    ranked = sorted(enumerate(rows), key=lambda item: (float(item[1][p_key]), int(item[0])))
    m = len(ranked)
    previous = 0.0
    for rank, (index, row) in enumerate(ranked, start=1):
        adjusted = max(previous, min(1.0, float(row[p_key]) * (m - rank + 1)))
        previous = adjusted
        rows[index][q_key] = float(adjusted)
        rows[index][pass_key] = bool(adjusted <= alpha)


def _by_adjust(rows: list[dict[str, Any]], p_key: str, q_key: str, pass_key: str, q_threshold: float) -> None:
    c_m = sum(1.0 / i for i in range(1, len(rows) + 1))
    ranked = sorted(enumerate(rows), key=lambda item: (float(item[1][p_key]), int(item[0])))
    m = len(ranked)
    previous = 1.0
    adjusted_by_index: dict[int, float] = {}
    for rank_from_end, (index, row) in enumerate(reversed(ranked), start=1):
        rank = m - rank_from_end + 1
        adjusted = min(previous, float(row[p_key]) * m * c_m / rank)
        previous = adjusted
        adjusted_by_index[index] = float(min(1.0, adjusted))
    for index, row in enumerate(rows):
        value = adjusted_by_index[index]
        row[q_key] = value
        row[pass_key] = bool(value <= q_threshold)


def _criterion_2(rows: list[dict[str, Any]], fdr_key: str) -> bool:
    discordant = [row for row in rows if row["discordant_pre_fdr"]]
    discordant_fdr = [row for row in rows if row[fdr_key]]
    spread = bool(
        len({row["instance"] for row in discordant}) > 1
        and (len({row["noise"] for row in discordant}) > 1 or len({row["shots_total"] for row in discordant}) > 1)
    )
    return len(discordant) >= 5 and len(discordant_fdr) >= 3 and spread


def _instance_endpoint(rows: list[dict[str, Any]], root: Path, result_dir: Path) -> dict[str, Any]:
    endpoint_rows: list[dict[str, Any]] = []
    for instance in sorted({str(row["instance"]) for row in rows}, key=lambda label: int(label[1:]) if label.startswith("H") else label):
        instance_rows = [row for row in rows if str(row["instance"]) == instance]
        stress_rows = [row for row in instance_rows if float(row["noise"]) >= 0.005 and int(row["shots_total"]) >= 1024 and not row["excluded"]]
        positive_stress_count = sum(bool(row["discordant_pre_fdr"]) for row in stress_rows)
        q05_count = sum(bool(row["discordant_fdr"]) for row in instance_rows)
        q10_count = sum(bool(row["discordant_fdr_q10"]) for row in instance_rows)
        positive_direction_fraction = float(np.mean([bool(row["cell_positive_direction"]) for row in stress_rows])) if stress_rows else 0.0
        endpoint_rows.append({
            "instance": instance,
            "block_id": instance_rows[0].get("block_id"),
            "stress_cells_evaluated": len(stress_rows),
            "discordant_pre_fdr_stress_count": positive_stress_count,
            "instance_positive_stress": positive_stress_count >= 3,
            "instance_positive_direction": positive_direction_fraction >= 0.50,
            "positive_direction_fraction_stress": positive_direction_fraction,
            "discordant_fdr_q05_count": q05_count,
            "discordant_fdr_q10_count": q10_count,
            "instance_supportive_q05": positive_stress_count >= 3 and q05_count >= 1,
            "instance_supportive_q10": positive_stress_count >= 3 and q10_count >= 1,
        })
    successes = sum(bool(row["instance_positive_stress"]) for row in endpoint_rows)
    p_value = float(binomtest(successes, len(endpoint_rows), p=0.5, alternative="greater").pvalue) if endpoint_rows else 1.0
    summary = {
        "analysis_version": PHASE_D_ANALYSIS_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_d_instance_primary_endpoint_complete",
        "n_instances": len(endpoint_rows),
        "successes_instance_positive_stress": successes,
        "alpha": 0.05,
        "null": "P(instance_positive_stress) <= 0.50",
        "alternative": "P(instance_positive_stress) > 0.50",
        "exact_binomial_p_value": p_value,
        "instance_level_binomial_endpoint_pass": bool(successes >= 10 and p_value <= 0.05),
        "instance_level_moderate_endpoint_pass": bool(successes >= 9 and not (successes >= 10 and p_value <= 0.05)),
    }
    _write_csv(result_dir / INSTANCE_ENDPOINT_CSV, endpoint_rows, overwrite=True)
    write_json_atomic(result_dir / INSTANCE_ENDPOINT_JSON, {**summary, "rows": endpoint_rows}, overwrite=True)
    return {**summary, "rows": endpoint_rows}


def run_phase_d_statistics(scope: str, overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    result_dir = _result_dir(scope, root)
    metrics_path = result_dir / METRICS_CSV
    comparisons_path = result_dir / COMPARISONS_CSV
    aggregation_summary_path = result_dir / AGGREGATION_SUMMARY
    output_paths = [result_dir / BOOTSTRAP_BY_CELL_CSV, result_dir / CLASSIFICATION_MATRIX_CSV]
    if scope == "global":
        output_paths.extend([result_dir / GLOBAL_SUMMARY, result_dir / INSTANCE_ENDPOINT_JSON, result_dir / INDEPENDENT_REPLICATION_JSON])
    else:
        output_paths.append(result_dir / BLOCK_GO_SUMMARY)
    if not overwrite:
        for path in output_paths:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing Phase D statistics output: {path}")
    if not all(path.is_file() for path in (metrics_path, comparisons_path, aggregation_summary_path)):
        raise FileNotFoundError("Run Phase D metric aggregation before statistics.")

    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    comparison_rows = _read_csv(comparisons_path)
    metric_rows = _read_csv(metrics_path)
    bootstrap_cfg = cfg["statistics"]["bootstrap"]
    resamples = int(bootstrap_cfg["resamples"])
    bootstrap_seed = int(cfg["rng"]["bootstrap_seed"])
    tau_r = float(cfg["phase_c"]["tau_R"])
    discordant_cfg = cfg["go_no_go"]["discordant"]
    strong_cfg = cfg["go_no_go"]["strongly_discordant"]
    metrics_by_key = {
        (row["instance"], _as_float(row["noise"]), _as_int(row["shots_total"]), _as_int(row["seed"]), row["method"]): row
        for row in metric_rows
    }
    grouped: dict[tuple[str, float, int, str], list[dict[str, str]]] = {}
    for row in comparison_rows:
        key = (row["instance"], _as_float(row["noise"]), _as_int(row["shots_total"]), row["zne_order"])
        grouped.setdefault(key, []).append(row)

    bootstrap_rows: list[dict[str, Any]] = []
    for cell_index, key in enumerate(sorted(grouped), start=1):
        instance, p_noise, shots_total, zne_order = key
        rows = sorted(grouped[key], key=lambda row: _as_int(row["seed"]))
        method = f"zne_{zne_order}"
        r_values = np.asarray([_as_float(row["r_mse"]) for row in rows], dtype=float)
        d_values = np.asarray([_as_float(row["d_R"]) for row in rows], dtype=float)
        stats = _bootstrap_pair_stats(r_values, d_values, resamples=resamples, seed=bootstrap_seed + cell_index)
        zne_metric_rows = [metrics_by_key[(instance, p_noise, shots_total, _as_int(row["seed"]), method)] for row in rows]
        below_count = sum(_csv_bool(row["below_detection_floor"]) for row in zne_metric_rows)
        breakdown_count = sum(_csv_bool(row["extrapolation_breakdown"]) for row in zne_metric_rows)
        block_id = rows[0].get("block_id") or ""
        row_out = {
            "cell_index": cell_index,
            "cell_id": f"{instance}|p={p_noise:g}|B={shots_total}|zne={zne_order}",
            "scope": scope,
            "block_id": block_id,
            "instance": instance,
            "noise": p_noise,
            "shots_total": shots_total,
            "zne_order": zne_order,
            "allocation": CORE_ALLOCATION,
            "tau_R": tau_r,
            "below_detection_floor_count": below_count,
            "below_detection_floor_rate": float(below_count / len(rows)),
            "excluded_below_detection_floor": bool(below_count > 0),
            "extrapolation_breakdown_count": breakdown_count,
            "extrapolation_breakdown_rate": float(breakdown_count / len(rows)),
            "excluded_extrapolation_breakdown": bool(breakdown_count > 0),
            **stats,
        }
        bootstrap_rows.append(row_out)

    _benjamini_hochberg(bootstrap_rows, "p_conjunction", "q_conjunction_bh_q05", "fdr_conjunction_pass_q05", PRIMARY_Q)
    _benjamini_hochberg(bootstrap_rows, "p_conjunction", "q_conjunction_bh_q10", "fdr_conjunction_pass_q10", CONTINUITY_Q)
    _holm_adjust(bootstrap_rows, "p_conjunction", "q_conjunction_holm", "fdr_conjunction_pass_holm", PRIMARY_Q)
    _by_adjust(bootstrap_rows, "p_conjunction", "q_conjunction_by", "fdr_conjunction_pass_by", PRIMARY_Q)

    classification_rows: list[dict[str, Any]] = []
    tau_rows: list[dict[str, Any]] = []
    for row in bootstrap_rows:
        excluded = bool(row["excluded_below_detection_floor"]) or bool(row["excluded_extrapolation_breakdown"])
        cell_positive_direction = bool(float(row["median_r_mse"]) > 0.0 and float(row["median_d_R"]) > 0.0)
        discordant_pre_fdr = bool(
            not excluded
            and float(row["median_r_mse"]) >= float(discordant_cfg["r_mse_min"])
            and float(row["ci_low_r_mse"]) > 0.0
            and float(row["median_d_R"]) > 0.0
            and float(row["ci_low_d_R"]) >= 0.0
            and float(row["prob_d_R_positive_seed"]) >= float(discordant_cfg["prob_d_R_positive_min"])
        )
        discordant_fdr = bool(discordant_pre_fdr and bool(row["fdr_conjunction_pass_q05"]))
        discordant_fdr_q10 = bool(discordant_pre_fdr and bool(row["fdr_conjunction_pass_q10"]))
        strong = bool(
            discordant_fdr
            and float(row["median_r_mse"]) >= float(strong_cfg["r_mse_min"])
            and float(row["median_d_R"]) >= tau_r
            and float(row["ci_low_r_mse"]) > 0.0
            and float(row["ci_low_d_R"]) > 0.0
            and float(row["prob_d_R_positive_seed"]) >= float(strong_cfg["prob_d_R_positive_min"])
        )
        tau_flags: dict[str, bool] = {}
        for label, multiplier in (("0_5x", 0.5), ("1_0x", 1.0), ("1_5x", 1.5)):
            tau_value = tau_r * multiplier
            tau_pre = bool(discordant_pre_fdr and float(row["median_d_R"]) >= tau_value)
            tau_fdr = bool(tau_pre and bool(row["fdr_conjunction_pass_q05"]))
            tau_flags[f"discordant_pre_fdr_tau_{label}"] = tau_pre
            tau_flags[f"discordant_fdr_tau_{label}_q05"] = tau_fdr
            tau_rows.append({
                "cell_id": row["cell_id"],
                "instance": row["instance"],
                "noise": row["noise"],
                "shots_total": row["shots_total"],
                "zne_order": row["zne_order"],
                "tau_multiplier": multiplier,
                "tau_value": tau_value,
                "median_d_R": row["median_d_R"],
                "discordant_pre_fdr_tau": tau_pre,
                "discordant_fdr_tau_q05": tau_fdr,
            })
        if row["excluded_below_detection_floor"]:
            label = "excluded_below_detection_floor"
        elif row["excluded_extrapolation_breakdown"]:
            label = "excluded_extrapolation_breakdown"
        elif strong:
            label = "strongly_discordant_fdr"
        elif discordant_fdr:
            label = "discordant_fdr"
        elif discordant_fdr_q10:
            label = "discordant_fdr_q10_only"
        elif discordant_pre_fdr:
            label = "discordant_not_fdr"
        else:
            label = "not_discordant"
        classification_rows.append({
            **row,
            "q_conjunction_bh": row["q_conjunction_bh_q05"],
            "fdr_conjunction_pass": row["fdr_conjunction_pass_q05"],
            "excluded": excluded,
            "cell_positive_direction": cell_positive_direction,
            "discordant_pre_fdr": discordant_pre_fdr,
            "discordant_fdr": discordant_fdr,
            "discordant_fdr_q10": discordant_fdr_q10,
            "discordant_fdr_holm": bool(discordant_pre_fdr and row["fdr_conjunction_pass_holm"]),
            "discordant_fdr_by": bool(discordant_pre_fdr and row["fdr_conjunction_pass_by"]),
            "strongly_discordant_fdr": strong,
            "classification": label,
            **tau_flags,
        })

    _write_csv(result_dir / BOOTSTRAP_BY_CELL_CSV, bootstrap_rows, overwrite=True)
    _write_csv(result_dir / CLASSIFICATION_MATRIX_CSV, classification_rows, overwrite=True)
    _write_csv(result_dir / FDR_SENSITIVITY_CSV, [
        {
            "cell_id": row["cell_id"],
            "instance": row["instance"],
            "noise": row["noise"],
            "shots_total": row["shots_total"],
            "zne_order": row["zne_order"],
            "p_conjunction": row["p_conjunction"],
            "discordant_pre_fdr": row["discordant_pre_fdr"],
            "discordant_fdr_q05": row["discordant_fdr"],
            "discordant_fdr_q10": row["discordant_fdr_q10"],
            "discordant_fdr_holm": row["discordant_fdr_holm"],
            "discordant_fdr_by": row["discordant_fdr_by"],
            "q_bh_q05": row["q_conjunction_bh_q05"],
            "q_bh_q10": row["q_conjunction_bh_q10"],
            "q_holm": row["q_conjunction_holm"],
            "q_by": row["q_conjunction_by"],
        }
        for row in classification_rows
    ], overwrite=True)
    _write_csv(result_dir / TAU_SENSITIVITY_CSV, tau_rows, overwrite=True)

    discordant = [row for row in classification_rows if row["discordant_pre_fdr"]]
    discordant_fdr = [row for row in classification_rows if row["discordant_fdr"]]
    discordant_fdr_q10 = [row for row in classification_rows if row["discordant_fdr_q10"]]
    strongly = [row for row in classification_rows if row["strongly_discordant_fdr"]]
    non_excluded = [row for row in classification_rows if not row["excluded"]]
    signs_mostly_positive = float(np.mean([bool(row["cell_positive_direction"]) for row in non_excluded])) >= 0.60 if non_excluded else False
    directionally_positive = bool(
        non_excluded
        and float(np.mean([bool(row["cell_positive_direction"]) for row in non_excluded])) >= 0.50
        and float(np.median([float(row["median_r_mse"]) for row in non_excluded])) > 0.0
        and float(np.median([float(row["median_d_R"]) for row in non_excluded])) > 0.0
    )
    directional_absent = bool(
        not non_excluded
        or float(np.mean([bool(row["cell_positive_direction"]) for row in non_excluded])) < 0.40
        or float(np.median([float(row["median_r_mse"]) for row in non_excluded])) <= 0.0
        or float(np.median([float(row["median_d_R"]) for row in non_excluded])) <= 0.0
    )
    criterion_q05 = _criterion_2(classification_rows, "discordant_fdr")
    criterion_q10 = _criterion_2(classification_rows, "discordant_fdr_q10")

    if scope == "global":
        instance_endpoint = _instance_endpoint(classification_rows, root, result_dir)
        endpoint_rows = instance_endpoint["rows"]
        q05_instances = sum(bool(row["instance_supportive_q05"]) for row in endpoint_rows)
        q10_instances = sum(bool(row["instance_supportive_q10"]) for row in endpoint_rows)
        pre_instances = sum(bool(row["instance_positive_stress"]) for row in endpoint_rows)
        blocks_with_q05 = len({row["block_id"] for row in discordant_fdr})
        blocks_with_q10 = len({row["block_id"] for row in discordant_fdr_q10})
        by_instance_q05 = Counter(str(row["instance"]) for row in discordant_fdr)
        top3_share = 0.0 if not discordant_fdr else sum(count for _, count in by_instance_q05.most_common(3)) / len(discordant_fdr)
        concentrated_few = bool(q05_instances < 4 or top3_share > 0.70)
        tau_stable_15 = bool(
            signs_mostly_positive
            and not directional_absent
            and len({str(row["instance"]) for row in classification_rows if row["discordant_pre_fdr_tau_1_5x"]}) >= 4
        )
        tau_weaker = bool(criterion_q05 and not tau_stable_15)
        q10_only = bool(criterion_q10 and not criterion_q05 and directionally_positive and q10_instances >= 6)
        distributed_q10 = bool(pre_instances >= 8 and q10_instances >= 6 and blocks_with_q10 >= 3)
        distributed_q05 = bool(pre_instances >= 8 and q05_instances >= 6 and blocks_with_q05 >= 3)
        if criterion_q05 and instance_endpoint["instance_level_binomial_endpoint_pass"] and blocks_with_q05 >= 3 and tau_stable_15 and not concentrated_few:
            label = "strong_independent_replication"
        elif ((criterion_q05 and instance_endpoint["instance_level_moderate_endpoint_pass"]) or (criterion_q05 and instance_endpoint["instance_level_binomial_endpoint_pass"] and tau_weaker) or (q10_only and distributed_q10)) and directionally_positive and not concentrated_few:
            label = "moderate_independent_replication"
        elif signs_mostly_positive or q10_only or criterion_q05:
            label = "weak_independent_replication"
        else:
            label = "not_supported_as_independent_replication"
        global_summary = {
            "analysis_version": PHASE_D_ANALYSIS_VERSION,
            "created_at_utc": _utc_now(),
            "status": "phase_d_global_540_analysis_complete",
            "scope": scope,
            "fdr_family_size": len(classification_rows),
            "primary_fdr_method": "benjamini_hochberg",
            "primary_q": PRIMARY_Q,
            "continuity_q": CONTINUITY_Q,
            "cell_level_primary_endpoint_pass": criterion_q05,
            "cell_level_continuity_q10_pass": criterion_q10,
            "instance_level_primary_endpoint_pass": instance_endpoint["instance_level_binomial_endpoint_pass"],
            "instance_level_moderate_endpoint_pass": instance_endpoint["instance_level_moderate_endpoint_pass"],
            "discordant_pre_fdr_count": len(discordant),
            "discordant_fdr_count": len(discordant_fdr),
            "discordant_fdr_q10_count": len(discordant_fdr_q10),
            "strongly_discordant_fdr_count": len(strongly),
            "classification_counts": dict(sorted(Counter(str(row["classification"]) for row in classification_rows).items())),
            "directionally_positive_phase_d": directionally_positive,
            "distributed_positive_pattern_phase_d": distributed_q05,
            "distributed_positive_pattern_phase_d_q10": distributed_q10,
            "phase_d_q10_only_support": q10_only,
            "tau_R_directionally_stable_at_1_5x": tau_stable_15,
            "tau_R_sensitivity_weaker": tau_weaker,
            "phase_d_evidence_concentrated_few_instances": concentrated_few,
            "top_3_instance_share": top3_share,
            "assigned_label": label,
            "outputs": {
                "classification_matrix": (result_dir / CLASSIFICATION_MATRIX_CSV).relative_to(root).as_posix(),
                "bootstrap_by_cell": (result_dir / BOOTSTRAP_BY_CELL_CSV).relative_to(root).as_posix(),
                "instance_primary_endpoint": (result_dir / INSTANCE_ENDPOINT_JSON).relative_to(root).as_posix(),
            },
        }
        tau_null = {
            "analysis_version": PHASE_D_ANALYSIS_VERSION,
            "created_at_utc": _utc_now(),
            "status": "phase_d_tau_r_null_diagnostic_deferred_to_robustness_stage",
            "reason": "Formal noisy-vs-noisy null replicas are computed by the Phase D robustness script using the same completed comparisons.",
            "tau_R": tau_r,
        }
        write_json_atomic(result_dir / GLOBAL_SUMMARY, global_summary, overwrite=True)
        write_json_atomic(result_dir / INDEPENDENT_REPLICATION_JSON, global_summary, overwrite=True)
        write_json_atomic(result_dir / TAU_NULL_JSON, tau_null, overwrite=True)
        return global_summary

    block_summary = {
        "analysis_version": PHASE_D_ANALYSIS_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_d_block_analysis_complete",
        "scope": scope,
        "fdr_family_size": len(classification_rows),
        "primary_q": PRIMARY_Q,
        "criterion_1_at_least_2_strongly_discordant_fdr": len(strongly) >= 2,
        "criterion_2_at_least_5_discordant_with_3_fdr_and_spread": criterion_q05,
        "discordant_pre_fdr_count": len(discordant),
        "discordant_fdr_count": len(discordant_fdr),
        "discordant_fdr_q10_count": len(discordant_fdr_q10),
        "directionally_positive_block": directionally_positive,
        "directional_pattern_absent": directional_absent,
        "classification_counts": dict(sorted(Counter(str(row["classification"]) for row in classification_rows).items())),
        "block_label": "block_strong_support" if (len(strongly) >= 2 or criterion_q05) else ("block_weak_support" if len(discordant) >= 3 else "block_not_supportive"),
    }
    write_json_atomic(result_dir / BLOCK_GO_SUMMARY, block_summary, overwrite=True)
    return block_summary


def write_phase_d_analysis_preflight(overwrite: bool = True, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    planning_dir = ensure_dir(root / "results" / "phase_d" / "planning")
    required_sources = [
        "src/phase_d_analysis.py",
        "scripts/36_phase_d_analysis.py",
        "src/phase_d_completion.py",
        "scripts/35_phase_d_completion.py",
        "src/metrics.py",
        "src/bootstrap.py",
    ]
    checks = {
        "analysis_sources_exist": all((root / rel).is_file() for rel in required_sources),
        "primary_q_is_0_05": PRIMARY_Q == 0.05,
        "continuity_q_is_0_10": CONTINUITY_Q == 0.10,
        "expected_global_cells_540": _expected_counts("global")["cells"] == 540,
        "expected_block_cells_135": _expected_counts("D1")["cells"] == 135,
        "global_manifest_input_contract": _manifest_path("global", root).relative_to(root).as_posix() == f"results/phase_d/{GLOBAL_RESULT_DIR}/manifest_final.json",
        "runner_version_declared": PHASE_D_KERNEL_RUNNER_VERSION.startswith("phase_d_kernel_runner"),
        "protocol_version_declared": isinstance(PROTOCOL_VERSION, str),
    }
    report = {
        "analysis_version": PHASE_D_ANALYSIS_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_d_analysis_preflight_passed" if all(checks.values()) else "phase_d_analysis_preflight_failed",
        "decision": "analysis_scripts_ready_for_phase_d_v1_0_freeze" if all(checks.values()) else "block_phase_d_v1_0_freeze_until_analysis_preflight_passes",
        "scope": "pre_official_execution_analysis_script_readiness",
        "checks": checks,
        "analysis_hashes_current": _current_analysis_hashes(root),
        "post_execution_output_contract": {
            "block_outputs": [
                f"results/phase_d/{block_id}/{name}"
                for block_id in PHASE_D_BLOCK_IDS
                for name in (METRICS_CSV, COMPARISONS_CSV, BOOTSTRAP_BY_CELL_CSV, CLASSIFICATION_MATRIX_CSV, BLOCK_GO_SUMMARY)
            ],
            "global_outputs": [
                f"results/phase_d/{GLOBAL_RESULT_DIR}/{name}"
                for name in (
                    METRICS_CSV,
                    COMPARISONS_CSV,
                    BOOTSTRAP_BY_CELL_CSV,
                    CLASSIFICATION_MATRIX_CSV,
                    GLOBAL_SUMMARY,
                    INSTANCE_ENDPOINT_CSV,
                    INSTANCE_ENDPOINT_JSON,
                    FDR_SENSITIVITY_CSV,
                    TAU_SENSITIVITY_CSV,
                    TAU_NULL_JSON,
                    INDEPENDENT_REPLICATION_JSON,
                )
            ],
        },
    }
    output = planning_dir / "phase_d_analysis_preflight.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Phase D analysis preflight: {output}")
    write_json_atomic(output, report, overwrite=True)
    return report
