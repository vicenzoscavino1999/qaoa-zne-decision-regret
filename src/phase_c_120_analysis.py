from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import MetricsConfig
from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, sha256_file, write_json_atomic
from .metrics import comparison_metrics, method_metrics
from .phase_c_120_completion import FINAL_MANIFEST_NAME, REPORT_NAME
from .phase_c_120_kernel_runner import CORE_ALLOCATION, OFFICIAL_RUN_NAME, TARGET_GRID_SIZE


ANALYSIS_VERSION = "phase_c_120_analysis_v1"
ANALYSIS_RESULT_DIR = "phase_c_120_analysis"
VALIDATION_REPORT = "analysis_input_validation.json"
INVENTORY_CSV = "analysis_input_inventory.csv"
AGGREGATION_SUMMARY = "aggregation_summary.json"
METRICS_CSV = "metrics_per_seed.csv"
COMPARISONS_CSV = "comparisons_per_seed.csv"
ANALYSIS_MANIFEST = "analysis_manifest.json"
STATISTICS_SUMMARY = "statistics_summary.json"
BOOTSTRAP_BY_CELL_CSV = "bootstrap_by_cell.csv"
CLASSIFICATION_MATRIX_CSV = "classification_matrix.csv"
PAPER_GO_NO_GO_JSON = "paper_go_no_go_summary.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_csv(path: Path, rows: list[dict[str, Any]], overwrite: bool = False) -> None:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _csv_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _as_float(value: Any) -> float:
    return float(str(value).strip())


def _as_int(value: Any) -> int:
    return int(str(value).strip())


def _result_dir(root: Path) -> Path:
    return ensure_dir(root / "results" / ANALYSIS_RESULT_DIR)


def _manifest_path(root: Path) -> Path:
    return root / "results" / OFFICIAL_RUN_NAME / FINAL_MANIFEST_NAME


def _completion_report_path(root: Path) -> Path:
    return root / "results" / OFFICIAL_RUN_NAME / REPORT_NAME


def _condition_key(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        task.get("instance"),
        task.get("p_noise"),
        task.get("shots_total"),
        task.get("seed"),
    )


def _zne_key(task: dict[str, Any]) -> tuple[Any, ...]:
    return (*_condition_key(task), task.get("zne_order"))


def _task_path(root: Path, task: dict[str, Any]) -> Path:
    return root / str(task["output_path"])


def _done_path(root: Path, task: dict[str, Any]) -> Path:
    return root / str(task["done_path"])


def _load_array(root: Path, task: dict[str, Any], mmap: bool = False) -> np.ndarray:
    kwargs = {"allow_pickle": False}
    if mmap:
        kwargs["mmap_mode"] = "r"
    return np.load(_task_path(root, task), **kwargs)


def _current_analysis_hashes(root: Path) -> dict[str, str | None]:
    rels = [
        "scripts/04_aggregate_metrics.py",
        "src/phase_c_120_analysis.py",
        "src/metrics.py",
        "src/bootstrap.py",
        "config/config_official_pilot_v1.6.yaml",
        "richardson_weights_v1.6.json",
    ]
    return {
        rel: sha256_file(root / rel) if (root / rel).is_file() else None
        for rel in rels
    }


def _analysis_hash_changes(root: Path, manifest: dict[str, Any]) -> list[dict[str, str | None]]:
    frozen = manifest.get("engine", {}).get("analysis_hashes", {})
    current = _current_analysis_hashes(root)
    out: list[dict[str, str | None]] = []
    for rel, digest in current.items():
        frozen_digest = frozen.get(rel)
        if frozen_digest is not None and frozen_digest != digest:
            out.append({
                "path": rel,
                "frozen_sha256": frozen_digest,
                "current_sha256": digest,
                "classification": "pre_analysis_pipeline_adaptation_after_final_freeze",
            })
    return out


def _validate_inventory(
    root: Path,
    manifest: dict[str, Any],
    cfg: dict[str, Any],
    check_hashes: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    tasks = list(manifest.get("tasks", []))
    task_ids = [str(task.get("task_id")) for task in tasks]
    kind_counts = dict(sorted(Counter(str(task.get("kind")) for task in tasks).items()))
    expected_instances = [str(value) for value in cfg["phase_c"]["instances"]]
    expected_noise = [float(value) for value in cfg["noise"]["p_values"]]
    expected_shots = [int(value) for value in cfg["shots"]["total_budget_primary"]]
    expected_seeds = [int(value) for value in manifest.get("engine", {}).get("seeds", [])]
    if not expected_seeds:
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
        array_path = _task_path(root, task)
        done_path = _done_path(root, task)
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
        "manifest_status_complete": manifest.get("status") == "phase_c_120_post_merge_complete",
        "grid_size_120": int(manifest.get("summary", {}).get("grid_size", -1)) == TARGET_GRID_SIZE,
        "task_count_5403": len(tasks) == 5403,
        "kind_counts_expected": kind_counts == {"ideal": 3, "noisy": 1350, "zne": 4050},
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
        },
        "kind_counts": kind_counts,
    }, inventory_rows


def validate_phase_c_120_analysis_inputs(
    overwrite: bool = False,
    check_hashes: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    result_dir = _result_dir(root)
    report_path = result_dir / VALIDATION_REPORT
    inventory_path = result_dir / INVENTORY_CSV
    if not overwrite:
        for path in [report_path, inventory_path]:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing Phase C 120 analysis validation output: {path}")

    manifest = _load_json(_manifest_path(root))
    completion_report = _load_json(_completion_report_path(root))
    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    inventory_report, inventory_rows = _validate_inventory(root, manifest, cfg, check_hashes=check_hashes)
    hash_changes = _analysis_hash_changes(root, manifest)
    checks = inventory_report["checks"]
    readiness_pass = (
        completion_report.get("decision") == "ready_for_phase_11_official_analysis"
        and checks["manifest_status_complete"] is True
        and checks["grid_size_120"] is True
        and checks["task_count_5403"] is True
        and checks["kind_counts_expected"] is True
        and checks["missing_ideal"] == []
        and checks["missing_noisy_count"] == 0
        and checks["missing_zne_count"] == 0
        and checks["duplicate_task_ids"] == []
        and checks["duplicate_noisy_conditions"] == []
        and checks["duplicate_zne_conditions"] == []
        and checks["file_failure_count"] == 0
    )
    report = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_c_120_analysis_inputs_validated" if readiness_pass else "phase_c_120_analysis_inputs_blocked",
        "decision": "ready_to_run_metric_aggregation_only" if readiness_pass else "block_metric_aggregation_until_input_issues_are_fixed",
        "scope": "pre_metric_pre_bootstrap_pre_fdr_input_validation",
        "completion_report_decision": completion_report.get("decision"),
        "manifest_path": _manifest_path(root).relative_to(root).as_posix(),
        "checks": checks,
        "expected": inventory_report["expected"],
        "kind_counts": inventory_report["kind_counts"],
        "file_failures": inventory_report["file_failures"],
        "analysis_hashes_current": _current_analysis_hashes(root),
        "analysis_hash_changes_after_final_freeze": hash_changes,
        "hash_change_policy": "allowed_before_official_phase_11_metrics_if_documented; raw scientific arrays are unchanged",
        "outputs": {
            "inventory_csv": inventory_path.relative_to(root).as_posix(),
        },
    }
    _write_csv(inventory_path, inventory_rows, overwrite=True)
    write_json_atomic(report_path, report, overwrite=True)
    return report


def _add_landscape_stats(row: dict[str, Any], array: np.ndarray) -> None:
    row.update({
        "landscape_min": float(np.min(array)),
        "landscape_max": float(np.max(array)),
        "landscape_range": float(np.max(array) - np.min(array)),
    })


def aggregate_phase_c_120_metrics(
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    result_dir = _result_dir(root)
    metrics_path = result_dir / METRICS_CSV
    comparisons_path = result_dir / COMPARISONS_CSV
    summary_path = result_dir / AGGREGATION_SUMMARY
    manifest_path = result_dir / ANALYSIS_MANIFEST
    if not overwrite:
        for path in [metrics_path, comparisons_path, summary_path, manifest_path]:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing Phase C 120 analysis output: {path}")

    validation = validate_phase_c_120_analysis_inputs(overwrite=True, check_hashes=False, root=root)
    if validation.get("status") != "phase_c_120_analysis_inputs_validated":
        raise RuntimeError("Phase C 120 analysis input validation did not pass.")

    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    metrics_config = MetricsConfig.from_experiment_config(cfg)
    manifest = _load_json(_manifest_path(root))
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
            "phase": "C120",
            "experiment_id": experiment_id,
            "run_name": OFFICIAL_RUN_NAME,
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
                "phase": "C120",
                "experiment_id": experiment_id,
                "run_name": OFFICIAL_RUN_NAME,
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
                "phase": "C120",
                "experiment_id": experiment_id,
                "run_name": OFFICIAL_RUN_NAME,
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

    _write_csv(metrics_path, metrics_rows, overwrite=True)
    _write_csv(comparisons_path, comparison_rows, overwrite=True)
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_c_120_metric_aggregation_complete",
        "scope": "metrics_and_per_seed_comparisons_only_no_bootstrap_no_fdr",
        "metrics_rows": len(metrics_rows),
        "comparison_rows": len(comparison_rows),
        "expected_metrics_rows": 5400,
        "expected_comparison_rows": 4050,
        "outputs": {
            "metrics_per_seed": metrics_path.relative_to(root).as_posix(),
            "comparisons_per_seed": comparisons_path.relative_to(root).as_posix(),
            "analysis_manifest": manifest_path.relative_to(root).as_posix(),
        },
        "input_validation": validation,
    }
    analysis_manifest = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at_utc": summary["created_at_utc"],
        "status": summary["status"],
        "source_manifest_final": _manifest_path(root).relative_to(root).as_posix(),
        "metrics_per_seed_sha256": sha256_file(metrics_path),
        "comparisons_per_seed_sha256": sha256_file(comparisons_path),
        "analysis_hashes_current": _current_analysis_hashes(root),
    }
    write_json_atomic(summary_path, summary, overwrite=True)
    write_json_atomic(manifest_path, analysis_manifest, overwrite=True)
    return summary


def _bootstrap_pair_stats(
    r_values: np.ndarray,
    d_values: np.ndarray,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    if r_values.shape != d_values.shape:
        raise ValueError("r_MSE and d_R arrays must have the same shape for paired bootstrap.")
    if r_values.size == 0:
        raise ValueError("Cannot bootstrap an empty cell.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, r_values.size, size=(int(resamples), int(r_values.size)))
    boot_r = np.median(r_values[indices], axis=1)
    boot_d = np.median(d_values[indices], axis=1)
    alpha = 1.0 - confidence
    p_r_positive = float((np.count_nonzero(boot_r <= 0.0) + 1) / (boot_r.size + 1))
    p_d_positive = float((np.count_nonzero(boot_d <= 0.0) + 1) / (boot_d.size + 1))
    return {
        "n_seeds": int(r_values.size),
        "resamples": int(resamples),
        "median_r_mse": float(np.median(r_values)),
        "mean_r_mse": float(np.mean(r_values)),
        "ci_low_r_mse": float(np.quantile(boot_r, alpha / 2.0)),
        "ci_high_r_mse": float(np.quantile(boot_r, 1.0 - alpha / 2.0)),
        "p_r_mse_positive": p_r_positive,
        "prob_r_mse_positive_seed": float(np.mean(r_values > 0.0)),
        "median_d_R": float(np.median(d_values)),
        "mean_d_R": float(np.mean(d_values)),
        "ci_low_d_R": float(np.quantile(boot_d, alpha / 2.0)),
        "ci_high_d_R": float(np.quantile(boot_d, 1.0 - alpha / 2.0)),
        "p_d_R_positive": p_d_positive,
        "prob_d_R_positive_seed": float(np.mean(d_values > 0.0)),
        "p_conjunction": max(p_r_positive, p_d_positive),
    }


def _benjamini_hochberg(rows: list[dict[str, Any]], p_key: str, q_key: str, pass_key: str, q_threshold: float) -> None:
    ranked = sorted(enumerate(rows), key=lambda item: (float(item[1][p_key]), int(item[0])))
    m = len(ranked)
    previous = 1.0
    adjusted_by_index: dict[int, float] = {}
    for rank_from_end, (index, row) in enumerate(reversed(ranked), start=1):
        rank = m - rank_from_end + 1
        adjusted = min(previous, float(row[p_key]) * m / rank)
        previous = adjusted
        adjusted_by_index[index] = float(min(1.0, adjusted))
    for index, row in enumerate(rows):
        q_value = adjusted_by_index[index]
        row[q_key] = q_value
        row[pass_key] = bool(q_value <= q_threshold)


def run_phase_c_120_statistics(
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    result_dir = _result_dir(root)
    metrics_path = result_dir / METRICS_CSV
    comparisons_path = result_dir / COMPARISONS_CSV
    aggregation_summary_path = result_dir / AGGREGATION_SUMMARY
    bootstrap_path = result_dir / BOOTSTRAP_BY_CELL_CSV
    classification_path = result_dir / CLASSIFICATION_MATRIX_CSV
    paper_summary_path = result_dir / PAPER_GO_NO_GO_JSON
    statistics_summary_path = result_dir / STATISTICS_SUMMARY
    for path in [bootstrap_path, classification_path, paper_summary_path, statistics_summary_path]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing Phase C 120 statistics output: {path}")
    if not all(path.is_file() for path in [metrics_path, comparisons_path, aggregation_summary_path]):
        raise FileNotFoundError("Run scripts/04_aggregate_metrics.py --phase-c-120 before statistics.")

    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    aggregation = _load_json(aggregation_summary_path)
    if aggregation.get("status") != "phase_c_120_metric_aggregation_complete":
        raise RuntimeError("Metric aggregation is not complete.")

    comparison_rows = _read_csv(comparisons_path)
    metric_rows = _read_csv(metrics_path)
    bootstrap_cfg = cfg["statistics"]["bootstrap"]
    if bootstrap_cfg.get("method") != "percentile" or bootstrap_cfg.get("unit") != "seed" or bootstrap_cfg.get("paired_by_seed") is not True:
        raise RuntimeError("Unsupported Phase C 120 bootstrap configuration.")
    resamples = int(bootstrap_cfg["resamples"])
    bootstrap_seed = int(cfg["rng"]["bootstrap_seed"])
    q_threshold = float(cfg["statistics"]["multiple_comparisons"]["q"])
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
        stats = _bootstrap_pair_stats(
            r_values,
            d_values,
            resamples=resamples,
            seed=bootstrap_seed + cell_index,
        )
        zne_metric_rows = [
            metrics_by_key[(instance, p_noise, shots_total, _as_int(row["seed"]), method)]
            for row in rows
        ]
        below_count = sum(_csv_bool(row["below_detection_floor"]) for row in zne_metric_rows)
        breakdown_count = sum(_csv_bool(row["extrapolation_breakdown"]) for row in zne_metric_rows)
        row_out = {
            "cell_index": cell_index,
            "cell_id": f"{instance}|p={p_noise:g}|B={shots_total}|zne={zne_order}",
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

    _benjamini_hochberg(
        bootstrap_rows,
        p_key="p_conjunction",
        q_key="q_conjunction_bh",
        pass_key="fdr_conjunction_pass",
        q_threshold=q_threshold,
    )

    classification_rows: list[dict[str, Any]] = []
    for row in bootstrap_rows:
        excluded = bool(row["excluded_below_detection_floor"]) or bool(row["excluded_extrapolation_breakdown"])
        discordant_pre_fdr = bool(
            not excluded
            and float(row["median_r_mse"]) >= float(discordant_cfg["r_mse_min"])
            and float(row["ci_low_r_mse"]) > 0.0
            and float(row["median_d_R"]) > 0.0
            and float(row["ci_low_d_R"]) >= 0.0
            and float(row["prob_d_R_positive_seed"]) >= float(discordant_cfg["prob_d_R_positive_min"])
        )
        discordant_fdr = bool(discordant_pre_fdr and bool(row["fdr_conjunction_pass"]))
        strong = bool(
            not excluded
            and float(row["median_r_mse"]) >= float(strong_cfg["r_mse_min"])
            and float(row["median_d_R"]) >= tau_r
            and float(row["ci_low_r_mse"]) > 0.0
            and float(row["ci_low_d_R"]) > 0.0
            and float(row["prob_d_R_positive_seed"]) >= float(strong_cfg["prob_d_R_positive_min"])
            and float(row["q_conjunction_bh"]) <= float(strong_cfg["requires_fdr_conjunction_q_max"])
        )
        if row["excluded_below_detection_floor"]:
            label = "excluded_below_detection_floor"
        elif row["excluded_extrapolation_breakdown"]:
            label = "excluded_extrapolation_breakdown"
        elif strong:
            label = "strongly_discordant_fdr"
        elif discordant_fdr:
            label = "discordant_fdr"
        elif discordant_pre_fdr:
            label = "discordant_not_fdr"
        else:
            label = "not_discordant"
        classification_rows.append({
            **row,
            "discordant_pre_fdr": discordant_pre_fdr,
            "discordant_fdr": discordant_fdr,
            "strongly_discordant_fdr": strong,
            "classification": label,
        })

    strongly = [row for row in classification_rows if row["strongly_discordant_fdr"]]
    discordant = [row for row in classification_rows if row["discordant_pre_fdr"]]
    discordant_fdr = [row for row in classification_rows if row["discordant_fdr"]]
    spread = bool(
        len({row["instance"] for row in discordant}) > 1
        or len({row["noise"] for row in discordant}) > 1
        or len({row["shots_total"] for row in discordant}) > 1
    )
    criterion_1 = len(strongly) >= 2
    criterion_2 = len(discordant) >= 5 and len(discordant_fdr) >= 3 and spread
    paper_summary = {
        "analysis_version": ANALYSIS_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_c_120_go_no_go_complete",
        "bootstrap_method": "paired_seed_percentile_median",
        "bootstrap_resamples": resamples,
        "p_value_method": "one_sided_bootstrap_median_positive",
        "fdr_method": "benjamini_hochberg_on_conjunction_p_values",
        "fdr_q": q_threshold,
        "fdr_family_size": len(classification_rows),
        "paper_go_strong": bool(criterion_1 or criterion_2),
        "criterion_1_at_least_2_strongly_discordant_fdr": criterion_1,
        "criterion_2_at_least_5_discordant_with_3_fdr_and_spread": criterion_2,
        "strongly_discordant_fdr_count": len(strongly),
        "discordant_pre_fdr_count": len(discordant),
        "discordant_fdr_count": len(discordant_fdr),
        "excluded_below_detection_floor_count": sum(bool(row["excluded_below_detection_floor"]) for row in classification_rows),
        "excluded_extrapolation_breakdown_count": sum(bool(row["excluded_extrapolation_breakdown"]) for row in classification_rows),
        "classification_counts": dict(sorted(Counter(str(row["classification"]) for row in classification_rows).items())),
        "discordant_distinct_instances": sorted({str(row["instance"]) for row in discordant}),
        "discordant_distinct_noise_values": sorted({float(row["noise"]) for row in discordant}),
        "discordant_distinct_shot_budgets": sorted({int(row["shots_total"]) for row in discordant}),
        "outputs": {
            "bootstrap_by_cell": (result_dir / BOOTSTRAP_BY_CELL_CSV).relative_to(root).as_posix(),
            "classification_matrix": (result_dir / CLASSIFICATION_MATRIX_CSV).relative_to(root).as_posix(),
            "statistics_summary": (result_dir / STATISTICS_SUMMARY).relative_to(root).as_posix(),
        },
    }
    statistics_summary = {
        **paper_summary,
        "scope": "bootstrap_pvalues_fdr_go_no_go_no_figures",
        "source_comparisons": comparisons_path.relative_to(root).as_posix(),
        "source_metrics": metrics_path.relative_to(root).as_posix(),
        "classification_rows": len(classification_rows),
        "expected_classification_rows": 135,
        "analysis_hashes_current": _current_analysis_hashes(root),
    }

    _write_csv(bootstrap_path, bootstrap_rows, overwrite=True)
    _write_csv(classification_path, classification_rows, overwrite=True)
    write_json_atomic(paper_summary_path, paper_summary, overwrite=True)
    write_json_atomic(statistics_summary_path, statistics_summary, overwrite=True)
    return statistics_summary
