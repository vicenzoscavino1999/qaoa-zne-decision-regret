from __future__ import annotations

import csv
import json
import math
import re
import shutil
import time
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .mitiq_wrapper import richardson_weights


PHASE_E_FREEZE_GATE_VERSION = "phase_e_bias_variance_runtime_gate_v0_1"
VARIANCE_DOMINATED_CELL_THRESHOLD = 0.50
VARIANCE_DOMINATED_FRACTION_GATE = 0.60
ACCEPTED_LOCAL_WALL_HOURS = 72.0
CONSERVATIVE_EFFECTIVE_SHARDS = 6.0
NOMINAL_SHARDS = 8
ENDPOINT_ROW_BYTES = 600
LHS_POINT_SCALE_ROW_BYTES = 300
SUMMARY_JSON_LOG_BYTES = 2 * 1024**3
HASH_MANIFEST_BYTES = 256 * 1024**2
STORAGE_SAFETY_FACTOR = 2.0
BACKUP_FACTOR = 1.0


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _group_mean(rows: list[dict[str, str]], key_fields: tuple[str, ...], value_field: str) -> dict[tuple[Any, ...], float]:
    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        value = _finite_float(row.get(value_field))
        if value is None:
            continue
        key = []
        for field in key_fields:
            raw = row[field]
            if field in {"p_dep"}:
                key.append(float(raw))
            elif field in {"shots_total", "N_LHS"}:
                key.append(int(float(raw)))
            else:
                key.append(raw)
        grouped.setdefault(tuple(key), []).append(value)
    return {key: float(mean(values)) for key, values in grouped.items() if values}


def _latest_selected_lhs_csv(root: Path, selected_n_lhs: int) -> Path:
    calibration = root / "results" / "phase_e" / "calibration"
    candidates = []
    for path in calibration.glob("lhs_sanity_combined_*/lhs_size_calibration.csv"):
        rows = _read_csv(path)
        observed = sorted({int(float(row["N_LHS"])) for row in rows})
        if int(selected_n_lhs) in observed:
            candidates.append((len(observed), max(observed), path.stat().st_mtime, path))
    if not candidates:
        fallback = calibration / "lhs_size_calibration.csv"
        if fallback.exists():
            return fallback
        raise FileNotFoundError("No LHS calibration CSV containing the selected N_LHS was found.")
    return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def _order_vifs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    orders = config["zne"]["scale_factors"]
    result = {}
    for order, scales in orders.items():
        weights = richardson_weights(scales)
        abs_sum = float(sum(abs(weight) for weight in weights))
        result[order] = {
            "scale_factors": list(scales),
            "weights": weights,
            "abs_weight_sum": abs_sum,
            "equal_per_scale_variance_vif": float(abs_sum**2),
        }
    return result


def _bias_variance_diagnostic(root: Path, config: dict[str, Any], selected_n_lhs: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    benchmark_rows = _read_csv(root / "results" / "phase_e" / "calibration" / "backend_benchmark_measurements.csv")
    lhs_csv = _latest_selected_lhs_csv(root, selected_n_lhs)
    lhs_rows = [row for row in _read_csv(lhs_csv) if int(float(row["N_LHS"])) == int(selected_n_lhs)]

    linear_variance = _group_mean(
        [row for row in benchmark_rows if row.get("objective_family") == "zne_richardson_linear"],
        ("instance", "p_dep", "shots_total"),
        "variance_estimate",
    )
    mse_zne = _group_mean(lhs_rows, ("instance", "p_dep", "shots_total"), "MSE_ZNE_Richardson")
    mse_noisy = _group_mean(lhs_rows, ("instance", "p_dep", "shots_total"), "MSE_noisy")
    r_mse = _group_mean(lhs_rows, ("instance", "p_dep", "shots_total"), "r_MSE_Richardson")

    vifs = _order_vifs(config)
    linear_vif = vifs["linear"]["equal_per_scale_variance_vif"]
    rows: list[dict[str, Any]] = []
    for key in sorted(mse_zne):
        instance, p_dep, shots_total = key
        if key not in linear_variance:
            continue
        for order, vif_info in vifs.items():
            scale = float(vif_info["equal_per_scale_variance_vif"]) / float(linear_vif)
            variance_proxy = float(linear_variance[key]) * scale
            ratio = float(variance_proxy / mse_zne[key]) if mse_zne[key] > 0.0 else math.inf
            rows.append(
                {
                    "instance": instance,
                    "p_dep": p_dep,
                    "shots_total": shots_total,
                    "richardson_order": order,
                    "selected_N_LHS": int(selected_n_lhs),
                    "MSE_noisy_mean": mse_noisy.get(key),
                    "MSE_ZNE_Richardson_linear_mean": mse_zne[key],
                    "r_MSE_Richardson_linear_mean": r_mse.get(key),
                    "linear_zne_shot_variance_estimate": linear_variance[key],
                    "equal_variance_vif": vif_info["equal_per_scale_variance_vif"],
                    "variance_proxy_for_order": variance_proxy,
                    "variance_to_MSE_ZNE_ratio_proxy": ratio,
                    "variance_dominated_cell_threshold": VARIANCE_DOMINATED_CELL_THRESHOLD,
                    "variance_dominated": bool(ratio >= VARIANCE_DOMINATED_CELL_THRESHOLD),
                    "interpretation_scope": "linear ratio is measured from pilot benchmark variance; quadratic/cubic ratios are VIF-scaled proxies against linear MSE_ZNE",
                }
            )

    by_order: dict[str, dict[str, Any]] = {}
    for order in vifs:
        order_rows = [row for row in rows if row["richardson_order"] == order]
        ratios = [float(row["variance_to_MSE_ZNE_ratio_proxy"]) for row in order_rows]
        dominated = [row for row in order_rows if bool(row["variance_dominated"])]
        by_order[order] = {
            "cell_count": len(order_rows),
            "variance_dominated_cells": len(dominated),
            "variance_dominated_fraction": float(len(dominated) / len(order_rows)) if order_rows else math.nan,
            "ratio_min": float(min(ratios)) if ratios else math.nan,
            "ratio_mean": float(mean(ratios)) if ratios else math.nan,
            "ratio_max": float(max(ratios)) if ratios else math.nan,
            "fraction_gate": VARIANCE_DOMINATED_FRACTION_GATE,
            "fraction_gate_exceeded": bool(order_rows and len(dominated) / len(order_rows) > VARIANCE_DOMINATED_FRACTION_GATE),
        }

    summary = {
        "selected_N_LHS": int(selected_n_lhs),
        "input_lhs_csv": str(lhs_csv.relative_to(root)).replace("\\", "/"),
        "variance_dominated_cell_threshold": VARIANCE_DOMINATED_CELL_THRESHOLD,
        "variance_dominated_fraction_gate": VARIANCE_DOMINATED_FRACTION_GATE,
        "richardson_vifs": vifs,
        "by_order": by_order,
        "status": "variance_dominated_warning"
        if any(item["fraction_gate_exceeded"] for item in by_order.values())
        else "variance_dominated_gate_projection_passed",
        "interpretation": (
            "The measured linear ZNE shot-variance proxy is comparable to MSE_ZNE in the pilot cells. "
            "Quadratic and cubic Richardson orders have larger coefficient VIFs and should be treated as variance-sensitive unless a later dedicated variance probe contradicts this."
        ),
    }
    return summary, rows


def _eval_stats(root: Path, selected_e: int, selected_k: int) -> dict[str, Any]:
    rows = [
        row
        for row in _read_csv(root / "results" / "phase_e" / "calibration" / "k_random_start_calibration.csv")
        if int(float(row["E_COBYLA"])) == int(selected_e) and int(float(row["K"])) == int(selected_k)
    ]
    if not rows:
        rows = [
            row
            for row in _read_csv(root / "results" / "phase_e" / "calibration" / "cobyla_budget_calibration.csv")
            if int(float(row["E_COBYLA"])) == int(selected_e) and int(float(row["K"])) == int(selected_k)
        ]
    if not rows:
        raise ValueError("No optimizer calibration rows match selected E_COBYLA and K.")

    def values(column: str) -> list[float]:
        return [float(row[column]) for row in rows if _finite_float(row.get(column)) is not None]

    return {
        "rows": len(rows),
        "noisy_eval_mean_per_start": float(mean(values("noisy_num_evaluations_mean"))),
        "zne_eval_mean_per_start": float(mean(values("zne_num_evaluations_mean"))),
        "noisy_eval_max_observed": float(max(values("noisy_num_evaluations_max"))),
        "zne_eval_max_observed": float(max(values("zne_num_evaluations_max"))),
        "actual_evaluations_hit_requested_cap_any": any(row.get("actual_evaluations_hit_requested_cap") == "True" for row in rows),
        "requested_max_evaluations": int(selected_e),
    }


def _mean_objective_seconds(root: Path, status: dict[str, Any]) -> dict[str, Any]:
    metrics = status.get("pilot_metrics", {})
    noisy = _finite_float(metrics.get("mean_noisy_objective_seconds"))
    zne = _finite_float(metrics.get("mean_zne_objective_seconds"))
    if noisy is not None and zne is not None:
        return {"noisy_objective_seconds": noisy, "zne_linear_objective_seconds": zne, "source": "phase_e_calibration_status"}
    rows = _read_csv(root / "results" / "phase_e" / "calibration" / "backend_benchmark_measurements.csv")
    noisy_values = [float(row["elapsed_seconds"]) for row in rows if row.get("objective_family") == "noisy"]
    zne_values = [float(row["elapsed_seconds"]) for row in rows if row.get("objective_family") == "zne_richardson_linear"]
    return {
        "noisy_objective_seconds": float(mean(noisy_values)),
        "zne_linear_objective_seconds": float(mean(zne_values)),
        "source": "backend_benchmark_measurements",
    }


def _accepted_local_wall_hours(root: Path) -> tuple[float, str]:
    amendment = root / "PHASE_E_RUNTIME_BUDGET_AMENDMENT_v0.1-accepted.md"
    if amendment.exists():
        text = amendment.read_text(encoding="utf-8-sig")
        match = re.search(r"accepted_local_wall_hours\s*=\s*([0-9]+(?:\.[0-9]+)?)", text)
        if match:
            return float(match.group(1)), "PHASE_E_RUNTIME_BUDGET_AMENDMENT_v0.1-accepted.md"
    return ACCEPTED_LOCAL_WALL_HOURS, "default_72h_budget"


def _runtime_storage_projection(
    root: Path,
    config: dict[str, Any],
    status: dict[str, Any],
    selected_e: int,
    selected_k: int,
    selected_n_lhs: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    instance_count = int(config["depth2_design"]["selected_instance_count"])
    noise_count = len(config["noise_rates"])
    shot_count = len(config["shot_budgets"])
    seed_count = int(config["depth2_design"]["seed_count"])
    orders = _order_vifs(config)
    base_cells = instance_count * noise_count * shot_count
    full_cells = base_cells * len(orders)

    eval_stats = _eval_stats(root, selected_e, selected_k)
    timings = _mean_objective_seconds(root, status)
    accepted_wall_hours, accepted_budget_source = _accepted_local_wall_hours(root)
    zne_linear_seconds = float(timings["zne_linear_objective_seconds"])
    noisy_seconds = float(timings["noisy_objective_seconds"])

    zne_order_seconds = {
        order: zne_linear_seconds * (len(info["scale_factors"]) / len(orders["linear"]["scale_factors"]))
        for order, info in orders.items()
    }

    noisy_starts_reuse = base_cells * seed_count * selected_k
    noisy_starts_conservative = full_cells * seed_count * selected_k
    zne_starts_by_order = base_cells * seed_count * selected_k
    reference_optimizer_runs = instance_count * 16 * 2

    noisy_eval_mean = float(eval_stats["noisy_eval_mean_per_start"])
    zne_eval_mean = float(eval_stats["zne_eval_mean_per_start"])
    cap = int(selected_e)

    optimizer_reuse_serial = noisy_starts_reuse * noisy_eval_mean * noisy_seconds
    optimizer_conservative_serial = noisy_starts_conservative * noisy_eval_mean * noisy_seconds
    zne_serial_by_order = {
        order: zne_starts_by_order * zne_eval_mean * seconds for order, seconds in zne_order_seconds.items()
    }
    reference_serial = reference_optimizer_runs * noisy_eval_mean * noisy_seconds

    lhs_csv = _latest_selected_lhs_csv(root, selected_n_lhs)
    lhs_rows = [row for row in _read_csv(lhs_csv) if int(float(row["N_LHS"])) == int(selected_n_lhs)]
    lhs_linear_row_seconds = float(mean(float(row["elapsed_seconds"]) for row in lhs_rows))
    linear_total_component = noisy_seconds + noisy_seconds + zne_linear_seconds
    lhs_order_seconds = {
        order: lhs_linear_row_seconds * ((noisy_seconds + noisy_seconds + zne_seconds) / linear_total_component)
        for order, zne_seconds in zne_order_seconds.items()
    }
    lhs_serial = base_cells * sum(lhs_order_seconds.values())

    primary_serial = optimizer_reuse_serial + sum(zne_serial_by_order.values()) + reference_serial + lhs_serial
    conservative_serial = optimizer_conservative_serial + sum(zne_serial_by_order.values()) + reference_serial + lhs_serial
    cap_serial = (
        noisy_starts_reuse * cap * noisy_seconds
        + sum(zne_starts_by_order * cap * seconds for seconds in zne_order_seconds.values())
        + reference_optimizer_runs * cap * noisy_seconds
        + lhs_serial
    )

    ordered_order_names = [name for name in ("linear", "quadratic", "cubic") if name in orders]
    scope_sensitivity = []
    for stop in range(1, len(ordered_order_names) + 1):
        subset = ordered_order_names[:stop]
        subset_serial = (
            optimizer_reuse_serial
            + sum(zne_serial_by_order[order] for order in subset)
            + reference_serial
            + base_cells * sum(lhs_order_seconds[order] for order in subset)
        )
        subset_wall_hours = float(subset_serial / CONSERVATIVE_EFFECTIVE_SHARDS / 3600.0)
        scope_sensitivity.append(
            {
                "richardson_orders_full_factorial": list(subset),
                "projected_wall_hours": subset_wall_hours,
                "accepted_local_wall_hours": accepted_wall_hours,
                "projection_passes_accepted_budget": bool(subset_wall_hours <= accepted_wall_hours),
            }
        )

    runtime_rows = []
    for label, serial in [
        ("observed_mean_eval_reuse_noisy_across_richardson_orders", primary_serial),
        ("observed_mean_eval_conservative_noisy_recomputed_per_order", conservative_serial),
        ("cap_worst_case_reuse_noisy_across_orders", cap_serial),
    ]:
        wall = float(serial / CONSERVATIVE_EFFECTIVE_SHARDS / 3600.0)
        runtime_rows.append(
            {
                "projection_model": label,
                "serial_hours": float(serial / 3600.0),
                "effective_shards": CONSERVATIVE_EFFECTIVE_SHARDS,
                "projected_wall_hours": wall,
                "accepted_local_wall_hours": accepted_wall_hours,
                "accepted_budget_source": accepted_budget_source,
                "projection_passes_accepted_budget": bool(wall <= accepted_wall_hours),
                "wall_hours_over_budget": float(wall - accepted_wall_hours),
            }
        )

    scale_records_per_base_cell = 1 + 1 + sum(len(info["scale_factors"]) for info in orders.values())
    lhs_point_scale_records = base_cells * selected_n_lhs * scale_records_per_base_cell
    endpoint_rows = noisy_starts_reuse + zne_starts_by_order * len(orders) + reference_optimizer_runs
    primary_storage = (
        lhs_point_scale_records * LHS_POINT_SCALE_ROW_BYTES
        + endpoint_rows * ENDPOINT_ROW_BYTES
        + SUMMARY_JSON_LOG_BYTES
        + HASH_MANIFEST_BYTES
    )
    required_with_safety = int(math.ceil(primary_storage * STORAGE_SAFETY_FACTOR))
    required_total_with_backup = int(math.ceil(required_with_safety * (1.0 + BACKUP_FACTOR)))
    disk = shutil.disk_usage(root)
    storage_rows = [
        {
            "component": "lhs_point_scale_records",
            "record_count": lhs_point_scale_records,
            "bytes_per_record": LHS_POINT_SCALE_ROW_BYTES,
            "projected_bytes": lhs_point_scale_records * LHS_POINT_SCALE_ROW_BYTES,
        },
        {
            "component": "optimizer_endpoint_start_summaries",
            "record_count": endpoint_rows,
            "bytes_per_record": ENDPOINT_ROW_BYTES,
            "projected_bytes": endpoint_rows * ENDPOINT_ROW_BYTES,
        },
        {
            "component": "logs_and_summary_json",
            "record_count": 1,
            "bytes_per_record": SUMMARY_JSON_LOG_BYTES,
            "projected_bytes": SUMMARY_JSON_LOG_BYTES,
        },
        {
            "component": "hashes_and_manifests",
            "record_count": 1,
            "bytes_per_record": HASH_MANIFEST_BYTES,
            "projected_bytes": HASH_MANIFEST_BYTES,
        },
        {
            "component": "safety_and_backup_total",
            "record_count": 1,
            "bytes_per_record": required_total_with_backup,
            "projected_bytes": required_total_with_backup,
        },
    ]

    primary = runtime_rows[0]
    projection = {
        "design_counts": {
            "selected_instances": instance_count,
            "noise_rates": noise_count,
            "shot_budgets": shot_count,
            "richardson_orders": len(orders),
            "seeds": seed_count,
            "base_cells_without_order": base_cells,
            "full_cells_with_order": full_cells,
            "selected_K": int(selected_k),
            "selected_E_COBYLA": int(selected_e),
            "selected_N_LHS": int(selected_n_lhs),
        },
        "timings": timings,
        "eval_stats": eval_stats,
        "order_objective_seconds": zne_order_seconds,
        "lhs_linear_row_seconds_at_selected_N": lhs_linear_row_seconds,
        "lhs_order_row_seconds": lhs_order_seconds,
        "runtime_status": "projection_fails_accepted_local_budget"
        if not primary["projection_passes_accepted_budget"]
        else "projection_passes_accepted_budget_not_full_gate",
        "primary_projected_wall_hours": primary["projected_wall_hours"],
        "accepted_local_wall_hours": accepted_wall_hours,
        "accepted_budget_source": accepted_budget_source,
        "richardson_order_scope_sensitivity": scope_sensitivity,
        "runtime_gate_complete": False,
        "runtime_gate_pass": False,
        "storage": {
            "primary_storage_bytes_before_safety": int(primary_storage),
            "required_bytes_with_safety": required_with_safety,
            "required_total_bytes_with_backup": required_total_with_backup,
            "required_total_gib_with_backup": float(required_total_with_backup / 1024**3),
            "disk_total_bytes": int(disk.total),
            "disk_free_bytes": int(disk.free),
            "required_fraction_of_free": float(required_total_with_backup / disk.free) if disk.free > 0 else math.inf,
            "projected_storage_pass": bool(required_total_with_backup <= 0.50 * disk.free),
            "storage_gate_complete": False,
            "storage_gate_pass": False,
        },
        "recommended_next_action": (
            "write_runtime_scope_or_resource_amendment_before_freeze"
            if not primary["projection_passes_accepted_budget"]
            else "run_full_runtime_dry_run_gate_before_freeze"
        ),
    }
    return projection, runtime_rows, storage_rows


def _markdown(summary: dict[str, Any]) -> str:
    runtime = summary["runtime_storage_projection"]
    variance = summary["bias_variance_diagnostic"]
    storage = runtime["storage"]
    lines = [
        "# Phase E Bias-Variance And Runtime Gate",
        "",
        f"Created UTC: {summary['created_at_utc']}",
        f"Status: `{summary['status']}`",
        "",
        "This report uses existing pilot/calibration artifacts only. It does not authorize official Phase E execution.",
        "",
        "## Selected Calibration Values",
        "",
        f"- E_COBYLA: `{summary['selected_values']['E_COBYLA']}`",
        f"- K: `{summary['selected_values']['K']}`",
        f"- reference strategy: `{summary['selected_values']['reference_strategy']}`",
        f"- N_LHS: `{summary['selected_values']['N_LHS']}`",
        "",
        "## Bias-Variance / VIF",
        "",
        f"- Cell variance-dominated threshold: `{variance['variance_dominated_cell_threshold']}`",
        f"- Fraction gate: `{variance['variance_dominated_fraction_gate']}`",
    ]
    for order, row in variance["by_order"].items():
        lines.append(
            f"- {order}: `{row['variance_dominated_cells']}/{row['cell_count']}` cells dominated, "
            f"mean ratio `{row['ratio_mean']:.3f}`, max ratio `{row['ratio_max']:.3f}`"
        )
    lines.extend(
        [
            "",
            "Interpretation: ZNE shot variance is already comparable to `MSE_ZNE` for the measured linear pilot cells; higher Richardson orders carry larger coefficient VIF. Treat `r_MSE` magnitude as variance-sensitive.",
            "",
            "## Runtime Projection",
            "",
            f"- Primary projected wall time: `{runtime['primary_projected_wall_hours']:.2f}` h",
            f"- Accepted local wall budget: `{runtime['accepted_local_wall_hours']:.2f}` h",
            f"- Runtime status: `{runtime['runtime_status']}`",
            f"- Accepted budget source: `{runtime['accepted_budget_source']}`",
            f"- Runtime gate complete: `{runtime['runtime_gate_complete']}`",
            "",
            "| Full-factorial Richardson orders | Projected wall hours | Passes accepted budget |",
            "|---|---:|---|",
        ]
    )
    for row in runtime["richardson_order_scope_sensitivity"]:
        orders = "+".join(row["richardson_orders_full_factorial"])
        lines.append(f"| {orders} | {row['projected_wall_hours']:.2f} | `{row['projection_passes_accepted_budget']}` |")
    lines.extend(
        [
            "",
            "## Storage Projection",
            "",
            f"- Required total with safety+backup: `{storage['required_total_gib_with_backup']:.2f}` GiB",
            f"- Fraction of free disk: `{storage['required_fraction_of_free']:.3f}`",
            f"- Storage projection pass: `{storage['projected_storage_pass']}`",
            f"- Storage gate complete: `{storage['storage_gate_complete']}`",
            "",
            "## Decision",
            "",
            f"- Official execution authorized: `{summary['official_execution_authorized']}`",
            f"- Recommended next action: `{runtime['recommended_next_action']}`",
            "",
        ]
    )
    return "\n".join(lines)


def run_phase_e_bias_variance_runtime_gate(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration" / "freeze_gates")
    paths = {
        "summary": output_dir / "phase_e_bias_variance_runtime_gate.json",
        "report": output_dir / "phase_e_bias_variance_runtime_gate.md",
        "variance_rows": output_dir / "phase_e_bias_variance_vif_rows.csv",
        "runtime_rows": output_dir / "phase_e_runtime_projection_rows.csv",
        "storage_rows": output_dir / "phase_e_storage_projection_rows.csv",
        "hashes": output_dir / "phase_e_bias_variance_runtime_gate_hashes.json",
    }
    if any(path.exists() for path in paths.values()) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase E freeze-gate outputs.")

    status = _read_json(root / "results" / "phase_e" / "calibration" / "phase_e_calibration_status.json")
    config = _read_json(root / "config" / "phase_e_config_v0.1-draft.json")
    selected_e = int(status["selected_cobyla_evaluation_budget"])
    selected_k = int(status["selected_random_start_count"])
    selected_n_lhs = int(status["selected_N_LHS"])

    variance_summary, variance_rows = _bias_variance_diagnostic(root, config, selected_n_lhs)
    runtime_storage, runtime_rows, storage_rows = _runtime_storage_projection(
        root,
        config,
        status,
        selected_e,
        selected_k,
        selected_n_lhs,
    )

    status_flags = []
    if variance_summary["status"] == "variance_dominated_warning":
        status_flags.append("variance_dominated_warning")
    if runtime_storage["runtime_status"] == "projection_fails_accepted_local_budget":
        status_flags.append("runtime_projection_failed")
    if not runtime_storage["storage"]["projected_storage_pass"]:
        status_flags.append("storage_projection_failed")
    status_text = "phase_e_freeze_gate_projection_requires_amendment" if status_flags else "phase_e_freeze_gate_projection_ready_for_full_gate"

    summary = {
        "created_at_utc": _utc_now(),
        "gate_version": PHASE_E_FREEZE_GATE_VERSION,
        "status": status_text,
        "status_flags": status_flags,
        "official_execution_authorized": False,
        "selected_values": {
            "E_COBYLA": selected_e,
            "K": selected_k,
            "reference_strategy": status["selected_reference_strategy"],
            "N_LHS": selected_n_lhs,
        },
        "bias_variance_diagnostic": variance_summary,
        "runtime_storage_projection": runtime_storage,
        "inputs": {
            "phase_e_calibration_status": "results/phase_e/calibration/phase_e_calibration_status.json",
            "phase_e_config": "config/phase_e_config_v0.1-draft.json",
            "backend_benchmark_measurements": "results/phase_e/calibration/backend_benchmark_measurements.csv",
            "k_random_start_calibration": "results/phase_e/calibration/k_random_start_calibration.csv",
            "selected_lhs_csv": variance_summary["input_lhs_csv"],
        },
        "limitations": [
            "No new circuit simulation is executed by this gate.",
            "Quadratic and cubic variance ratios are VIF-scaled proxies because the existing benchmark measured the linear Richardson objective.",
            "Runtime is a projection from existing objective timings and observed COBYLA evaluation counts; it is not a full runner dry-run.",
            "Storage projection assumes compact optimizer summaries plus per-scale LHS records; exact storage policy must still be frozen.",
        ],
    }

    _write_csv(paths["variance_rows"], variance_rows)
    _write_csv(paths["runtime_rows"], runtime_rows)
    _write_csv(paths["storage_rows"], storage_rows)
    write_json_atomic(paths["summary"], summary, overwrite=True)
    paths["report"].write_text(_markdown(summary), encoding="utf-8")
    hashes = {
        "created_at_utc": _utc_now(),
        "gate_version": PHASE_E_FREEZE_GATE_VERSION,
        "hashes": {key: sha256_file(path) for key, path in paths.items() if key != "hashes"},
    }
    write_json_atomic(paths["hashes"], hashes, overwrite=True)
    summary["output_hashes"] = hashes["hashes"]
    return summary
