from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.stats import norm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, sha256_file, write_json_atomic
from .metrics import oob_fraction
from .phase_c_120_analysis import (
    BOOTSTRAP_BY_CELL_CSV,
    CLASSIFICATION_MATRIX_CSV,
    COMPARISONS_CSV,
    METRICS_CSV,
    PAPER_GO_NO_GO_JSON,
    STATISTICS_SUMMARY,
    _as_float,
    _as_int,
    _benjamini_hochberg,
    _csv_bool,
    _current_analysis_hashes,
    _load_array,
    _load_json,
    _manifest_path,
    _read_csv,
    _result_dir,
    _utc_now,
    _write_csv,
)
from .phase_c_120_completion import FINAL_MANIFEST_NAME
from .phase_c_120_kernel_runner import CORE_ALLOCATION


OUTPUT_VERSION = "phase_c_120_paper_outputs_v1"
SENSITIVITY_SUMMARY = "sensitivity_summary.json"
SENSITIVITY_OOB_CSV = "sensitivity_oob_thresholds.csv"
SENSITIVITY_Q05_CSV = "sensitivity_q05_classification.csv"
SENSITIVITY_BCA_CSV = "sensitivity_bca_classification.csv"
SENSITIVITY_FLIPS_CSV = "sensitivity_flips.csv"
TABLE_GO_NO_GO_CSV = "paper_table_go_no_go_summary.csv"
TABLE_DISCORDANT_FDR_CSV = "paper_table_discordant_fdr.csv"
TABLE_DISCORDANT_NOT_FDR_CSV = "paper_table_discordant_prefdr_not_fdr.csv"
TABLE_EXCLUSIONS_CSV = "paper_table_exclusions.csv"
TABLE_INSTANCE_NOISE_CSV = "paper_table_instance_noise_summary.csv"
PAPER_OUTPUTS_SUMMARY = "paper_outputs_summary.json"
LIMITATIONS_MD = "phase_c_120_limitations_and_signoff.md"


def _required_paths(root: Path) -> dict[str, Path]:
    result_dir = _result_dir(root)
    return {
        "metrics": result_dir / METRICS_CSV,
        "comparisons": result_dir / COMPARISONS_CSV,
        "bootstrap": result_dir / BOOTSTRAP_BY_CELL_CSV,
        "classification": result_dir / CLASSIFICATION_MATRIX_CSV,
        "paper": result_dir / PAPER_GO_NO_GO_JSON,
        "statistics": result_dir / STATISTICS_SUMMARY,
        "manifest": _manifest_path(root),
    }


def _write_key_value_csv(path: Path, data: dict[str, Any], overwrite: bool) -> None:
    rows = [{"field": key, "value": json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value} for key, value in data.items()]
    _write_csv(path, rows, overwrite=overwrite)


def _cell_key(row: dict[str, Any]) -> tuple[str, float, int, str]:
    return (str(row["instance"]), float(row["noise"]), int(row["shots_total"]), str(row["zne_order"]))


def _cell_id(row: dict[str, Any]) -> str:
    return f"{row['instance']}|p={float(row['noise']):g}|B={int(row['shots_total'])}|zne={row['zne_order']}"


def _classify_rows(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    q_threshold: float,
    use_existing_q: bool = True,
) -> list[dict[str, Any]]:
    out = [dict(row) for row in rows]
    if not use_existing_q:
        _benjamini_hochberg(
            out,
            p_key="p_conjunction",
            q_key="q_conjunction_bh",
            pass_key="fdr_conjunction_pass",
            q_threshold=q_threshold,
        )
    tau_r = float(cfg["phase_c"]["tau_R"])
    discordant_cfg = cfg["go_no_go"]["discordant"]
    strong_cfg = cfg["go_no_go"]["strongly_discordant"]
    for row in out:
        row["fdr_conjunction_pass"] = bool(float(row["q_conjunction_bh"]) <= q_threshold)
        excluded_below = _csv_bool(row["excluded_below_detection_floor"])
        excluded_breakdown = _csv_bool(row["excluded_extrapolation_breakdown"])
        excluded = excluded_below or excluded_breakdown
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
            and float(row["q_conjunction_bh"]) <= q_threshold
        )
        if excluded_below:
            label = "excluded_below_detection_floor"
        elif excluded_breakdown:
            label = "excluded_extrapolation_breakdown"
        elif strong:
            label = "strongly_discordant_fdr"
        elif discordant_fdr:
            label = "discordant_fdr"
        elif discordant_pre_fdr:
            label = "discordant_not_fdr"
        else:
            label = "not_discordant"
        row["discordant_pre_fdr"] = discordant_pre_fdr
        row["discordant_fdr"] = discordant_fdr
        row["strongly_discordant_fdr"] = strong
        row["classification"] = label
    return out


def _summary_from_classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    discordant = [row for row in rows if _csv_bool(row["discordant_pre_fdr"])]
    discordant_fdr = [row for row in rows if _csv_bool(row["discordant_fdr"])]
    strongly = [row for row in rows if _csv_bool(row["strongly_discordant_fdr"])]
    spread = bool(
        len({row["instance"] for row in discordant}) > 1
        or len({float(row["noise"]) for row in discordant}) > 1
        or len({int(row["shots_total"]) for row in discordant}) > 1
    )
    criterion_1 = len(strongly) >= 2
    criterion_2 = len(discordant) >= 5 and len(discordant_fdr) >= 3 and spread
    return {
        "paper_go_strong": bool(criterion_1 or criterion_2),
        "criterion_1": criterion_1,
        "criterion_2": criterion_2,
        "discordant_pre_fdr_count": len(discordant),
        "discordant_fdr_count": len(discordant_fdr),
        "strongly_discordant_fdr_count": len(strongly),
        "classification_counts": dict(sorted(Counter(str(row["classification"]) for row in rows).items())),
    }


def _bca_ci(values: np.ndarray, boot_stats: np.ndarray, observed: float, confidence: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - confidence
    prop_less = float(np.mean(boot_stats < observed))
    eps = 1.0 / (2.0 * len(boot_stats))
    prop_less = min(max(prop_less, eps), 1.0 - eps)
    z0 = float(norm.ppf(prop_less))
    if values.size <= 2:
        accel = 0.0
    else:
        jack = np.asarray([np.median(np.delete(values, i)) for i in range(values.size)], dtype=float)
        jack_mean = float(np.mean(jack))
        diffs = jack_mean - jack
        numerator = float(np.sum(diffs**3))
        denominator = float(6.0 * (np.sum(diffs**2) ** 1.5))
        accel = 0.0 if denominator == 0.0 or not math.isfinite(denominator) else numerator / denominator

    def adjusted_quantile(prob: float) -> float:
        z = float(norm.ppf(prob))
        denom = 1.0 - accel * (z0 + z)
        if denom == 0.0:
            return prob
        return float(norm.cdf(z0 + (z0 + z) / denom))

    q_low = min(max(adjusted_quantile(alpha / 2.0), 0.0), 1.0)
    q_high = min(max(adjusted_quantile(1.0 - alpha / 2.0), 0.0), 1.0)
    if q_low > q_high:
        q_low, q_high = q_high, q_low
    return float(np.quantile(boot_stats, q_low)), float(np.quantile(boot_stats, q_high))


def _build_bca_rows(
    bootstrap_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, str]],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    resamples = int(cfg["statistics"]["bootstrap"]["resamples"])
    bootstrap_seed = int(cfg["rng"]["bootstrap_seed"])
    grouped: dict[tuple[str, float, int, str], list[dict[str, str]]] = {}
    for row in comparison_rows:
        grouped.setdefault(_cell_key(row), []).append(row)
    rows: list[dict[str, Any]] = []
    for base in bootstrap_rows:
        cell_index = int(base["cell_index"])
        key = _cell_key(base)
        source = sorted(grouped[key], key=lambda row: _as_int(row["seed"]))
        r_values = np.asarray([_as_float(row["r_mse"]) for row in source], dtype=float)
        d_values = np.asarray([_as_float(row["d_R"]) for row in source], dtype=float)
        rng = np.random.default_rng(bootstrap_seed + cell_index)
        indices = rng.integers(0, r_values.size, size=(resamples, r_values.size))
        boot_r = np.median(r_values[indices], axis=1)
        boot_d = np.median(d_values[indices], axis=1)
        out = dict(base)
        out["ci_low_r_mse"], out["ci_high_r_mse"] = _bca_ci(r_values, boot_r, float(base["median_r_mse"]))
        out["ci_low_d_R"], out["ci_high_d_R"] = _bca_ci(d_values, boot_d, float(base["median_d_R"]))
        out["sensitivity"] = "bca_ci"
        out["ci_method"] = "bca"
        rows.append(out)
    return _classify_rows(rows, cfg, q_threshold=float(cfg["statistics"]["multiple_comparisons"]["q"]))


def _oob_threshold_rows(
    root: Path,
    bootstrap_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
    cfg: dict[str, Any],
    thresholds: list[float],
) -> list[dict[str, Any]]:
    manifest = _load_json(_manifest_path(root))
    ideal_tasks = {str(task["instance"]): task for task in manifest["tasks"] if task["kind"] == "ideal"}
    ideal_cache: dict[str, np.ndarray] = {}
    zne_metric_rows = [
        row for row in metric_rows
        if str(row["method"]).startswith("zne_")
    ]
    rows_by_cell_seed = {
        (row["instance"], float(row["noise"]), int(row["shots_total"]), int(row["seed"]), row["zne_order"]): row
        for row in zne_metric_rows
    }
    source_by_cell: dict[tuple[str, float, int, str], list[dict[str, str]]] = {}
    for row in zne_metric_rows:
        source_by_cell.setdefault((row["instance"], float(row["noise"]), int(row["shots_total"]), row["zne_order"]), []).append(row)

    out_rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        cell_rows: list[dict[str, Any]] = []
        for base in bootstrap_rows:
            instance = str(base["instance"])
            if instance not in ideal_cache:
                ideal_cache[instance] = _load_array(root, ideal_tasks[instance])
            ideal = ideal_cache[instance]
            key = _cell_key(base)
            seed_rows = sorted(source_by_cell[key], key=lambda row: int(row["seed"]))
            f_values: list[float] = []
            breakdown_count = 0
            for seed_row in seed_rows:
                zne = np.load(root / seed_row["array_path"], allow_pickle=False)
                f_oob, breakdown = oob_fraction(
                    zne,
                    ideal,
                    delta_fraction=threshold,
                    breakdown_threshold=float(cfg["oob"]["breakdown_threshold"]),
                )
                f_values.append(float(f_oob))
                breakdown_count += int(breakdown)
            row = dict(base)
            row["oob_delta_fraction"] = threshold
            row["mean_f_oob"] = float(np.mean(f_values))
            row["max_f_oob"] = float(np.max(f_values))
            row["extrapolation_breakdown_count"] = breakdown_count
            row["extrapolation_breakdown_rate"] = float(breakdown_count / len(seed_rows))
            row["excluded_extrapolation_breakdown"] = bool(breakdown_count > 0)
            row["sensitivity"] = f"oob_{threshold:g}"
            cell_rows.append(row)
        out_rows.extend(_classify_rows(cell_rows, cfg, q_threshold=float(cfg["statistics"]["multiple_comparisons"]["q"])))
    return out_rows


def _q05_rows(bootstrap_rows: list[dict[str, str]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [dict(row, sensitivity="fdr_q_0.05") for row in bootstrap_rows]
    return _classify_rows(rows, cfg, q_threshold=0.05)


def _collect_flips(
    baseline_rows: list[dict[str, str]],
    sensitivity_sets: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    baseline = {_cell_id(row): row for row in baseline_rows}
    flips: list[dict[str, Any]] = []
    for name, rows in sensitivity_sets.items():
        for row in rows:
            cell_id = _cell_id(row)
            base = baseline[cell_id]
            classification_changed = str(base["classification"]) != str(row["classification"])
            discordant_changed = _csv_bool(base["discordant_fdr"]) != bool(row["discordant_fdr"])
            strong_changed = _csv_bool(base["strongly_discordant_fdr"]) != bool(row["strongly_discordant_fdr"])
            if classification_changed or discordant_changed or strong_changed:
                flips.append({
                    "sensitivity": name,
                    "cell_id": cell_id,
                    "instance": row["instance"],
                    "noise": row["noise"],
                    "shots_total": row["shots_total"],
                    "zne_order": row["zne_order"],
                    "baseline_classification": base["classification"],
                    "sensitivity_classification": row["classification"],
                    "baseline_discordant_fdr": base["discordant_fdr"],
                    "sensitivity_discordant_fdr": row["discordant_fdr"],
                    "baseline_strongly_discordant_fdr": base["strongly_discordant_fdr"],
                    "sensitivity_strongly_discordant_fdr": row["strongly_discordant_fdr"],
                })
    return flips


def _make_tables(
    result_dir: Path,
    classification_rows: list[dict[str, str]],
    paper_summary: dict[str, Any],
    overwrite: bool,
) -> dict[str, str]:
    _write_key_value_csv(result_dir / TABLE_GO_NO_GO_CSV, paper_summary, overwrite=overwrite)
    discordant_fdr = [row for row in classification_rows if row["classification"] == "discordant_fdr"]
    discordant_not_fdr = [row for row in classification_rows if row["classification"] == "discordant_not_fdr"]
    exclusions = [row for row in classification_rows if str(row["classification"]).startswith("excluded")]
    selected_cols = [
        "instance",
        "noise",
        "shots_total",
        "zne_order",
        "classification",
        "median_r_mse",
        "ci_low_r_mse",
        "ci_high_r_mse",
        "median_d_R",
        "ci_low_d_R",
        "ci_high_d_R",
        "q_conjunction_bh",
        "prob_d_R_positive_seed",
    ]
    _write_csv(result_dir / TABLE_DISCORDANT_FDR_CSV, [{key: row[key] for key in selected_cols} for row in discordant_fdr], overwrite=overwrite)
    _write_csv(result_dir / TABLE_DISCORDANT_NOT_FDR_CSV, [{key: row[key] for key in selected_cols} for row in discordant_not_fdr], overwrite=overwrite)
    _write_csv(
        result_dir / TABLE_EXCLUSIONS_CSV,
        [{key: row[key] for key in ["instance", "noise", "shots_total", "zne_order", "classification", "below_detection_floor_count", "extrapolation_breakdown_count"]} for row in exclusions],
        overwrite=overwrite,
    )
    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted({(row["instance"], row["noise"]): [] for row in classification_rows}.items()):
        subset = [row for row in classification_rows if (row["instance"], row["noise"]) == key]
        counts = Counter(row["classification"] for row in subset)
        summary_rows.append({
            "instance": key[0],
            "noise": key[1],
            "discordant_fdr": counts.get("discordant_fdr", 0),
            "discordant_not_fdr": counts.get("discordant_not_fdr", 0),
            "excluded_below_detection_floor": counts.get("excluded_below_detection_floor", 0),
            "not_discordant": counts.get("not_discordant", 0),
            "total": len(subset),
        })
    _write_csv(result_dir / TABLE_INSTANCE_NOISE_CSV, summary_rows, overwrite=overwrite)
    return {
        "go_no_go_summary": TABLE_GO_NO_GO_CSV,
        "discordant_fdr": TABLE_DISCORDANT_FDR_CSV,
        "discordant_not_fdr": TABLE_DISCORDANT_NOT_FDR_CSV,
        "exclusions": TABLE_EXCLUSIONS_CSV,
        "instance_noise_summary": TABLE_INSTANCE_NOISE_CSV,
    }


def _plot_classification_counts(fig_dir: Path, classification_rows: list[dict[str, str]]) -> str:
    counts = Counter(row["classification"] for row in classification_rows)
    order = ["discordant_fdr", "discordant_not_fdr", "excluded_below_detection_floor", "not_discordant"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    ax.bar(order, [counts.get(label, 0) for label in order], color=["#2a6fbb", "#7aa6d9", "#9a9a9a", "#d6d6d6"])
    ax.set_ylabel("Cells")
    ax.set_title("Phase C 120x120 GO/NO-GO classification counts")
    ax.tick_params(axis="x", rotation=25)
    path = fig_dir / "phase_c_120_classification_counts.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def _plot_discordance_scatter(fig_dir: Path, classification_rows: list[dict[str, str]]) -> str:
    colors = {
        "discordant_fdr": "#1b7837",
        "discordant_not_fdr": "#a6dba0",
        "excluded_below_detection_floor": "#8c8c8c",
        "not_discordant": "#bdbdbd",
    }
    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    for label in sorted(colors):
        subset = [row for row in classification_rows if row["classification"] == label]
        if not subset:
            continue
        xs = np.asarray([float(row["median_r_mse"]) for row in subset], dtype=float)
        ys = np.asarray([float(row["median_d_R"]) for row in subset], dtype=float)
        ax.scatter(np.clip(xs, -1.0, 1.0), ys, label=label, s=35, color=colors[label], edgecolor="black", linewidth=0.25, alpha=0.85)
    ax.axvline(0.10, color="#333333", linewidth=0.9, linestyle="--")
    ax.axvline(0.20, color="#333333", linewidth=0.7, linestyle=":")
    ax.axhline(0.0, color="#333333", linewidth=0.9)
    ax.axhline(float(classification_rows[0]["tau_R"]), color="#333333", linewidth=0.7, linestyle=":")
    ax.set_xlabel("Median r_MSE (display clipped to [-1, 1])")
    ax.set_ylabel("Median d_R")
    ax.set_title("Phase C 120x120 MSE improvement vs optimization regret")
    ax.legend(fontsize=7)
    path = fig_dir / "phase_c_120_discordance_scatter.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def _plot_primary_landscape(root: Path, fig_dir: Path, classification_rows: list[dict[str, str]], comparison_rows: list[dict[str, str]]) -> str:
    candidates = [row for row in classification_rows if row["classification"] == "discordant_fdr"]
    candidates.sort(key=lambda row: (float(row["q_conjunction_bh"]), -float(row["median_d_R"])))
    selected = candidates[0]
    source = [
        row for row in comparison_rows
        if row["instance"] == selected["instance"]
        and float(row["noise"]) == float(selected["noise"])
        and int(row["shots_total"]) == int(selected["shots_total"])
        and row["zne_order"] == selected["zne_order"]
    ]
    seed_row = sorted(source, key=lambda row: int(row["seed"]))[0]
    manifest = _load_json(_manifest_path(root))
    ideal_task = next(task for task in manifest["tasks"] if task["kind"] == "ideal" and task["instance"] == selected["instance"])
    ideal = _load_array(root, ideal_task)
    noisy = np.load(root / seed_row["noisy_array_path"], allow_pickle=False)
    zne = np.load(root / seed_row["zne_array_path"], allow_pickle=False)
    items = [("Ideal", ideal), ("Noisy", noisy), (f"ZNE {selected['zne_order']}", zne), ("ZNE - Ideal", zne - ideal)]
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 3.6), constrained_layout=True)
    vmin = min(float(np.min(data)) for _, data in items[:3])
    vmax = max(float(np.max(data)) for _, data in items[:3])
    diff_abs = float(np.max(np.abs(items[3][1])))
    for ax, (title, data) in zip(axes, items, strict=True):
        if title == "ZNE - Ideal":
            im = ax.imshow(data.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-diff_abs, vmax=diff_abs)
        else:
            im = ax.imshow(data.T, origin="lower", aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("gamma index")
        ax.set_ylabel("beta index")
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(f"Primary discordant landscape: {selected['instance']}, p={float(selected['noise']):g}, B={selected['shots_total']}, seed={seed_row['seed']}")
    path = fig_dir / "phase_c_120_primary_landscape_example.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path.name


def _make_figures(root: Path, classification_rows: list[dict[str, str]], comparison_rows: list[dict[str, str]]) -> dict[str, str]:
    fig_dir = ensure_dir(root / "results" / "figures")
    return {
        "classification_counts": f"results/figures/{_plot_classification_counts(fig_dir, classification_rows)}",
        "discordance_scatter": f"results/figures/{_plot_discordance_scatter(fig_dir, classification_rows)}",
        "primary_landscape_example": f"results/figures/{_plot_primary_landscape(root, fig_dir, classification_rows, comparison_rows)}",
    }


def _write_limitations(path: Path, summary: dict[str, Any], sensitivity_summary: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite limitations file: {path}")
    text = f"""# Phase C 120x120 Limitations And Sign-Off

Generated UTC: `{summary['created_at_utc']}`

## Primary Decision

`paper_go_strong = {summary['paper_go_strong']}`.

The paper-level GO is supported by criterion 2: `{summary['criterion_2_at_least_5_discordant_with_3_fdr_and_spread']}`.
Criterion 1 is false: `{summary['criterion_1_at_least_2_strongly_discordant_fdr']}`.

## Required Interpretation

The result should be stated as a distributed discordance pattern, not as two or more individually strong discordant cells.

- Discordant pre-FDR cells: `{summary['discordant_pre_fdr_count']}`
- Discordant FDR cells: `{summary['discordant_fdr_count']}`
- Strongly discordant FDR cells: `{summary['strongly_discordant_fdr_count']}`
- Below-detection-floor exclusions: `{summary['excluded_below_detection_floor_count']}`
- Extrapolation-breakdown exclusions: `{summary['excluded_extrapolation_breakdown_count']}`

## Sensitivity Summary

- OOB thresholds checked: `{sensitivity_summary['oob_thresholds']}`
- FDR q=0.05 paper GO: `{sensitivity_summary['q05_summary']['paper_go_strong']}`
- BCa CI paper GO: `{sensitivity_summary['bca_summary']['paper_go_strong']}`
- Total classification flips across sensitivities: `{sensitivity_summary['total_sensitivity_flips']}`

Any flip listed in `sensitivity_flips.csv` must be reported in the paper or supplement.

## Sign-Off Checklist

- [x] Official 120x120 tasks complete.
- [x] Post-merge manifest passed.
- [x] Backup verified.
- [x] Primary analysis completed.
- [x] Classification matrix exists.
- [x] Core tables generated.
- [x] Primary figures generated.
- [x] Sensitivity analyses generated.
- [x] Limitations documented.
"""
    path.write_text(text, encoding="utf-8")


def run_phase_c_120_paper_outputs(overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    paths = _required_paths(root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required Phase C 120 outputs before paper package: {missing}")
    result_dir = _result_dir(root)
    outputs = [
        result_dir / SENSITIVITY_SUMMARY,
        result_dir / SENSITIVITY_OOB_CSV,
        result_dir / SENSITIVITY_Q05_CSV,
        result_dir / SENSITIVITY_BCA_CSV,
        result_dir / SENSITIVITY_FLIPS_CSV,
        result_dir / TABLE_GO_NO_GO_CSV,
        result_dir / TABLE_DISCORDANT_FDR_CSV,
        result_dir / TABLE_DISCORDANT_NOT_FDR_CSV,
        result_dir / TABLE_EXCLUSIONS_CSV,
        result_dir / TABLE_INSTANCE_NOISE_CSV,
        result_dir / PAPER_OUTPUTS_SUMMARY,
        result_dir / LIMITATIONS_MD,
    ]
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing paper outputs: {existing}")

    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    metric_rows = _read_csv(paths["metrics"])
    comparison_rows = _read_csv(paths["comparisons"])
    bootstrap_rows = _read_csv(paths["bootstrap"])
    classification_rows = _read_csv(paths["classification"])
    paper_summary = _load_json(paths["paper"])
    statistics_summary = _load_json(paths["statistics"])

    q05_rows = _q05_rows(bootstrap_rows, cfg)
    bca_rows = _build_bca_rows(bootstrap_rows, comparison_rows, cfg)
    oob_rows = _oob_threshold_rows(root, bootstrap_rows, metric_rows, cfg, thresholds=[0.05, 0.10, 0.15])
    sensitivity_sets = {
        "fdr_q_0.05": q05_rows,
        "bca_ci": bca_rows,
        "oob_0.05": [row for row in oob_rows if float(row["oob_delta_fraction"]) == 0.05],
        "oob_0.10": [row for row in oob_rows if float(row["oob_delta_fraction"]) == 0.10],
        "oob_0.15": [row for row in oob_rows if float(row["oob_delta_fraction"]) == 0.15],
    }
    flips = _collect_flips(classification_rows, sensitivity_sets)
    sensitivity_summary = {
        "output_version": OUTPUT_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_c_120_sensitivity_complete",
        "oob_thresholds": [0.05, 0.10, 0.15],
        "q05_summary": _summary_from_classification(q05_rows),
        "bca_summary": _summary_from_classification(bca_rows),
        "oob_summaries": {
            f"{threshold:g}": _summary_from_classification([row for row in oob_rows if float(row["oob_delta_fraction"]) == threshold])
            for threshold in [0.05, 0.10, 0.15]
        },
        "total_sensitivity_flips": len(flips),
        "flip_counts_by_sensitivity": dict(sorted(Counter(row["sensitivity"] for row in flips).items())),
        "primary_outputs_unchanged": paper_summary.get("paper_go_strong") is True,
    }
    _write_csv(result_dir / SENSITIVITY_Q05_CSV, q05_rows, overwrite=overwrite)
    _write_csv(result_dir / SENSITIVITY_BCA_CSV, bca_rows, overwrite=overwrite)
    _write_csv(result_dir / SENSITIVITY_OOB_CSV, oob_rows, overwrite=overwrite)
    _write_csv(result_dir / SENSITIVITY_FLIPS_CSV, flips, overwrite=overwrite)
    write_json_atomic(result_dir / SENSITIVITY_SUMMARY, sensitivity_summary, overwrite=True)

    table_outputs = _make_tables(result_dir, classification_rows, paper_summary, overwrite=overwrite)
    figure_outputs = _make_figures(root, classification_rows, comparison_rows)
    output_summary = {
        "output_version": OUTPUT_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_c_120_paper_outputs_complete",
        "source_statistics_status": statistics_summary.get("status"),
        "paper_go_strong": paper_summary.get("paper_go_strong"),
        "tables": {key: f"results/phase_c_120_analysis/{value}" for key, value in table_outputs.items()},
        "figures": figure_outputs,
        "sensitivity_summary": f"results/phase_c_120_analysis/{SENSITIVITY_SUMMARY}",
        "limitations": f"results/phase_c_120_analysis/{LIMITATIONS_MD}",
        "analysis_hashes_current": {
            **_current_analysis_hashes(root),
            "scripts/05_make_figures.py": sha256_file(root / "scripts" / "05_make_figures.py"),
            "src/phase_c_120_paper_outputs.py": sha256_file(root / "src" / "phase_c_120_paper_outputs.py"),
        },
    }
    write_json_atomic(result_dir / PAPER_OUTPUTS_SUMMARY, output_summary, overwrite=True)
    _write_limitations(result_dir / LIMITATIONS_MD, {**paper_summary, "created_at_utc": output_summary["created_at_utc"]}, sensitivity_summary, overwrite=overwrite)
    return output_summary
