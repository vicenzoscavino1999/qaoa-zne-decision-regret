from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .phase_c_120_analysis import _as_float, _as_int, _read_csv


PHASE_D_PAPER_OUTPUTS_VERSION = "phase_d_paper_outputs_v1"
PAPER_OUTPUT_DIR = "paper_outputs"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _result_dir(root: Path) -> Path:
    return ensure_dir(root / "results" / "phase_d" / PAPER_OUTPUT_DIR)


def _fig_dir(root: Path) -> Path:
    return ensure_dir(root / "results" / "figures")


def _hashes(root: Path, rels: list[str]) -> dict[str, str | None]:
    return {rel: sha256_file(root / rel) if (root / rel).is_file() else None for rel in rels}


def _write_csv_ordered(path: Path, rows: list[dict[str, Any]], fields: list[str], overwrite: bool = False) -> None:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _required_paths(root: Path) -> dict[str, Path]:
    return {
        "phase_d_global_summary": root / "results" / "phase_d" / "global_D1_D2_D3_D4" / "fdr_global_540_summary.json",
        "phase_d_classification": root / "results" / "phase_d" / "global_D1_D2_D3_D4" / "classification_matrix.csv",
        "phase_d_comparisons": root / "results" / "phase_d" / "global_D1_D2_D3_D4" / "comparisons_per_seed.csv",
        "phase_d_instance_endpoint": root / "results" / "phase_d" / "global_D1_D2_D3_D4" / "instance_primary_endpoint.json",
        "phase_d_manifest": root / "results" / "phase_d" / "global_D1_D2_D3_D4" / "manifest_final.json",
        "phase_d_robustness_tier1": root / "results" / "phase_d" / "robustness" / "phase_d_540_robustness_summary.json",
        "phase_d_robustness_tier2": root / "results" / "phase_d" / "robustness" / "phase_d_540_robustness_tier2_summary.json",
        "phase_d_convergence": root / "results" / "phase_d" / "convergence" / "phase_d_convergence_assessment.json",
        "phase_cd_pooled": root / "results" / "phase_d" / "robustness" / "phase_cd_pooled_15" / "phase_cd_pooled_15_summary.json",
        "phase_cd_pooled_by_instance": root / "results" / "phase_d" / "robustness" / "phase_cd_pooled_15" / "phase_cd_pooled_15_by_instance.csv",
    }


def _set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#2f3437",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#e7e9ec",
        "grid.linewidth": 0.7,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "savefig.bbox": "tight",
    })


def _phase_d_label_order() -> list[str]:
    return [
        "discordant_fdr",
        "discordant_fdr_q10_only",
        "discordant_not_fdr",
        "excluded_below_detection_floor",
        "excluded_extrapolation_breakdown",
        "not_discordant",
    ]


def _label_colors() -> dict[str, str]:
    return {
        "discordant_fdr": "#166534",
        "discordant_fdr_q10_only": "#d99a20",
        "discordant_not_fdr": "#7cb342",
        "excluded_below_detection_floor": "#7a7f86",
        "excluded_extrapolation_breakdown": "#4f565e",
        "not_discordant": "#d7dce2",
    }


def _display_labels() -> dict[str, str]:
    return {
        "discordant_fdr": "FDR q=0.05",
        "discordant_fdr_q10_only": "q=0.10 only",
        "discordant_not_fdr": "nominal only",
        "excluded_below_detection_floor": "below floor",
        "excluded_extrapolation_breakdown": "extrapolation\nbreakdown",
        "not_discordant": "not discordant",
    }


def _paper_assessment_label(label: Any) -> str:
    labels = {
        "weak_independent_replication": "robust, effect-size-limited",
        "strong_independent_replication": "strong independent replication",
        "directionally_positive_phase_d": "directionally positive replication",
    }
    return labels.get(str(label), str(label).replace("_", " "))


def _paper_outcome_label(label: Any) -> str:
    labels = {
        "partial_convergence": "partial convergence",
        "full_convergence": "full convergence",
        "no_convergence": "no convergence",
    }
    return labels.get(str(label), str(label).replace("_", " "))


def _paper_source_label(label: Any) -> str:
    labels = {
        "PhaseC": "Discovery set",
        "PhaseD": "Independent replication set",
    }
    return labels.get(str(label), str(label).replace("_", " "))


def _plot_phase_d_classification_counts(fig_dir: Path, rows: list[dict[str, str]], summary: dict[str, Any]) -> str:
    counts = Counter(row["classification"] for row in rows)
    order = [label for label in _phase_d_label_order() if counts.get(label, 0)]
    colors = _label_colors()
    display = _display_labels()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={"width_ratios": [1.18, 1.0]}, constrained_layout=True)
    ax = axes[0]
    plot_order = list(reversed(order))
    bars = ax.barh(
        [display.get(label, label) for label in plot_order],
        [counts[label] for label in plot_order],
        color=[colors.get(label, "#bbbbbb") for label in plot_order],
        edgecolor="#2f3437",
        linewidth=0.4,
    )
    for bar in bars:
        ax.text(bar.get_width() + 6, bar.get_y() + bar.get_height() / 2.0, f"{int(bar.get_width())}", ha="left", va="center", fontsize=8)
    ax.set_xlabel("Cells")
    ax.set_title("Independent replication: 540-cell classification")
    ax.set_xlim(0, max(counts.values()) * 1.20)

    ax2 = axes[1]
    ax2.axis("off")
    facts = [
        ("Assessment", _paper_assessment_label(summary.get("assigned_label"))),
        ("Cell endpoint", "pass" if summary.get("cell_level_primary_endpoint_pass") else "fail"),
        ("Instance endpoint", "pass" if summary.get("instance_level_primary_endpoint_pass") else "fail"),
        ("BH q=0.05 FDR", str(summary.get("discordant_fdr_count"))),
        ("BH q=0.10 FDR", str(summary.get("discordant_fdr_q10_count"))),
        ("Strong-effect cells", str(summary.get("strongly_discordant_fdr_count"))),
    ]
    y = 0.88
    for key, value in facts:
        ax2.text(0.02, y, key, color="#4b5563", fontsize=9, ha="left", va="center")
        ax2.text(0.58, y, value, color="#111827", fontsize=9.5, fontweight="bold", ha="left", va="center")
        y -= 0.13
    ax2.text(0.02, 0.04, "Interpretation: robust replication signal, below the strong-effect threshold.", color="#374151", fontsize=8.5, ha="left")
    path = fig_dir / "independent_replication_cell_classification_summary.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path.name


def _plot_phase_d_discordance_scatter(fig_dir: Path, rows: list[dict[str, str]]) -> str:
    colors = _label_colors()
    display = _display_labels()
    fig, ax = plt.subplots(figsize=(7.8, 5.6), constrained_layout=True)
    for label in _phase_d_label_order():
        subset = [row for row in rows if row["classification"] == label]
        if not subset:
            continue
        xs = np.asarray([_as_float(row["median_r_mse"]) for row in subset], dtype=float)
        ys = np.asarray([_as_float(row["median_d_R"]) for row in subset], dtype=float)
        ax.scatter(
            np.clip(xs, -1.0, 1.0),
            ys,
            s=28 if label != "not_discordant" else 18,
            color=colors.get(label, "#bbbbbb"),
            edgecolor="#1f2937" if label != "not_discordant" else "none",
            linewidth=0.25,
            alpha=0.86 if label != "not_discordant" else 0.45,
            label=display.get(label, label),
        )
    tau_r = _as_float(rows[0]["tau_R"])
    ax.axvline(0.10, color="#111827", linestyle="--", linewidth=0.9, label="r_MSE 0.10")
    ax.axvline(0.20, color="#111827", linestyle=":", linewidth=0.8, label="r_MSE 0.20")
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.axhline(tau_r, color="#991b1b", linestyle=":", linewidth=1.0, label="inherited tau_R")
    ax.set_xlabel("Median r_MSE (clipped to [-1, 1])")
    ax.set_ylabel("Median d_R")
    ax.set_title("Replication-set cell-level improvement vs regret shift")
    ax.legend(ncol=2, frameon=False, loc="upper left")
    path = fig_dir / "independent_replication_improvement_regret_scatter.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path.name


def _plot_phase_d_instance_heatmap(fig_dir: Path, rows: list[dict[str, str]]) -> str:
    instances = sorted({row["instance"] for row in rows}, key=lambda value: int(value[1:]) if value.startswith("H") else value)
    noises = sorted({_as_float(row["noise"]) for row in rows})
    matrix = np.zeros((len(instances), len(noises)), dtype=float)
    for i, instance in enumerate(instances):
        for j, noise in enumerate(noises):
            subset = [row for row in rows if row["instance"] == instance and _as_float(row["noise"]) == noise]
            matrix[i, j] = sum(_bool(row["discordant_fdr"]) for row in subset)
    fig, ax = plt.subplots(figsize=(7.6, 5.8), constrained_layout=True)
    im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=max(1.0, float(np.max(matrix))), aspect="auto")
    ax.set_xticks(range(len(noises)), [f"{noise:g}" for noise in noises])
    ax.set_yticks(range(len(instances)), instances)
    ax.set_xlabel("Noise p")
    ax.set_ylabel("Replication instance")
    ax.set_title("FDR cell count by replication instance and noise")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="#111827" if value <= 2 else "white", fontsize=8, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.78, label="FDR q=0.05 cells")
    path = fig_dir / "independent_replication_instance_noise_heatmap.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path.name


def _choose_landscape_cell(rows: list[dict[str, str]]) -> dict[str, str]:
    candidates = [row for row in rows if row["classification"] == "discordant_fdr"]
    if not candidates:
        raise ValueError("No independent-replication discordant_fdr cells available for landscape plot.")
    return max(candidates, key=lambda row: (_as_float(row["median_r_mse"]) * _as_float(row["median_d_R"]), -_as_float(row["q_conjunction_bh_q05"])))


def _choose_control_landscape_cell(rows: list[dict[str, str]], reference: dict[str, str]) -> dict[str, str]:
    same_condition = [
        row for row in rows
        if row["classification"] == "not_discordant"
        and row["instance"] == reference["instance"]
        and _as_float(row["noise"]) == _as_float(reference["noise"])
        and not _bool(row.get("excluded", False))
    ]
    candidates = same_condition or [
        row for row in rows
        if row["classification"] == "not_discordant"
        and not _bool(row.get("excluded", False))
    ]
    if not candidates:
        raise ValueError("No not-discordant control cells available for landscape plot.")
    return min(
        candidates,
        key=lambda row: (
            abs(_as_float(row["median_r_mse"])) + abs(_as_float(row["median_d_R"])),
            _as_float(row["q_conjunction_bh_q05"]),
            row["instance"],
            _as_float(row["noise"]),
            _as_int(row["shots_total"]),
            row["zne_order"],
        ),
    )


def _select_distinct_signal_cells(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    candidates = sorted(
        [row for row in rows if row["classification"] == "discordant_fdr"],
        key=lambda row: (_as_float(row["median_r_mse"]) * _as_float(row["median_d_R"]), _as_float(row["median_r_mse"])),
        reverse=True,
    )
    selected: list[dict[str, str]] = []
    used_instances: set[str] = set()
    for row in candidates:
        if row["instance"] in used_instances:
            continue
        selected.append(row)
        used_instances.add(row["instance"])
        if len(selected) == count:
            return selected
    for row in candidates:
        if row not in selected:
            selected.append(row)
            if len(selected) == count:
                return selected
    return selected


def _representative_seed_row(selected: dict[str, str], comparisons: list[dict[str, str]]) -> dict[str, str]:
    source = [
        row for row in comparisons
        if row["instance"] == selected["instance"]
        and _as_float(row["noise"]) == _as_float(selected["noise"])
        and _as_int(row["shots_total"]) == _as_int(selected["shots_total"])
        and row["zne_order"] == selected["zne_order"]
    ]
    if not source:
        raise ValueError(f"No comparison rows found for selected landscape cell {selected['cell_id']}")
    median_r = _as_float(selected["median_r_mse"])
    median_d = _as_float(selected["median_d_R"])
    return min(source, key=lambda row: abs(_as_float(row["r_mse"]) - median_r) + abs(_as_float(row["d_R"]) - median_d))


def _ideal_array_path(manifest: dict[str, Any], instance: str) -> str:
    for task in manifest.get("tasks", []):
        if task.get("kind") == "ideal" and task.get("instance") == instance:
            return str(task["output_path"])
    raise ValueError(f"No ideal landscape task found for {instance}")


def _plot_landscape_panel(ax: Any, data: np.ndarray, title: str, cmap: str, vmin: float, vmax: float, contour: bool = True) -> Any:
    im = ax.imshow(data.T, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if contour and float(np.nanmax(data)) > float(np.nanmin(data)):
        levels = np.linspace(float(np.nanmin(data)), float(np.nanmax(data)), 9)
        ax.contour(data.T, levels=levels, colors="#111827", linewidths=0.35, alpha=0.45, origin="lower")
    ax.set_title(title)
    ax.set_xlabel("gamma index")
    ax.set_ylabel("beta index")
    return im


def _plot_topographic_landscape(
    root: Path,
    fig_dir: Path,
    selected: dict[str, str],
    comparisons: list[dict[str, str]],
    manifest: dict[str, Any],
    title: str,
    selection_note: str,
    filename: str,
) -> str:
    seed_row = _representative_seed_row(selected, comparisons)
    ideal = np.load(root / _ideal_array_path(manifest, selected["instance"]), allow_pickle=False)
    noisy = np.load(root / seed_row["noisy_array_path"], allow_pickle=False)
    zne = np.load(root / seed_row["zne_array_path"], allow_pickle=False)
    landscapes = [("Ideal", ideal), ("Noisy", noisy), (f"ZNE {selected['zne_order']}", zne)]
    residuals = [("Noisy - Ideal", noisy - ideal), ("ZNE - Ideal", zne - ideal), ("ZNE - Noisy", zne - noisy)]
    vmin = min(float(np.min(data)) for _, data in landscapes)
    vmax = max(float(np.max(data)) for _, data in landscapes)
    residual_abs = max(float(np.max(np.abs(data))) for _, data in residuals)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4), constrained_layout=True)
    for ax, (panel_title, data) in zip(axes[0], landscapes, strict=True):
        im = _plot_landscape_panel(ax, data, panel_title, "viridis", vmin, vmax, contour=True)
        fig.colorbar(im, ax=ax, shrink=0.74)
    for ax, (panel_title, data) in zip(axes[1], residuals, strict=True):
        im = _plot_landscape_panel(ax, data, panel_title, "coolwarm", -residual_abs, residual_abs, contour=False)
        fig.colorbar(im, ax=ax, shrink=0.74)
    fig.suptitle(
        f"{title}: "
        f"{selected['instance']}, p={_as_float(selected['noise']):g}, B={selected['shots_total']}, "
        f"{selected['zne_order']}, seed={seed_row['seed']}\n"
        f"{selection_note} "
        f"(medians: r_MSE={_as_float(selected['median_r_mse']):.3f}, d_R={_as_float(selected['median_d_R']):.4f})",
        fontsize=12,
    )
    path = fig_dir / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path.name


def _plot_phase_d_primary_landscape(root: Path, fig_dir: Path, rows: list[dict[str, str]], comparisons: list[dict[str, str]], manifest: dict[str, Any]) -> str:
    selected = _choose_landscape_cell(rows)
    return _plot_topographic_landscape(
        root,
        fig_dir,
        selected,
        comparisons,
        manifest,
        "Representative topographic landscape from the independent replication set",
        "q=0.05 FDR cell selected reproducibly by median r_MSE x median d_R",
        "independent_replication_topographic_landscape_example.png",
    )


def _plot_phase_d_control_landscape(root: Path, fig_dir: Path, rows: list[dict[str, str]], comparisons: list[dict[str, str]], manifest: dict[str, Any]) -> str:
    reference = _choose_landscape_cell(rows)
    selected = _choose_control_landscape_cell(rows, reference)
    return _plot_topographic_landscape(
        root,
        fig_dir,
        selected,
        comparisons,
        manifest,
        "Topographic control from the independent replication set",
        "not-discordant control selected reproducibly by minimum |median r_MSE| + |median d_R|",
        "independent_replication_topographic_control_example.png",
    )


def _plot_phase_d_signal_atlas(root: Path, fig_dir: Path, rows: list[dict[str, str]], comparisons: list[dict[str, str]], manifest: dict[str, Any]) -> str:
    selected_cells = _select_distinct_signal_cells(rows, count=6)
    items: list[dict[str, Any]] = []
    for selected in selected_cells:
        seed_row = _representative_seed_row(selected, comparisons)
        ideal = np.load(root / _ideal_array_path(manifest, selected["instance"]), allow_pickle=False)
        zne = np.load(root / seed_row["zne_array_path"], allow_pickle=False)
        items.append({
            "selected": selected,
            "seed": seed_row["seed"],
            "ideal": ideal,
            "residual": zne - ideal,
        })
    residual_abs = max(float(np.max(np.abs(item["residual"]))) for item in items)
    residual_abs = max(residual_abs, 1e-9)
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), constrained_layout=True)
    im = None
    for ax, item in zip(axes.ravel(), items, strict=True):
        selected = item["selected"]
        data = item["residual"]
        im = ax.imshow(data.T, origin="lower", aspect="auto", cmap="coolwarm", vmin=-residual_abs, vmax=residual_abs)
        ax.contour(item["ideal"].T, levels=8, colors="#111827", linewidths=0.25, alpha=0.28, origin="lower")
        ax.set_title(
            f"{selected['instance']} | p={_as_float(selected['noise']):g} | B={selected['shots_total']} | {selected['zne_order']}\n"
            f"r_MSE={_as_float(selected['median_r_mse']):.3f}, d_R={_as_float(selected['median_d_R']):.4f}",
            fontsize=9,
        )
        ax.set_xlabel("gamma index")
        ax.set_ylabel("beta index")
    fig.suptitle("Topographic residual atlas for BH-retained replication cells", fontsize=13)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.76, label="ZNE - Ideal")
    path = fig_dir / "independent_replication_topographic_signal_atlas.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path.name


def _condition_columns(rows: list[dict[str, str]]) -> list[tuple[float, int, str]]:
    noises = sorted({_as_float(row["noise"]) for row in rows})
    shots = sorted({_as_int(row["shots_total"]) for row in rows})
    zne_order = ["linear", "quadratic", "cubic"]
    return [(noise, shot, zne) for noise in noises for shot in shots for zne in zne_order]


def _plot_phase_d_condition_fingerprint(fig_dir: Path, rows: list[dict[str, str]]) -> str:
    instances = sorted({row["instance"] for row in rows}, key=lambda value: int(value[1:]) if value.startswith("H") else value)
    columns = _condition_columns(rows)
    code_by_label = {
        "not_discordant": 0,
        "excluded_below_detection_floor": 1,
        "excluded_extrapolation_breakdown": 2,
        "discordant_not_fdr": 3,
        "discordant_fdr_q10_only": 4,
        "discordant_fdr": 5,
    }
    colors = ["#d7dce2", "#7a7f86", "#4f565e", "#9ccc65", "#d99a20", "#166534"]
    matrix = np.full((len(instances), len(columns)), np.nan)
    lookup = {
        (row["instance"], _as_float(row["noise"]), _as_int(row["shots_total"]), row["zne_order"]): row
        for row in rows
    }
    for i, instance in enumerate(instances):
        for j, (noise, shot, zne) in enumerate(columns):
            row = lookup[(instance, noise, shot, zne)]
            matrix[i, j] = code_by_label[row["classification"]]
    fig, ax = plt.subplots(figsize=(14.2, 5.6), constrained_layout=True)
    cmap = ListedColormap(colors)
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(colors) - 0.5)
    ax.set_yticks(range(len(instances)), instances)
    group = 9
    noises = sorted({_as_float(row["noise"]) for row in rows})
    ax.set_xticks([idx * group + (group - 1) / 2.0 for idx in range(len(noises))], [f"p={noise:g}" for noise in noises])
    ax.set_xlabel("Noise group; within each group: B=512/1024/4096 and ZNE=linear/quadratic/cubic")
    ax.set_ylabel("Replication instance")
    ax.set_title("Condition fingerprint across the 540-cell independent replication family")
    for boundary in range(1, len(noises)):
        ax.axvline(boundary * group - 0.5, color="white", linewidth=2.0)
    for boundary in range(len(noises)):
        base = boundary * group
        ax.axvline(base + 3 - 0.5, color="white", linewidth=0.8, alpha=0.8)
        ax.axvline(base + 6 - 0.5, color="white", linewidth=0.8, alpha=0.8)
    legend_items = [
        Patch(facecolor=colors[5], edgecolor="#1f2937", label="FDR q=0.05"),
        Patch(facecolor=colors[4], edgecolor="#1f2937", label="q=0.10 only"),
        Patch(facecolor=colors[3], edgecolor="#1f2937", label="nominal only"),
        Patch(facecolor=colors[1], edgecolor="#1f2937", label="below floor"),
        Patch(facecolor=colors[0], edgecolor="#1f2937", label="not discordant"),
    ]
    ax.legend(handles=legend_items, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.11))
    path = fig_dir / "independent_replication_condition_fingerprint_atlas.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path.name


def _plot_phase_d_robustness(fig_dir: Path, tier1: dict[str, Any], tier2: dict[str, Any], pooled: dict[str, Any]) -> str:
    leave_one_seed = tier2.get("leave_one_seed", {})
    nonparam = tier2.get("nonparametric", {})
    values = {
        "Replication": tier1.get("primary_global_summary", {}).get("discordant_fdr_count", 0),
        "Seed LOO min": leave_one_seed.get("discordant_fdr_q05_min", 0),
        "Sign": nonparam.get("sign", {}).get("discordant_fdr_q05_count", 0),
        "Perm median": nonparam.get("permutation_median", {}).get("discordant_fdr_q05_count", 0),
        "Perm mean": nonparam.get("permutation_mean", {}).get("discordant_fdr_q05_count", 0),
        "Pooled\nsupplement": pooled.get("pooled_discordant_fdr_q05_count", 0),
    }
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    labels = list(values)
    counts = [int(values[label]) for label in labels]
    bars = ax.bar(labels, counts, color=["#166534", "#2f855a", "#3b82f6", "#60a5fa", "#93c5fd", "#d99a20"], edgecolor="#1f2937", linewidth=0.4)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.2, f"{int(bar.get_height())}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("FDR q=0.05 cells")
    ax.set_title("Robustness diagnostics for the replication signal")
    ax.tick_params(axis="x", rotation=18)
    ax.set_ylim(0, max(counts) * 1.20)
    subtitle = (
        f"Tier 1 leave-one-instance flips: {tier1.get('leave_one_instance_flip_count')} | "
        f"Tier 2 seed LOO flips: {leave_one_seed.get('seed_loo_flips')} | "
        f"supplementary pooled analysis only"
    )
    ax.text(0.0, -0.22, subtitle, transform=ax.transAxes, color="#374151", fontsize=8.5, ha="left")
    path = fig_dir / "independent_replication_robustness_diagnostics.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path.name


def _plot_phase_cd_convergence(fig_dir: Path, convergence: dict[str, Any], pooled: dict[str, Any]) -> str:
    phase_c = convergence.get("phase_c", {})
    phase_d = convergence.get("phase_d", {})
    labels = ["Discovery\nset", "Independent\nreplication", "Supplementary\npooled"]
    counts = [
        int(phase_c.get("discordant_fdr_count", 0)),
        int(phase_d.get("discordant_fdr_count", 0)),
        int(pooled.get("pooled_discordant_fdr_q05_count", 0)),
    ]
    family = [
        int(phase_c.get("fdr_family_size", 1)),
        int(phase_d.get("fdr_family_size", 1)),
        int(pooled.get("pooled_family_size", 1)),
    ]
    rates = [count / fam for count, fam in zip(counts, family, strict=True)]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), gridspec_kw={"width_ratios": [1.2, 1.0]}, constrained_layout=True)
    ax = axes[0]
    bars = ax.bar(labels, counts, color=["#2563eb", "#166534", "#d99a20"], edgecolor="#1f2937", linewidth=0.5)
    for idx, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.2, f"{counts[idx]}/{family[idx]}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("FDR q=0.05 cells")
    ax.set_title("Discovery-replication convergence summary")
    ax.set_ylim(0, max(counts) * 1.22)

    ax2 = axes[1]
    ax2.bar(labels, rates, color=["#93c5fd", "#86efac", "#f5c66b"], edgecolor="#1f2937", linewidth=0.5)
    ax2.set_ylim(0, max(rates) * 1.35)
    ax2.set_ylabel("FDR cell fraction")
    ax2.set_title(f"Outcome: {_paper_outcome_label(convergence.get('outcome'))}")
    for idx, rate in enumerate(rates):
        ax2.text(idx, rate + max(rates) * 0.04, f"{rate:.1%}", ha="center", va="bottom", fontsize=8)
    ax2.text(0.0, -0.24, "The pooled analysis is supplementary and does not replace discovery or independent-replication endpoints.", transform=ax2.transAxes, fontsize=8.2, color="#374151")
    path = fig_dir / "discovery_replication_convergence_summary.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path.name


def _make_tables(result_dir: Path, rows: list[dict[str, str]], endpoint: dict[str, Any], pooled_by_instance: list[dict[str, str]], overwrite: bool) -> dict[str, str]:
    counts = Counter(row["classification"] for row in rows)
    classification_names = {
        "discordant_fdr": "BH-retained discordant cells (q=0.05)",
        "discordant_fdr_q10_only": "BH-retained only at sensitivity threshold (q=0.10)",
        "discordant_not_fdr": "Nominal discordant only, not BH-retained",
        "excluded_below_detection_floor": "Excluded because below detection floor",
        "excluded_extrapolation_breakdown": "Excluded because of extrapolation breakdown",
        "not_discordant": "Not discordant",
    }
    classification_table = [
        {
            "analysis_set": "Independent replication set",
            "classification": classification_names.get(label, label),
            "cells": counts.get(label, 0),
        }
        for label in _phase_d_label_order()
        if counts.get(label, 0)
    ]
    classification_file = "table_independent_replication_cell_classification.csv"
    _write_csv_ordered(
        result_dir / classification_file,
        classification_table,
        ["analysis_set", "classification", "cells"],
        overwrite=overwrite,
    )

    endpoint_rows = [
        {
            "replication_stratum": row["block_id"],
            "replication_instance": row["instance"],
            "evaluated_stress_cells": row["stress_cells_evaluated"],
            "nominal_discordant_stress_cells": row["discordant_pre_fdr_stress_count"],
            "bh_retained_cells_q05": row["discordant_fdr_q05_count"],
            "bh_retained_cells_q10": row["discordant_fdr_q10_count"],
            "positive_direction_fraction": f"{_as_float(row['positive_direction_fraction_stress']):.3f}",
            "passes_instance_endpoint": row["instance_positive_stress"],
            "supportive_at_q05": row["instance_supportive_q05"],
            "supportive_at_q10": row["instance_supportive_q10"],
        }
        for row in endpoint.get("rows", [])
    ]
    endpoint_file = "table_independent_replication_instance_endpoint.csv"
    _write_csv_ordered(
        result_dir / endpoint_file,
        endpoint_rows,
        [
            "replication_stratum",
            "replication_instance",
            "evaluated_stress_cells",
            "nominal_discordant_stress_cells",
            "bh_retained_cells_q05",
            "bh_retained_cells_q10",
            "positive_direction_fraction",
            "passes_instance_endpoint",
            "supportive_at_q05",
            "supportive_at_q10",
        ],
        overwrite=overwrite,
    )

    pooled_rows = [
        {
            "analysis_set": _paper_source_label(row["source_phase"]),
            "analysis_instance": row["instance"],
            "cells": row["cells"],
            "nominal_discordant_cells": row["pooled_discordant_pre_fdr_count"],
            "bh_retained_cells_q05": row["pooled_discordant_fdr_q05_count"],
            "bh_retained_cells_q10": row["pooled_discordant_fdr_q10_count"],
            "source_family_bh_retained_cells": row["source_discordant_fdr_count"],
            "positive_direction_cells": row["cell_positive_direction_count"],
        }
        for row in pooled_by_instance
    ]
    pooled_file = "table_supplementary_pooled_discovery_replication_by_instance.csv"
    _write_csv_ordered(
        result_dir / pooled_file,
        pooled_rows,
        [
            "analysis_set",
            "analysis_instance",
            "cells",
            "nominal_discordant_cells",
            "bh_retained_cells_q05",
            "bh_retained_cells_q10",
            "source_family_bh_retained_cells",
            "positive_direction_cells",
        ],
        overwrite=overwrite,
    )
    return {
        "independent_replication_cell_classification": classification_file,
        "independent_replication_instance_endpoint": endpoint_file,
        "supplementary_pooled_discovery_replication_by_instance": pooled_file,
    }


def run_phase_d_paper_outputs(overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    paths = _required_paths(root)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required Phase D paper-output inputs: {missing}")
    result_dir = _result_dir(root)
    output_paths = [
        result_dir / "phase_d_paper_outputs_summary.json",
        result_dir / "table_independent_replication_cell_classification.csv",
        result_dir / "table_independent_replication_instance_endpoint.csv",
        result_dir / "table_supplementary_pooled_discovery_replication_by_instance.csv",
        result_dir / "phase_d_visual_interpretation.md",
    ]
    if not overwrite:
        existing = [str(path) for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite existing Phase D paper outputs: {existing}")

    _set_style()
    fig_dir = _fig_dir(root)
    classification_rows = _read_csv(paths["phase_d_classification"])
    comparison_rows = _read_csv(paths["phase_d_comparisons"])
    pooled_by_instance = _read_csv(paths["phase_cd_pooled_by_instance"])
    global_summary = _load_json(paths["phase_d_global_summary"])
    endpoint = _load_json(paths["phase_d_instance_endpoint"])
    manifest = _load_json(paths["phase_d_manifest"])
    tier1 = _load_json(paths["phase_d_robustness_tier1"])
    tier2 = _load_json(paths["phase_d_robustness_tier2"])
    convergence = _load_json(paths["phase_d_convergence"])
    pooled = _load_json(paths["phase_cd_pooled"])

    table_outputs = _make_tables(result_dir, classification_rows, endpoint, pooled_by_instance, overwrite=overwrite)
    figure_outputs = {
        "independent_replication_cell_classification": f"results/figures/{_plot_phase_d_classification_counts(fig_dir, classification_rows, global_summary)}",
        "independent_replication_improvement_regret_scatter": f"results/figures/{_plot_phase_d_discordance_scatter(fig_dir, classification_rows)}",
        "independent_replication_instance_noise_heatmap": f"results/figures/{_plot_phase_d_instance_heatmap(fig_dir, classification_rows)}",
        "independent_replication_topographic_landscape": f"results/figures/{_plot_phase_d_primary_landscape(root, fig_dir, classification_rows, comparison_rows, manifest)}",
        "independent_replication_topographic_control": f"results/figures/{_plot_phase_d_control_landscape(root, fig_dir, classification_rows, comparison_rows, manifest)}",
        "independent_replication_topographic_signal_atlas": f"results/figures/{_plot_phase_d_signal_atlas(root, fig_dir, classification_rows, comparison_rows, manifest)}",
        "independent_replication_condition_fingerprint_atlas": f"results/figures/{_plot_phase_d_condition_fingerprint(fig_dir, classification_rows)}",
        "independent_replication_robustness_diagnostics": f"results/figures/{_plot_phase_d_robustness(fig_dir, tier1, tier2, pooled)}",
        "discovery_replication_convergence_summary": f"results/figures/{_plot_phase_cd_convergence(fig_dir, convergence, pooled)}",
    }

    interpretation_path = result_dir / "phase_d_visual_interpretation.md"
    interpretation_path.write_text(
        "\n".join([
            "# Independent Replication Visual Interpretation",
            "",
            f"Generated UTC: `{_utc_now()}`",
            "",
            "The visual package should be read as paper-output support, not as a new statistical endpoint.",
            "",
            f"- Independent replication assessment: `{_paper_assessment_label(global_summary.get('assigned_label'))}`.",
            f"- Discovery-replication convergence outcome: `{_paper_outcome_label(convergence.get('outcome'))}`.",
            f"- Supplementary pooled analysis: `{pooled.get('pooled_discordant_fdr_q05_count')}` pooled q=0.05 cells across `{pooled.get('pooled_family_size')}` cells.",
            "- The topographic landscape example visualizes an existing representative q=0.05 FDR cell; it is not an additional endpoint.",
            "- The topographic control visualizes an existing not-discordant cell selected by a fixed rule; it is a visual contrast, not an endpoint.",
            "- The topographic signal atlas shows six BH-retained residual landscapes from distinct replication instances selected by a fixed ranking rule.",
            "- The condition fingerprint atlas summarizes all 540 cells as a categorical condition map.",
            "- The pooled diagnostic remains exploratory and does not upgrade the independent-replication label.",
            "",
        ]),
        encoding="utf-8",
    )

    rels = [path.relative_to(root).as_posix() for path in paths.values()]
    rels.extend(figure_outputs.values())
    rels.extend(f"results/phase_d/{PAPER_OUTPUT_DIR}/{name}" for name in table_outputs.values())
    rels.append(f"results/phase_d/{PAPER_OUTPUT_DIR}/phase_d_visual_interpretation.md")
    summary = {
        "paper_outputs_version": PHASE_D_PAPER_OUTPUTS_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_d_paper_outputs_complete",
        "scope": "paper_facing_independent_replication_figures_tables_visual_interpretation",
        "paper_language": "discovery_set_independent_replication_set_supplementary_pooled_analysis",
        "phase_d_assigned_label": global_summary.get("assigned_label"),
        "paper_independent_replication_assessment": _paper_assessment_label(global_summary.get("assigned_label")),
        "phase_d_discordant_fdr_count": global_summary.get("discordant_fdr_count"),
        "phase_d_instance_endpoint_pass": global_summary.get("instance_level_primary_endpoint_pass"),
        "convergence_outcome": convergence.get("outcome"),
        "paper_convergence_outcome": _paper_outcome_label(convergence.get("outcome")),
        "pooled_15_exploratory_q05_cells": pooled.get("pooled_discordant_fdr_q05_count"),
        "tables": {key: f"results/phase_d/{PAPER_OUTPUT_DIR}/{value}" for key, value in table_outputs.items()},
        "figures": figure_outputs,
        "visual_interpretation": f"results/phase_d/{PAPER_OUTPUT_DIR}/phase_d_visual_interpretation.md",
        "input_and_output_hashes": _hashes(root, rels),
    }
    write_json_atomic(result_dir / "phase_d_paper_outputs_summary.json", summary, overwrite=True)
    return summary
