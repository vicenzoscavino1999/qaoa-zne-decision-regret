from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .io_utils import ensure_dir


def save_smoke_heatmaps(path: str | Path, ideal: np.ndarray, noisy: np.ndarray, zne: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    items = [("Ideal C_I", ideal), ("Noisy C_N", noisy), ("ZNE linear C_Z", zne)]
    vmin = min(float(np.min(x)) for _, x in items)
    vmax = max(float(np.max(x)) for _, x in items)
    for ax, (title, data) in zip(axes, items, strict=True):
        im = ax.imshow(data.T, origin="lower", aspect="auto", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("gamma index")
        ax.set_ylabel("beta index")
    fig.colorbar(im, ax=axes, shrink=0.85)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_phase_b_scatter(path: str | Path, comparison_rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    rows = [
        row
        for row in comparison_rows
        if row.get("comparison_type") == "zne_vs_noisy"
        and np.isfinite(float(row["r_mse"]))
        and np.isfinite(float(row["d_R"]))
    ]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)
    markers = {"linear": "o", "cubic": "s"}
    colors = {"proportional_to_abs_weights": "#1f77b4", "uniform": "#d62728"}
    for allocation in sorted({str(row["allocation"]) for row in rows}):
        for order in sorted({str(row["zne_order"]) for row in rows}):
            subset = [row for row in rows if row["allocation"] == allocation and row["zne_order"] == order]
            if not subset:
                continue
            ax.scatter(
                [float(row["r_mse"]) for row in subset],
                [float(row["d_R"]) for row in subset],
                label=f"{order} / {allocation}",
                marker=markers.get(order, "o"),
                color=colors.get(allocation, "#555555"),
                alpha=0.85,
                edgecolors="black",
                linewidths=0.4,
            )
    ax.axvline(0.0, color="#333333", linewidth=0.8)
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_xlabel("r_MSE")
    ax.set_ylabel("d_R")
    ax.set_title("Phase B preliminary discordance")
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=160)
    plt.close(fig)
