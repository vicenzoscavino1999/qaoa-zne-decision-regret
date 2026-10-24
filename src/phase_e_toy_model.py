from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .circuits import GraphInstance
from .io_utils import PROJECT_ROOT, ensure_dir, write_json_atomic
from .mitiq_wrapper import richardson_weights
from .phase_e_config import DEFAULT_NOISE_RATES
from .phase_e_sim_backend import evaluate_custom_density_objective


PHASE_E_TOY_MODEL_VERSION = "phase_e_toy_model_v0_1"


def toy_ideal_expected_cut(gamma: np.ndarray | float, beta: np.ndarray | float) -> np.ndarray | float:
    return 0.5 + 0.5 * np.sin(gamma) * np.sin(4.0 * beta)


def toy_ideal_objective(gamma: np.ndarray | float, beta: np.ndarray | float) -> np.ndarray | float:
    return -toy_ideal_expected_cut(gamma, beta)


def toy_affine_factor(scale_factor: int, p_dep: float, p_1q_ratio: float = 1.0) -> float:
    """Affine contraction factor for the 2-qubit single-edge control model.

    This is the analytic control model for the toy proof. It captures the
    lightcone action on the traceless landscape component:
    four relevant one-qubit depolarizing hits and two relevant two-qubit hits
    per unfolded circuit copy.
    """
    p_1q = float(p_dep) * float(p_1q_ratio)
    p_2q = float(p_dep)
    if not (0.0 <= p_1q <= 1.0 and 0.0 <= p_2q <= 1.0):
        raise ValueError("Toy affine control requires 0 <= p_1q,p_2q <= 1.")
    scale = int(scale_factor)
    return float(((1.0 - p_1q) ** (4 * scale)) * ((1.0 - p_2q) ** (2 * scale)))


def toy_affine_objective(
    gamma: np.ndarray | float,
    beta: np.ndarray | float,
    scale_factor: int,
    p_dep: float,
    p_1q_ratio: float = 1.0,
) -> np.ndarray | float:
    factor = toy_affine_factor(scale_factor, p_dep, p_1q_ratio=p_1q_ratio)
    return -0.5 - 0.5 * factor * np.sin(gamma) * np.sin(4.0 * beta)


def toy_richardson_affine_factor(
    scale_factors: list[int],
    p_dep: float,
    p_1q_ratio: float = 1.0,
) -> float:
    weights = richardson_weights(scale_factors)
    factors = [toy_affine_factor(scale, p_dep, p_1q_ratio=p_1q_ratio) for scale in scale_factors]
    return float(sum(weight * factor for weight, factor in zip(weights, factors, strict=True)))


def toy_zne_affine_objective(
    gamma: np.ndarray | float,
    beta: np.ndarray | float,
    scale_factors: list[int],
    p_dep: float,
    p_1q_ratio: float = 1.0,
) -> np.ndarray | float:
    factor = toy_richardson_affine_factor(scale_factors, p_dep, p_1q_ratio=p_1q_ratio)
    return -0.5 - 0.5 * factor * np.sin(gamma) * np.sin(4.0 * beta)


def toy_uniform_mse_from_factor(factor: float) -> float:
    # Uniform average of sin(gamma)^2 sin(4 beta)^2 over the torus is 1/4.
    return float(((factor - 1.0) ** 2) / 16.0)


def _grid(grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    gammas = np.linspace(0.0, 2.0 * math.pi, int(grid_size), endpoint=False)
    betas = np.linspace(0.0, math.pi, int(grid_size), endpoint=False)
    return np.meshgrid(gammas, betas, indexing="ij")


def _argmin_location(array: np.ndarray, gammas: np.ndarray, betas: np.ndarray) -> dict[str, Any]:
    index = np.unravel_index(int(np.argmin(array)), array.shape)
    return {
        "index": [int(index[0]), int(index[1])],
        "gamma": float(gammas[index]),
        "beta": float(betas[index]),
        "value": float(array[index]),
    }


def _ideal_backend_validation() -> dict[str, Any]:
    graph = GraphInstance(name="phase_e_toy_single_edge", edges=((0, 1, 1.0),))
    points = [
        (0.0, 0.0),
        (0.37, 0.22),
        (math.pi / 2.0, math.pi / 8.0),
        (1.31, 0.77),
        (5.02, 2.41),
    ]
    rows = []
    for gamma, beta in points:
        analytic = float(toy_ideal_objective(gamma, beta))
        backend = evaluate_custom_density_objective(graph, (gamma, beta), p_dep=0.0).objective_value
        rows.append(
            {
                "gamma": gamma,
                "beta": beta,
                "analytic_objective": analytic,
                "backend_objective": backend,
                "abs_diff": abs(analytic - backend),
            }
        )
    max_abs_diff = max(row["abs_diff"] for row in rows)
    return {
        "status": "passed" if max_abs_diff <= 1e-10 else "failed",
        "max_abs_diff": max_abs_diff,
        "tolerance": 1e-10,
        "rows": rows,
    }


def _noise_table(p_values: tuple[float, ...], p_1q_ratio: float, scale_factors: list[int]) -> list[dict[str, Any]]:
    weights = richardson_weights(scale_factors)
    rows = []
    for p_dep in p_values:
        factor_noisy = toy_affine_factor(1, p_dep, p_1q_ratio=p_1q_ratio)
        factor_zne = toy_richardson_affine_factor(scale_factors, p_dep, p_1q_ratio=p_1q_ratio)
        mse_noisy = toy_uniform_mse_from_factor(factor_noisy)
        mse_zne = toy_uniform_mse_from_factor(factor_zne)
        rows.append(
            {
                "p_dep": float(p_dep),
                "p_1q_ratio": float(p_1q_ratio),
                "scale_factors": scale_factors,
                "richardson_weights": weights,
                "affine_factor_noisy": factor_noisy,
                "affine_factor_zne": factor_zne,
                "MSE_noisy_uniform": mse_noisy,
                "MSE_ZNE_uniform": mse_zne,
                "r_MSE_uniform": float((mse_noisy - mse_zne) / mse_noisy) if mse_noisy != 0.0 else float("nan"),
                "argmin_preserved_noisy": bool(factor_noisy > 0.0),
                "argmin_preserved_zne": bool(factor_zne > 0.0),
                "discordance_possible_in_affine_model": bool(factor_noisy <= 0.0 or factor_zne <= 0.0),
                "zne_mse_improves": bool(mse_zne < mse_noisy),
            }
        )
    return rows


def _asymmetric_ratio_scan(scale_factors: list[int]) -> list[dict[str, Any]]:
    ratios = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0)
    rows = []
    for ratio in ratios:
        for p_dep in DEFAULT_NOISE_RATES:
            if p_dep * ratio > 1.0:
                continue
            factor_noisy = toy_affine_factor(1, p_dep, p_1q_ratio=ratio)
            factor_zne = toy_richardson_affine_factor(scale_factors, p_dep, p_1q_ratio=ratio)
            rows.append(
                {
                    "p_dep": float(p_dep),
                    "p_1q_ratio": float(ratio),
                    "affine_factor_noisy": factor_noisy,
                    "affine_factor_zne": factor_zne,
                    "argmin_preserved_noisy": bool(factor_noisy > 0.0),
                    "argmin_preserved_zne": bool(factor_zne > 0.0),
                    "zne_mse_improves": bool(
                        toy_uniform_mse_from_factor(factor_zne) < toy_uniform_mse_from_factor(factor_noisy)
                    ),
                }
            )
    return rows


def _write_expression_record(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase E Toy Model Expression Record",
        "",
        "Status: draft_not_frozen",
        "",
        "Model:",
        "",
        "- Graph: two-qubit single-edge MaxCut.",
        "- QAOA depth: q=1.",
        "- Circuit convention: H, RZZ(-gamma), RX(2 beta).",
        "- Objective convention: minimize L = -E[Cut].",
        "- First-order Richardson scale factors: [1, 3].",
        "",
        "Closed-form ideal landscape:",
        "",
        "```text",
        "E0[Cut](gamma,beta) = 1/2 + (1/2) sin(gamma) sin(4 beta)",
        "L0(gamma,beta) = -1/2 - (1/2) sin(gamma) sin(4 beta)",
        "```",
        "",
        "Affine depolarizing control model:",
        "",
        "```text",
        "A_lambda(epsilon,rho) = (1 - rho epsilon)^(4 lambda) (1 - epsilon)^(2 lambda)",
        "L_lambda(gamma,beta) = -1/2 - (A_lambda/2) sin(gamma) sin(4 beta)",
        "L_ZNE = sum_i c_i L_lambda_i",
        "A_ZNE = sum_i c_i A_lambda_i",
        "MSE_uniform(A) = (A - 1)^2 / 16",
        "```",
        "",
        "Negative-control conclusion:",
        "",
        "If A_lambda > 0 and A_ZNE > 0, noisy and ZNE landscapes are positive affine contractions of the same",
        "non-constant component of L0. They may reduce MSE, but they cannot move the argmin.",
        "",
        "Backend validation:",
        "",
        f"- status: {summary['ideal_backend_validation']['status']}",
        f"- max abs diff: {summary['ideal_backend_validation']['max_abs_diff']:.3e}",
        "",
        "Extension status:",
        "",
        summary["extension_path_status"],
    ]
    (output_dir / "phase_e_toy_model_expression_record.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plots(output_dir: Path, grid_size: int, p_dep: float, p_1q_ratio: float, scale_factors: list[int]) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gamma_grid, beta_grid = _grid(grid_size)
    ideal = np.asarray(toy_ideal_objective(gamma_grid, beta_grid), dtype=float)
    noisy = np.asarray(toy_affine_objective(gamma_grid, beta_grid, 1, p_dep, p_1q_ratio=p_1q_ratio), dtype=float)
    zne = np.asarray(toy_zne_affine_objective(gamma_grid, beta_grid, scale_factors, p_dep, p_1q_ratio=p_1q_ratio), dtype=float)
    residual = zne - ideal

    np.savez_compressed(output_dir / "phase_e_toy_model_landscapes.npz", ideal=ideal, noisy=noisy, zne=zne, residual=residual)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2), constrained_layout=True)
    panels = [
        ("Ideal L0", ideal),
        ("Noisy affine", noisy),
        ("Richardson ZNE", zne),
        ("ZNE residual", residual),
    ]
    for ax, (title, data) in zip(axes, panels, strict=True):
        im = ax.imshow(data.T, origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("gamma index")
        ax.set_ylabel("beta index")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    png = output_dir / "phase_e_toy_model_landscapes.png"
    pdf = output_dir / "phase_e_toy_model_landscapes.pdf"
    fig.savefig(png, dpi=180)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": png.relative_to(PROJECT_ROOT).as_posix(), "pdf": pdf.relative_to(PROJECT_ROOT).as_posix()}


def run_phase_e_toy_model(
    root: Path = PROJECT_ROOT,
    grid_size: int = 121,
    p_1q_ratio: float = 1.0,
    overwrite: bool = True,
) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_e" / "toy_model")
    summary_path = output_dir / "phase_e_toy_model_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing toy-model summary: {summary_path}")

    started = time.perf_counter()
    scale_factors = [1, 3]
    gamma_grid, beta_grid = _grid(grid_size)
    ideal = np.asarray(toy_ideal_objective(gamma_grid, beta_grid), dtype=float)
    noisy_for_argmin = np.asarray(toy_affine_objective(gamma_grid, beta_grid, 1, 0.01, p_1q_ratio=p_1q_ratio), dtype=float)
    zne_for_argmin = np.asarray(toy_zne_affine_objective(gamma_grid, beta_grid, scale_factors, 0.01, p_1q_ratio=p_1q_ratio), dtype=float)

    ideal_argmin = _argmin_location(ideal, gamma_grid, beta_grid)
    noisy_argmin = _argmin_location(noisy_for_argmin, gamma_grid, beta_grid)
    zne_argmin = _argmin_location(zne_for_argmin, gamma_grid, beta_grid)
    noise_table = _noise_table(tuple(float(x) for x in DEFAULT_NOISE_RATES), p_1q_ratio, scale_factors)
    asymmetric_scan = _asymmetric_ratio_scan(scale_factors)
    asymmetric_discordant_rows = [
        row for row in asymmetric_scan if not (row["argmin_preserved_noisy"] and row["argmin_preserved_zne"])
    ]

    summary = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "phase_e_toy_model_complete",
        "toy_model_version": PHASE_E_TOY_MODEL_VERSION,
        "scope": "two_qubit_single_edge_depth1_affine_depolarizing_negative_control",
        "not_depth2_outcome_dependent": True,
        "closed_form": {
            "ideal_expected_cut": "1/2 + (1/2) sin(gamma) sin(4 beta)",
            "ideal_objective": "-1/2 - (1/2) sin(gamma) sin(4 beta)",
            "affine_factor": "A_lambda(epsilon,rho) = (1-rho epsilon)^(4 lambda) (1-epsilon)^(2 lambda)",
            "affine_objective": "-1/2 - (A_lambda/2) sin(gamma) sin(4 beta)",
            "uniform_mse": "(A-1)^2/16",
            "negative_control_proposition": (
                "For A_noisy>0 and A_ZNE>0, noisy and Richardson-ZNE landscapes are positive affine "
                "contractions of the ideal non-constant component, so argmin preservation is guaranteed."
            ),
        },
        "ideal_backend_validation": _ideal_backend_validation(),
        "grid_size": int(grid_size),
        "argmin_check_at_p_dep_0p01": {
            "p_dep": 0.01,
            "p_1q_ratio": float(p_1q_ratio),
            "ideal_argmin": ideal_argmin,
            "noisy_argmin": noisy_argmin,
            "zne_argmin": zne_argmin,
            "noisy_argmin_matches_ideal_grid": bool(noisy_argmin["index"] == ideal_argmin["index"]),
            "zne_argmin_matches_ideal_grid": bool(zne_argmin["index"] == ideal_argmin["index"]),
        },
        "noise_table": noise_table,
        "all_default_noise_rows_argmin_preserved": bool(
            all(row["argmin_preserved_noisy"] and row["argmin_preserved_zne"] for row in noise_table)
        ),
        "all_default_noise_rows_zne_mse_improves": bool(all(row["zne_mse_improves"] for row in noise_table)),
        "asymmetric_affine_ratio_scan": asymmetric_scan,
        "asymmetric_affine_discordant_rows": asymmetric_discordant_rows,
        "extension_path_status": (
            "The two-qubit affine depolarizing control, including p_1q/p_2q ratio changes, did not produce "
            "argmin discordance. This is a negative control, not a failure. Positive-existence search should "
            "continue with 3-qubit structure or non-unital amplitude-damping/dephasing before freeze."
        ),
        "plots": _write_plots(output_dir, grid_size=grid_size, p_dep=0.01, p_1q_ratio=p_1q_ratio, scale_factors=scale_factors),
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_expression_record(output_dir, summary)
    write_json_atomic(summary_path, summary, overwrite=overwrite)
    return summary
