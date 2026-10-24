from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr

from .config import MetricsConfig
from .gradients import gradient_alignment


def mse(observed: np.ndarray, ideal: np.ndarray) -> float:
    return float(np.mean((observed - ideal) ** 2))


def normalized_ideal_regret(observed: np.ndarray, ideal: np.ndarray) -> float:
    data_range = float(np.max(ideal) - np.min(ideal))
    if data_range == 0:
        return float("nan")
    theta_m = np.unravel_index(int(np.argmin(observed)), observed.shape)
    theta_i = np.unravel_index(int(np.argmin(ideal)), ideal.shape)
    return float((ideal[theta_m] - ideal[theta_i]) / data_range)


def argmin_preserved(observed: np.ndarray, ideal: np.ndarray, epsilon_fraction: float) -> bool:
    data_range = float(np.max(ideal) - np.min(ideal))
    eps = epsilon_fraction * data_range
    theta_m = np.unravel_index(int(np.argmin(observed)), observed.shape)
    return bool(ideal[theta_m] - np.min(ideal) <= eps)


def spearman_corr(observed: np.ndarray, ideal: np.ndarray) -> float:
    rho = spearmanr(ideal.ravel(), observed.ravel()).correlation
    return float(rho) if rho is not None and math.isfinite(float(rho)) else float("nan")


def good_region_overlap(observed: np.ndarray, ideal: np.ndarray, fraction: float) -> float:
    n = ideal.size
    k = max(1, int(round(n * fraction)))
    ideal_idx = set(np.argpartition(ideal.ravel(), k - 1)[:k].tolist())
    observed_idx = set(np.argpartition(observed.ravel(), k - 1)[:k].tolist())
    union = ideal_idx | observed_idx
    if not union:
        return float("nan")
    return float(len(ideal_idx & observed_idx) / len(union))


def oob_fraction(observed: np.ndarray, ideal: np.ndarray, delta_fraction: float, breakdown_threshold: float) -> tuple[float, bool]:
    c_min = float(np.min(ideal))
    c_max = float(np.max(ideal))
    delta = delta_fraction * (c_max - c_min)
    mask = (observed < c_min - delta) | (observed > c_max + delta)
    frac = float(np.mean(mask))
    return frac, bool(frac > breakdown_threshold)


def below_detection_floor(noisy: np.ndarray, ideal: np.ndarray, threshold: float) -> bool:
    data_range = float(np.max(ideal) - np.min(ideal))
    if data_range == 0:
        return False
    return bool(mse(noisy, ideal) / (data_range**2) < threshold)


def method_metrics(
    observed: np.ndarray,
    ideal: np.ndarray,
    metrics_config: MetricsConfig,
    noisy_reference: np.ndarray | None = None,
) -> dict[str, object]:
    size = ideal.shape[0]
    d_gamma = 2.0 * math.pi / size
    d_beta = math.pi / size
    f_oob, breakdown = oob_fraction(
        observed,
        ideal,
        delta_fraction=metrics_config.oob_delta_fraction,
        breakdown_threshold=metrics_config.oob_breakdown_threshold,
    )
    grad_align, grad_meta = gradient_alignment(
        ideal,
        observed,
        d_gamma,
        d_beta,
        boundary_tolerance_fraction=metrics_config.gradient_boundary_tolerance_fraction,
        min_ideal_norm=metrics_config.gradient_min_ideal_norm,
        denominator_epsilon=metrics_config.gradient_denominator_epsilon,
    )
    out = {
        "mse": mse(observed, ideal),
        "norm_regret": normalized_ideal_regret(observed, ideal),
        "argmin_preserved": argmin_preserved(observed, ideal, metrics_config.epsilon_tie_fraction),
        "spearman": spearman_corr(observed, ideal),
        "good_region_jaccard": good_region_overlap(observed, ideal, metrics_config.good_region_fraction),
        "grad_align": grad_align,
        "f_oob": f_oob,
        "extrapolation_breakdown": breakdown,
        "gradient_periodic_gamma": grad_meta["periodic_gamma"],
        "gradient_periodic_beta": grad_meta["periodic_beta"],
    }
    if noisy_reference is not None:
        out["below_detection_floor"] = below_detection_floor(
            noisy_reference,
            ideal,
            threshold=metrics_config.below_detection_floor_threshold,
        )
    return out


def comparison_metrics(noisy: np.ndarray, zne: np.ndarray, ideal: np.ndarray) -> dict[str, float]:
    noisy_mse = mse(noisy, ideal)
    zne_mse = mse(zne, ideal)
    noisy_regret = normalized_ideal_regret(noisy, ideal)
    zne_regret = normalized_ideal_regret(zne, ideal)
    return {
        "noisy_mse": noisy_mse,
        "zne_mse": zne_mse,
        "r_mse": float((noisy_mse - zne_mse) / noisy_mse) if noisy_mse != 0 else float("nan"),
        "noisy_norm_regret": noisy_regret,
        "zne_norm_regret": zne_regret,
        "d_R": float(zne_regret - noisy_regret),
    }
