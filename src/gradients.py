from __future__ import annotations

import numpy as np


def boundary_continuity_ok(values: np.ndarray, axis: int, tolerance_fraction: float) -> bool:
    data_range = float(np.max(values) - np.min(values))
    if data_range == 0:
        return True
    first = np.take(values, 0, axis=axis)
    last = np.take(values, -1, axis=axis)
    return float(np.mean(np.abs(first - last))) <= tolerance_fraction * data_range


def central_gradients(values: np.ndarray, d_gamma: float, d_beta: float, periodic_gamma: bool, periodic_beta: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grad_gamma = np.full(values.shape, np.nan, dtype=float)
    grad_beta = np.full(values.shape, np.nan, dtype=float)
    valid = np.ones(values.shape, dtype=bool)

    if periodic_gamma:
        grad_gamma = (np.roll(values, -1, axis=0) - np.roll(values, 1, axis=0)) / (2.0 * d_gamma)
    else:
        grad_gamma[1:-1, :] = (values[2:, :] - values[:-2, :]) / (2.0 * d_gamma)
        valid[[0, -1], :] = False

    if periodic_beta:
        grad_beta = (np.roll(values, -1, axis=1) - np.roll(values, 1, axis=1)) / (2.0 * d_beta)
    else:
        grad_beta[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2.0 * d_beta)
        valid[:, [0, -1]] = False

    return grad_gamma, grad_beta, valid


def gradient_alignment(
    ideal: np.ndarray,
    observed: np.ndarray,
    d_gamma: float,
    d_beta: float,
    boundary_tolerance_fraction: float,
    min_ideal_norm: float,
    denominator_epsilon: float,
) -> tuple[float, dict[str, bool]]:
    periodic_gamma = boundary_continuity_ok(ideal, axis=0, tolerance_fraction=boundary_tolerance_fraction)
    periodic_beta = boundary_continuity_ok(ideal, axis=1, tolerance_fraction=boundary_tolerance_fraction)
    gi_g, gi_b, valid_i = central_gradients(ideal, d_gamma, d_beta, periodic_gamma, periodic_beta)
    gm_g, gm_b, valid_m = central_gradients(observed, d_gamma, d_beta, periodic_gamma, periodic_beta)
    valid = valid_i & valid_m & np.isfinite(gi_g) & np.isfinite(gi_b) & np.isfinite(gm_g) & np.isfinite(gm_b)
    ideal_norm = np.sqrt(gi_g**2 + gi_b**2)
    observed_norm = np.sqrt(gm_g**2 + gm_b**2)
    valid &= ideal_norm >= min_ideal_norm
    if not np.any(valid):
        return float("nan"), {"periodic_gamma": periodic_gamma, "periodic_beta": periodic_beta}
    numerator = gi_g * gm_g + gi_b * gm_b
    alignment = numerator / (ideal_norm * observed_norm + denominator_epsilon)
    return float(np.nanmean(alignment[valid])), {"periodic_gamma": periodic_gamma, "periodic_beta": periodic_beta}
