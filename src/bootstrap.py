from __future__ import annotations

import numpy as np


def percentile_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - confidence
    return (float(np.quantile(values, alpha / 2.0)), float(np.quantile(values, 1.0 - alpha / 2.0)))


def bootstrap_median_ci(
    values: np.ndarray,
    resamples: int,
    seed: int,
    sample_size: int | None = None,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty finite sample.")
    if sample_size is None:
        sample_size = int(values.size)
    if sample_size <= 0:
        raise ValueError("Bootstrap sample_size must be positive.")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(int(resamples), int(sample_size)))
    medians = np.median(values[indices], axis=1)
    ci_low, ci_high = percentile_ci(medians, confidence=confidence)
    return {
        "n_observed": int(values.size),
        "sample_size": int(sample_size),
        "resamples": int(resamples),
        "median_observed": float(np.median(values)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_width": float(ci_high - ci_low),
    }
