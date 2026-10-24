from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .metrics import comparison_metrics


PHASE_E_DEPTH1_ADAPTER_VERSION = "phase_e_depth1_adapter_v0_1"
GAMMA_PERIOD = 2.0 * math.pi
BETA_PERIOD = math.pi


InterpolationMode = Literal["nearest", "bilinear"]


@dataclass(frozen=True)
class Depth1LandscapeBundle:
    ideal: np.ndarray
    noisy: np.ndarray
    zne: np.ndarray
    ideal_path: Path
    noisy_path: Path
    zne_path: Path
    interpolation: InterpolationMode = "nearest"

    @property
    def grid_size(self) -> int:
        if self.ideal.shape != self.noisy.shape or self.ideal.shape != self.zne.shape:
            raise ValueError("Depth-1 landscape arrays must have identical shapes.")
        if len(self.ideal.shape) != 2 or self.ideal.shape[0] != self.ideal.shape[1]:
            raise ValueError("Depth-1 landscape arrays must be square 2D grids.")
        return int(self.ideal.shape[0])

    def evaluate(self, theta: tuple[float, float], objective: Literal["ideal", "noisy", "zne"]) -> float:
        array = {"ideal": self.ideal, "noisy": self.noisy, "zne": self.zne}[objective]
        if self.interpolation == "nearest":
            return nearest_grid_value(array, theta)
        if self.interpolation == "bilinear":
            return bilinear_periodic_value(array, theta)
        raise ValueError(f"Unsupported interpolation mode: {self.interpolation}")


def theta_from_grid_index(i: int, j: int, grid_size: int) -> tuple[float, float]:
    return (GAMMA_PERIOD * int(i) / int(grid_size), BETA_PERIOD * int(j) / int(grid_size))


def _fractional_indices(theta: tuple[float, float], grid_size: int) -> tuple[float, float]:
    gamma, beta = theta
    return (float(gamma % GAMMA_PERIOD) / GAMMA_PERIOD * grid_size, float(beta % BETA_PERIOD) / BETA_PERIOD * grid_size)


def nearest_grid_indices(theta: tuple[float, float], grid_size: int) -> tuple[int, int]:
    gamma_index, beta_index = _fractional_indices(theta, grid_size)
    return (int(round(gamma_index)) % grid_size, int(round(beta_index)) % grid_size)


def nearest_grid_value(array: np.ndarray, theta: tuple[float, float]) -> float:
    i, j = nearest_grid_indices(theta, int(array.shape[0]))
    return float(array[i, j])


def bilinear_periodic_value(array: np.ndarray, theta: tuple[float, float]) -> float:
    size = int(array.shape[0])
    gamma_index, beta_index = _fractional_indices(theta, size)
    i0 = int(math.floor(gamma_index)) % size
    j0 = int(math.floor(beta_index)) % size
    i1 = (i0 + 1) % size
    j1 = (j0 + 1) % size
    di = float(gamma_index - math.floor(gamma_index))
    dj = float(beta_index - math.floor(beta_index))
    return float(
        (1.0 - di) * (1.0 - dj) * array[i0, j0]
        + di * (1.0 - dj) * array[i1, j0]
        + (1.0 - di) * dj * array[i0, j1]
        + di * dj * array[i1, j1]
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _infer_ideal_path(noisy_path: Path) -> Path:
    grid_root = noisy_path
    while grid_root.name and not grid_root.name.startswith("grid_"):
        parent = grid_root.parent
        if parent == grid_root:
            break
        grid_root = parent
    if not grid_root.name.startswith("grid_"):
        raise ValueError(f"Could not infer grid root from noisy path: {noisy_path}")
    return grid_root / "C_ideal.npy"


def load_depth1_bundle_from_comparison_row(
    row: dict[str, str],
    root: Path = PROJECT_ROOT,
    interpolation: InterpolationMode = "nearest",
) -> Depth1LandscapeBundle:
    noisy_path = root / row["noisy_array_path"]
    zne_path = root / row["zne_array_path"]
    ideal_path = _infer_ideal_path(noisy_path)
    return Depth1LandscapeBundle(
        ideal=np.load(ideal_path, allow_pickle=False),
        noisy=np.load(noisy_path, allow_pickle=False),
        zne=np.load(zne_path, allow_pickle=False),
        ideal_path=ideal_path,
        noisy_path=noisy_path,
        zne_path=zne_path,
        interpolation=interpolation,
    )


def _sample_global_comparison_rows(root: Path, sample_rows: int) -> list[dict[str, str]]:
    path = root / "results" / "phase_d" / "global_D1_D2_D3_D4" / "comparisons_per_seed.csv"
    rows = _read_csv(path)
    if not rows:
        raise RuntimeError(f"No Phase D comparison rows found: {path}")
    if sample_rows <= 0 or sample_rows >= len(rows):
        return rows

    selected_indices = np.linspace(0, len(rows) - 1, int(sample_rows), dtype=int)
    selected = [rows[int(index)] for index in selected_indices]

    # Ensure all Richardson orders are represented even when sample_rows is small.
    by_order: dict[str, dict[str, str]] = {}
    for row in rows:
        by_order.setdefault(str(row["zne_order"]), row)
    seen = {str(row["zne_order"]) for row in selected}
    for order, row in sorted(by_order.items()):
        if order not in seen:
            selected.append(row)
    return selected


def _compare_float(name: str, actual: float, expected: float, tolerance: float) -> dict[str, Any]:
    diff = abs(float(actual) - float(expected))
    return {
        "name": name,
        "actual": float(actual),
        "expected": float(expected),
        "abs_diff": diff,
        "tolerance": float(tolerance),
        "passed": bool(diff <= tolerance),
    }


def run_phase_e_depth1_regression(
    root: Path = PROJECT_ROOT,
    sample_rows: int = 18,
    overwrite: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = _sample_global_comparison_rows(root, sample_rows)
    regression_rows: list[dict[str, Any]] = []
    metric_checks: list[dict[str, Any]] = []
    grid_checks: list[dict[str, Any]] = []
    hash_checks: list[dict[str, Any]] = []

    fixed_indices = ((0, 0), (7, 11), (61, 43), (119, 118))
    for row_index, row in enumerate(rows):
        bundle = load_depth1_bundle_from_comparison_row(row, root=root, interpolation="nearest")
        metrics = comparison_metrics(bundle.noisy, bundle.zne, bundle.ideal)
        metric_checks.extend(
            [
                _compare_float("noisy_mse", metrics["noisy_mse"], float(row["noisy_mse"]), 1e-12),
                _compare_float("zne_mse", metrics["zne_mse"], float(row["zne_mse"]), 1e-12),
                _compare_float("r_mse", metrics["r_mse"], float(row["r_mse"]), 1e-10),
                _compare_float("noisy_norm_regret", metrics["noisy_norm_regret"], float(row["noisy_norm_regret"]), 1e-12),
                _compare_float("zne_norm_regret", metrics["zne_norm_regret"], float(row["zne_norm_regret"]), 1e-12),
                _compare_float("d_R", metrics["d_R"], float(row["d_R"]), 1e-12),
            ]
        )

        hash_checks.append(
            {
                "row_index": row_index,
                "instance": row["instance"],
                "seed": row["seed"],
                "zne_order": row["zne_order"],
                "noisy_hash_matches": sha256_file(bundle.noisy_path) == row["noisy_array_sha256"],
                "zne_hash_matches": sha256_file(bundle.zne_path) == row["zne_array_sha256"],
            }
        )

        for i, j in fixed_indices:
            theta = theta_from_grid_index(i, j, bundle.grid_size)
            nearest_indices = nearest_grid_indices(theta, bundle.grid_size)
            grid_checks.append(
                {
                    "row_index": row_index,
                    "instance": row["instance"],
                    "seed": row["seed"],
                    "zne_order": row["zne_order"],
                    "grid_index": [i, j],
                    "nearest_indices": list(nearest_indices),
                    "ideal_exact_match": bundle.evaluate(theta, "ideal") == float(bundle.ideal[i, j]),
                    "noisy_exact_match": bundle.evaluate(theta, "noisy") == float(bundle.noisy[i, j]),
                    "zne_exact_match": bundle.evaluate(theta, "zne") == float(bundle.zne[i, j]),
                }
            )

        regression_rows.append(
            {
                "row_index": row_index,
                "block_id": row["block_id"],
                "instance": row["instance"],
                "noise": row["noise"],
                "shots_total": row["shots_total"],
                "seed": row["seed"],
                "zne_order": row["zne_order"],
                "grid_size": bundle.grid_size,
                "ideal_path": bundle.ideal_path.relative_to(root).as_posix(),
                "noisy_path": bundle.noisy_path.relative_to(root).as_posix(),
                "zne_path": bundle.zne_path.relative_to(root).as_posix(),
            }
        )

    failed_metric_checks = [check for check in metric_checks if not check["passed"]]
    failed_hash_checks = [check for check in hash_checks if not (check["noisy_hash_matches"] and check["zne_hash_matches"])]
    failed_grid_checks = [
        check
        for check in grid_checks
        if not (check["ideal_exact_match"] and check["noisy_exact_match"] and check["zne_exact_match"])
    ]
    status = "passed" if not failed_metric_checks and not failed_hash_checks and not failed_grid_checks else "failed"

    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": status,
        "adapter_version": PHASE_E_DEPTH1_ADAPTER_VERSION,
        "scope": "read_only_regression_against_frozen_phase_d_global_comparisons",
        "sampled_rows": len(rows),
        "metric_checks": len(metric_checks),
        "failed_metric_checks": failed_metric_checks,
        "hash_checks": hash_checks,
        "failed_hash_checks": failed_hash_checks,
        "grid_exact_checks": len(grid_checks),
        "failed_grid_exact_checks": failed_grid_checks,
        "interpolation_default": "nearest",
        "regression_rows": regression_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }

    output_dir = ensure_dir(root / "results" / "phase_e" / "planning")
    write_json_atomic(output_dir / "phase_e_depth1_regression_summary.json", report, overwrite=overwrite)
    return report
