from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .circuits import graph_from_config
from .gradients import central_gradients
from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, save_array_atomic, sha256_file, write_json_atomic
from .landscapes import generate_ideal_landscape


DEFAULT_REFERENCE_GRID = 240
DEFAULT_CANDIDATE_GRIDS = (40, 60, 80, 120)

STRICT_REGRET_TAU_FRACTION = 0.25
STRICT_EPSILON_TIE_FRACTION = 0.001
STRICT_GOOD_REGION_JACCARD_MIN = 0.90
STRICT_SPEARMAN_MIN = 0.995
STRICT_GRADIENT_ALIGNMENT_MIN = 0.99


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _array_path(root: Path, instance: str, grid_size: int) -> Path:
    return root / "data" / "raw" / "grid_convergence_ideal" / instance / f"C_ideal_grid_{grid_size}.npy"


def _metadata_path(array_path: Path) -> Path:
    return array_path.with_suffix(".json")


def _load_or_generate_ideal(
    root: Path,
    cfg: dict[str, Any],
    instance: str,
    grid_size: int,
    overwrite: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    array_path = _array_path(root, instance, grid_size)
    metadata_path = _metadata_path(array_path)
    if array_path.exists() and metadata_path.exists() and not overwrite:
        arr = np.load(array_path, allow_pickle=False)
        metadata = {
            "path": array_path.relative_to(root).as_posix(),
            "loaded_existing": True,
            "sha256": sha256_file(array_path),
            "grid_size": grid_size,
            "shape": list(arr.shape),
        }
        return arr, metadata

    if array_path.exists() and not overwrite:
        raise RuntimeError(f"Existing ideal grid array is missing metadata: {array_path}")
    if metadata_path.exists() and not overwrite:
        raise RuntimeError(f"Existing ideal grid metadata is missing array: {metadata_path}")

    graph = graph_from_config(instance, cfg)
    started = time.perf_counter()
    arr = generate_ideal_landscape(graph, grid_size)
    elapsed = time.perf_counter() - started
    save_array_atomic(array_path, arr, overwrite=overwrite)
    metadata = {
        "path": array_path.relative_to(root).as_posix(),
        "loaded_existing": False,
        "sha256": sha256_file(array_path),
        "grid_size": grid_size,
        "shape": list(arr.shape),
        "elapsed_seconds": elapsed,
        "array_min": float(np.min(arr)),
        "array_max": float(np.max(arr)),
        "array_range": float(np.max(arr) - np.min(arr)),
    }
    write_json_atomic(metadata_path, metadata, overwrite=True)
    return arr, metadata


def _good_region_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    k = max(1, int(round(values.size * fraction)))
    idx = np.argpartition(values.ravel(), k - 1)[:k]
    mask = np.zeros(values.size, dtype=bool)
    mask[idx] = True
    return mask.reshape(values.shape)


def _jaccard(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.logical_or(mask_a, mask_b)
    if not bool(np.any(union)):
        return float("nan")
    return float(np.logical_and(mask_a, mask_b).sum() / union.sum())


def _angular_coordinates(index: tuple[int, int], grid_size: int) -> tuple[float, float]:
    i, j = index
    return (2.0 * math.pi * i / grid_size, math.pi * j / grid_size)


def _angular_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    d_gamma = abs(a[0] - b[0])
    d_gamma = min(d_gamma, 2.0 * math.pi - d_gamma)
    d_beta = abs(a[1] - b[1])
    return float(math.hypot(d_gamma, d_beta))


def _gradient_alignment_nested(candidate: np.ndarray, reference: np.ndarray, step: int) -> float:
    c_size = candidate.shape[0]
    r_size = reference.shape[0]
    c_d_gamma = 2.0 * math.pi / c_size
    c_d_beta = math.pi / c_size
    r_d_gamma = 2.0 * math.pi / r_size
    r_d_beta = math.pi / r_size
    c_g, c_b, c_valid = central_gradients(candidate, c_d_gamma, c_d_beta, periodic_gamma=True, periodic_beta=False)
    r_g, r_b, r_valid = central_gradients(reference, r_d_gamma, r_d_beta, periodic_gamma=True, periodic_beta=False)
    r_g = r_g[::step, ::step]
    r_b = r_b[::step, ::step]
    r_valid = r_valid[::step, ::step]
    c_norm = np.sqrt(c_g**2 + c_b**2)
    r_norm = np.sqrt(r_g**2 + r_b**2)
    valid = c_valid & r_valid & (c_norm > 1e-10) & (r_norm > 1e-10)
    if not bool(np.any(valid)):
        return float("nan")
    cos = (c_g[valid] * r_g[valid] + c_b[valid] * r_b[valid]) / (c_norm[valid] * r_norm[valid])
    return float(np.mean(cos))


def _local_minima_mask(values: np.ndarray, max_rel_height: float) -> np.ndarray:
    threshold = float(np.min(values) + max_rel_height * (np.max(values) - np.min(values)))
    minima = np.zeros(values.shape, dtype=bool)
    rows, cols = values.shape
    for i in range(rows):
        for j in range(cols):
            value = values[i, j]
            if value > threshold:
                continue
            row_neighbors = [(i - 1) % rows, i, (i + 1) % rows]
            col_neighbors = [j]
            if j > 0:
                col_neighbors.append(j - 1)
            if j < cols - 1:
                col_neighbors.append(j + 1)
            neighborhood = values[np.ix_(row_neighbors, col_neighbors)]
            minima[i, j] = bool(value <= np.min(neighborhood))
    return minima


def _compare_candidate_to_reference(
    instance: str,
    candidate_grid: int,
    candidate: np.ndarray,
    reference_grid: int,
    reference: np.ndarray,
    tau_r: float,
    good_region_fraction: float,
) -> dict[str, object]:
    if reference_grid % candidate_grid != 0:
        raise ValueError(f"reference_grid={reference_grid} is not divisible by candidate_grid={candidate_grid}")
    step = reference_grid // candidate_grid
    reference_on_candidate = reference[::step, ::step]
    nested_max_abs_diff = float(np.max(np.abs(candidate - reference_on_candidate)))
    reference_range = float(np.max(reference) - np.min(reference))
    reference_min = float(np.min(reference))
    candidate_argmin = tuple(int(x) for x in np.unravel_index(int(np.argmin(candidate)), candidate.shape))
    reference_argmin = tuple(int(x) for x in np.unravel_index(int(np.argmin(reference)), reference.shape))
    mapped_argmin = (candidate_argmin[0] * step, candidate_argmin[1] * step)
    reference_value_at_candidate_argmin = float(reference[mapped_argmin])
    reference_regret_abs = float(reference_value_at_candidate_argmin - reference_min)
    reference_regret = float(reference_regret_abs / reference_range)

    upsampled = np.repeat(np.repeat(candidate, step, axis=0), step, axis=1)
    mse_over_range2 = float(np.mean((upsampled - reference) ** 2) / (reference_range**2))
    spearman = spearmanr(reference.ravel(), upsampled.ravel()).correlation
    spearman = float(spearman) if spearman is not None and math.isfinite(float(spearman)) else float("nan")
    j10 = _jaccard(_good_region_mask(upsampled, good_region_fraction), _good_region_mask(reference, good_region_fraction))
    grad_align = _gradient_alignment_nested(candidate, reference, step)

    reference_good10 = _good_region_mask(reference, good_region_fraction)
    candidate_argmin_in_reference_good10 = bool(reference_good10[mapped_argmin])

    tau_half = tau_r / 2.0
    tau_quarter = STRICT_REGRET_TAU_FRACTION * tau_r
    epsilon_tie_abs = STRICT_EPSILON_TIE_FRACTION * reference_range
    important_height = tau_half
    candidate_minima = _local_minima_mask(candidate, important_height)
    reference_minima = _local_minima_mask(reference, important_height)
    candidate_minima_count = int(candidate_minima.sum())
    reference_minima_count = int(reference_minima.sum())

    candidate_coords = _angular_coordinates(candidate_argmin, candidate_grid)
    reference_coords = _angular_coordinates(reference_argmin, reference_grid)
    angular_distance = _angular_distance(candidate_coords, reference_coords)
    cell_diagonal = math.hypot(2.0 * math.pi / candidate_grid, math.pi / candidate_grid)

    pragmatic_passes = {
        "nested_alignment": nested_max_abs_diff <= 1e-10,
        "reference_regret_le_tau_half": reference_regret <= tau_half,
        "argmin_in_reference_good10": candidate_argmin_in_reference_good10,
        "spearman_ge_0_98": spearman >= 0.98,
        "j10_ge_0_85": j10 >= 0.85,
        "gradient_alignment_ge_0_95": grad_align >= 0.95,
        "argmin_within_two_cell_diagonals": angular_distance <= 2.0 * cell_diagonal,
    }
    strict_passes = {
        "nested_alignment": nested_max_abs_diff <= 1e-10,
        "reference_regret_lt_tau_quarter": reference_regret < tau_quarter,
        "argmin_within_epsilon_tie": reference_regret_abs <= epsilon_tie_abs,
        "j10_ge_0_90": j10 >= STRICT_GOOD_REGION_JACCARD_MIN,
        "spearman_ge_0_995": spearman >= STRICT_SPEARMAN_MIN,
        "gradient_alignment_ge_0_99": grad_align >= STRICT_GRADIENT_ALIGNMENT_MIN,
    }

    return {
        "instance": instance,
        "candidate_grid": candidate_grid,
        "reference_grid": reference_grid,
        "step": step,
        "nested_max_abs_diff": nested_max_abs_diff,
        "reference_regret_abs_at_candidate_argmin": reference_regret_abs,
        "reference_regret_at_candidate_argmin": reference_regret,
        "tau_R": tau_r,
        "tau_R_half": tau_half,
        "tau_R_quarter": tau_quarter,
        "epsilon_tie_fraction": STRICT_EPSILON_TIE_FRACTION,
        "epsilon_tie_abs": epsilon_tie_abs,
        "candidate_argmin_i": candidate_argmin[0],
        "candidate_argmin_j": candidate_argmin[1],
        "reference_argmin_i": reference_argmin[0],
        "reference_argmin_j": reference_argmin[1],
        "candidate_argmin_gamma": candidate_coords[0],
        "candidate_argmin_beta": candidate_coords[1],
        "reference_argmin_gamma": reference_coords[0],
        "reference_argmin_beta": reference_coords[1],
        "argmin_angular_distance": angular_distance,
        "candidate_cell_diagonal": cell_diagonal,
        "candidate_argmin_in_reference_good10": candidate_argmin_in_reference_good10,
        "upsampled_mse_over_range2": mse_over_range2,
        "upsampled_spearman": spearman,
        "upsampled_good_region_jaccard": j10,
        "nested_gradient_alignment": grad_align,
        "candidate_important_local_minima": candidate_minima_count,
        "reference_important_local_minima": reference_minima_count,
        "passes_pragmatic_acceptance_checks": all(pragmatic_passes.values()),
        "passes_strict_acceptance_checks": all(strict_passes.values()),
        "passes_all_acceptance_checks": all(strict_passes.values()),
        **{f"pass_pragmatic_{key}": value for key, value in pragmatic_passes.items()},
        **{f"pass_strict_{key}": value for key, value in strict_passes.items()},
    }


def run_grid_convergence(
    config_path: str | Path | None = None,
    reference_grid: int = DEFAULT_REFERENCE_GRID,
    candidate_grids: tuple[int, ...] = DEFAULT_CANDIDATE_GRIDS,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = PROJECT_ROOT
    cfg = read_yaml(config_path or root / "config" / "config_official_pilot_v1.6.yaml")
    phase_c = cfg["phase_c"]
    instances = tuple(str(value) for value in phase_c["instances"])
    tau_r = float(phase_c["tau_R"])
    good_region_fraction = float(cfg["metrics"]["good_region_fraction"])
    if any(reference_grid % candidate != 0 for candidate in candidate_grids):
        raise ValueError("reference_grid must be divisible by every candidate grid.")

    result_dir = root / "results" / "grid_convergence"
    ensure_dir(result_dir)
    metadata: list[dict[str, Any]] = []
    rows: list[dict[str, object]] = []
    recommendation_by_instance: dict[str, str] = {}
    for instance in instances:
        arrays: dict[int, np.ndarray] = {}
        for grid_size in sorted(set(candidate_grids + (reference_grid,))):
            arrays[grid_size], item = _load_or_generate_ideal(root, cfg, instance, grid_size, overwrite=overwrite)
            metadata.append({"instance": instance, **item})
        reference = arrays[reference_grid]
        instance_rows = []
        for candidate in candidate_grids:
            row = _compare_candidate_to_reference(
                instance,
                candidate,
                arrays[candidate],
                reference_grid,
                reference,
                tau_r,
                good_region_fraction,
            )
            rows.append(row)
            instance_rows.append(row)
        passing = [row for row in instance_rows if bool(row["passes_strict_acceptance_checks"])]
        recommendation_by_instance[instance] = f"{min(int(row['candidate_grid']) for row in passing)}x{min(int(row['candidate_grid']) for row in passing)}" if passing else "80x80_or_higher"

    pass_by_grid: dict[int, bool] = {}
    pragmatic_pass_by_grid: dict[int, bool] = {}
    for candidate in candidate_grids:
        pass_by_grid[candidate] = all(bool(row["passes_strict_acceptance_checks"]) for row in rows if int(row["candidate_grid"]) == candidate)
        pragmatic_pass_by_grid[candidate] = all(bool(row["passes_pragmatic_acceptance_checks"]) for row in rows if int(row["candidate_grid"]) == candidate)
    selected_grid = None
    for candidate in sorted(candidate_grids):
        if pass_by_grid[candidate]:
            selected_grid = candidate
            break
    if selected_grid is None:
        selected_grid = max(candidate_grids)

    report = {
        "status": "grid_convergence_complete",
        "reference_grid": reference_grid,
        "candidate_grids": list(candidate_grids),
        "instances": list(instances),
        "tau_R": tau_r,
        "good_region_fraction": good_region_fraction,
        "acceptance_thresholds": {
            "reference_regret_at_candidate_argmin_max": "0.25 * tau_R",
            "argmin_tie_max": "0.001 * range(C_I_ref)",
            "upsampled_good_region_jaccard_min": STRICT_GOOD_REGION_JACCARD_MIN,
            "upsampled_spearman_min": STRICT_SPEARMAN_MIN,
            "nested_gradient_alignment_min": STRICT_GRADIENT_ALIGNMENT_MIN,
            "nested_max_abs_diff_max": 1e-10,
        },
        "pragmatic_acceptance_thresholds_for_context": {
            "reference_regret_at_candidate_argmin_max": "tau_R / 2",
            "upsampled_spearman_min": 0.98,
            "upsampled_good_region_jaccard_min": 0.85,
            "nested_gradient_alignment_min": 0.95,
            "nested_max_abs_diff_max": 1e-10,
            "argmin_distance_max": "2 candidate cell diagonals",
        },
        "array_metadata": metadata,
        "rows": rows,
        "pass_by_grid": {str(key): value for key, value in pass_by_grid.items()},
        "strict_pass_by_grid": {str(key): value for key, value in pass_by_grid.items()},
        "pragmatic_pass_by_grid": {str(key): value for key, value in pragmatic_pass_by_grid.items()},
        "recommendation_by_instance": recommendation_by_instance,
        "selected_grid_by_strict_rule": selected_grid,
        "selected_fallback_grid_if_needed": selected_grid,
        "decision_rule": (
            "Select the smallest candidate grid that passes all strict ideal-only grid adequacy checks for every "
            "Phase C instance. This is a grid adequacy result, not a runtime feasibility result."
        ),
    }
    write_json_atomic(result_dir / "grid_convergence_report.json", report, overwrite=True)
    _write_csv(result_dir / "grid_convergence_table.csv", rows)
    return report
