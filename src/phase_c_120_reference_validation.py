from __future__ import annotations

import csv
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .circuits import graph_from_config, ideal_expected_cut
from .grid_convergence import _array_path, _load_or_generate_ideal
from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, sha256_file, write_json_atomic
from .phase_c_120_preflight import PROTOCOL_VERSION


REFERENCE_VALIDATION_VERSION = "phase_c_120_reference_validation_v1"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_c_120_reference_validation"
REFERENCE_GRID = 240
LOCALIZED_GRID = 480
WINDOW_240_CELLS = 8
EPSILON_TIE_FRACTION = 0.001
REGRET_TAU_FRACTION = 0.25


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _argmin(values: np.ndarray) -> tuple[int, int]:
    return tuple(int(x) for x in np.unravel_index(int(np.argmin(values)), values.shape))


def _basin_mask(values: np.ndarray, epsilon_tie_abs: float) -> np.ndarray:
    rows, cols = values.shape
    start = _argmin(values)
    threshold = float(np.min(values) + epsilon_tie_abs)
    mask = np.zeros(values.shape, dtype=bool)
    queue: deque[tuple[int, int]] = deque([start])
    while queue:
        i, j = queue.popleft()
        if mask[i, j] or values[i, j] > threshold:
            continue
        mask[i, j] = True
        neighbors = [((i - 1) % rows, j), ((i + 1) % rows, j)]
        if j > 0:
            neighbors.append((i, j - 1))
        if j < cols - 1:
            neighbors.append((i, j + 1))
        queue.extend(neighbors)
    return mask


def _localized_480_indices(center_240: tuple[int, int]) -> list[tuple[int, int]]:
    center_i, center_j = center_240
    out: list[tuple[int, int]] = []
    half_window_480 = 2 * WINDOW_240_CELLS
    for di in range(-half_window_480, half_window_480 + 1):
        i480 = (2 * center_i + di) % LOCALIZED_GRID
        for dj in range(-half_window_480, half_window_480 + 1):
            j480 = 2 * center_j + dj
            if 0 <= j480 < LOCALIZED_GRID:
                out.append((i480, j480))
    return out


def _map_480_to_240(index_480: tuple[int, int]) -> tuple[int, int]:
    i480, j480 = index_480
    i240 = int(round(i480 / 2.0)) % REFERENCE_GRID
    j240 = min(REFERENCE_GRID - 1, max(0, int(round(j480 / 2.0))))
    return i240, j240


def _coords(index: tuple[int, int], grid: int) -> tuple[float, float]:
    return 2.0 * math.pi * index[0] / grid, math.pi * index[1] / grid


def _validate_instance(root: Path, cfg: dict[str, Any], instance: str, overwrite: bool) -> dict[str, Any]:
    tau_r = float(cfg["phase_c"]["tau_R"])
    reference, ref_meta = _load_or_generate_ideal(root, cfg, instance, REFERENCE_GRID, overwrite=False)
    graph = graph_from_config(instance, cfg)
    reference_min = float(np.min(reference))
    reference_range = float(np.max(reference) - np.min(reference))
    epsilon_tie_abs = EPSILON_TIE_FRACTION * reference_range
    tau_quarter = REGRET_TAU_FRACTION * tau_r
    reference_argmin = _argmin(reference)
    basin = _basin_mask(reference, epsilon_tie_abs)
    basin_size = int(np.sum(basin))

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for i480, j480 in _localized_480_indices(reference_argmin):
        gamma, beta = _coords((i480, j480), LOCALIZED_GRID)
        value = -ideal_expected_cut(graph, gamma, beta)
        mapped = _map_480_to_240((i480, j480))
        rows.append({
            "instance": instance,
            "i480": i480,
            "j480": j480,
            "gamma": gamma,
            "beta": beta,
            "ideal_value_480": value,
            "mapped_i240": mapped[0],
            "mapped_j240": mapped[1],
            "mapped_value_240": float(reference[mapped]),
            "mapped_in_240_basin": bool(basin[mapped]),
            "mapped_epsilon_tied_240": bool(reference[mapped] <= reference_min + epsilon_tie_abs),
        })
    elapsed = time.perf_counter() - started
    local_min_row = min(rows, key=lambda row: float(row["ideal_value_480"]))
    mapped = (int(local_min_row["mapped_i240"]), int(local_min_row["mapped_j240"]))
    mapped_reference_value = float(reference[mapped])
    normalized_regret_shift = float((mapped_reference_value - reference_min) / reference_range) if reference_range else math.nan
    optimum_energy_shift_abs = abs(float(local_min_row["ideal_value_480"]) - reference_min)
    pass_basin = bool(basin[mapped] or mapped_reference_value <= reference_min + epsilon_tie_abs)
    pass_regret = bool(normalized_regret_shift <= tau_quarter)
    pass_energy = bool(optimum_energy_shift_abs <= epsilon_tie_abs)

    result_dir = ensure_dir(root / "results" / "phase_c_120_reference_validation")
    csv_path = result_dir / f"{instance}_localized_480.csv"
    _write_csv(csv_path, rows)
    return {
        "instance": instance,
        "status": "pass" if pass_basin and pass_regret and pass_energy else "fail",
        "reference_grid": REFERENCE_GRID,
        "localized_grid": LOCALIZED_GRID,
        "localized_points": len(rows),
        "window_240_cells": WINDOW_240_CELLS,
        "reference_argmin_i": reference_argmin[0],
        "reference_argmin_j": reference_argmin[1],
        "reference_argmin_gamma": _coords(reference_argmin, REFERENCE_GRID)[0],
        "reference_argmin_beta": _coords(reference_argmin, REFERENCE_GRID)[1],
        "reference_min": reference_min,
        "reference_range": reference_range,
        "epsilon_tie_abs": epsilon_tie_abs,
        "epsilon_tie_fraction": EPSILON_TIE_FRACTION,
        "basin_size": basin_size,
        "local_480_argmin_i": int(local_min_row["i480"]),
        "local_480_argmin_j": int(local_min_row["j480"]),
        "local_480_argmin_gamma": float(local_min_row["gamma"]),
        "local_480_argmin_beta": float(local_min_row["beta"]),
        "local_480_min": float(local_min_row["ideal_value_480"]),
        "mapped_i240": mapped[0],
        "mapped_j240": mapped[1],
        "mapped_reference_value": mapped_reference_value,
        "mapped_in_240_basin": bool(basin[mapped]),
        "mapped_epsilon_tied_240": bool(mapped_reference_value <= reference_min + epsilon_tie_abs),
        "normalized_regret_shift": normalized_regret_shift,
        "tau_R": tau_r,
        "tau_R_quarter": tau_quarter,
        "optimum_energy_shift_abs": optimum_energy_shift_abs,
        "pass_basin_or_epsilon_tie": pass_basin,
        "pass_regret_shift": pass_regret,
        "pass_energy_shift": pass_energy,
        "elapsed_seconds": elapsed,
        "reference_array": ref_meta["path"],
        "reference_array_sha256": sha256_file(_array_path(root, instance, REFERENCE_GRID)),
        "localized_csv": csv_path.relative_to(root).as_posix(),
        "localized_csv_sha256": sha256_file(csv_path),
    }


def run_phase_c_120_reference_validation(overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    result_dir = ensure_dir(root / "results" / "phase_c_120_reference_validation")
    summary_path = result_dir / "reference_validation_report.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing reference validation report: {summary_path}")

    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    instances = [str(instance) for instance in cfg["phase_c"]["instances"]]
    started = time.perf_counter()
    rows = [_validate_instance(root, cfg, instance, overwrite=overwrite) for instance in instances]
    status = "phase_c_120_reference_validation_passed" if all(row["status"] == "pass" for row in rows) else "phase_c_120_reference_validation_failed"
    report = {
        "created_at_utc": _now_utc(),
        "reference_validation_version": REFERENCE_VALIDATION_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "scope": "phase_minus_1_localized_480_reference_validation",
        "instances": instances,
        "reference_grid": REFERENCE_GRID,
        "localized_grid": LOCALIZED_GRID,
        "window_240_cells": WINDOW_240_CELLS,
        "acceptance": {
            "mapped_480_optimum": "same 240 basin or epsilon-tied compatible basin",
            "normalized_ideal_regret_shift_max": "0.25 * tau_R",
            "optimum_energy_shift_abs_max": "epsilon_tie = 0.001 * range(C_I_240)",
        },
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "decision": "phase_minus_1_reference_gate_passed" if status.endswith("passed") else "do_not_start_phase_9_until_reference_gate_is_resolved",
    }
    write_json_atomic(summary_path, report, overwrite=True)
    _write_csv(result_dir / "reference_validation_table.csv", rows)
    return report
