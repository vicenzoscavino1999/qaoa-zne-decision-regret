from __future__ import annotations

import csv
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .circuits import GraphInstance, ideal_expected_cut
from .gradients import central_gradients
from .io_utils import PROJECT_ROOT, ensure_dir, save_array_atomic, sha256_file, write_json_atomic
from .landscapes import generate_ideal_landscape


GRID_ADEQUACY_VERSION = "phase_d_grid_adequacy_v1"
TARGET_GRID = 120
REFERENCE_GRID = 240
LOCALIZED_GRID = 480
WINDOW_240_CELLS = 8
EPSILON_TIE_FRACTION = 0.001
GOOD_REGION_FRACTION = 0.10
TAU_R = 0.03996218710369882

STRICT = {
    "normalized_regret_error_lt": 0.25 * TAU_R,
    "j10_min": 0.90,
    "spearman_min": 0.995,
    "gradient_alignment_min": 0.99,
}
BORDERLINE = {
    "normalized_regret_error_lt": 0.50 * TAU_R,
    "j10_min": 0.85,
    "spearman_min": 0.990,
    "gradient_alignment_min": 0.985,
    "max_strict_soft_metric_misses": 1,
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_grid_limitations(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Phase D Grid Adequacy Limitations",
        "",
        f"Created UTC: {summary['created_at_utc']}",
        "",
        "This gate uses ideal-only landscapes. It does not inspect noisy or ZNE outcomes.",
        "",
        "The fixed 120x120 production grid can restrict Phase D to the subset of",
        "generated candidates whose ideal landscapes are resolved by the inherited",
        "Phase C grid. This is reported as a grid-resolved generated-instance",
        "replication envelope, not as an unbiased sample of all generated graphs.",
        "",
        f"Grid status: `{summary['status']}`.",
        f"Selected evaluated: `{summary['selected_evaluated']}`.",
        f"Reserves evaluated: `{summary['reserves_evaluated']}`.",
        f"Replacement events: `{len(summary['replacement_events'])}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _graph_from_payload(name: str, edges: list[list[Any]]) -> GraphInstance:
    return GraphInstance(
        name=name,
        edges=tuple((int(u), int(v), float(w)) for u, v, w in edges),
    )


def _array_path(root: Path, run_name: str, instance_id: str, grid_size: int) -> Path:
    return root / "data" / "raw" / run_name / instance_id / f"C_ideal_grid_{grid_size}.npy"


def _metadata_path(array_path: Path) -> Path:
    return array_path.with_suffix(".json")


def _load_or_generate_ideal(
    root: Path,
    run_name: str,
    graph: GraphInstance,
    grid_size: int,
    overwrite: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    array_path = _array_path(root, run_name, graph.name, grid_size)
    metadata_path = _metadata_path(array_path)
    if array_path.exists() and metadata_path.exists() and not overwrite:
        arr = np.load(array_path, allow_pickle=False)
        metadata = _load_json(metadata_path)
        metadata["loaded_existing"] = True
        return arr, metadata
    if array_path.exists() and not overwrite:
        raise RuntimeError(f"Existing ideal grid array is missing metadata: {array_path}")
    if metadata_path.exists() and not overwrite:
        raise RuntimeError(f"Existing ideal grid metadata is missing array: {metadata_path}")

    started = time.perf_counter()
    arr = generate_ideal_landscape(graph, grid_size)
    elapsed = time.perf_counter() - started
    save_array_atomic(array_path, arr, overwrite=overwrite)
    metadata = {
        "created_at_utc": _utc_now(),
        "grid_adequacy_version": GRID_ADEQUACY_VERSION,
        "run_name": run_name,
        "instance": graph.name,
        "grid_size": grid_size,
        "shape": list(arr.shape),
        "path": array_path.relative_to(root).as_posix(),
        "sha256": sha256_file(array_path),
        "elapsed_seconds": elapsed,
        "array_min": float(np.min(arr)),
        "array_max": float(np.max(arr)),
        "array_range": float(np.max(arr) - np.min(arr)),
        "loaded_existing": False,
    }
    write_json_atomic(metadata_path, metadata, overwrite=True)
    return arr, metadata


def _argmin(values: np.ndarray) -> tuple[int, int]:
    return tuple(int(x) for x in np.unravel_index(int(np.argmin(values)), values.shape))


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


def _local_minima_count(values: np.ndarray, max_rel_height: float) -> int:
    threshold = float(np.min(values) + max_rel_height * (np.max(values) - np.min(values)))
    count = 0
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
            if bool(value <= np.min(values[np.ix_(row_neighbors, col_neighbors)])):
                count += 1
    return count


def _roughness_proxy(values: np.ndarray) -> float:
    d_gamma = 2.0 * math.pi / values.shape[0]
    d_beta = math.pi / values.shape[1]
    g, b, valid = central_gradients(values, d_gamma, d_beta, periodic_gamma=True, periodic_beta=False)
    norm = np.sqrt(g**2 + b**2)
    if not bool(np.any(valid)):
        return float("nan")
    data_range = float(np.max(values) - np.min(values))
    return float(np.nanmean(norm[valid]) / data_range) if data_range else float("nan")


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


def _coords(index: tuple[int, int], grid: int) -> tuple[float, float]:
    return 2.0 * math.pi * index[0] / grid, math.pi * index[1] / grid


def _map_480_to_240(index_480: tuple[int, int]) -> tuple[int, int]:
    i480, j480 = index_480
    i240 = int(round(i480 / 2.0)) % REFERENCE_GRID
    j240 = min(REFERENCE_GRID - 1, max(0, int(round(j480 / 2.0))))
    return i240, j240


def _localized_480_check(
    graph: GraphInstance,
    reference: np.ndarray,
    basin: np.ndarray,
    epsilon_tie_abs: float,
) -> dict[str, Any]:
    reference_min = float(np.min(reference))
    reference_range = float(np.max(reference) - np.min(reference))
    reference_argmin = _argmin(reference)
    rows = []
    for i480, j480 in _localized_480_indices(reference_argmin):
        gamma, beta = _coords((i480, j480), LOCALIZED_GRID)
        value = -ideal_expected_cut(graph, gamma, beta)
        mapped = _map_480_to_240((i480, j480))
        rows.append({
            "i480": i480,
            "j480": j480,
            "ideal_value_480": value,
            "mapped_i240": mapped[0],
            "mapped_j240": mapped[1],
            "mapped_in_240_basin": bool(basin[mapped]),
            "mapped_epsilon_tied_240": bool(reference[mapped] <= reference_min + epsilon_tie_abs),
        })
    local_min_row = min(rows, key=lambda row: float(row["ideal_value_480"]))
    mapped = (int(local_min_row["mapped_i240"]), int(local_min_row["mapped_j240"]))
    mapped_reference_value = float(reference[mapped])
    normalized_shift = float((mapped_reference_value - reference_min) / reference_range) if reference_range else float("nan")
    energy_shift = abs(float(local_min_row["ideal_value_480"]) - reference_min)
    pass_basin = bool(basin[mapped] or mapped_reference_value <= reference_min + epsilon_tie_abs)
    pass_regret = bool(normalized_shift <= STRICT["normalized_regret_error_lt"])
    pass_energy = bool(energy_shift <= epsilon_tie_abs)
    return {
        "localized_480_triggered": True,
        "localized_480_pass": bool(pass_basin and pass_regret and pass_energy),
        "localized_480_points": len(rows),
        "localized_480_argmin_i": int(local_min_row["i480"]),
        "localized_480_argmin_j": int(local_min_row["j480"]),
        "localized_480_mapped_i240": mapped[0],
        "localized_480_mapped_j240": mapped[1],
        "localized_480_normalized_regret_shift": normalized_shift,
        "localized_480_optimum_energy_shift_abs": energy_shift,
        "localized_480_pass_basin": pass_basin,
        "localized_480_pass_regret": pass_regret,
        "localized_480_pass_energy": pass_energy,
    }


def _compare_120_to_240(graph: GraphInstance, candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    step = REFERENCE_GRID // TARGET_GRID
    reference_on_candidate = reference[::step, ::step]
    nested_max_abs_diff = float(np.max(np.abs(candidate - reference_on_candidate)))
    reference_range = float(np.max(reference) - np.min(reference))
    reference_min = float(np.min(reference))
    epsilon_tie_abs = EPSILON_TIE_FRACTION * reference_range
    candidate_argmin = _argmin(candidate)
    reference_argmin = _argmin(reference)
    mapped_argmin = (candidate_argmin[0] * step, candidate_argmin[1] * step)
    reference_value_at_candidate_argmin = float(reference[mapped_argmin])
    normalized_regret_error = float((reference_value_at_candidate_argmin - reference_min) / reference_range) if reference_range else float("nan")
    reference_regret_abs = float(reference_value_at_candidate_argmin - reference_min)
    basin = _basin_mask(reference, epsilon_tie_abs)
    argmin_compatible = bool(basin[mapped_argmin] or reference_value_at_candidate_argmin <= reference_min + epsilon_tie_abs)
    upsampled = np.repeat(np.repeat(candidate, step, axis=0), step, axis=1)
    spearman = spearmanr(reference.ravel(), upsampled.ravel()).correlation
    spearman = float(spearman) if spearman is not None and math.isfinite(float(spearman)) else float("nan")
    j10 = _jaccard(_good_region_mask(upsampled, GOOD_REGION_FRACTION), _good_region_mask(reference, GOOD_REGION_FRACTION))
    grad_align = _gradient_alignment_nested(candidate, reference, step)
    basin_size = int(np.sum(basin))
    near_optimal = reference <= reference_min + STRICT["normalized_regret_error_lt"] * reference_range
    strict_checks = {
        "nested_alignment": nested_max_abs_diff <= 1e-10,
        "normalized_regret": normalized_regret_error < STRICT["normalized_regret_error_lt"],
        "argmin_compatible": argmin_compatible,
        "j10": j10 >= STRICT["j10_min"],
        "spearman": spearman >= STRICT["spearman_min"],
        "gradient_alignment": grad_align >= STRICT["gradient_alignment_min"],
    }
    soft_misses = sum(
        not strict_checks[name]
        for name in ("normalized_regret", "j10", "spearman", "gradient_alignment")
    )
    borderline_candidate = bool(
        nested_max_abs_diff <= 1e-10
        and normalized_regret_error < BORDERLINE["normalized_regret_error_lt"]
        and argmin_compatible
        and j10 >= BORDERLINE["j10_min"]
        and spearman >= BORDERLINE["spearman_min"]
        and grad_align >= BORDERLINE["gradient_alignment_min"]
        and soft_misses <= BORDERLINE["max_strict_soft_metric_misses"]
    )
    localized = {
        "localized_480_triggered": False,
        "localized_480_pass": "",
        "localized_480_points": 0,
    }
    if not all(strict_checks.values()) and borderline_candidate:
        localized = _localized_480_check(graph, reference, basin, epsilon_tie_abs)
    grid_pass = bool(all(strict_checks.values()))
    grid_borderline = bool(borderline_candidate and localized.get("localized_480_pass") is True)
    if grid_pass:
        grid_label = "grid_pass"
    elif grid_borderline:
        grid_label = "grid_borderline_acceptable"
    else:
        grid_label = "grid_fail"
    return {
        "grid_label": grid_label,
        "grid_pass": grid_pass,
        "grid_borderline_acceptable": grid_borderline,
        "grid_fail": not (grid_pass or grid_borderline),
        "reference_grid": REFERENCE_GRID,
        "production_grid": TARGET_GRID,
        "nested_max_abs_diff": nested_max_abs_diff,
        "normalized_regret_error": normalized_regret_error,
        "reference_regret_abs_at_120_argmin": reference_regret_abs,
        "reference_range": reference_range,
        "epsilon_tie_abs": epsilon_tie_abs,
        "tau_R": TAU_R,
        "tau_R_quarter": STRICT["normalized_regret_error_lt"],
        "tau_R_half": BORDERLINE["normalized_regret_error_lt"],
        "argmin_compatible": argmin_compatible,
        "candidate_argmin_i": candidate_argmin[0],
        "candidate_argmin_j": candidate_argmin[1],
        "mapped_argmin_i240": mapped_argmin[0],
        "mapped_argmin_j240": mapped_argmin[1],
        "reference_argmin_i": reference_argmin[0],
        "reference_argmin_j": reference_argmin[1],
        "J10_120_vs_240": j10,
        "Spearman_120_vs_240": spearman,
        "gradient_alignment_120_vs_240": grad_align,
        "basin_width_proxy": basin_size,
        "near_optimal_grid_point_count": int(np.sum(near_optimal)),
        "near_optimal_area_fraction": float(np.mean(near_optimal)),
        "ideal_landscape_roughness_proxy": _roughness_proxy(reference),
        "basin_count_proxy": _local_minima_count(reference, max_rel_height=STRICT["normalized_regret_error_lt"]),
        "strict_soft_metric_misses": soft_misses,
        **{f"pass_strict_{key}": value for key, value in strict_checks.items()},
        **localized,
    }


def _edge_payload_from_csv(row: dict[str, str]) -> list[list[Any]]:
    return json.loads(row["edges_json"])


def _topology_metrics(edges: list[list[Any]], n_nodes: int) -> dict[str, Any]:
    weighted = [(int(u), int(v), float(w)) for u, v, w in edges]
    degree = np.zeros(n_nodes, dtype=float)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=float)
    for u, v, weight in weighted:
        degree[u] += 1.0
        degree[v] += 1.0
        adjacency[u, v] = weight
        adjacency[v, u] = weight
    laplacian = np.diag(np.sum(adjacency, axis=1)) - adjacency
    adjacency_eigs = np.linalg.eigvalsh(adjacency)
    lap_eigs = np.linalg.eigvalsh(laplacian)
    triangle_count = int(round(float(np.trace((adjacency > 0).astype(float) @ (adjacency > 0).astype(float) @ (adjacency > 0).astype(float))) / 6.0))
    possible_triads = max(1, n_nodes * (n_nodes - 1) * (n_nodes - 2) // 6)
    return {
        "spectral_radius": float(np.max(np.abs(adjacency_eigs))) if adjacency_eigs.size else 0.0,
        "laplacian_algebraic_connectivity": float(lap_eigs[1]) if len(lap_eigs) > 1 else 0.0,
        "laplacian_spectral_gap": float(lap_eigs[1]) if len(lap_eigs) > 1 else 0.0,
        "triangle_count": triangle_count,
        "clustering_coefficient": float(triangle_count / possible_triads),
        "max_degree": int(np.max(degree)) if degree.size else 0,
    }


def _evaluate_candidate(
    root: Path,
    run_name: str,
    row: dict[str, str],
    overwrite_arrays: bool,
) -> dict[str, Any]:
    edges = _edge_payload_from_csv(row)
    graph = _graph_from_payload(str(row["candidate_id"]), edges)
    candidate, candidate_meta = _load_or_generate_ideal(root, run_name, graph, TARGET_GRID, overwrite=overwrite_arrays)
    reference, reference_meta = _load_or_generate_ideal(root, run_name, graph, REFERENCE_GRID, overwrite=overwrite_arrays)
    metrics = _compare_120_to_240(graph, candidate, reference)
    return {
        "candidate_id": row["candidate_id"],
        "selected_label": row.get("selected_label", ""),
        "reserve_rank": row.get("reserve_rank", ""),
        "candidate_status": row["candidate_status"],
        "stratum_id": row["stratum_id"],
        "stratum_queue_rank": row.get("stratum_queue_rank", ""),
        "n_nodes": int(row["n_nodes"]),
        "n_edges": int(row["n_edges"]),
        "graph_signature": row["graph_signature"],
        "production_array": candidate_meta["path"],
        "production_array_sha256": candidate_meta["sha256"],
        "reference_array": reference_meta["path"],
        "reference_array_sha256": reference_meta["sha256"],
        **metrics,
        **_topology_metrics(edges, int(row["n_nodes"])),
    }


def _candidate_sort_key(row: dict[str, str]) -> tuple[str, int]:
    rank = row.get("stratum_queue_rank", "")
    return row["stratum_id"], int(rank) if str(rank).strip() else 999999


def _group_mean(rows: list[dict[str, Any]], key: str) -> float | str:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "" or value is None:
            continue
        try:
            f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            values.append(f)
    return float(np.mean(values)) if values else ""


def _summary_rows(audit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in audit_rows:
        groups.setdefault((str(row.get("candidate_status", "")), str(row.get("grid_label", ""))), []).append(row)
    out = []
    for (candidate_status, grid_label), rows in sorted(groups.items()):
        out.append({
            "candidate_status": candidate_status,
            "grid_label": grid_label,
            "count": len(rows),
            "mean_n_nodes": _group_mean(rows, "n_nodes"),
            "mean_n_edges": _group_mean(rows, "n_edges"),
            "mean_density": _group_mean(rows, "density"),
            "mean_degree_std": _group_mean(rows, "degree_std"),
            "mean_weight_std": _group_mean(rows, "weight_std"),
            "mean_spectral_radius": _group_mean(rows, "spectral_radius"),
            "mean_laplacian_spectral_gap": _group_mean(rows, "laplacian_spectral_gap"),
            "mean_roughness_proxy": _group_mean(rows, "ideal_landscape_roughness_proxy"),
            "mean_basin_width_proxy": _group_mean(rows, "basin_width_proxy"),
        })
    return out


def _final_selection_details(
    final_selection: dict[str, Any],
    candidate_rows: list[dict[str, str]],
    evaluated: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows_by_id = {row["candidate_id"]: row for row in candidate_rows}
    out: dict[str, Any] = {}
    for label, payload in sorted(final_selection.items(), key=lambda item: int(item[0][1:])):
        candidate_id = payload["candidate_id"]
        row = rows_by_id[candidate_id]
        eval_row = evaluated.get(candidate_id, {})
        out[label] = {
            "selected_label": label,
            "candidate_id": candidate_id,
            "source_candidate_status": row["candidate_status"],
            "replacement_used": bool(payload.get("replacement_used", False)),
            "replaces_candidate_id": payload.get("replaces_candidate_id", ""),
            "stratum_id": row["stratum_id"],
            "stratum_name": row["stratum_name"],
            "stratum_queue_rank": row.get("stratum_queue_rank", ""),
            "reserve_rank": row.get("reserve_rank", ""),
            "grid_label": payload["grid_label"],
            "n_nodes": int(row["n_nodes"]),
            "n_edges": int(row["n_edges"]),
            "density": float(row["density"]),
            "weighted_flag": row["weighted_flag"] == "True",
            "weight_min": float(row["weight_min"]),
            "weight_max": float(row["weight_max"]),
            "weight_mean": float(row["weight_mean"]),
            "weight_std": float(row["weight_std"]),
            "degree_min": int(row["degree_min"]),
            "degree_max": int(row["degree_max"]),
            "degree_mean": float(row["degree_mean"]),
            "degree_std": float(row["degree_std"]),
            "graph_signature": row["graph_signature"],
            "edges": _edge_payload_from_csv(row),
            "grid_metrics": {
                key: eval_row.get(key)
                for key in (
                    "normalized_regret_error",
                    "J10_120_vs_240",
                    "Spearman_120_vs_240",
                    "gradient_alignment_120_vs_240",
                    "localized_480_triggered",
                    "localized_480_pass",
                    "production_array",
                    "production_array_sha256",
                    "reference_array",
                    "reference_array_sha256",
                )
            },
        }
    return out


def _write_phase_d_instance_config(
    root: Path,
    planning_dir: Path,
    final_instances: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    generator_summary = _load_json(planning_dir / "phase_d_generator_summary.json")
    generator_config = _load_json(planning_dir / "phase_d_generator_config.json")
    stratum_queues = _load_json(planning_dir / "phase_d_stratum_queues.json")
    payload = {
        "created_at_utc": _utc_now(),
        "config_version": "phase_d_instances_v1_0_candidate_pre_freeze",
        "scope": "Phase D final H1-H12 instances and same-stratum reserves after ideal-only grid adequacy",
        "source_files": {
            "generator_config": "results/phase_d/planning/phase_d_generator_config.json",
            "candidate_queue": "results/phase_d/planning/phase_d_candidate_queue.csv",
            "stratum_queues": "results/phase_d/planning/phase_d_stratum_queues.json",
            "grid_adequacy_summary": "results/phase_d/planning/phase_d_grid_adequacy_summary.json",
            "selected_instances_final": "results/phase_d/planning/phase_d_selected_instances_final.json",
        },
        "hashes": {
            "generator_config_hash": generator_summary.get("config_hash"),
            "candidate_queue_hash": sha256_file(planning_dir / "phase_d_candidate_queue.csv"),
            "stratum_queues_hash": generator_summary.get("stratum_queues_hash"),
            "grid_adequacy_summary_hash": "available_in_phase_d_scope_snapshot_after_summary_write",
        },
        "generator_seed": generator_config.get("master_seed"),
        "structural_constraints": generator_config.get("envelope", {}),
        "grid_adequacy_status": summary["status"],
        "replacement_events": summary["replacement_events"],
        "selected_instances_H1_to_H12": final_instances,
        "reserve_instances_by_stratum": generator_summary.get("reserves", {}),
        "stratum_queues": stratum_queues,
    }
    config_path = root / "config" / "phase_d_instances_v1.0.json"
    write_json_atomic(config_path, payload, overwrite=True)
    return {
        "path": config_path.relative_to(root).as_posix(),
        "sha256": sha256_file(config_path),
    }


def run_phase_d_grid_adequacy(
    root: Path = PROJECT_ROOT,
    probe: bool = False,
    max_selected: int | None = None,
    overwrite_arrays: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    planning_dir = root / "results" / "phase_d" / "planning"
    output_dir = ensure_dir(planning_dir / "grid_adequacy_probe") if probe else planning_dir
    summary_path = output_dir / "phase_d_grid_adequacy_summary.json"
    if summary_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite grid adequacy output: {summary_path}")

    queue_path = planning_dir / "phase_d_candidate_queue.csv"
    candidate_rows = _read_csv(queue_path)
    selected_rows = [row for row in candidate_rows if row.get("candidate_status") == "selected"]
    if max_selected is not None:
        allowed_labels = {f"H{i}" for i in range(1, max_selected + 1)}
        selected_rows = [row for row in selected_rows if row.get("selected_label") in allowed_labels]
    selected_strata = {row["stratum_id"] for row in selected_rows}
    reserve_rows = [
        row for row in candidate_rows
        if row.get("candidate_status") == "reserve" and row["stratum_id"] in selected_strata
    ]
    eval_rows = sorted(selected_rows + reserve_rows, key=_candidate_sort_key)
    run_name = "phase_d_grid_adequacy_probe" if probe else "phase_d_grid_adequacy"
    evaluated: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for row in eval_rows:
        evaluated[row["candidate_id"]] = _evaluate_candidate(root, run_name, row, overwrite_arrays=overwrite_arrays)

    final_selection: dict[str, Any] = {}
    replacement_events: list[dict[str, Any]] = []
    for row in selected_rows:
        candidate_id = row["candidate_id"]
        label = row["selected_label"]
        eval_row = evaluated[candidate_id]
        if eval_row["grid_label"] in {"grid_pass", "grid_borderline_acceptable"}:
            final_selection[label] = {
                "candidate_id": candidate_id,
                "grid_label": eval_row["grid_label"],
                "replacement_used": False,
            }
            continue
        same_reserves = sorted(
            [r for r in reserve_rows if r["stratum_id"] == row["stratum_id"]],
            key=lambda r: int(r["reserve_rank"]),
        )
        replacement = None
        for reserve in same_reserves:
            reserve_eval = evaluated[reserve["candidate_id"]]
            if reserve_eval["grid_label"] in {"grid_pass", "grid_borderline_acceptable"}:
                replacement = reserve
                break
        if replacement is None:
            final_selection[label] = {
                "candidate_id": candidate_id,
                "grid_label": "grid_fail_exclude",
                "replacement_used": False,
            }
        else:
            final_selection[label] = {
                "candidate_id": replacement["candidate_id"],
                "grid_label": evaluated[replacement["candidate_id"]]["grid_label"],
                "replacement_used": True,
                "replaces_candidate_id": candidate_id,
            }
            replacement_events.append({
                "selected_label": label,
                "failed_candidate_id": candidate_id,
                "replacement_candidate_id": replacement["candidate_id"],
                "stratum_id": row["stratum_id"],
                "reserve_rank": replacement["reserve_rank"],
                "replacement_grid_label": evaluated[replacement["candidate_id"]]["grid_label"],
            })

    audit_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        base: dict[str, Any] = dict(row)
        base.update(_topology_metrics(_edge_payload_from_csv(row), int(row["n_nodes"])))
        if row["candidate_id"] in evaluated:
            base.update(evaluated[row["candidate_id"]])
            if base["candidate_status"] == "selected" and base["grid_label"] == "grid_fail":
                base["grid_label"] = "grid_fail_replace_with_reserve" if any(ev["failed_candidate_id"] == row["candidate_id"] for ev in replacement_events) else "grid_fail_exclude"
        elif row["candidate_status"].startswith("rejected"):
            base["grid_label"] = "structural_rejected_not_grid_evaluated"
        else:
            base["grid_label"] = "not_grid_evaluated"
        audit_rows.append(base)

    by_instance = [evaluated[row["candidate_id"]] for row in eval_rows]
    failure_modes: dict[str, int] = {}
    for row in by_instance:
        if row["grid_label"] in {"grid_pass", "grid_borderline_acceptable"}:
            continue
        for key, value in row.items():
            if key.startswith("pass_strict_") and value is False:
                failure_modes[key] = failure_modes.get(key, 0) + 1

    selected_failures = [row for row in by_instance if row["candidate_status"] == "selected" and row["grid_label"] == "grid_fail"]
    final_exclusions = [
        label for label, payload in final_selection.items()
        if payload["grid_label"] == "grid_fail_exclude"
    ]
    if final_exclusions:
        status = "fail_unresolved_grid_exclusions"
    elif replacement_events:
        status = "pass_with_mechanical_replacement"
    elif selected_failures:
        status = "replacement_required"
    else:
        status = "pass"
    summary = {
        "created_at_utc": _utc_now(),
        "grid_adequacy_version": GRID_ADEQUACY_VERSION,
        "mode": "probe" if probe else "full",
        "run_name": run_name,
        "status": status,
        "final_selection_complete": len(final_selection) == len(selected_rows) and not final_exclusions,
        "final_exclusions": final_exclusions,
        "selected_evaluated": len(selected_rows),
        "reserves_evaluated": len(reserve_rows),
        "candidate_queue_rows": len(candidate_rows),
        "grid_pass_count": sum(1 for row in by_instance if row["grid_label"] == "grid_pass"),
        "grid_borderline_acceptable_count": sum(1 for row in by_instance if row["grid_label"] == "grid_borderline_acceptable"),
        "grid_fail_count": sum(1 for row in by_instance if row["grid_label"] == "grid_fail"),
        "replacement_events": replacement_events,
        "final_selection": final_selection,
        "failure_modes": failure_modes,
        "elapsed_seconds": time.perf_counter() - started,
        "outputs": {
            "grid_adequacy_by_instance": "phase_d_grid_adequacy_by_instance.csv",
            "grid_rejection_audit": "phase_d_grid_rejection_audit.csv",
            "selected_vs_rejected_topology_summary": "phase_d_selected_vs_rejected_topology_summary.csv",
            "grid_failure_modes": "phase_d_grid_failure_modes.json",
            "grid_adequacy_limitations": "grid_adequacy_limitations.md",
            "selected_instances_final": "phase_d_selected_instances_final.json",
        },
        "limitations": [
            "This is ideal-only grid adequacy and does not use noisy or ZNE outcomes.",
            "In probe mode, only a subset of selected instances is evaluated.",
        ] if probe else [
            "This is ideal-only grid adequacy and does not use noisy or ZNE outcomes.",
            "The fixed 120x120 gate may restrict Phase D to grid-resolved generated instances.",
        ],
    }
    _write_csv(output_dir / "phase_d_grid_adequacy_by_instance.csv", by_instance)
    _write_csv(output_dir / "phase_d_grid_rejection_audit.csv", audit_rows)
    _write_csv(output_dir / "phase_d_selected_vs_rejected_topology_summary.csv", _summary_rows(audit_rows))
    write_json_atomic(output_dir / "phase_d_grid_failure_modes.json", failure_modes, overwrite=True)
    _write_grid_limitations(output_dir / "grid_adequacy_limitations.md", summary)
    final_details = _final_selection_details(final_selection, candidate_rows, evaluated)
    write_json_atomic(
        output_dir / "phase_d_selected_instances_final.json",
        final_details,
        overwrite=True,
    )
    if not probe:
        summary["phase_d_instance_config"] = _write_phase_d_instance_config(root, planning_dir, final_details, summary)
    write_json_atomic(summary_path, summary, overwrite=True)
    return summary
