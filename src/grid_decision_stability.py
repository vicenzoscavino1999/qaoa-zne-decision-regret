from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from qiskit_aer import AerSimulator

from .circuits import build_qaoa_circuit, graph_from_config
from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, write_json_atomic
from .landscapes import _sample_expected_cut, parameter_grid
from .mitiq_wrapper import allocate_shots, deterministic_seed, fold_global_odd_integer, richardson_weights
from .noise_models import build_depolarizing_noise_model


DEFAULT_REFERENCE_GRID = 240
DEFAULT_CANDIDATE_GRIDS = (80, 120)
DEFAULT_NOISE_VALUES = (0.005, 0.01)
DEFAULT_SHOTS_TOTAL = 1024
DEFAULT_ZNE_ORDERS = ("linear", "quadratic", "cubic")
DEFAULT_MAX_POINTS_PER_GRID = 48
DEFAULT_PILOT_HOLDOUT_SEEDS = (91011, 91021, 91031, 91041, 91051)
SIMULATOR_PROFILE = "default_threads_1"
SIMULATOR_OPTIONS: dict[str, object] = {"max_parallel_threads": 1}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_ideal(root: Path, instance: str, grid: int) -> np.ndarray:
    path = root / "data" / "raw" / "grid_convergence_ideal" / instance / f"C_ideal_grid_{grid}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"Missing ideal grid array: {path}")
    arr = np.load(path, allow_pickle=False)
    if arr.shape != (grid, grid):
        raise ValueError(f"Unexpected shape for {path}: {arr.shape}")
    if not bool(np.isfinite(arr).all()):
        raise ValueError(f"Non-finite values in {path}")
    return arr


def _add_neighborhood(points: list[tuple[int, int, str]], center: tuple[int, int], grid: int, radius: int, source: str) -> None:
    ci, cj = center
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            j = cj + dj
            if 0 <= j < grid:
                points.append(((ci + di) % grid, j, source))


def _evenly_spaced_best_indices(values: np.ndarray, count: int, fraction: float) -> list[tuple[int, int]]:
    k = max(1, int(round(values.size * fraction)))
    order = np.argpartition(values.ravel(), k - 1)[:k]
    sorted_order = order[np.argsort(values.ravel()[order])]
    if len(sorted_order) <= count:
        selected = sorted_order
    else:
        positions = np.linspace(0, len(sorted_order) - 1, count, dtype=int)
        selected = sorted_order[positions]
    return [tuple(int(x) for x in np.unravel_index(int(idx), values.shape)) for idx in selected]


def _map_reference_index_to_grid(index: tuple[int, int], reference_grid: int, grid: int) -> tuple[int, int]:
    ri, rj = index
    i = int(round(ri * grid / reference_grid)) % grid
    j = int(round(rj * grid / reference_grid))
    return i, min(max(j, 0), grid - 1)


def _try_add_continuous_point(
    out: list[dict[str, object]],
    seen_by_grid: dict[int, set[tuple[int, int]]],
    reference: np.ndarray,
    reference_grid: int,
    candidate_grids: tuple[int, ...],
    ref_index: tuple[int, int],
    source: str,
    max_points: int,
) -> bool:
    ri = ref_index[0] % reference_grid
    rj = min(max(ref_index[1], 0), reference_grid - 1)
    mapped = {grid: _map_reference_index_to_grid((ri, rj), reference_grid, grid) for grid in candidate_grids}
    if any(mapped[grid] in seen_by_grid[grid] for grid in candidate_grids):
        return False
    for grid in candidate_grids:
        seen_by_grid[grid].add(mapped[grid])
    out.append({
        "continuous_point_id": len(out),
        "reference_i": ri,
        "reference_j": rj,
        "gamma": float(2.0 * math.pi * ri / reference_grid),
        "beta": float(math.pi * rj / reference_grid),
        "source": source,
        "reference_ideal_value": float(reference[ri, rj]),
        "mapped_indices": {str(grid): {"i": mapped[grid][0], "j": mapped[grid][1]} for grid in candidate_grids},
    })
    return len(out) >= max_points


def _select_continuous_probe_points(
    reference: np.ndarray,
    reference_grid: int,
    candidate_grids: tuple[int, ...],
    max_points: int,
    seed: int,
) -> list[dict[str, object]]:
    reference_argmin = tuple(int(x) for x in np.unravel_index(int(np.argmin(reference)), reference.shape))
    out: list[dict[str, object]] = []
    seen_by_grid: dict[int, set[tuple[int, int]]] = {grid: set() for grid in candidate_grids}

    local_target = min(24, max_points)
    for radius in range(0, 24):
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                if max(abs(di), abs(dj)) != radius:
                    continue
                if _try_add_continuous_point(
                    out,
                    seen_by_grid,
                    reference,
                    reference_grid,
                    candidate_grids,
                    (reference_argmin[0] + di, reference_argmin[1] + dj),
                    "reference_argmin_local_basin",
                    max_points,
                ):
                    return out
                if len(out) >= local_target:
                    break
            if len(out) >= local_target:
                break
        if len(out) >= local_target:
            break

    for idx in _evenly_spaced_best_indices(reference, count=48, fraction=0.10):
        if _try_add_continuous_point(out, seen_by_grid, reference, reference_grid, candidate_grids, idx, "reference_top10_even", max_points):
            return out

    flat_sorted = np.argsort(reference.ravel())
    for quantile, source in [(0.50, "reference_mid_control"), (0.90, "reference_high_cost_control")]:
        center = int(round((len(flat_sorted) - 1) * quantile))
        span = range(max(0, center - 24), min(len(flat_sorted), center + 25))
        for pos in span:
            idx = tuple(int(x) for x in np.unravel_index(int(flat_sorted[pos]), reference.shape))
            if _try_add_continuous_point(out, seen_by_grid, reference, reference_grid, candidate_grids, idx, source, max_points):
                return out

    rng = np.random.default_rng(seed)
    for flat in rng.permutation(reference.size):
        idx = tuple(int(x) for x in np.unravel_index(int(flat), reference.shape))
        if _try_add_continuous_point(out, seen_by_grid, reference, reference_grid, candidate_grids, idx, "deterministic_global_control", max_points):
            return out
    return out


def _select_probe_points(
    candidate: np.ndarray,
    reference: np.ndarray,
    grid: int,
    reference_grid: int,
    max_points: int,
    seed: int,
) -> list[dict[str, object]]:
    candidate_argmin = tuple(int(x) for x in np.unravel_index(int(np.argmin(candidate)), candidate.shape))
    reference_argmin = tuple(int(x) for x in np.unravel_index(int(np.argmin(reference)), reference.shape))
    mapped_reference_argmin = _map_reference_index_to_grid(reference_argmin, reference_grid, grid)

    raw_points: list[tuple[int, int, str]] = []
    _add_neighborhood(raw_points, mapped_reference_argmin, grid, radius=2, source="mapped_reference_argmin_radius2")
    _add_neighborhood(raw_points, candidate_argmin, grid, radius=1, source="candidate_argmin_radius1")
    for idx in _evenly_spaced_best_indices(candidate, count=12, fraction=0.10):
        raw_points.append((idx[0], idx[1], "candidate_top10_even"))

    rng = np.random.default_rng(seed)
    for flat in rng.choice(candidate.size, size=min(12, candidate.size), replace=False):
        idx = tuple(int(x) for x in np.unravel_index(int(flat), candidate.shape))
        raw_points.append((idx[0], idx[1], "deterministic_global_control"))

    seen: set[tuple[int, int]] = set()
    out: list[dict[str, object]] = []
    for i, j, source in raw_points:
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "i": i,
            "j": j,
            "theta_index": i * grid + j,
            "source": source,
            "ideal_value": float(candidate[i, j]),
        })
        if len(out) >= max_points:
            break
    return out


def _point_noisy_value(
    graph,
    grid: int,
    i: int,
    j: int,
    p_noise: float,
    shots_total: int,
    seed: int,
    simulator: AerSimulator,
    transpile_cache: dict[tuple[object, ...], object],
) -> float:
    gammas, betas = parameter_grid(grid)
    theta_index = i * grid + j
    circuit = build_qaoa_circuit(graph, float(gammas[i]), float(betas[j]), measure=False)
    circuit_seed = deterministic_seed(graph.name, p_noise, shots_total, seed, "phase_c_decision_noisy", "none", 1, theta_index)
    return -_sample_expected_cut(
        graph,
        circuit,
        shots_total,
        p_noise,
        circuit_seed,
        simulator=simulator,
        simulator_options=SIMULATOR_OPTIONS,
        transpile_cache=transpile_cache,
        transpile_cache_key=(graph.name, grid, theta_index, 1),
    )


def _point_zne_value(
    graph,
    grid: int,
    i: int,
    j: int,
    p_noise: float,
    shots_total: int,
    seed: int,
    zne_order: str,
    scale_factors: list[int],
    simulator: AerSimulator,
    transpile_cache: dict[tuple[object, ...], object],
) -> tuple[float, list[int]]:
    gammas, betas = parameter_grid(grid)
    theta_index = i * grid + j
    base = build_qaoa_circuit(graph, float(gammas[i]), float(betas[j]), measure=False)
    weights = richardson_weights(scale_factors)
    shot_allocation = allocate_shots(shots_total, weights, strategy="proportional_to_abs_weights")
    extrapolated = 0.0
    for scale, weight, shots in zip(scale_factors, weights, shot_allocation, strict=True):
        folded = fold_global_odd_integer(base, int(scale))
        circuit_seed = deterministic_seed(graph.name, p_noise, shots_total, seed, "zne", zne_order, scale, theta_index)
        value = -_sample_expected_cut(
            graph,
            folded,
            int(shots),
            p_noise,
            circuit_seed,
            simulator=simulator,
            simulator_options=SIMULATOR_OPTIONS,
            transpile_cache=transpile_cache,
            transpile_cache_key=(graph.name, grid, theta_index, int(scale)),
        )
        extrapolated += weight * value
    return float(extrapolated), [int(x) for x in shot_allocation]


def _sample_comparison(noisy: np.ndarray, zne: np.ndarray, ideal: np.ndarray, ideal_range: float, ideal_min: float) -> dict[str, float]:
    noisy_mse = float(np.mean((noisy - ideal) ** 2))
    zne_mse = float(np.mean((zne - ideal) ** 2))
    noisy_argmin = int(np.argmin(noisy))
    zne_argmin = int(np.argmin(zne))
    noisy_regret = float((ideal[noisy_argmin] - ideal_min) / ideal_range) if ideal_range > 0 else float("nan")
    zne_regret = float((ideal[zne_argmin] - ideal_min) / ideal_range) if ideal_range > 0 else float("nan")
    return {
        "probe_noisy_mse": noisy_mse,
        "probe_zne_mse": zne_mse,
        "probe_r_mse": float((noisy_mse - zne_mse) / noisy_mse) if noisy_mse != 0 else float("nan"),
        "probe_noisy_norm_regret": noisy_regret,
        "probe_zne_norm_regret": zne_regret,
        "probe_d_R": float(zne_regret - noisy_regret),
    }


def _classify_condition(seed_rows: list[dict[str, object]], tau_r: float) -> dict[str, object]:
    r_values = np.asarray([float(row["probe_r_mse"]) for row in seed_rows], dtype=float)
    d_values = np.asarray([float(row["probe_d_R"]) for row in seed_rows], dtype=float)
    median_r = float(np.median(r_values))
    median_d = float(np.median(d_values))
    return {
        "median_probe_r_mse": median_r,
        "median_probe_d_R": median_d,
        "sign_median_probe_r_mse": _sign_label(median_r),
        "sign_median_probe_d_R": _sign_label(median_d),
        "prob_probe_d_R_positive": float(np.mean(d_values > 0.0)),
        "prob_probe_r_mse_positive": float(np.mean(r_values > 0.0)),
        "pilot_discordant": bool(median_r > 0.10 and median_d > 0.0 and np.mean(d_values > 0.0) >= 0.60),
        "pilot_strongly_discordant": bool(median_r > 0.20 and median_d > tau_r and np.mean(d_values > 0.0) >= 0.70),
        "seed_count": len(seed_rows),
    }


def _sign_label(value: float) -> str:
    if value > 0.0:
        return "positive"
    if value < 0.0:
        return "negative"
    return "zero"


def run_grid_decision_stability(
    config_path: str | Path | None = None,
    candidate_grids: tuple[int, ...] = DEFAULT_CANDIDATE_GRIDS,
    reference_grid: int = DEFAULT_REFERENCE_GRID,
    max_points_per_grid: int = DEFAULT_MAX_POINTS_PER_GRID,
    overwrite: bool = True,
) -> dict[str, Any]:
    root = PROJECT_ROOT
    cfg = read_yaml(config_path or root / "config" / "config_official_pilot_v1.6.yaml")
    phase_c = cfg["phase_c"]
    seeds_cfg = read_yaml(root / "config" / "seeds.yaml")
    official_seeds = {int(x) for x in seeds_cfg["official_candidates"]}
    seeds = [int(x) for x in DEFAULT_PILOT_HOLDOUT_SEEDS]
    if any(seed in official_seeds for seed in seeds):
        raise ValueError("Pilot holdout seeds must not overlap official Phase C seeds.")
    instances = [str(x) for x in phase_c["instances"]]
    tau_r = float(phase_c["tau_R"])
    noise_values = [float(x) for x in DEFAULT_NOISE_VALUES]
    shots_total = DEFAULT_SHOTS_TOTAL
    zne_methods = {key: [int(x) for x in cfg["zne"]["methods"][key]] for key in DEFAULT_ZNE_ORDERS}

    result_dir = root / "results" / "grid_decision_stability"
    ensure_dir(result_dir)

    value_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    classification_rows: list[dict[str, object]] = []
    probe_point_rows: list[dict[str, object]] = []
    started_all = time.perf_counter()

    selected_points_by_instance: dict[str, list[dict[str, object]]] = {}
    for instance in instances:
        reference = _load_ideal(root, instance, reference_grid)
        selected_points_by_instance[instance] = _select_continuous_probe_points(
            reference,
            reference_grid,
            tuple(candidate_grids),
            max_points=max_points_per_grid,
            seed=20260510 + len(instance),
        )
    write_json_atomic(
        result_dir / "selected_continuous_points.json",
        {
            "status": "selected_before_noisy_zne",
            "selection_uses": "ideal reference landscape only",
            "reference_grid": reference_grid,
            "candidate_grids": list(candidate_grids),
            "max_points_per_grid": max_points_per_grid,
            "selected_points_by_instance": selected_points_by_instance,
        },
        overwrite=True,
    )

    for instance in instances:
        graph = graph_from_config(instance, cfg)
        for grid in candidate_grids:
            ideal_grid = _load_ideal(root, instance, grid)
            points: list[dict[str, object]] = []
            for selected in selected_points_by_instance[instance]:
                ref_index = (int(selected["reference_i"]), int(selected["reference_j"]))
                i, j = _map_reference_index_to_grid(ref_index, reference_grid, grid)
                points.append({
                    "continuous_point_id": int(selected["continuous_point_id"]),
                    "i": i,
                    "j": j,
                    "theta_index": i * grid + j,
                    "source": str(selected["source"]),
                    "gamma": float(selected["gamma"]),
                    "beta": float(selected["beta"]),
                    "reference_i": ref_index[0],
                    "reference_j": ref_index[1],
                    "reference_ideal_value": float(selected["reference_ideal_value"]),
                    "ideal_value": float(ideal_grid[i, j]),
                })
            ideal_vector = np.asarray([float(point["ideal_value"]) for point in points], dtype=float)
            ideal_range = float(np.max(ideal_grid) - np.min(ideal_grid))
            ideal_min = float(np.min(ideal_grid))
            for point_id, point in enumerate(points):
                probe_point_rows.append({
                    "instance": instance,
                    "grid": grid,
                    "point_id": point_id,
                    **point,
                })

            for p_noise in noise_values:
                noise_model = build_depolarizing_noise_model(p_noise)
                simulator = AerSimulator(noise_model=noise_model, **SIMULATOR_OPTIONS)
                transpile_cache: dict[tuple[object, ...], object] = {}
                for seed in seeds:
                    noisy_values: list[float] = []
                    zne_values_by_order: dict[str, list[float]] = {order: [] for order in zne_methods}
                    started_cell = time.perf_counter()
                    for point_id, point in enumerate(points):
                        i = int(point["i"])
                        j = int(point["j"])
                        noisy_value = _point_noisy_value(
                            graph,
                            grid,
                            i,
                            j,
                            p_noise,
                            shots_total,
                            seed,
                            simulator,
                            transpile_cache,
                        )
                        noisy_values.append(noisy_value)
                        value_rows.append({
                            "instance": instance,
                            "grid": grid,
                            "noise": p_noise,
                            "shots_total": shots_total,
                        "seed": seed,
                        "point_id": point_id,
                        "continuous_point_id": int(point["continuous_point_id"]),
                        "method": "noisy",
                        "zne_order": "none",
                            "value": noisy_value,
                            "ideal_value": float(ideal_vector[point_id]),
                        })
                        for order, scales in zne_methods.items():
                            zne_value, shot_allocation = _point_zne_value(
                                graph,
                                grid,
                                i,
                                j,
                                p_noise,
                                shots_total,
                                seed,
                                order,
                                scales,
                                simulator,
                                transpile_cache,
                            )
                            zne_values_by_order[order].append(zne_value)
                            value_rows.append({
                                "instance": instance,
                                "grid": grid,
                                "noise": p_noise,
                                "shots_total": shots_total,
                                "seed": seed,
                                "point_id": point_id,
                                "continuous_point_id": int(point["continuous_point_id"]),
                                "method": f"zne_{order}",
                                "zne_order": order,
                                "value": zne_value,
                                "ideal_value": float(ideal_vector[point_id]),
                                "shot_allocation": "|".join(str(x) for x in shot_allocation),
                            })
                    noisy = np.asarray(noisy_values, dtype=float)
                    for order, zne_values in zne_values_by_order.items():
                        row = {
                            "instance": instance,
                            "grid": grid,
                            "noise": p_noise,
                            "shots_total": shots_total,
                            "seed": seed,
                            "zne_order": order,
                            "n_probe_points": len(points),
                            "elapsed_seconds": time.perf_counter() - started_cell,
                            **_sample_comparison(noisy, np.asarray(zne_values, dtype=float), ideal_vector, ideal_range, ideal_min),
                        }
                        seed_rows.append(row)

    by_condition: dict[tuple[str, int, float, str], list[dict[str, object]]] = {}
    for row in seed_rows:
        key = (str(row["instance"]), int(row["grid"]), float(row["noise"]), str(row["zne_order"]))
        by_condition.setdefault(key, []).append(row)
    for key, rows in sorted(by_condition.items()):
        instance, grid, p_noise, order = key
        classification_rows.append({
            "instance": instance,
            "grid": grid,
            "noise": p_noise,
            "zne_order": order,
            **_classify_condition(rows, tau_r),
        })

    stability_rows: list[dict[str, object]] = []
    by_no_grid = {(row["instance"], float(row["noise"]), row["zne_order"], int(row["grid"])): row for row in classification_rows}
    compared_grids = tuple(sorted(candidate_grids))
    if len(compared_grids) >= 2:
        baseline_grid = compared_grids[0]
        reference_candidate_grid = compared_grids[-1]
        for instance in instances:
            for p_noise in noise_values:
                for order in zne_methods:
                    left = by_no_grid[(instance, p_noise, order, baseline_grid)]
                    right = by_no_grid[(instance, p_noise, order, reference_candidate_grid)]
                    delta_d = float(left["median_probe_d_R"]) - float(right["median_probe_d_R"])
                    stability_rows.append({
                        "instance": instance,
                        "noise": p_noise,
                        "zne_order": order,
                        "baseline_grid": baseline_grid,
                        "reference_grid": reference_candidate_grid,
                        "baseline_pilot_discordant": bool(left["pilot_discordant"]),
                        "reference_pilot_discordant": bool(right["pilot_discordant"]),
                        "baseline_pilot_strongly_discordant": bool(left["pilot_strongly_discordant"]),
                        "reference_pilot_strongly_discordant": bool(right["pilot_strongly_discordant"]),
                        "discordant_classification_stable": bool(left["pilot_discordant"]) == bool(right["pilot_discordant"]),
                        "strong_classification_stable": bool(left["pilot_strongly_discordant"]) == bool(right["pilot_strongly_discordant"]),
                        "sign_median_r_mse_stable": str(left["sign_median_probe_r_mse"]) == str(right["sign_median_probe_r_mse"]),
                        "sign_median_d_R_stable": str(left["sign_median_probe_d_R"]) == str(right["sign_median_probe_d_R"]),
                        "abs_delta_median_probe_d_R_le_tau_quarter": abs(delta_d) <= 0.25 * tau_r,
                        "abs_delta_median_probe_d_R_le_tau_half": abs(delta_d) <= 0.50 * tau_r,
                        "delta_median_probe_r_mse": float(left["median_probe_r_mse"]) - float(right["median_probe_r_mse"]),
                        "delta_median_probe_d_R": delta_d,
                    })

    decision_stable = all(
        bool(row["discordant_classification_stable"])
        and bool(row["strong_classification_stable"])
        and bool(row["sign_median_r_mse_stable"])
        and bool(row["sign_median_d_R_stable"])
        and bool(row["abs_delta_median_probe_d_R_le_tau_quarter"])
        for row in stability_rows
    )
    recommended_grid = min(candidate_grids) if decision_stable else max(candidate_grids)
    report = {
        "status": "grid_decision_stability_complete",
        "scope": "finite-shot decision-probe, not full Phase C",
        "candidate_grids": list(candidate_grids),
        "reference_grid_for_comparison": max(candidate_grids),
        "instances": instances,
        "noise_values": noise_values,
        "shots_total": shots_total,
        "seeds": seeds,
        "seeds_source": "pilot_holdout_seeds_not_used_by_official_phase_c_seed_list",
        "zne_orders": list(zne_methods),
        "simulator_profile": SIMULATOR_PROFILE,
        "simulator_options": SIMULATOR_OPTIONS,
        "max_points_per_grid": max_points_per_grid,
        "tau_R": tau_r,
        "classification_rule": {
            "pilot_discordant": "median_probe_r_mse > 0.10 and median_probe_d_R > 0 and P(d_R > 0) >= 0.60",
            "pilot_strongly_discordant": "median_probe_r_mse > 0.20 and median_probe_d_R > tau_R and P(d_R > 0) >= 0.70",
            "accept_min_grid": (
                "all rows preserve discordant and strong-discordant classification, median r_MSE sign, "
                "median d_R sign, and abs(delta median d_R) <= 0.25 * tau_R"
            ),
        },
        "selected_continuous_points_written_before_noisy_zne": True,
        "n_probe_point_rows": len(probe_point_rows),
        "n_value_rows": len(value_rows),
        "n_seed_rows": len(seed_rows),
        "n_classification_rows": len(classification_rows),
        "n_stability_rows": len(stability_rows),
        "decision_stable_between_min_and_max_grid": decision_stable,
        "recommended_grid_if_this_probe_is_accepted": recommended_grid,
        "elapsed_seconds": time.perf_counter() - started_all,
        "classification_rows": classification_rows,
        "stability_rows": stability_rows,
    }

    _write_csv(result_dir / "probe_points.csv", probe_point_rows)
    _write_csv(result_dir / "probe_point_values.csv", value_rows)
    _write_csv(result_dir / "seed_comparisons.csv", seed_rows)
    _write_csv(result_dir / "classification_by_condition.csv", classification_rows)
    _write_csv(result_dir / "stability_80_vs_120.csv", stability_rows)
    write_json_atomic(result_dir / "decision_stability_report.json", report, overwrite=overwrite)
    return report
