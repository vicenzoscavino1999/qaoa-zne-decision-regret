from __future__ import annotations

import csv
import json
import math
import os
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from .circuits import graph_from_config
from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, write_json_atomic
from .landscapes import parameter_grid
from .mitiq_wrapper import deterministic_seed
from .phase_c_120_custom_kernel import (
    CUSTOM_KERNEL_VERSION,
    expected_cut_from_probabilities,
    new_probability_cache,
    sampled_expected_cut_from_probabilities,
)
from .phase_c_120_kernel_runner import KERNEL_RUNNER_VERSION
from .phase_c_120_preflight import PROTOCOL_VERSION, TARGET_GRID_SIZE


TIMED_PROBE_VERSION = "phase_c_120_timed_probe_v1"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_c_120_timed_probe"
DEFAULT_INSTANCE = "G3_weighted6"
DEFAULT_NOISE = 0.005
DEFAULT_SHOTS = 1024
DEFAULT_SEED = 42
DEFAULT_THETA_COUNT = 32
DEFAULT_SCALE_FACTORS = (1, 3, 5, 7)
TARGET_LOCAL_HOURS = 72.0


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _selected_theta_indices(grid_size: int, theta_count: int) -> list[int]:
    total = grid_size * grid_size
    if theta_count < 1 or theta_count > total:
        raise ValueError(f"theta_count must satisfy 1 <= theta_count <= {total}.")
    indices = np.linspace(0, total - 1, theta_count, dtype=int).tolist()
    return sorted(set(int(index) for index in indices))


def _load_official_estimate(root: Path) -> dict[str, Any]:
    path = root / "results" / "phase_c_120_kernel_official" / "runtime_estimate.json"
    if not path.is_file():
        raise FileNotFoundError(f"Official kernel estimate not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def run_phase_c_120_timed_probe(
    theta_count: int = DEFAULT_THETA_COUNT,
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    result_dir = ensure_dir(root / "results" / "phase_c_120_timed_probe")
    summary_path = result_dir / "timed_probe_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing timed probe output: {summary_path}")

    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    official_estimate = _load_official_estimate(root)
    official_cache = official_estimate.get("probability_cache_model", {})
    official_misses = int(official_estimate.get("official_probability_cache_misses") or official_cache.get("expected_misses"))
    local_workers = max(1, int(os.cpu_count() or 1))

    graph = graph_from_config(DEFAULT_INSTANCE, cfg)
    gammas, betas = parameter_grid(TARGET_GRID_SIZE)
    theta_indices = _selected_theta_indices(TARGET_GRID_SIZE, theta_count)
    cache = new_probability_cache()
    rows: list[dict[str, Any]] = []
    aggregate_expected_cut = 0.0
    aggregate_sampled_cut = 0.0

    tracemalloc.start()
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    for theta_index in theta_indices:
        i = theta_index // TARGET_GRID_SIZE
        j = theta_index % TARGET_GRID_SIZE
        gamma = float(gammas[i])
        beta = float(betas[j])
        for scale_factor in DEFAULT_SCALE_FACTORS:
            started = time.perf_counter()
            result = cache.get(graph, theta_index, gamma, beta, DEFAULT_NOISE, scale_factor)
            probability_seconds = time.perf_counter() - started
            expected_cut = expected_cut_from_probabilities(graph, result.probabilities)
            sample_seed = deterministic_seed(
                graph.name,
                DEFAULT_NOISE,
                DEFAULT_SHOTS,
                DEFAULT_SEED,
                "timed_probe",
                "probability_throughput",
                scale_factor,
                theta_index,
            )
            sampled_cut = sampled_expected_cut_from_probabilities(
                graph,
                result.probabilities,
                DEFAULT_SHOTS,
                sample_seed,
            )
            aggregate_expected_cut += expected_cut
            aggregate_sampled_cut += sampled_cut
            rows.append(
                {
                    "theta_index": theta_index,
                    "gamma_index": i,
                    "beta_index": j,
                    "gamma": gamma,
                    "beta": beta,
                    "instance": graph.name,
                    "n_qubits": graph.n_qubits,
                    "p_noise": DEFAULT_NOISE,
                    "scale_factor": scale_factor,
                    "shots_total": DEFAULT_SHOTS,
                    "seed": DEFAULT_SEED,
                    "probability_seconds": probability_seconds,
                    "probability_sum": float(np.sum(result.probabilities)),
                    "probability_min": float(np.min(result.probabilities)),
                    "probability_max": float(np.max(result.probabilities)),
                    "expected_cut": expected_cut,
                    "sampled_expected_cut": sampled_cut,
                    "transpiled_gate_count": result.transpiled_gate_count,
                    "transpiled_depth": result.transpiled_depth,
                }
            )

    wall_elapsed = time.perf_counter() - wall_started
    cpu_elapsed = time.process_time() - cpu_started
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    _write_csv(result_dir / "timed_probe_measurements.csv", rows)
    cache_summary = cache.summary()
    probability_misses = int(cache_summary["misses"])
    seconds_per_probability_miss = wall_elapsed / probability_misses if probability_misses else math.nan
    serial_hours = seconds_per_probability_miss * official_misses / 3600.0
    local_75pct_parallel_hours = serial_hours / (local_workers * 0.75)
    previous_local_hours = float(official_estimate.get("local_75pct_parallel_hours", math.nan))
    projection_ratio_vs_previous = local_75pct_parallel_hours / previous_local_hours if math.isfinite(previous_local_hours) and previous_local_hours > 0.0 else math.nan
    probabilities_valid = all(
        math.isclose(float(row["probability_sum"]), 1.0, rel_tol=0.0, abs_tol=1e-9)
        and float(row["probability_min"]) >= -1e-12
        and np.isfinite(float(row["sampled_expected_cut"]))
        for row in rows
    )
    passes_72h_gate = bool(local_75pct_parallel_hours <= TARGET_LOCAL_HOURS)
    status = "phase_c_120_timed_probe_passed" if probabilities_valid and passes_72h_gate else "phase_c_120_timed_probe_failed"

    summary = {
        "created_at_utc": _now_utc(),
        "timed_probe_version": TIMED_PROBE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "custom_kernel_version": CUSTOM_KERNEL_VERSION,
        "runner_version": KERNEL_RUNNER_VERSION,
        "status": status,
        "scope": "120x120_probability_throughput_probe_not_official_phase_c_data",
        "grid_size": TARGET_GRID_SIZE,
        "instance": DEFAULT_INSTANCE,
        "p_noise": DEFAULT_NOISE,
        "shot_budget": DEFAULT_SHOTS,
        "seed": DEFAULT_SEED,
        "theta_count": len(theta_indices),
        "theta_indices": theta_indices,
        "scale_factors": list(DEFAULT_SCALE_FACTORS),
        "probability_cache": cache_summary,
        "expected_probability_misses": len(theta_indices) * len(DEFAULT_SCALE_FACTORS),
        "wall_elapsed_seconds": wall_elapsed,
        "cpu_elapsed_seconds": cpu_elapsed,
        "seconds_per_probability_cache_miss": seconds_per_probability_miss,
        "official_probability_cache_misses": official_misses,
        "projected_serial_hours": serial_hours,
        "local_workers": local_workers,
        "projected_local_75pct_parallel_hours": local_75pct_parallel_hours,
        "previous_projected_local_75pct_parallel_hours": previous_local_hours,
        "projection_ratio_vs_previous": projection_ratio_vs_previous,
        "passes_72h_gate": passes_72h_gate,
        "probabilities_valid": probabilities_valid,
        "tracemalloc_current_bytes": current_bytes,
        "tracemalloc_peak_bytes": peak_bytes,
        "aggregate_expected_cut": aggregate_expected_cut,
        "aggregate_sampled_cut": aggregate_sampled_cut,
        "measurements_csv": "results/phase_c_120_timed_probe/timed_probe_measurements.csv",
        "decision": "phase_8_throughput_probe_accepts_kernel_runner_for_limited_parallel_pilot"
        if status == "phase_c_120_timed_probe_passed"
        else "do_not_start_official_phase_c_until_throughput_issue_is_resolved",
        "limitations": [
            "This probe uses real 120 x 120 grid coordinates but only a deterministic subset of theta points.",
            "It does not create official Phase C landscapes and must not be used as scientific result data.",
            "It measures probability-cache miss throughput and sampling overhead, not long-run thermal throttling.",
        ],
    }
    write_json_atomic(summary_path, summary, overwrite=True)
    return summary
