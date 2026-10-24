from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import shutil
import statistics
import time
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
from .phase_c_120_operational_dry_run import _process_memory_sample, _thermal_snapshot
from .phase_c_120_preflight import PROTOCOL_VERSION, TARGET_GRID_SIZE


SOAK_TEST_VERSION = "phase_c_120_operational_soak_test_v1"
SOAK_TEST_NAME = "phase_c_120_operational_soak_test"
DEFAULT_WORKERS = min(8, max(1, os.cpu_count() or 1))
DEFAULT_DURATION_SECONDS = 30 * 60
DEFAULT_SAMPLE_INTERVAL_SECONDS = 30.0
DEFAULT_INSTANCE = "G3_weighted6"
DEFAULT_NOISE_VALUES = (0.005, 0.01)
DEFAULT_SCALE_FACTORS = (1, 3, 5, 7)
DEFAULT_SHOTS = 1024
DEFAULT_SEED = 42
CACHE_RESET_MISSES = 256
TARGET_THROUGHPUT_DROP_FRACTION = 0.20


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_reset(path: Path, root: Path) -> None:
    resolved = path.resolve()
    allowed = {
        (root / "results" / SOAK_TEST_NAME).resolve(),
    }
    if resolved not in allowed:
        raise RuntimeError(f"Refusing to remove non-soak-test path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except json.JSONDecodeError:
        return {}


def _worker_progress_payload(
    worker_index: int,
    status: str,
    started_perf: float,
    iterations: int,
    hits: int,
    misses: int,
    requests: int,
    cache_resets: int,
    probability_failures: int,
    finite_failures: int,
    checksum: float,
    error: str | None = None,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started_perf
    return {
        "worker_index": worker_index,
        "status": status,
        "updated_at_utc": _now_utc(),
        "elapsed_seconds": elapsed,
        "probability_evaluations": iterations,
        "hits": hits,
        "misses": misses,
        "requests": requests,
        "cache_resets": cache_resets,
        "probability_failures": probability_failures,
        "finite_failures": finite_failures,
        "checksum": checksum,
        "throughput_evaluations_per_second": iterations / elapsed if elapsed > 0.0 else 0.0,
        "error": error,
    }


def _soak_worker(
    worker_index: int,
    duration_seconds: float,
    sample_interval_seconds: float,
    root_text: str,
) -> None:
    root = Path(root_text)
    result_dir = ensure_dir(root / "results" / SOAK_TEST_NAME)
    progress_path = result_dir / f"worker_{worker_index:03d}_progress.json"
    summary_path = result_dir / f"worker_{worker_index:03d}_summary.json"
    started_perf = time.perf_counter()
    hits = 0
    misses = 0
    requests = 0
    iterations = 0
    probability_failures = 0
    finite_failures = 0
    cache_resets = 0
    checksum = 0.0
    error: str | None = None

    try:
        cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
        graph = graph_from_config(DEFAULT_INSTANCE, cfg)
        gammas, betas = parameter_grid(TARGET_GRID_SIZE)
        total_points = TARGET_GRID_SIZE * TARGET_GRID_SIZE
        cache = new_probability_cache()
        misses_at_reset = 0
        next_progress = started_perf
        end_perf = started_perf + float(duration_seconds)
        while time.perf_counter() < end_perf:
            theta_index = int((worker_index * 1009 + iterations * 7919) % total_points)
            i = theta_index // TARGET_GRID_SIZE
            j = theta_index % TARGET_GRID_SIZE
            scale_factor = int(DEFAULT_SCALE_FACTORS[(iterations + worker_index) % len(DEFAULT_SCALE_FACTORS)])
            p_noise = float(DEFAULT_NOISE_VALUES[((iterations // len(DEFAULT_SCALE_FACTORS)) + worker_index) % len(DEFAULT_NOISE_VALUES)])
            before = cache.summary()
            result = cache.get(graph, theta_index, float(gammas[i]), float(betas[j]), p_noise, scale_factor)
            after = cache.summary()
            hits += int(after["hits"]) - int(before["hits"])
            misses += int(after["misses"]) - int(before["misses"])
            requests += int(after["requests"]) - int(before["requests"])
            probability_sum = float(np.sum(result.probabilities))
            probability_min = float(np.min(result.probabilities))
            expected_cut = expected_cut_from_probabilities(graph, result.probabilities)
            sample_seed = deterministic_seed(
                graph.name,
                p_noise,
                DEFAULT_SHOTS,
                DEFAULT_SEED,
                "operational_soak",
                f"worker_{worker_index}",
                scale_factor,
                theta_index,
                iterations,
            )
            sampled_cut = sampled_expected_cut_from_probabilities(graph, result.probabilities, DEFAULT_SHOTS, sample_seed)
            if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-9) or probability_min < -1e-12:
                probability_failures += 1
            if not (math.isfinite(expected_cut) and math.isfinite(sampled_cut)):
                finite_failures += 1
            checksum += 1e-6 * expected_cut + 1e-7 * sampled_cut
            iterations += 1

            if misses - misses_at_reset >= CACHE_RESET_MISSES:
                cache = new_probability_cache()
                misses_at_reset = misses
                cache_resets += 1

            now = time.perf_counter()
            if now >= next_progress:
                write_json_atomic(
                    progress_path,
                    _worker_progress_payload(
                        worker_index,
                        "running",
                        started_perf,
                        iterations,
                        hits,
                        misses,
                        requests,
                        cache_resets,
                        probability_failures,
                        finite_failures,
                        checksum,
                    ),
                    overwrite=True,
                )
                next_progress = now + float(sample_interval_seconds)
    except Exception as exc:  # pragma: no cover - surfaced through JSON in operational run.
        error = repr(exc)

    status = "passed" if error is None and probability_failures == 0 and finite_failures == 0 and iterations > 0 else "failed"
    payload = _worker_progress_payload(
        worker_index,
        status,
        started_perf,
        iterations,
        hits,
        misses,
        requests,
        cache_resets,
        probability_failures,
        finite_failures,
        checksum,
        error,
    )
    write_json_atomic(progress_path, payload, overwrite=True)
    write_json_atomic(summary_path, payload, overwrite=True)


def _aggregate_worker_progress(result_dir: Path) -> dict[str, Any]:
    rows = [_load_json(path) for path in sorted(result_dir.glob("worker_*_progress.json"))]
    rows = [row for row in rows if row]
    total = {
        "workers_seen": len(rows),
        "workers_passed": sum(row.get("status") == "passed" for row in rows),
        "workers_running": sum(row.get("status") == "running" for row in rows),
        "probability_evaluations": sum(int(row.get("probability_evaluations", 0)) for row in rows),
        "hits": sum(int(row.get("hits", 0)) for row in rows),
        "misses": sum(int(row.get("misses", 0)) for row in rows),
        "requests": sum(int(row.get("requests", 0)) for row in rows),
        "cache_resets": sum(int(row.get("cache_resets", 0)) for row in rows),
        "probability_failures": sum(int(row.get("probability_failures", 0)) for row in rows),
        "finite_failures": sum(int(row.get("finite_failures", 0)) for row in rows),
        "checksum": sum(float(row.get("checksum", 0.0)) for row in rows),
    }
    total["reuse_factor"] = float(total["requests"] / total["misses"]) if int(total["misses"]) else 0.0
    return total


def _throughput_stats(samples: list[dict[str, Any]], expected_workers: int) -> dict[str, Any]:
    full_load_samples = [row for row in samples if int(row.get("running_workers", 0)) == expected_workers]
    rates: list[float] = []
    for previous, current in zip(full_load_samples, full_load_samples[1:], strict=False):
        dt = float(current["elapsed_seconds"]) - float(previous["elapsed_seconds"])
        if dt <= 0.0:
            continue
        delta = int(current["aggregate"].get("probability_evaluations", 0)) - int(previous["aggregate"].get("probability_evaluations", 0))
        if delta > 0:
            rates.append(delta / dt)
    if len(rates) < 2:
        return {
            "rates": rates,
            "first_half_median": None,
            "second_half_median": None,
            "drop_fraction": None,
            "passes_throughput_gate": False,
        }
    midpoint = max(1, len(rates) // 2)
    first = statistics.median(rates[:midpoint])
    second = statistics.median(rates[midpoint:]) if rates[midpoint:] else rates[-1]
    drop = max(0.0, (first - second) / first) if first > 0.0 else math.inf
    return {
        "rates": rates,
        "first_half_median": first,
        "second_half_median": second,
        "drop_fraction": drop,
        "passes_throughput_gate": bool(drop <= TARGET_THROUGHPUT_DROP_FRACTION),
    }


def run_phase_c_120_operational_soak_test(
    duration_seconds: int = DEFAULT_DURATION_SECONDS,
    workers: int = DEFAULT_WORKERS,
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    result_dir = root / "results" / SOAK_TEST_NAME
    summary_path = result_dir / "operational_soak_test_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing soak test output: {summary_path}")
    if overwrite:
        _safe_reset(result_dir, root)
    ensure_dir(result_dir)

    thermal_before = _thermal_snapshot(root)
    ctx = mp.get_context("spawn")
    processes: list[mp.Process] = []
    for worker_index in range(int(workers)):
        proc = ctx.Process(
            target=_soak_worker,
            args=(worker_index, float(duration_seconds), float(sample_interval_seconds), str(root)),
            name=f"phase-c-120-soak-{worker_index:03d}",
        )
        proc.start()
        processes.append(proc)

    started = time.perf_counter()
    monitor_samples: list[dict[str, Any]] = []
    while any(proc.is_alive() for proc in processes):
        pids = [int(proc.pid) for proc in processes if proc.pid and proc.is_alive()]
        memory_samples = _process_memory_sample(pids, root)
        aggregate = _aggregate_worker_progress(result_dir)
        peak_working_set = max((int(row.get("PeakWorkingSet64") or 0) for row in memory_samples), default=0)
        sample = {
            "sampled_at_utc": _now_utc(),
            "elapsed_seconds": time.perf_counter() - started,
            "running_workers": len(pids),
            "aggregate": aggregate,
            "memory_samples": memory_samples,
            "peak_working_set_bytes": peak_working_set,
        }
        monitor_samples.append(sample)
        write_json_atomic(result_dir / "monitor_progress.json", {"samples": monitor_samples[-240:]}, overwrite=True)
        time.sleep(float(sample_interval_seconds))

    for proc in processes:
        proc.join(timeout=120)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=30)

    thermal_after = _thermal_snapshot(root)
    elapsed = time.perf_counter() - started
    worker_summaries = [_load_json(path) for path in sorted(result_dir.glob("worker_*_summary.json"))]
    aggregate = _aggregate_worker_progress(result_dir)
    exitcodes = [proc.exitcode for proc in processes]
    throughput = _throughput_stats(monitor_samples, int(workers))
    peak_memory = max((int(sample.get("peak_working_set_bytes", 0)) for sample in monitor_samples), default=0)
    ram_bytes = 15.8 * 1024**3
    memory_pass = bool(peak_memory > 0 and peak_memory < 0.80 * ram_bytes)
    workers_pass = bool(len(worker_summaries) == int(workers) and all(row.get("status") == "passed" for row in worker_summaries) and all(code == 0 for code in exitcodes))
    correctness_pass = bool(aggregate["probability_failures"] == 0 and aggregate["finite_failures"] == 0 and aggregate["probability_evaluations"] > 0)
    throughput_pass = bool(throughput.get("passes_throughput_gate") is True)
    status = "phase_c_120_operational_soak_test_passed" if (
        workers_pass and correctness_pass and throughput_pass and memory_pass
    ) else "phase_c_120_operational_soak_test_failed"
    summary = {
        "created_at_utc": _now_utc(),
        "soak_test_version": SOAK_TEST_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "custom_kernel_version": CUSTOM_KERNEL_VERSION,
        "status": status,
        "scope": "parallel_longer_soak_test_not_official_phase_c_data",
        "instance": DEFAULT_INSTANCE,
        "grid_size": TARGET_GRID_SIZE,
        "workers": int(workers),
        "requested_duration_seconds": int(duration_seconds),
        "observed_elapsed_seconds": elapsed,
        "sample_interval_seconds": float(sample_interval_seconds),
        "noise_values": list(DEFAULT_NOISE_VALUES),
        "scale_factors": list(DEFAULT_SCALE_FACTORS),
        "shot_budget": DEFAULT_SHOTS,
        "seed": DEFAULT_SEED,
        "cache_reset_misses": CACHE_RESET_MISSES,
        "aggregate": aggregate,
        "worker_exitcodes": exitcodes,
        "worker_summaries": worker_summaries,
        "monitor_samples_count": len(monitor_samples),
        "throughput": throughput,
        "memory": {
            "peak_working_set_bytes": peak_memory,
            "ram_reference_bytes": int(ram_bytes),
            "peak_working_set_fraction_of_ram": peak_memory / ram_bytes if peak_memory else None,
            "passes_80pct_ram_gate": memory_pass,
        },
        "thermal": {
            "before": thermal_before,
            "after": thermal_after,
            "status": "sensor_available" if thermal_before.get("available") or thermal_after.get("available") else "temperature_sensor_unavailable_throughput_proxy_only",
        },
        "checks": {
            "workers_passed": workers_pass,
            "probabilities_valid": correctness_pass,
            "throughput_drop_fraction_le_20pct": throughput_pass,
            "memory_gate_passed": memory_pass,
            "not_official_phase_c_data": True,
        },
        "decision": "phase_8_soak_gate_passed" if status.endswith("passed") else "do_not_start_phase_9_until_soak_gate_is_resolved",
        "limitations": [
            "This is a load/thermal proxy over custom-kernel probability evaluations, not official Phase C result data.",
            "It does not write official 120x120 landscapes.",
            "If Windows temperature sensors are unavailable, throughput stability is used as the thermal-throttling proxy.",
        ],
    }
    write_json_atomic(summary_path, summary, overwrite=True)
    return summary
