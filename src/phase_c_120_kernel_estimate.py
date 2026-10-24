from __future__ import annotations

import ast
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, write_json_atomic
from .phase_c_120_preflight import PROTOCOL_VERSION, TARGET_GRID_SIZE, _write_csv, generate_task_inventory


KERNEL_ESTIMATE_VERSION = "phase_c_120_kernel_estimate_v1"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_c_120_kernel_estimate"
NO_REUSE_SCALE_SUM = 30
PROBABILITY_REUSE_SCALE_SUM = 16
SCALE_EVALUATIONS_PER_CONDITION = 10
TARGET_HOURS = {
    "full_run_gate": 72.0,
    "single_machine_preferred": 7.0 * 24.0,
    "single_machine_upper_soft": 14.0 * 24.0,
}


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _random_unitary(rng: np.random.Generator, dim: int) -> np.ndarray:
    matrix = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(matrix)
    phases = np.diag(r)
    phases = phases / np.where(np.abs(phases) == 0.0, 1.0, np.abs(phases))
    return (q * phases).astype(np.complex128)


def _random_density(rng: np.random.Generator, dim: int) -> np.ndarray:
    matrix = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = matrix @ matrix.conj().T
    rho /= np.trace(rho)
    return rho.astype(np.complex128)


def _apply_gate_on_axes(tensor: np.ndarray, gate: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    axes = tuple(int(axis) for axis in axes)
    moved = np.moveaxis(tensor, axes, range(len(axes)))
    front_dim = 2 ** len(axes)
    rest_shape = moved.shape[len(axes):]
    updated = gate @ moved.reshape(front_dim, -1)
    moved_updated = updated.reshape((2,) * len(axes) + rest_shape)
    return np.moveaxis(moved_updated, range(len(axes)), axes)


def _density_local_unitary(rho: np.ndarray, gate: np.ndarray, targets: tuple[int, ...], n_qubits: int) -> np.ndarray:
    tensor = rho.reshape((2,) * (2 * n_qubits))
    tensor = _apply_gate_on_axes(tensor, gate, targets)
    tensor = _apply_gate_on_axes(tensor, gate.conj(), tuple(n_qubits + target for target in targets))
    return tensor.reshape(rho.shape)


def _pauli_matrices() -> dict[str, np.ndarray]:
    return {
        "I": np.array([[1, 0], [0, 1]], dtype=np.complex128),
        "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
    }


def _density_1q_depolarize_proxy(rho: np.ndarray, target: int, n_qubits: int, p_noise: float) -> np.ndarray:
    paulis = _pauli_matrices()
    acc = (1.0 - p_noise) * rho
    for key in ("X", "Y", "Z"):
        acc = acc + (p_noise / 3.0) * _density_local_unitary(rho, paulis[key], (target,), n_qubits)
    return acc


def _density_2q_depolarize_proxy(rho: np.ndarray, targets: tuple[int, int], n_qubits: int, p_noise: float) -> np.ndarray:
    paulis = _pauli_matrices()
    acc = (1.0 - p_noise) * rho
    for left_name, left in paulis.items():
        for right_name, right in paulis.items():
            if left_name == "I" and right_name == "I":
                continue
            gate = np.kron(left, right).astype(np.complex128)
            acc = acc + (p_noise / 15.0) * _density_local_unitary(rho, gate, targets, n_qubits)
    return acc


def _bench(label: str, fn, min_seconds: float = 0.25, max_repeats: int = 1_000_000) -> dict[str, Any]:
    fn()
    repeats = 1
    elapsed = 0.0
    while repeats <= max_repeats:
        started = time.perf_counter()
        for _ in range(repeats):
            fn()
        elapsed = time.perf_counter() - started
        if elapsed >= min_seconds:
            break
        repeats *= 2
    return {
        "operation": label,
        "repeats": repeats,
        "elapsed_seconds": elapsed,
        "seconds_per_operation": elapsed / repeats,
        "operations_per_second": repeats / elapsed if elapsed > 0.0 else math.inf,
    }


def _micro_benchmarks() -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260511)
    rows: list[dict[str, Any]] = []
    for n_qubits in (4, 6):
        dim = 2**n_qubits
        unitary = _random_unitary(rng, dim)
        oneq_unitary = _random_unitary(rng, 2)
        twoq_unitary = _random_unitary(rng, 4)
        rho = _random_density(rng, dim)
        psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        psi = (psi / np.linalg.norm(psi)).astype(np.complex128)
        identity = np.eye(dim, dtype=np.complex128) / dim
        probabilities = np.abs(psi) ** 2
        probabilities /= probabilities.sum()
        cut_values = rng.normal(size=dim).astype(float)
        sample_rng = np.random.default_rng(271828 + n_qubits)

        density_holder = {"rho": rho.copy()}
        state_holder = {"psi": psi.copy()}

        def density_unitary() -> None:
            density_holder["rho"] = unitary @ density_holder["rho"] @ unitary.conj().T

        def density_global_depolarize() -> None:
            density_holder["rho"] = 0.995 * density_holder["rho"] + 0.005 * identity

        def density_local_1q_unitary() -> None:
            density_holder["rho"] = _density_local_unitary(density_holder["rho"], oneq_unitary, (0,), n_qubits)

        def density_local_2q_unitary() -> None:
            density_holder["rho"] = _density_local_unitary(density_holder["rho"], twoq_unitary, (0, 1), n_qubits)

        def density_local_1q_depolarize() -> None:
            density_holder["rho"] = _density_1q_depolarize_proxy(density_holder["rho"], 0, n_qubits, 0.005)

        def density_local_2q_depolarize() -> None:
            density_holder["rho"] = _density_2q_depolarize_proxy(density_holder["rho"], (0, 1), n_qubits, 0.01)

        def statevector_unitary() -> None:
            state_holder["psi"] = unitary @ state_holder["psi"]

        def diagonal_probabilities() -> None:
            probs = np.real(np.diag(density_holder["rho"]))
            probs = np.maximum(probs, 0.0)
            probs / np.sum(probs)

        def multinomial_expectation() -> None:
            counts = sample_rng.multinomial(1024, probabilities)
            float(np.dot(counts / 1024.0, cut_values))

        for row in [
            _bench("density_full_unitary_update", density_unitary),
            _bench("density_global_depolarize_proxy", density_global_depolarize),
            _bench("density_local_1q_unitary_update", density_local_1q_unitary),
            _bench("density_local_2q_unitary_update", density_local_2q_unitary),
            _bench("density_local_1q_depolarize_proxy", density_local_1q_depolarize),
            _bench("density_local_2q_depolarize_proxy", density_local_2q_depolarize),
            _bench("statevector_full_unitary_update", statevector_unitary),
            _bench("diagonal_probability_extract", diagonal_probabilities),
            _bench("multinomial_expectation_B1024", multinomial_expectation),
        ]:
            row["n_qubits"] = n_qubits
            row["dimension"] = dim
            row["created_at_utc"] = _now_utc()
            rows.append(row)
    return rows


def _load_runtime_model(root: Path) -> dict[str, Any]:
    path = root / "results" / "phase_c_120_profiling" / "runtime_model_120.json"
    if not path.exists():
        raise FileNotFoundError(f"Phase 1 profiling runtime model not found: {path}")
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_circuit_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "results" / "phase_c_120_planning" / "circuit_size_report.csv"
    if not path.exists():
        raise FileNotFoundError(f"Circuit size report not found: {path}")
    import csv

    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _base_gate_counts(circuit_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in circuit_rows:
        if row["circuit_kind"] != "base_unfolded_unmeasured":
            continue
        ops = ast.literal_eval(row["ops"])
        out[row["instance"]] = {
            "n_qubits": int(row["n_qubits"]),
            "n_edges": int(row["n_edges"]),
            "logical_gate_count_base": int(row["gate_count"]),
            "ops": ops,
        }
    return out


def _operation_lookup(rows: list[dict[str, Any]]) -> dict[tuple[int, str], float]:
    return {
        (int(row["n_qubits"]), str(row["operation"])): float(row["seconds_per_operation"])
        for row in rows
    }


def _runtime_total_from_seconds(serial_seconds: float, aer_local_75pct_hours: float) -> dict[str, Any]:
    local_workers = max(1, int(os.cpu_count() or 1))
    local_ideal_hours = serial_seconds / 3600.0 / local_workers
    local_75pct_hours = serial_seconds / 3600.0 / (local_workers * 0.75)
    return {
        "serial_hours": serial_seconds / 3600.0,
        "local_workers": local_workers,
        "local_ideal_parallel_hours": local_ideal_hours,
        "local_75pct_parallel_hours": local_75pct_hours,
        "speedup_vs_aer_local_75pct": aer_local_75pct_hours / local_75pct_hours if local_75pct_hours > 0 else math.inf,
        "passes_72h_gate": local_75pct_hours <= TARGET_HOURS["full_run_gate"],
        "passes_7d_single_machine": local_75pct_hours <= TARGET_HOURS["single_machine_preferred"],
        "passes_14d_single_machine": local_75pct_hours <= TARGET_HOURS["single_machine_upper_soft"],
    }


def _decision(summary: dict[str, Any]) -> str:
    reuse = summary["local_pauli_validated_probability_reuse"]
    no_reuse = summary["local_pauli_drop_in_no_probability_reuse"]
    if reuse["passes_72h_gate"]:
        return "custom_kernel_worth_implementing_if_equivalence_validates"
    if reuse["passes_14d_single_machine"]:
        return "custom_kernel_worth_implementing_but_sharding_may_still_be_needed"
    if no_reuse["passes_14d_single_machine"]:
        return "custom_kernel_possible_without_probability_reuse_but_needs_runner_sharding"
    return "custom_kernel_microbenchmark_insufficient_use_sharding_or_cloud"


def estimate_custom_kernel_runtime(
    benchmark_rows: list[dict[str, Any]],
    root: Path = PROJECT_ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    seeds_cfg = read_yaml(root / "config" / "seeds.yaml")
    task_inventory = generate_task_inventory(cfg, seeds_cfg)
    runtime_model = _load_runtime_model(root)
    circuit_rows = _load_circuit_rows(root)
    base_counts = _base_gate_counts(circuit_rows)
    op = _operation_lookup(benchmark_rows)
    grid_points = TARGET_GRID_SIZE * TARGET_GRID_SIZE
    noise_count = len(cfg["noise"]["p_values"])
    shot_budget_count = len(cfg["shots"]["total_budget_primary"])
    seed_count = int(task_inventory["seed_count"])
    conditions_per_instance = noise_count * shot_budget_count * seed_count
    aer_local_75 = float(
        runtime_model["runtime_estimates"]["max_measured_per_point_recommended_conservative"]["local_75pct_parallel_hours"]
    )

    estimate_rows: list[dict[str, Any]] = []
    drop_in_serial = 0.0
    reuse_probability_serial = 0.0
    local_drop_in_serial = 0.0
    local_reuse_probability_serial = 0.0
    sampling_serial = 0.0
    for instance, spec in sorted(base_counts.items()):
        n_qubits = int(spec["n_qubits"])
        base_gate_count = int(spec["logical_gate_count_base"])
        ops = spec["ops"]
        oneq_gate_count = int(ops.get("h", 0)) + int(ops.get("rx", 0))
        twoq_gate_count = int(ops.get("rzz", 0))
        gate_seconds = (
            op[(n_qubits, "density_full_unitary_update")]
            + op[(n_qubits, "density_global_depolarize_proxy")]
        )
        local_gate_seconds_per_scale = (
            oneq_gate_count
            * (
                op[(n_qubits, "density_local_1q_unitary_update")]
                + op[(n_qubits, "density_local_1q_depolarize_proxy")]
            )
            + twoq_gate_count
            * (
                op[(n_qubits, "density_local_2q_unitary_update")]
                + op[(n_qubits, "density_local_2q_depolarize_proxy")]
            )
        )
        sampling_seconds = (
            op[(n_qubits, "diagonal_probability_extract")]
            + op[(n_qubits, "multinomial_expectation_B1024")]
        )
        drop_in_one_condition = grid_points * base_gate_count * NO_REUSE_SCALE_SUM * gate_seconds
        reuse_one_noise = grid_points * base_gate_count * PROBABILITY_REUSE_SCALE_SUM * gate_seconds
        local_drop_in_one_condition = grid_points * NO_REUSE_SCALE_SUM * local_gate_seconds_per_scale
        local_reuse_one_noise = grid_points * PROBABILITY_REUSE_SCALE_SUM * local_gate_seconds_per_scale
        sampling_all_conditions = (
            grid_points
            * SCALE_EVALUATIONS_PER_CONDITION
            * conditions_per_instance
            * sampling_seconds
        )
        drop_in_instance = drop_in_one_condition * conditions_per_instance
        reuse_instance = reuse_one_noise * noise_count
        local_drop_in_instance = local_drop_in_one_condition * conditions_per_instance
        local_reuse_instance = local_reuse_one_noise * noise_count
        drop_in_serial += drop_in_instance
        reuse_probability_serial += reuse_instance
        local_drop_in_serial += local_drop_in_instance
        local_reuse_probability_serial += local_reuse_instance
        sampling_serial += sampling_all_conditions
        estimate_rows.append({
            "instance": instance,
            "n_qubits": n_qubits,
            "base_logical_gates": base_gate_count,
            "oneq_logical_gates": oneq_gate_count,
            "twoq_logical_gates": twoq_gate_count,
            "density_gate_proxy_seconds": gate_seconds,
            "local_gate_proxy_seconds_per_scale1_base_circuit": local_gate_seconds_per_scale,
            "sampling_proxy_seconds": sampling_seconds,
            "dense_drop_in_no_probability_reuse_seconds": drop_in_instance,
            "dense_validated_probability_reuse_compute_seconds": reuse_instance,
            "local_pauli_drop_in_no_probability_reuse_seconds": local_drop_in_instance,
            "local_pauli_validated_probability_reuse_compute_seconds": local_reuse_instance,
            "sampling_all_outputs_seconds": sampling_all_conditions,
            "grid_points": grid_points,
            "noise_count": noise_count,
            "conditions_per_instance": conditions_per_instance,
        })

    total_drop_in = drop_in_serial + sampling_serial
    total_reuse = reuse_probability_serial + sampling_serial
    total_local_drop_in = local_drop_in_serial + sampling_serial
    total_local_reuse = local_reuse_probability_serial + sampling_serial
    totals = {
        "created_at_utc": _now_utc(),
        "protocol_version": PROTOCOL_VERSION,
        "kernel_estimate_version": KERNEL_ESTIMATE_VERSION,
        "benchmark_interpretation": "Microbenchmark only. It estimates feasibility; it does not validate scientific equivalence.",
        "target_grid_size": TARGET_GRID_SIZE,
        "grid_points": grid_points,
        "aer_local_75pct_parallel_hours": aer_local_75,
        "required_speedup_vs_aer_for_72h": aer_local_75 / TARGET_HOURS["full_run_gate"],
        "assumptions": {
            "density_full_unitary_update": "Proxy for applying a local gate to a density matrix. Conservative if the final kernel uses tensor-local updates.",
            "density_global_depolarize_proxy": "Timing proxy only; final kernel must implement the exact Aer-equivalent local depolarizing channel.",
            "local_pauli_proxy": "Uses tensor-axis local 1q/2q unitary updates and explicit Pauli-mixture depolarizing proxies.",
            "drop_in_no_probability_reuse": "Recomputes every noisy/ZNE scale evaluation per seed and shot budget.",
            "validated_probability_reuse": "Computes probability landscapes once per instance/noise/scale and reuses them for seeds and shot budgets; allowed only after equivalence validation.",
            "sampling": "Finite-shot outputs still require deterministic multinomial draws per official seed/budget/method/scale.",
        },
        "raw_seconds": {
            "dense_drop_in_compute": drop_in_serial,
            "dense_validated_probability_reuse_compute": reuse_probability_serial,
            "local_pauli_drop_in_compute": local_drop_in_serial,
            "local_pauli_validated_probability_reuse_compute": local_reuse_probability_serial,
            "sampling_all_outputs": sampling_serial,
        },
        "dense_drop_in_no_probability_reuse": _runtime_total_from_seconds(total_drop_in, aer_local_75),
        "dense_validated_probability_reuse": _runtime_total_from_seconds(total_reuse, aer_local_75),
        "local_pauli_drop_in_no_probability_reuse": _runtime_total_from_seconds(total_local_drop_in, aer_local_75),
        "local_pauli_validated_probability_reuse": _runtime_total_from_seconds(total_local_reuse, aer_local_75),
    }
    totals["phase2_decision"] = _decision(totals)
    return estimate_rows, totals


def run_phase_c_120_kernel_estimate(
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_c_120_kernel_estimate")
    summary_path = output_dir / "kernel_estimate_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing kernel estimate output: {summary_path}")

    benchmark_rows = _micro_benchmarks()
    estimate_rows, totals = estimate_custom_kernel_runtime(benchmark_rows, root=root)
    report = {
        "created_at_utc": _now_utc(),
        "protocol_version": PROTOCOL_VERSION,
        "kernel_estimate_version": KERNEL_ESTIMATE_VERSION,
        "status": "phase_c_120_kernel_estimate_complete",
        "phase2_decision": totals["phase2_decision"],
        "dense_drop_in_local_75pct_parallel_hours": totals["dense_drop_in_no_probability_reuse"]["local_75pct_parallel_hours"],
        "dense_reuse_local_75pct_parallel_hours": totals["dense_validated_probability_reuse"]["local_75pct_parallel_hours"],
        "local_drop_in_local_75pct_parallel_hours": totals["local_pauli_drop_in_no_probability_reuse"]["local_75pct_parallel_hours"],
        "local_reuse_local_75pct_parallel_hours": totals["local_pauli_validated_probability_reuse"]["local_75pct_parallel_hours"],
        "local_reuse_speedup_vs_aer": totals["local_pauli_validated_probability_reuse"]["speedup_vs_aer_local_75pct"],
        "passes_72h_gate_with_local_reuse": totals["local_pauli_validated_probability_reuse"]["passes_72h_gate"],
        "outputs": {
            "micro_benchmark": "micro_benchmark.csv",
            "kernel_runtime_estimates": "kernel_runtime_estimates.csv",
            "kernel_runtime_model": "kernel_runtime_model.json",
        },
    }
    _write_csv(output_dir / "micro_benchmark.csv", benchmark_rows)
    _write_csv(output_dir / "kernel_runtime_estimates.csv", estimate_rows)
    write_json_atomic(output_dir / "kernel_runtime_model.json", totals)
    write_json_atomic(summary_path, report)
    return report
