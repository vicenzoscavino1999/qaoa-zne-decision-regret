from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import CXGate, HGate, RXGate, RZGate
from qiskit_aer import AerSimulator

from .circuits import GraphInstance, build_qaoa_circuit, cut_value, graph_from_config
from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, write_json_atomic
from .landscapes import parameter_grid
from .mitiq_wrapper import allocate_shots, deterministic_seed, fold_global_odd_integer, richardson_weights
from .noise_models import build_depolarizing_noise_model
from .phase_c_120_preflight import PROTOCOL_VERSION


CUSTOM_KERNEL_VERSION = "phase_c_120_custom_kernel_v2"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_c_120_custom_kernel"
BASIS_GATES = ("rx", "rz", "h", "cx")
VALIDATION_INSTANCES = ("G1_C4", "G2_weighted4", "G3_weighted6")
VALIDATION_NOISE_VALUES = (0.0, 0.005, 0.01)
VALIDATION_SCALE_FACTORS = (1, 3, 5, 7)
VALIDATION_POINTS = (
    ("fixed_low", 0.123, 0.456),
    ("fixed_mid", 1.234, 0.789),
    ("fixed_high", 5.678, 2.345),
)
IDEAL_PROB_TOL = 1e-10
NOISY_PROB_TOL = 1e-8
OBS_TOL = 1e-8

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.asarray([[0, 1], [1, 0]], dtype=complex),
    "Y": np.asarray([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.asarray([[1, 0], [0, -1]], dtype=complex),
}


@dataclass(frozen=True)
class ProbabilityResult:
    probabilities: np.ndarray
    transpiled_gate_count: int
    transpiled_depth: int


@dataclass
class KernelProbabilityCache:
    data: dict[tuple[object, ...], ProbabilityResult]
    hits: int = 0
    misses: int = 0

    def get(
        self,
        graph: GraphInstance,
        theta_index: int,
        gamma: float,
        beta: float,
        p_noise: float,
        scale_factor: int,
    ) -> ProbabilityResult:
        key = (graph.name, int(theta_index), float(gamma), float(beta), float(p_noise), int(scale_factor))
        if key in self.data:
            self.hits += 1
            return self.data[key]
        self.misses += 1
        result = kernel_probabilities(graph, gamma, beta, p_noise, scale_factor)
        self.data[key] = result
        return result

    @property
    def requests(self) -> int:
        return self.hits + self.misses

    def summary(self) -> dict[str, int | float]:
        return {
            "entries": len(self.data),
            "hits": self.hits,
            "misses": self.misses,
            "requests": self.requests,
            "reuse_factor": float(self.requests / self.misses) if self.misses else 0.0,
        }


def new_probability_cache() -> KernelProbabilityCache:
    return KernelProbabilityCache(data={})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _transpile_for_kernel(circuit: QuantumCircuit, p_noise: float) -> QuantumCircuit:
    simulator = AerSimulator(noise_model=build_depolarizing_noise_model(p_noise))
    return transpile(
        circuit,
        simulator,
        basis_gates=list(BASIS_GATES),
        optimization_level=0,
        seed_transpiler=0,
    )


def _qargs_to_indices(circuit: QuantumCircuit, instruction: Any) -> tuple[int, ...]:
    return tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)


def _local_pauli_matrix(labels: tuple[str, ...]) -> np.ndarray:
    matrix = _PAULI[labels[-1]]
    for label in reversed(labels[:-1]):
        matrix = np.kron(matrix, _PAULI[label])
    return matrix


@lru_cache(maxsize=None)
def _target_blocks(n_qubits: int, targets: tuple[int, ...]) -> tuple[np.ndarray, ...]:
    dim = 1 << n_qubits
    mask = sum(1 << q for q in targets)
    blocks: list[np.ndarray] = []
    for base in range(dim):
        if base & mask:
            continue
        indices = []
        for local_index in range(1 << len(targets)):
            value = base
            for pos, qubit in enumerate(targets):
                if (local_index >> pos) & 1:
                    value |= 1 << qubit
            indices.append(value)
        blocks.append(np.asarray(indices, dtype=int))
    return tuple(blocks)


@lru_cache(maxsize=None)
def _target_blocks_array(n_qubits: int, targets: tuple[int, ...]) -> np.ndarray:
    blocks = _target_blocks(int(n_qubits), tuple(int(target) for target in targets))
    out = np.asarray(blocks, dtype=int)
    out.setflags(write=False)
    return out


def _apply_local_unitary_density_reference(rho: np.ndarray, gate: np.ndarray, targets: tuple[int, ...], n_qubits: int) -> np.ndarray:
    after_left = np.empty_like(rho)
    for indices in _target_blocks(n_qubits, targets):
        after_left[indices, :] = gate @ rho[indices, :]

    after_right = np.empty_like(rho)
    gate_dagger = gate.conj().T
    for indices in _target_blocks(n_qubits, targets):
        after_right[:, indices] = after_left[:, indices] @ gate_dagger
    return after_right


def _apply_local_unitary_density(rho: np.ndarray, gate: np.ndarray, targets: tuple[int, ...], n_qubits: int) -> np.ndarray:
    blocks = _target_blocks_array(int(n_qubits), tuple(int(target) for target in targets))
    after_left = np.empty_like(rho)
    after_left[blocks, :] = gate @ rho[blocks, :]

    after_right = np.empty_like(rho)
    after_right[:, blocks] = after_left[:, blocks] @ gate.conj().T
    return after_right


def _depolarizing_weights(param: float, n_targets: int) -> tuple[float, float]:
    num_terms = 4**n_targets
    max_param = num_terms / (num_terms - 1)
    return 1.0 - float(param) / max_param, float(param) / num_terms


def _apply_local_depolarizing_pauli_mixture(rho: np.ndarray, targets: tuple[int, ...], n_qubits: int, param: float) -> np.ndarray:
    if param <= 0.0:
        return rho
    n_targets = len(targets)
    prob_identity, prob_pauli = _depolarizing_weights(param, n_targets)
    acc = prob_identity * rho
    labels = ("I", "X", "Y", "Z")
    for flat in range(1, 4**n_targets):
        value = flat
        pauli_labels = []
        for _ in range(n_targets):
            pauli_labels.append(labels[value % 4])
            value //= 4
        gate = _local_pauli_matrix(tuple(pauli_labels))
        acc = acc + prob_pauli * _apply_local_unitary_density(rho, gate, targets, n_qubits)
    return acc


def _apply_local_depolarizing_reference(rho: np.ndarray, targets: tuple[int, ...], n_qubits: int, param: float) -> np.ndarray:
    if param <= 0.0:
        return rho
    ordered_targets = tuple(sorted(int(target) for target in targets))
    keep = tuple(qubit for qubit in range(n_qubits) if qubit not in ordered_targets)
    row_keep_axes = tuple(n_qubits - 1 - qubit for qubit in keep)
    row_target_axes = tuple(n_qubits - 1 - qubit for qubit in ordered_targets)
    col_keep_axes = tuple(n_qubits + n_qubits - 1 - qubit for qubit in keep)
    col_target_axes = tuple(n_qubits + n_qubits - 1 - qubit for qubit in ordered_targets)
    permutation = row_keep_axes + row_target_axes + col_keep_axes + col_target_axes
    inverse_permutation = np.argsort(permutation)
    keep_dim = 1 << len(keep)
    target_dim = 1 << len(ordered_targets)

    tensor = rho.reshape((2,) * (2 * n_qubits)).transpose(permutation)
    tensor = tensor.reshape(keep_dim, target_dim, keep_dim, target_dim)
    reduced = np.einsum("atbt->ab", tensor)
    mixed = np.zeros_like(tensor)
    for target_index in range(target_dim):
        mixed[:, target_index, :, target_index] = reduced / float(target_dim)
    mixed = mixed.reshape((2,) * (2 * n_qubits)).transpose(inverse_permutation).reshape(rho.shape)
    return (1.0 - float(param)) * rho + float(param) * mixed


def _apply_local_depolarizing(rho: np.ndarray, targets: tuple[int, ...], n_qubits: int, param: float) -> np.ndarray:
    return _apply_local_depolarizing_reference(rho, targets, n_qubits, param)


def _gate_matrix(name: str, params: list[Any]) -> np.ndarray:
    if name == "h":
        return HGate().to_matrix()
    if name == "rx":
        return RXGate(float(params[0])).to_matrix()
    if name == "rz":
        return RZGate(float(params[0])).to_matrix()
    if name == "cx":
        return CXGate().to_matrix()
    raise ValueError(f"Unsupported transpiled instruction for custom kernel: {name}")


def probabilities_from_transpiled_circuit(circuit: QuantumCircuit, p_noise: float) -> ProbabilityResult:
    n_qubits = circuit.num_qubits
    dim = 1 << n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    gate_count = 0

    for instruction in circuit.data:
        name = instruction.operation.name
        if name in {"barrier", "measure"}:
            continue
        targets = _qargs_to_indices(circuit, instruction)
        gate = _gate_matrix(name, list(instruction.operation.params))
        rho = _apply_local_unitary_density(rho, gate, targets, n_qubits)
        gate_count += 1
        if p_noise > 0.0:
            if len(targets) == 1 and name in {"h", "rx", "rz"}:
                rho = _apply_local_depolarizing(rho, targets, n_qubits, p_noise)
            elif len(targets) == 2 and name == "cx":
                rho = _apply_local_depolarizing(rho, targets, n_qubits, 2.0 * p_noise)
            else:
                raise ValueError(f"No depolarizing rule for instruction {name} on {len(targets)} qubits.")

    probs = np.real(np.diag(rho)).astype(float)
    min_prob = float(np.min(probs))
    if min_prob < -1e-10:
        raise RuntimeError(f"Custom kernel produced negative probability below tolerance: {min_prob}")
    probs = np.clip(probs, 0.0, 1.0)
    total = float(np.sum(probs))
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        probs = probs / total
    return ProbabilityResult(probabilities=probs, transpiled_gate_count=gate_count, transpiled_depth=int(circuit.depth() or 0))


def build_folded_transpiled_qaoa_uncached(graph: GraphInstance, gamma: float, beta: float, scale_factor: int, p_noise: float) -> QuantumCircuit:
    base = build_qaoa_circuit(graph, float(gamma), float(beta), measure=False)
    folded = fold_global_odd_integer(base, int(scale_factor))
    return _transpile_for_kernel(folded, p_noise)


def _build_parameterized_qaoa_circuit(graph: GraphInstance) -> tuple[QuantumCircuit, Parameter, Parameter]:
    gamma_param = Parameter("gamma")
    beta_param = Parameter("beta")
    qc = QuantumCircuit(graph.n_qubits)
    qc.h(range(graph.n_qubits))
    for u, v, weight in graph.edges:
        qc.rzz(-gamma_param * weight, u, v)
    for q in range(graph.n_qubits):
        qc.rx(2.0 * beta_param, q)
    return qc, gamma_param, beta_param


@lru_cache(maxsize=None)
def _folded_transpiled_qaoa_template(
    graph: GraphInstance,
    scale_factor: int,
    p_noise: float,
) -> tuple[QuantumCircuit, Parameter, Parameter]:
    base, gamma_param, beta_param = _build_parameterized_qaoa_circuit(graph)
    folded = fold_global_odd_integer(base, int(scale_factor))
    return _transpile_for_kernel(folded, float(p_noise)), gamma_param, beta_param


def build_folded_transpiled_qaoa(graph: GraphInstance, gamma: float, beta: float, scale_factor: int, p_noise: float) -> QuantumCircuit:
    template, gamma_param, beta_param = _folded_transpiled_qaoa_template(graph, int(scale_factor), float(p_noise))
    return template.assign_parameters({gamma_param: float(gamma), beta_param: float(beta)}, inplace=False)


def kernel_probabilities(graph: GraphInstance, gamma: float, beta: float, p_noise: float, scale_factor: int) -> ProbabilityResult:
    circuit = build_folded_transpiled_qaoa(graph, gamma, beta, scale_factor, p_noise)
    return probabilities_from_transpiled_circuit(circuit, p_noise)


def aer_density_probabilities(graph: GraphInstance, gamma: float, beta: float, p_noise: float, scale_factor: int) -> ProbabilityResult:
    circuit = build_folded_transpiled_qaoa(graph, gamma, beta, scale_factor, p_noise)
    circuit.save_probabilities()
    simulator = AerSimulator(noise_model=build_depolarizing_noise_model(p_noise), method="density_matrix")
    result = simulator.run(circuit).result()
    probs = np.asarray(result.data(0)["probabilities"], dtype=float)
    probs = probs / float(np.sum(probs))
    gate_count = sum(count for name, count in circuit.count_ops().items() if name not in {"save_probabilities", "barrier", "measure"})
    return ProbabilityResult(probabilities=probs, transpiled_gate_count=int(gate_count), transpiled_depth=int(circuit.depth() or 0))


def cut_values_vector(graph: GraphInstance) -> np.ndarray:
    dim = 1 << graph.n_qubits
    values = np.zeros(dim, dtype=float)
    for index in range(dim):
        bits = [(index >> q) & 1 for q in range(graph.n_qubits)]
        values[index] = cut_value(bits, graph.edges)
    return values


def expected_cut_from_probabilities(graph: GraphInstance, probabilities: np.ndarray) -> float:
    return float(np.dot(probabilities, cut_values_vector(graph)))


def sampled_expected_cut_from_probabilities(graph: GraphInstance, probabilities: np.ndarray, shots: int, seed: int) -> float:
    rng = np.random.default_rng(int(seed))
    counts = rng.multinomial(int(shots), probabilities)
    return float(np.dot(counts / float(shots), cut_values_vector(graph)))


def generate_custom_kernel_noisy_landscape(
    graph: GraphInstance,
    size: int,
    p_noise: float,
    shots_total: int,
    seed: int,
    method_tag: str = "phase_c_checkpoint_noisy",
    probability_cache: KernelProbabilityCache | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    gammas, betas = parameter_grid(size)
    cache = probability_cache or new_probability_cache()
    before = cache.summary()
    landscape = np.zeros((size, size), dtype=float)
    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):
            theta_index = i * size + j
            result = cache.get(graph, theta_index, float(gamma), float(beta), p_noise, 1)
            sample_seed = deterministic_seed(graph.name, p_noise, shots_total, seed, method_tag, "none", 1, theta_index)
            landscape[i, j] = -sampled_expected_cut_from_probabilities(graph, result.probabilities, shots_total, sample_seed)
    after = cache.summary()
    return landscape, {
        "custom_kernel_version": CUSTOM_KERNEL_VERSION,
        "engine": "custom_density_matrix_probability_reuse",
        "method": "noisy",
        "scale_factors": [1],
        "shots_total": int(shots_total),
        "seed": int(seed),
        "p_noise": float(p_noise),
        "cache_before": before,
        "cache_after": after,
        "cache_delta_hits": int(after["hits"]) - int(before["hits"]),
        "cache_delta_misses": int(after["misses"]) - int(before["misses"]),
        "sampling_rng": "numpy.default_rng.multinomial",
        "sampling_seed_policy": "deterministic_seed(instance,p,B_total,seed,method,order,scale,theta_index)",
    }


def generate_custom_kernel_zne_landscape(
    graph: GraphInstance,
    size: int,
    p_noise: float,
    shots_total: int,
    seed: int,
    scale_factors: list[int],
    zne_order: str,
    shot_allocation_strategy: str = "proportional_to_abs_weights",
    probability_cache: KernelProbabilityCache | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    gammas, betas = parameter_grid(size)
    weights = richardson_weights(scale_factors)
    shot_alloc = allocate_shots(shots_total, weights, strategy=shot_allocation_strategy)
    cache = probability_cache or new_probability_cache()
    before = cache.summary()
    landscape = np.zeros((size, size), dtype=float)
    gate_counts: dict[str, list[int]] = {str(scale): [] for scale in scale_factors}
    depths: dict[str, list[int]] = {str(scale): [] for scale in scale_factors}

    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):
            theta_index = i * size + j
            extrapolated = 0.0
            for scale, weight, shots in zip(scale_factors, weights, shot_alloc, strict=True):
                result = cache.get(graph, theta_index, float(gamma), float(beta), p_noise, int(scale))
                gate_counts[str(scale)].append(result.transpiled_gate_count)
                depths[str(scale)].append(result.transpiled_depth)
                sample_seed = deterministic_seed(graph.name, p_noise, shots_total, seed, "zne", zne_order, scale, theta_index)
                value = -sampled_expected_cut_from_probabilities(graph, result.probabilities, shots, sample_seed)
                extrapolated += weight * value
            landscape[i, j] = extrapolated

    after = cache.summary()
    return landscape, {
        "custom_kernel_version": CUSTOM_KERNEL_VERSION,
        "engine": "custom_density_matrix_probability_reuse",
        "method": f"zne_{zne_order}",
        "scale_factors": [int(scale) for scale in scale_factors],
        "folding_backend": "qiskit_local_global_u_u_dagger_u",
        "weights": weights,
        "shot_allocation": shot_alloc,
        "shot_allocation_strategy": shot_allocation_strategy,
        "shots_total": int(shots_total),
        "seed": int(seed),
        "p_noise": float(p_noise),
        "cache_before": before,
        "cache_after": after,
        "cache_delta_hits": int(after["hits"]) - int(before["hits"]),
        "cache_delta_misses": int(after["misses"]) - int(before["misses"]),
        "gate_count_mean": {scale: float(np.mean(values)) for scale, values in gate_counts.items()},
        "depth_mean": {scale: float(np.mean(values)) for scale, values in depths.items()},
        "sampling_rng": "numpy.default_rng.multinomial",
        "sampling_seed_policy": "deterministic_seed(instance,p,B_total,seed,method,order,scale,theta_index)",
    }


def kernel_sampled_landscape_value(
    graph: GraphInstance,
    gamma: float,
    beta: float,
    p_noise: float,
    shots: int,
    seed: int,
    method_tag: str,
    zne_order: str,
    scale_factor: int,
    theta_index: int,
) -> float:
    probs = kernel_probabilities(graph, gamma, beta, p_noise, scale_factor).probabilities
    sample_seed = deterministic_seed(graph.name, p_noise, shots, seed, method_tag, zne_order, scale_factor, theta_index)
    return -sampled_expected_cut_from_probabilities(graph, probs, shots, sample_seed)


def _validation_row(
    cfg: dict[str, Any],
    instance: str,
    point_name: str,
    gamma: float,
    beta: float,
    p_noise: float,
    scale_factor: int,
) -> dict[str, Any]:
    graph = graph_from_config(instance, cfg)
    started = time.perf_counter()
    kernel = kernel_probabilities(graph, gamma, beta, p_noise, scale_factor)
    kernel_seconds = time.perf_counter() - started
    started = time.perf_counter()
    aer = aer_density_probabilities(graph, gamma, beta, p_noise, scale_factor)
    aer_seconds = time.perf_counter() - started

    prob_diff = np.abs(kernel.probabilities - aer.probabilities)
    kernel_expected = expected_cut_from_probabilities(graph, kernel.probabilities)
    aer_expected = expected_cut_from_probabilities(graph, aer.probabilities)
    tolerance = IDEAL_PROB_TOL if p_noise == 0.0 else NOISY_PROB_TOL
    return {
        "instance": instance,
        "n_qubits": graph.n_qubits,
        "point_name": point_name,
        "gamma": gamma,
        "beta": beta,
        "p_noise": p_noise,
        "scale_factor": scale_factor,
        "kernel_seconds": kernel_seconds,
        "aer_density_seconds": aer_seconds,
        "kernel_gate_count": kernel.transpiled_gate_count,
        "aer_gate_count": aer.transpiled_gate_count,
        "kernel_depth": kernel.transpiled_depth,
        "aer_depth": aer.transpiled_depth,
        "kernel_prob_sum": float(np.sum(kernel.probabilities)),
        "aer_prob_sum": float(np.sum(aer.probabilities)),
        "max_abs_probability_diff": float(np.max(prob_diff)),
        "l1_probability_diff": float(np.sum(prob_diff)),
        "kernel_expected_cut": kernel_expected,
        "aer_expected_cut": aer_expected,
        "abs_expected_cut_diff": abs(kernel_expected - aer_expected),
        "probability_tolerance": tolerance,
        "observable_tolerance": OBS_TOL,
        "passes_probability_tolerance": bool(float(np.max(prob_diff)) <= tolerance),
        "passes_observable_tolerance": bool(abs(kernel_expected - aer_expected) <= OBS_TOL),
    }


def run_custom_kernel_validation(overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_c_120_custom_kernel")
    summary_path = output_dir / "custom_kernel_validation_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing custom-kernel validation output: {summary_path}")

    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for instance in VALIDATION_INSTANCES:
        for point_name, gamma, beta in VALIDATION_POINTS:
            for p_noise in VALIDATION_NOISE_VALUES:
                for scale_factor in VALIDATION_SCALE_FACTORS:
                    rows.append(_validation_row(cfg, instance, point_name, gamma, beta, p_noise, scale_factor))
                    _write_csv(output_dir / "probability_equivalence_partial.csv", rows)

    max_prob_diff = max(float(row["max_abs_probability_diff"]) for row in rows)
    max_obs_diff = max(float(row["abs_expected_cut_diff"]) for row in rows)
    probability_failures = [row for row in rows if not row["passes_probability_tolerance"]]
    observable_failures = [row for row in rows if not row["passes_observable_tolerance"]]
    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol_version": PROTOCOL_VERSION,
        "custom_kernel_version": CUSTOM_KERNEL_VERSION,
        "status": "custom_kernel_validation_passed" if not probability_failures and not observable_failures else "custom_kernel_validation_failed",
        "scope": "probability_and_observable_equivalence_against_aer_density_matrix_for_folded_transpiled_qaoa",
        "instances": list(VALIDATION_INSTANCES),
        "noise_values": list(VALIDATION_NOISE_VALUES),
        "scale_factors": list(VALIDATION_SCALE_FACTORS),
        "points": [{"name": name, "gamma": gamma, "beta": beta} for name, gamma, beta in VALIDATION_POINTS],
        "rows": len(rows),
        "max_abs_probability_diff": max_prob_diff,
        "max_abs_expected_cut_diff": max_obs_diff,
        "ideal_probability_tolerance": IDEAL_PROB_TOL,
        "noisy_probability_tolerance": NOISY_PROB_TOL,
        "observable_tolerance": OBS_TOL,
        "probability_failures": len(probability_failures),
        "observable_failures": len(observable_failures),
        "elapsed_seconds": time.perf_counter() - started,
        "decision": "custom_kernel_probability_engine_validated" if not probability_failures and not observable_failures else "do_not_use_custom_kernel_until_investigated",
        "limitations": [
            "This validates exact probability and observable equivalence, not full Phase C runner integration.",
            "Finite-shot NumPy sampling remains deterministic by seed but is not expected to be bit-identical to Aer sampling streams.",
            "Probability reuse is allowed only for probabilities; finite-shot sampled outputs must still be generated per official seed and shot budget.",
        ],
    }
    _write_csv(output_dir / "probability_equivalence.csv", rows)
    write_json_atomic(summary_path, report)
    return report
