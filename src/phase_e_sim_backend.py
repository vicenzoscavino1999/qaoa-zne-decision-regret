from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from .circuits import GraphInstance, cut_value, qiskit_count_key_to_bits
from .mitiq_wrapper import allocate_shots, deterministic_seed, fold_global_odd_integer, richardson_weights
from .phase_c_120_custom_kernel import (
    _apply_local_depolarizing,
    _apply_local_unitary_density,
    _gate_matrix,
    _qargs_to_indices,
    cut_values_vector,
)


PHASE_E_BACKEND_VERSION = "phase_e_custom_density_backend_v0_2"
PHASE_E_BASIS_GATES = ("rx", "rz", "h", "cx")
DEFAULT_GAMMA_PERIOD = 2.0 * math.pi
DEFAULT_BETA_PERIOD = math.pi


@dataclass(frozen=True)
class PhaseENoiseRates:
    p_dep: float
    p_1q: float
    p_2q: float
    model_version: str = "phase_e_per_gate_depolarizing_v0_1"

    @classmethod
    def from_p_dep(cls, p_dep: float, p_1q_ratio: float = 0.1) -> "PhaseENoiseRates":
        p_dep = float(p_dep)
        if p_dep < 0.0:
            raise ValueError("p_dep must be non-negative.")
        return cls(p_dep=p_dep, p_1q=p_dep * float(p_1q_ratio), p_2q=p_dep)


@dataclass(frozen=True)
class PhaseEProbabilityResult:
    probabilities: np.ndarray
    transpiled_gate_count: int
    transpiled_depth: int
    backend_version: str
    noise_rates: PhaseENoiseRates
    scale_factor: int
    qaoa_depth: int


@dataclass(frozen=True)
class PhaseEObjectiveResult:
    objective_value: float
    expected_cut: float
    shot_variance: float | None
    probabilities: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class PhaseEZNEResult:
    objective_value: float
    weights: list[float]
    scale_factors: list[int]
    shot_allocation: list[int] | None
    per_scale_values: list[float]
    per_scale_variances: list[float | None]
    variance_estimate: float | None
    derivation_record: dict[str, object]


@dataclass(frozen=True)
class PhaseELHSPointResult:
    ideal_objective: float
    noisy_objective: float
    zne_objective: float
    weights: list[float]
    scale_factors: list[int]
    shot_allocation: list[int]
    per_scale_values: list[float]
    per_scale_variances: list[float | None]


@dataclass(frozen=True)
class PhaseELHSCrossOrderResult:
    ideal_objective: float
    noisy_objective: float
    zne_objectives_by_order: dict[str, float]
    weights_by_order: dict[str, list[float]]
    scale_factors_by_order: dict[str, list[int]]
    shot_allocation_by_order: dict[str, list[int]]
    per_scale_values_by_order: dict[str, list[float]]
    per_scale_variances_by_order: dict[str, list[float | None]]
    probability_scale_factors: list[int]


def normalize_theta(theta: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(x) for x in theta)
    if len(values) == 0 or len(values) % 2 != 0:
        raise ValueError("theta must contain gamma,beta pairs.")
    return values


def qaoa_depth_from_theta(theta: Sequence[float]) -> int:
    return len(normalize_theta(theta)) // 2


def theta_periods(depth: int) -> tuple[float, ...]:
    periods: list[float] = []
    for _ in range(int(depth)):
        periods.extend([DEFAULT_GAMMA_PERIOD, DEFAULT_BETA_PERIOD])
    return tuple(periods)


def wrap_theta(theta: Sequence[float]) -> tuple[float, ...]:
    values = normalize_theta(theta)
    periods = theta_periods(len(values) // 2)
    return tuple(float(value % period) for value, period in zip(values, periods, strict=True))


def build_qaoa_circuit_depth(graph: GraphInstance, theta: Sequence[float], measure: bool = False) -> QuantumCircuit:
    """Build depth-q QAOA using theta=(gamma_1,beta_1,...,gamma_q,beta_q)."""
    values = normalize_theta(theta)
    qc = QuantumCircuit(graph.n_qubits)
    qc.h(range(graph.n_qubits))
    for layer in range(len(values) // 2):
        gamma = values[2 * layer]
        beta = values[2 * layer + 1]
        for u, v, weight in graph.edges:
            qc.rzz(-gamma * weight, u, v)
        for qubit in range(graph.n_qubits):
            qc.rx(2.0 * beta, qubit)
    if measure:
        qc.measure_all()
    return qc


def _build_parameterized_qaoa_circuit_depth(
    graph: GraphInstance,
    qaoa_depth: int,
) -> tuple[QuantumCircuit, tuple[object, ...]]:
    depth = int(qaoa_depth)
    if depth <= 0:
        raise ValueError("qaoa_depth must be positive.")
    gammas = ParameterVector("gamma", depth)
    betas = ParameterVector("beta", depth)
    qc = QuantumCircuit(graph.n_qubits)
    qc.h(range(graph.n_qubits))
    theta_parameters: list[object] = []
    for layer in range(depth):
        gamma_param = gammas[layer]
        beta_param = betas[layer]
        theta_parameters.extend([gamma_param, beta_param])
        for u, v, weight in graph.edges:
            qc.rzz(-gamma_param * weight, u, v)
        for qubit in range(graph.n_qubits):
            qc.rx(2.0 * beta_param, qubit)
    return qc, tuple(theta_parameters)


def build_phase_e_depolarizing_noise_model(noise_rates: PhaseENoiseRates) -> NoiseModel:
    noise_model = NoiseModel()
    if noise_rates.p_1q > 0.0:
        one_qubit = depolarizing_error(float(noise_rates.p_1q), 1)
        noise_model.add_all_qubit_quantum_error(one_qubit, ["h", "rx", "rz"])
    if noise_rates.p_2q > 0.0:
        two_qubit = depolarizing_error(float(noise_rates.p_2q), 2)
        noise_model.add_all_qubit_quantum_error(two_qubit, ["cx"])
    return noise_model


def _transpile_for_phase_e(circuit: QuantumCircuit, noise_rates: PhaseENoiseRates) -> QuantumCircuit:
    simulator = AerSimulator(noise_model=build_phase_e_depolarizing_noise_model(noise_rates))
    return transpile(
        circuit,
        simulator,
        basis_gates=list(PHASE_E_BASIS_GATES),
        optimization_level=0,
        seed_transpiler=0,
    )


@lru_cache(maxsize=None)
def _folded_transpiled_qaoa_depth_template(
    graph: GraphInstance,
    qaoa_depth: int,
    scale_factor: int,
    noise_rates: PhaseENoiseRates,
) -> tuple[QuantumCircuit, tuple[object, ...]]:
    base, theta_parameters = _build_parameterized_qaoa_circuit_depth(graph, int(qaoa_depth))
    folded = fold_global_odd_integer(base, int(scale_factor))
    return _transpile_for_phase_e(folded, noise_rates), theta_parameters


def build_folded_transpiled_qaoa_depth(
    graph: GraphInstance,
    theta: Sequence[float],
    scale_factor: int,
    noise_rates: PhaseENoiseRates,
) -> QuantumCircuit:
    values = wrap_theta(theta)
    template, theta_parameters = _folded_transpiled_qaoa_depth_template(
        graph,
        len(values) // 2,
        int(scale_factor),
        noise_rates,
    )
    return template.assign_parameters(dict(zip(theta_parameters, values, strict=True)), inplace=False)


def probabilities_from_transpiled_circuit_phase_e(
    circuit: QuantumCircuit,
    noise_rates: PhaseENoiseRates,
    scale_factor: int,
    qaoa_depth: int,
) -> PhaseEProbabilityResult:
    n_qubits = circuit.num_qubits
    dim = 1 << n_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    gate_count = 0

    for instruction in circuit.data:
        name = instruction.operation.name
        if name in {"barrier", "measure", "save_probabilities"}:
            continue
        targets = _qargs_to_indices(circuit, instruction)
        gate = _gate_matrix(name, list(instruction.operation.params))
        rho = _apply_local_unitary_density(rho, gate, targets, n_qubits)
        gate_count += 1
        if len(targets) == 1 and name in {"h", "rx", "rz"}:
            rho = _apply_local_depolarizing(rho, targets, n_qubits, noise_rates.p_1q)
        elif len(targets) == 2 and name == "cx":
            rho = _apply_local_depolarizing(rho, targets, n_qubits, noise_rates.p_2q)
        else:
            raise ValueError(f"No Phase E depolarizing rule for instruction {name} on {len(targets)} qubits.")

    probs = np.real(np.diag(rho)).astype(float)
    min_prob = float(np.min(probs))
    if min_prob < -1e-10:
        raise RuntimeError(f"Phase E custom backend produced negative probability below tolerance: {min_prob}")
    probs = np.clip(probs, 0.0, 1.0)
    total = float(np.sum(probs))
    if total <= 0.0:
        raise RuntimeError("Phase E custom backend produced zero total probability.")
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        probs = probs / total
    return PhaseEProbabilityResult(
        probabilities=probs,
        transpiled_gate_count=gate_count,
        transpiled_depth=int(circuit.depth() or 0),
        backend_version=PHASE_E_BACKEND_VERSION,
        noise_rates=noise_rates,
        scale_factor=int(scale_factor),
        qaoa_depth=int(qaoa_depth),
    )


def custom_density_probabilities(
    graph: GraphInstance,
    theta: Sequence[float],
    p_dep: float,
    scale_factor: int = 1,
    p_1q_ratio: float = 0.1,
) -> PhaseEProbabilityResult:
    values = wrap_theta(theta)
    noise_rates = PhaseENoiseRates.from_p_dep(p_dep, p_1q_ratio=p_1q_ratio)
    circuit = build_folded_transpiled_qaoa_depth(graph, values, scale_factor, noise_rates)
    return probabilities_from_transpiled_circuit_phase_e(
        circuit,
        noise_rates,
        scale_factor=int(scale_factor),
        qaoa_depth=len(values) // 2,
    )


def aer_density_probabilities_phase_e(
    graph: GraphInstance,
    theta: Sequence[float],
    p_dep: float,
    scale_factor: int = 1,
    p_1q_ratio: float = 0.1,
) -> PhaseEProbabilityResult:
    values = wrap_theta(theta)
    noise_rates = PhaseENoiseRates.from_p_dep(p_dep, p_1q_ratio=p_1q_ratio)
    circuit = build_folded_transpiled_qaoa_depth(graph, values, scale_factor, noise_rates)
    circuit.save_probabilities()
    simulator = AerSimulator(noise_model=build_phase_e_depolarizing_noise_model(noise_rates), method="density_matrix")
    result = simulator.run(circuit).result()
    probs = np.asarray(result.data(0)["probabilities"], dtype=float)
    probs = probs / float(np.sum(probs))
    gate_count = sum(count for name, count in circuit.count_ops().items() if name not in {"save_probabilities", "barrier", "measure"})
    return PhaseEProbabilityResult(
        probabilities=probs,
        transpiled_gate_count=int(gate_count),
        transpiled_depth=int(circuit.depth() or 0),
        backend_version="qiskit_aer_density_matrix_reference",
        noise_rates=noise_rates,
        scale_factor=int(scale_factor),
        qaoa_depth=len(values) // 2,
    )


def statevector_probabilities_phase_e(
    graph: GraphInstance,
    theta: Sequence[float],
    scale_factor: int = 1,
) -> np.ndarray:
    base = build_qaoa_circuit_depth(graph, wrap_theta(theta), measure=False)
    folded = fold_global_odd_integer(base, int(scale_factor))
    state = Statevector.from_instruction(folded)
    probs = np.asarray(state.probabilities(), dtype=float)
    return probs / float(np.sum(probs))


@lru_cache(maxsize=None)
def _cached_cut_values_vector(graph: GraphInstance) -> np.ndarray:
    values = np.asarray(cut_values_vector(graph), dtype=float)
    values.setflags(write=False)
    return values


def expected_cut_from_phase_e_probabilities(graph: GraphInstance, probabilities: np.ndarray) -> float:
    return float(np.dot(probabilities, _cached_cut_values_vector(graph)))


def shot_variance_from_probabilities(graph: GraphInstance, probabilities: np.ndarray, shots: int) -> float:
    values = _cached_cut_values_vector(graph)
    mean = float(np.dot(probabilities, values))
    variance_single_shot = float(np.dot(probabilities, (values - mean) ** 2))
    return variance_single_shot / float(shots)


def sampled_expected_cut_from_probabilities(
    graph: GraphInstance,
    probabilities: np.ndarray,
    shots: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(int(seed))
    counts = rng.multinomial(int(shots), probabilities)
    return float(np.dot(counts / float(shots), _cached_cut_values_vector(graph)))


def evaluate_objective_from_probabilities(
    graph: GraphInstance,
    probabilities: np.ndarray,
    shots: int | None = None,
    seed: int | None = None,
) -> PhaseEObjectiveResult:
    if shots is None:
        expected_cut = expected_cut_from_phase_e_probabilities(graph, probabilities)
        shot_variance = None
        shot_seed = None
    else:
        if seed is None:
            raise ValueError("seed is required when shots is not None.")
        expected_cut = sampled_expected_cut_from_probabilities(graph, probabilities, int(shots), int(seed))
        shot_variance = shot_variance_from_probabilities(graph, probabilities, int(shots))
        shot_seed = int(seed)
    return PhaseEObjectiveResult(
        objective_value=-float(expected_cut),
        expected_cut=float(expected_cut),
        shot_variance=shot_variance,
        probabilities=probabilities,
        metadata={
            "shots": None if shots is None else int(shots),
            "shot_seed": shot_seed,
            "objective_convention": "minimize_negative_expected_cut",
        },
    )


def evaluate_custom_density_objective(
    graph: GraphInstance,
    theta: Sequence[float],
    p_dep: float,
    scale_factor: int = 1,
    shots: int | None = None,
    seed: int | None = None,
    p_1q_ratio: float = 0.1,
) -> PhaseEObjectiveResult:
    result = custom_density_probabilities(graph, theta, p_dep=p_dep, scale_factor=scale_factor, p_1q_ratio=p_1q_ratio)
    objective = evaluate_objective_from_probabilities(graph, result.probabilities, shots=shots, seed=seed)
    metadata = dict(objective.metadata)
    metadata.update(
        {
            "backend_version": result.backend_version,
            "noise_model_version": result.noise_rates.model_version,
            "p_dep": float(p_dep),
            "p_1q": float(result.noise_rates.p_1q),
            "p_2q": float(result.noise_rates.p_2q),
            "scale_factor": int(scale_factor),
            "qaoa_depth": int(result.qaoa_depth),
            "transpiled_gate_count": int(result.transpiled_gate_count),
            "transpiled_depth": int(result.transpiled_depth),
        }
    )
    return PhaseEObjectiveResult(
        objective_value=objective.objective_value,
        expected_cut=objective.expected_cut,
        shot_variance=objective.shot_variance,
        probabilities=objective.probabilities,
        metadata=metadata,
    )


def richardson_zne_objective(
    graph: GraphInstance,
    theta: Sequence[float],
    p_dep: float,
    scale_factors: Iterable[int],
    shots_total: int | None = None,
    seed: int | None = None,
    shot_allocation_strategy: str = "proportional_to_abs_weights",
    p_1q_ratio: float = 0.1,
) -> PhaseEZNEResult:
    scales = [int(scale) for scale in scale_factors]
    weights = richardson_weights(scales)
    if shots_total is None:
        shot_allocation = None
    else:
        shot_allocation = allocate_shots(int(shots_total), weights, strategy=shot_allocation_strategy)

    values: list[float] = []
    variances: list[float | None] = []
    per_scale_metadata: list[dict[str, object]] = []
    for index, scale in enumerate(scales):
        shots = None if shot_allocation is None else int(shot_allocation[index])
        shot_seed = None
        if shots is not None:
            if seed is None:
                raise ValueError("seed is required when shots_total is provided.")
            shot_seed = deterministic_seed("phase_e_zne", graph.name, tuple(wrap_theta(theta)), p_dep, shots_total, seed, scale)
        objective = evaluate_custom_density_objective(
            graph,
            theta,
            p_dep=p_dep,
            scale_factor=scale,
            shots=shots,
            seed=shot_seed,
            p_1q_ratio=p_1q_ratio,
        )
        values.append(float(objective.objective_value))
        variances.append(objective.shot_variance)
        per_scale_metadata.append(objective.metadata)

    zne_value = float(sum(weight * value for weight, value in zip(weights, values, strict=True)))
    if any(variance is None for variance in variances):
        variance_estimate = None
    else:
        variance_estimate = float(sum((weight**2) * float(variance) for weight, variance in zip(weights, variances, strict=True)))

    derivation = {
        "backend_version": PHASE_E_BACKEND_VERSION,
        "graph": graph.name,
        "theta_json": list(wrap_theta(theta)),
        "p_dep": float(p_dep),
        "p_1q_ratio": float(p_1q_ratio),
        "zne_method": "richardson",
        "scale_factors_json": scales,
        "richardson_coefficients_json": weights,
        "per_scale_values_json": values,
        "shot_allocation_json": shot_allocation,
        "per_scale_variances_json": variances,
        "variance_estimate": variance_estimate,
        "per_scale_metadata": per_scale_metadata,
    }
    return PhaseEZNEResult(
        objective_value=zne_value,
        weights=weights,
        scale_factors=scales,
        shot_allocation=shot_allocation,
        per_scale_values=values,
        per_scale_variances=variances,
        variance_estimate=variance_estimate,
        derivation_record=derivation,
    )


def evaluate_lhs_point_with_scale1_probability_reuse(
    graph: GraphInstance,
    theta: Sequence[float],
    p_dep: float,
    scale_factors: Iterable[int],
    shots_total: int,
    noisy_seed: int,
    zne_seed: int,
    p_1q_ratio: float = 0.1,
) -> PhaseELHSPointResult:
    """Evaluate one LHS point while reusing the shared scale=1 probability vector.

    The noisy objective and Richardson scale=1 objective have different shot
    seeds, but the exact noisy probability vector is identical for the same
    graph, theta, p_dep and scale factor. Reusing only that probability vector
    preserves PRNG streams and sampled values while avoiding one density pass.
    """
    scales = [int(scale) for scale in scale_factors]
    if 1 not in scales:
        raise ValueError("scale_factors must include 1 for scale=1 probability reuse.")
    weights = richardson_weights(scales)
    shot_allocation = allocate_shots(int(shots_total), weights, strategy="proportional_to_abs_weights")

    ideal = evaluate_custom_density_objective(graph, theta, p_dep=0.0, p_1q_ratio=p_1q_ratio)
    scale1_probabilities = custom_density_probabilities(
        graph,
        theta,
        p_dep=p_dep,
        scale_factor=1,
        p_1q_ratio=p_1q_ratio,
    ).probabilities
    noisy = evaluate_objective_from_probabilities(
        graph,
        scale1_probabilities,
        shots=int(shots_total),
        seed=int(noisy_seed),
    )

    values: list[float] = []
    variances: list[float | None] = []
    for index, scale in enumerate(scales):
        shot_seed = deterministic_seed("phase_e_zne", graph.name, tuple(wrap_theta(theta)), p_dep, shots_total, int(zne_seed), scale)
        if scale == 1:
            probabilities = scale1_probabilities
        else:
            probabilities = custom_density_probabilities(
                graph,
                theta,
                p_dep=p_dep,
                scale_factor=scale,
                p_1q_ratio=p_1q_ratio,
            ).probabilities
        objective = evaluate_objective_from_probabilities(
            graph,
            probabilities,
            shots=int(shot_allocation[index]),
            seed=shot_seed,
        )
        values.append(float(objective.objective_value))
        variances.append(objective.shot_variance)

    zne_value = float(sum(weight * value for weight, value in zip(weights, values, strict=True)))
    return PhaseELHSPointResult(
        ideal_objective=float(ideal.objective_value),
        noisy_objective=float(noisy.objective_value),
        zne_objective=zne_value,
        weights=[float(weight) for weight in weights],
        scale_factors=scales,
        shot_allocation=[int(value) for value in shot_allocation],
        per_scale_values=values,
        per_scale_variances=variances,
    )


def evaluate_lhs_point_cross_order_probability_cache(
    graph: GraphInstance,
    theta: Sequence[float],
    p_dep: float,
    scale_factors_by_order: Mapping[str, Iterable[int]],
    shots_total: int,
    noisy_seed: int,
    zne_seed: int,
    p_1q_ratio: float = 0.1,
) -> PhaseELHSCrossOrderResult:
    """Evaluate one LHS point while caching exact probabilities across ZNE orders.

    This is an implementation-only optimization for fixed LHS robustness cells.
    It preserves the same Richardson weights, per-order shot allocation, and
    per-scale shot seeds that separate calls to ``richardson_zne_objective``
    would use. Only exact density probability vectors are reused.
    """
    values = wrap_theta(theta)
    normalized_scales = {
        str(order): [int(scale) for scale in scales]
        for order, scales in scale_factors_by_order.items()
    }
    if not normalized_scales:
        raise ValueError("scale_factors_by_order must contain at least one order.")
    for order, scales in normalized_scales.items():
        if not scales:
            raise ValueError(f"scale_factors_by_order[{order!r}] must not be empty.")
        if 1 not in scales:
            raise ValueError(f"scale_factors_by_order[{order!r}] must include scale factor 1.")

    probability_scale_factors = sorted({scale for scales in normalized_scales.values() for scale in scales})
    ideal_probabilities = custom_density_probabilities(
        graph,
        values,
        p_dep=0.0,
        scale_factor=1,
        p_1q_ratio=p_1q_ratio,
    ).probabilities
    probability_cache = {
        scale: custom_density_probabilities(
            graph,
            values,
            p_dep=p_dep,
            scale_factor=scale,
            p_1q_ratio=p_1q_ratio,
        ).probabilities
        for scale in probability_scale_factors
    }

    ideal = evaluate_objective_from_probabilities(graph, ideal_probabilities)
    noisy = evaluate_objective_from_probabilities(
        graph,
        probability_cache[1],
        shots=int(shots_total),
        seed=int(noisy_seed),
    )

    zne_objectives_by_order: dict[str, float] = {}
    weights_by_order: dict[str, list[float]] = {}
    shot_allocation_by_order: dict[str, list[int]] = {}
    per_scale_values_by_order: dict[str, list[float]] = {}
    per_scale_variances_by_order: dict[str, list[float | None]] = {}

    for order, scales in normalized_scales.items():
        weights = richardson_weights(scales)
        shot_allocation = allocate_shots(int(shots_total), weights, strategy="proportional_to_abs_weights")
        per_scale_values: list[float] = []
        per_scale_variances: list[float | None] = []
        for index, scale in enumerate(scales):
            shot_seed = deterministic_seed(
                "phase_e_zne",
                graph.name,
                tuple(values),
                p_dep,
                int(shots_total),
                int(zne_seed),
                int(scale),
            )
            objective = evaluate_objective_from_probabilities(
                graph,
                probability_cache[int(scale)],
                shots=int(shot_allocation[index]),
                seed=shot_seed,
            )
            per_scale_values.append(float(objective.objective_value))
            per_scale_variances.append(objective.shot_variance)

        zne_objectives_by_order[order] = float(
            sum(weight * value for weight, value in zip(weights, per_scale_values, strict=True))
        )
        weights_by_order[order] = [float(weight) for weight in weights]
        shot_allocation_by_order[order] = [int(value) for value in shot_allocation]
        per_scale_values_by_order[order] = per_scale_values
        per_scale_variances_by_order[order] = per_scale_variances

    return PhaseELHSCrossOrderResult(
        ideal_objective=float(ideal.objective_value),
        noisy_objective=float(noisy.objective_value),
        zne_objectives_by_order=zne_objectives_by_order,
        weights_by_order=weights_by_order,
        scale_factors_by_order=normalized_scales,
        shot_allocation_by_order=shot_allocation_by_order,
        per_scale_values_by_order=per_scale_values_by_order,
        per_scale_variances_by_order=per_scale_variances_by_order,
        probability_scale_factors=probability_scale_factors,
    )


def probabilities_dict_from_array(probabilities: np.ndarray, n_qubits: int) -> dict[str, float]:
    out: dict[str, float] = {}
    for index, probability in enumerate(probabilities):
        bitstring = format(index, f"0{n_qubits}b")[::-1]
        out[bitstring] = float(probability)
    return out


def expected_cut_from_statevector_dict(graph: GraphInstance, probabilities: dict[str, float]) -> float:
    expected = 0.0
    for bitstring, probability in probabilities.items():
        bits = qiskit_count_key_to_bits(bitstring, graph.n_qubits)
        expected += float(probability) * cut_value(bits, graph.edges)
    return float(expected)
