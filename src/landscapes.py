from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from .config import SUPPORTED_FOLDING_BACKEND
from .circuits import GraphInstance, build_qaoa_circuit, cut_value, ideal_expected_cut, qiskit_count_key_to_bits
from .mitiq_wrapper import add_measurements, allocate_shots, deterministic_seed, fold_global_odd_integer, richardson_weights
from .noise_models import build_depolarizing_noise_model


ProgressCallback = Callable[[int, int, str], None]


def parameter_grid(size: int, gamma_max: float = 2.0 * math.pi, beta_max: float = math.pi) -> tuple[np.ndarray, np.ndarray]:
    gammas = np.linspace(0.0, gamma_max, size, endpoint=False)
    betas = np.linspace(0.0, beta_max, size, endpoint=False)
    return gammas, betas


def generate_ideal_landscape(graph: GraphInstance, size: int, progress: ProgressCallback | None = None) -> np.ndarray:
    gammas, betas = parameter_grid(size)
    landscape = np.zeros((size, size), dtype=float)
    total = size * size
    k = 0
    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):
            landscape[i, j] = -ideal_expected_cut(graph, float(gamma), float(beta))
            k += 1
            if progress and (k == 1 or k == total or k % max(1, total // 10) == 0):
                progress(k, total, "ideal")
    return landscape


def _sample_expected_cut(
    graph: GraphInstance,
    circuit,
    shots: int,
    p_noise: float,
    seed: int,
    simulator: AerSimulator | None = None,
    simulator_options: dict[str, object] | None = None,
    transpile_cache: dict[tuple[object, ...], object] | None = None,
    transpile_cache_key: tuple[object, ...] | None = None,
) -> float:
    measured = add_measurements(circuit)
    if simulator is None:
        noise_model = build_depolarizing_noise_model(p_noise)
        simulator = AerSimulator(noise_model=noise_model, seed_simulator=seed, **(simulator_options or {}))
    if transpile_cache is not None and transpile_cache_key is not None and transpile_cache_key in transpile_cache:
        tqc = transpile_cache[transpile_cache_key]
    else:
        tqc = transpile(
            measured,
            simulator,
            basis_gates=["rx", "rz", "h", "cx"],
            optimization_level=0,
            seed_transpiler=0 if transpile_cache is not None else seed,
        )
        if transpile_cache is not None and transpile_cache_key is not None:
            transpile_cache[transpile_cache_key] = tqc
    result = simulator.run(tqc, shots=int(shots), seed_simulator=seed).result()
    counts = result.get_counts()
    expected = 0.0
    for key, count in counts.items():
        bits = qiskit_count_key_to_bits(key, graph.n_qubits)
        expected += (count / shots) * cut_value(bits, graph.edges)
    return float(expected)


def generate_noisy_landscape(
    graph: GraphInstance,
    size: int,
    p_noise: float,
    shots_total: int,
    seed: int,
    method_tag: str = "noisy",
    progress: ProgressCallback | None = None,
    transpile_cache: dict[tuple[object, ...], object] | None = None,
    simulator_options: dict[str, object] | None = None,
) -> np.ndarray:
    gammas, betas = parameter_grid(size)
    landscape = np.zeros((size, size), dtype=float)
    total = size * size
    k = 0
    noise_model = build_depolarizing_noise_model(p_noise)
    simulator = AerSimulator(noise_model=noise_model, **(simulator_options or {}))
    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):
            theta_index = i * size + j
            circuit = build_qaoa_circuit(graph, float(gamma), float(beta), measure=False)
            circuit_seed = deterministic_seed(graph.name, p_noise, shots_total, seed, method_tag, "none", 1, theta_index)
            landscape[i, j] = -_sample_expected_cut(
                graph,
                circuit,
                shots_total,
                p_noise,
                circuit_seed,
                simulator=simulator,
                simulator_options=simulator_options,
                transpile_cache=transpile_cache,
                transpile_cache_key=(graph.name, size, theta_index, 1),
            )
            k += 1
            if progress and (k == 1 or k == total or k % max(1, total // 10) == 0):
                progress(k, total, method_tag)
    return landscape


def generate_zne_landscape(
    graph: GraphInstance,
    size: int,
    p_noise: float,
    shots_total: int,
    seed: int,
    scale_factors: list[int],
    zne_order: str,
    folding_backend: str = SUPPORTED_FOLDING_BACKEND,
    shot_allocation_strategy: str = "proportional_to_abs_weights",
    progress: ProgressCallback | None = None,
    transpile_cache: dict[tuple[object, ...], object] | None = None,
    simulator_options: dict[str, object] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    if folding_backend != SUPPORTED_FOLDING_BACKEND:
        raise ValueError(f"Unsupported folding backend: {folding_backend}")
    gammas, betas = parameter_grid(size)
    weights = richardson_weights(scale_factors)
    shot_alloc = allocate_shots(shots_total, weights, strategy=shot_allocation_strategy)
    landscape = np.zeros((size, size), dtype=float)
    total = size * size
    k = 0
    gate_counts: dict[str, list[int]] = {str(s): [] for s in scale_factors}
    depths: dict[str, list[int]] = {str(s): [] for s in scale_factors}
    noise_model = build_depolarizing_noise_model(p_noise)
    simulator = AerSimulator(noise_model=noise_model, **(simulator_options or {}))

    for i, gamma in enumerate(gammas):
        for j, beta in enumerate(betas):
            theta_index = i * size + j
            base = build_qaoa_circuit(graph, float(gamma), float(beta), measure=False)
            extrapolated = 0.0
            for scale, weight, shots in zip(scale_factors, weights, shot_alloc, strict=True):
                folded = fold_global_odd_integer(base, int(scale))
                gate_counts[str(scale)].append(sum(folded.count_ops().values()))
                depths[str(scale)].append(folded.depth())
                circuit_seed = deterministic_seed(graph.name, p_noise, shots_total, seed, "zne", zne_order, scale, theta_index)
                value = -_sample_expected_cut(
                    graph,
                    folded,
                    shots,
                    p_noise,
                    circuit_seed,
                    simulator=simulator,
                    simulator_options=simulator_options,
                    transpile_cache=transpile_cache,
                    transpile_cache_key=(graph.name, size, theta_index, int(scale)),
                )
                extrapolated += weight * value
            landscape[i, j] = extrapolated
            k += 1
            if progress and (k == 1 or k == total or k % max(1, total // 10) == 0):
                progress(k, total, f"zne_{zne_order}")

    metadata = {
        "scale_factors": scale_factors,
        "folding_backend": folding_backend,
        "weights": weights,
        "shot_allocation": shot_alloc,
        "shot_allocation_strategy": shot_allocation_strategy,
        "gate_count_mean": {s: float(np.mean(v)) for s, v in gate_counts.items()},
        "depth_mean": {s: float(np.mean(v)) for s, v in depths.items()},
    }
    return landscape, metadata
