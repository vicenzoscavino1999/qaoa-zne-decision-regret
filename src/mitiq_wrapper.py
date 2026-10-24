from __future__ import annotations

import hashlib
from fractions import Fraction
from typing import Iterable

from qiskit import QuantumCircuit


def richardson_weights(scale_factors: Iterable[int]) -> list[float]:
    scales = [Fraction(int(s), 1) for s in scale_factors]
    weights: list[Fraction] = []
    for i, scale_i in enumerate(scales):
        numerator = Fraction(1, 1)
        for j, scale_j in enumerate(scales):
            if i == j:
                continue
            numerator *= Fraction(-scale_j, scale_i - scale_j)
        weights.append(numerator)
    return [float(w) for w in weights]


def all_protocol_weights() -> dict[str, list[float]]:
    return {
        "linear_[1,3]": richardson_weights([1, 3]),
        "quadratic_[1,3,5]": richardson_weights([1, 3, 5]),
        "cubic_[1,3,5,7]": richardson_weights([1, 3, 5, 7]),
    }


def _largest_remainder(total_budget: int, probabilities: list[float]) -> list[int]:
    raw = [total_budget * p for p in probabilities]
    floors = [int(x) for x in raw]
    remainder = total_budget - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors


def allocate_shots(total_budget: int, weights: Iterable[float], strategy: str = "proportional_to_abs_weights") -> list[int]:
    weights = [float(w) for w in weights]
    if strategy == "uniform":
        return _largest_remainder(total_budget, [1.0 / len(weights)] * len(weights))
    if strategy != "proportional_to_abs_weights":
        raise ValueError(f"Unsupported shot allocation strategy: {strategy}")
    abs_weights = [abs(w) for w in weights]
    denom = sum(abs_weights)
    if denom <= 0:
        raise ValueError("Richardson weights must not all be zero.")
    return _largest_remainder(total_budget, [w / denom for w in abs_weights])


def deterministic_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) & 0x7FFFFFFF


def fold_global_odd_integer(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    if scale_factor < 1 or scale_factor % 2 != 1:
        raise ValueError("This pilot only supports odd integer scale factors.")
    base = circuit.remove_final_measurements(inplace=False)
    folded = QuantumCircuit(base.num_qubits)
    folded.compose(base, inplace=True)
    repeats = (scale_factor - 1) // 2
    inverse = base.inverse()
    for _ in range(repeats):
        folded.compose(inverse, inplace=True)
        folded.compose(base, inplace=True)
    return folded


def add_measurements(circuit: QuantumCircuit) -> QuantumCircuit:
    measured = circuit.copy()
    measured.measure_all()
    return measured


def try_import_mitiq() -> tuple[bool, str]:
    try:
        import mitiq  # noqa: F401

        return True, getattr(mitiq, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, repr(exc)
