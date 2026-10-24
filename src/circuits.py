from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass(frozen=True)
class GraphInstance:
    name: str
    edges: tuple[tuple[int, int, float], ...]

    @property
    def n_qubits(self) -> int:
        return max(max(u, v) for u, v, _ in self.edges) + 1


def graph_from_config(name: str, cfg: dict) -> GraphInstance:
    edges = tuple((int(u), int(v), float(w)) for u, v, w in cfg["instances"][name]["edges"])
    return GraphInstance(name=name, edges=edges)


def build_qaoa_circuit(graph: GraphInstance, gamma: float, beta: float, measure: bool = False) -> QuantumCircuit:
    """Build p=1 QAOA for weighted MaxCut.

    The objective is C = -E[Cut], so downstream code minimizes C.
    """
    qc = QuantumCircuit(graph.n_qubits)
    qc.h(range(graph.n_qubits))
    for u, v, weight in graph.edges:
        qc.rzz(-gamma * weight, u, v)
    for q in range(graph.n_qubits):
        qc.rx(2.0 * beta, q)
    if measure:
        qc.measure_all()
    return qc


def cut_value(bits: Iterable[int], edges: tuple[tuple[int, int, float], ...]) -> float:
    values = list(bits)
    return float(sum(weight for u, v, weight in edges if values[u] != values[v]))


def qiskit_count_key_to_bits(key: str, n_qubits: int) -> list[int]:
    compact = key.replace(" ", "")
    return [int(x) for x in compact[::-1][:n_qubits]]


def ideal_expected_cut(graph: GraphInstance, gamma: float, beta: float) -> float:
    qc = build_qaoa_circuit(graph, gamma, beta, measure=False)
    state = Statevector.from_instruction(qc)
    probs = state.probabilities_dict()
    expected = 0.0
    for bitstring, prob in probs.items():
        bits = qiskit_count_key_to_bits(bitstring, graph.n_qubits)
        expected += float(prob) * cut_value(bits, graph.edges)
    return float(expected)

