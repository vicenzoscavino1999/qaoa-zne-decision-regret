from __future__ import annotations

from qiskit_aer.noise import NoiseModel, depolarizing_error


def build_depolarizing_noise_model(p: float) -> NoiseModel:
    noise_model = NoiseModel()
    if p <= 0:
        return noise_model
    one_qubit = depolarizing_error(float(p), 1)
    two_qubit = depolarizing_error(float(2.0 * p), 2)
    noise_model.add_all_qubit_quantum_error(one_qubit, ["h", "rx", "rz"])
    noise_model.add_all_qubit_quantum_error(two_qubit, ["cx"])
    return noise_model

