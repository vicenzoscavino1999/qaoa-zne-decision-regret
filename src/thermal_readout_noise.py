from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from qiskit_aer.noise import NoiseModel, ReadoutError, thermal_relaxation_error


THERMAL_READOUT_NOISE_MODEL_VERSION = "phase_d_thermal_relaxation_readout_v0_1"


@dataclass(frozen=True)
class ThermalReadoutNoiseConfig:
    """Phase D robustness noise-model parameters.

    Time values only need to be internally consistent; the defaults use seconds.
    RZ is treated as virtual and receives no thermal-relaxation channel.
    """

    model: str = "thermal_relaxation_plus_readout"
    model_version: str = THERMAL_READOUT_NOISE_MODEL_VERSION
    t1_seconds: float = 100.0e-6
    t2_seconds: float = 80.0e-6
    one_qubit_gate_time_seconds: float = 50.0e-9
    two_qubit_gate_time_seconds: float = 300.0e-9
    readout_p01: float = 0.02
    readout_p10: float = 0.03
    excited_state_population: float = 0.0
    enable_thermal_relaxation: bool = True
    enable_readout_error: bool = True
    noisy_one_qubit_gates: tuple[str, ...] = ("h", "rx", "sx", "x")
    virtual_one_qubit_gates: tuple[str, ...] = ("rz",)
    noisy_two_qubit_gates: tuple[str, ...] = ("cx",)
    basis_gates: tuple[str, ...] = ("h", "rx", "rz", "sx", "x", "cx")
    notes: tuple[str, ...] = field(
        default_factory=lambda: (
            "Exploratory Phase D robustness diagnostic, not a replacement for the primary depolarizing analysis.",
            "No idle-window scheduling model is included in v0.1.",
            "Readout noise is a symmetric all-qubit assignment channel only if p01 == p10.",
        )
    )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_THERMAL_READOUT_CONFIG = ThermalReadoutNoiseConfig()


def validate_thermal_readout_config(config: ThermalReadoutNoiseConfig) -> list[str]:
    errors: list[str] = []
    if config.t1_seconds <= 0.0:
        errors.append("t1_seconds must be positive.")
    if config.t2_seconds <= 0.0:
        errors.append("t2_seconds must be positive.")
    if config.t2_seconds > 2.0 * config.t1_seconds:
        errors.append("thermal_relaxation_error requires T2 <= 2*T1.")
    if config.one_qubit_gate_time_seconds < 0.0:
        errors.append("one_qubit_gate_time_seconds must be non-negative.")
    if config.two_qubit_gate_time_seconds < 0.0:
        errors.append("two_qubit_gate_time_seconds must be non-negative.")
    if not (0.0 <= config.readout_p01 <= 1.0):
        errors.append("readout_p01 must lie in [0, 1].")
    if not (0.0 <= config.readout_p10 <= 1.0):
        errors.append("readout_p10 must lie in [0, 1].")
    if not (0.0 <= config.excited_state_population <= 1.0):
        errors.append("excited_state_population must lie in [0, 1].")
    if not config.noisy_one_qubit_gates:
        errors.append("at least one noisy one-qubit gate must be declared.")
    if not config.noisy_two_qubit_gates:
        errors.append("at least one noisy two-qubit gate must be declared.")
    return errors


def readout_assignment_matrix(config: ThermalReadoutNoiseConfig) -> list[list[float]]:
    return [
        [1.0 - float(config.readout_p01), float(config.readout_p01)],
        [float(config.readout_p10), 1.0 - float(config.readout_p10)],
    ]


def build_thermal_relaxation_readout_noise_model(
    config: ThermalReadoutNoiseConfig = DEFAULT_THERMAL_READOUT_CONFIG,
) -> NoiseModel:
    errors = validate_thermal_readout_config(config)
    if errors:
        raise ValueError("; ".join(errors))

    noise_model = NoiseModel()
    if config.enable_thermal_relaxation and config.one_qubit_gate_time_seconds > 0.0:
        one_qubit_error = thermal_relaxation_error(
            float(config.t1_seconds),
            float(config.t2_seconds),
            float(config.one_qubit_gate_time_seconds),
            excited_state_population=float(config.excited_state_population),
        )
        noise_model.add_all_qubit_quantum_error(one_qubit_error, list(config.noisy_one_qubit_gates))

    if config.enable_thermal_relaxation and config.two_qubit_gate_time_seconds > 0.0:
        thermal_1q_for_2q = thermal_relaxation_error(
            float(config.t1_seconds),
            float(config.t2_seconds),
            float(config.two_qubit_gate_time_seconds),
            excited_state_population=float(config.excited_state_population),
        )
        two_qubit_error = thermal_1q_for_2q.tensor(thermal_1q_for_2q)
        noise_model.add_all_qubit_quantum_error(two_qubit_error, list(config.noisy_two_qubit_gates))

    if config.enable_readout_error and (config.readout_p01 > 0.0 or config.readout_p10 > 0.0):
        noise_model.add_all_qubit_readout_error(ReadoutError(readout_assignment_matrix(config)))

    return noise_model


def thermal_readout_model_summary(config: ThermalReadoutNoiseConfig = DEFAULT_THERMAL_READOUT_CONFIG) -> dict[str, Any]:
    noise_model = build_thermal_relaxation_readout_noise_model(config)
    payload = noise_model.to_dict()
    return {
        "model": config.model,
        "model_version": config.model_version,
        "basis_gates": list(noise_model.basis_gates),
        "declared_basis_gates": list(config.basis_gates),
        "noisy_one_qubit_gates": list(config.noisy_one_qubit_gates),
        "virtual_one_qubit_gates": list(config.virtual_one_qubit_gates),
        "noisy_two_qubit_gates": list(config.noisy_two_qubit_gates),
        "readout_assignment_matrix": readout_assignment_matrix(config),
        "noise_error_count": len(payload.get("errors", [])),
        "config": config.as_dict(),
    }
