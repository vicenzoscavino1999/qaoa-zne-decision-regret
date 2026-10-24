from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MetricsConfig:
    epsilon_tie_fraction: float = 0.001
    good_region_fraction: float = 0.10
    oob_delta_fraction: float = 0.10
    oob_breakdown_threshold: float = 0.50
    below_detection_floor_threshold: float = 0.0001
    gradient_boundary_tolerance_fraction: float = 0.05
    gradient_min_ideal_norm: float = 1e-8
    gradient_denominator_epsilon: float = 1e-12

    @classmethod
    def from_experiment_config(cls, cfg: dict[str, Any]) -> "MetricsConfig":
        gradient_rule = cfg["metrics"]["gradient_boundary_rule"]
        return cls(
            epsilon_tie_fraction=float(cfg["ties"]["epsilon_tie_fraction_of_range"]),
            good_region_fraction=float(cfg["metrics"]["good_region_fraction"]),
            oob_delta_fraction=float(cfg["oob"]["delta_fraction_of_range"]),
            oob_breakdown_threshold=float(cfg["oob"]["breakdown_threshold"]),
            below_detection_floor_threshold=float(cfg["noise"]["below_detection_floor_threshold"]),
            gradient_boundary_tolerance_fraction=float(gradient_rule["boundary_continuity_tolerance_fraction"]),
            gradient_min_ideal_norm=float(gradient_rule["min_ideal_gradient_norm"]),
            gradient_denominator_epsilon=float(gradient_rule["denominator_epsilon"]),
        )


SUPPORTED_FOLDING_BACKEND = "qiskit_local_global_u_u_dagger_u"

