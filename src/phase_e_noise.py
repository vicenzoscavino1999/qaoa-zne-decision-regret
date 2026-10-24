from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .io_utils import PROJECT_ROOT, ensure_dir, write_json_atomic
from .phase_e_config import DEFAULT_NOISE_RATES, PhaseENoiseModelConfig


PHASE_E_NOISE_SNAPSHOT_VERSION = "phase_e_noise_model_snapshot_v0_1"


def build_phase_e_noise_model_snapshot() -> dict[str, Any]:
    config = PhaseENoiseModelConfig()
    hardware_map = []
    for p_dep in DEFAULT_NOISE_RATES:
        hardware_map.append(
            {
                "p_dep": float(p_dep),
                "p_2q": float(p_dep),
                "p_1q": float(p_dep) * config.p_1q_ratio,
                "approx_2q_fidelity": float(1.0 - p_dep),
                "approx_1q_fidelity": float(1.0 - p_dep * config.p_1q_ratio),
            }
        )

    return {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snapshot_version": PHASE_E_NOISE_SNAPSHOT_VERSION,
        "status": "draft_not_frozen",
        "primary_noise_model": {
            "model": config.model,
            "model_version": config.model_version,
            "semantics": "per_gate_after_transpilation_to_rx_rz_h_cx",
            "p_2q_rule": config.p_2q_rule,
            "p_1q_rule": config.p_1q_rule,
            "p_1q_ratio": config.p_1q_ratio,
            "readout_noise": config.readout_noise,
            "idle_decoherence": config.idle_decoherence,
            "folding_rule": config.folding_rule,
            "phase_c_d_compatibility_note": (
                "Phase C/D used a different custom-kernel convention in which one-qubit errors were p "
                "and CX errors were 2p. Phase E uses the protocol draft convention p_2q=p_dep and p_1q=p_dep/10."
            ),
        },
        "hardware_regime_mapping": hardware_map,
        "targeted_secondary_models": [
            "amplitude_damping_plus_dephasing_subset",
            "asymmetric_depolarizing_subset",
        ],
    }


def write_phase_e_noise_model_snapshot(root: Path = PROJECT_ROOT, overwrite: bool = True) -> dict[str, Any]:
    snapshot = build_phase_e_noise_model_snapshot()
    output_dir = ensure_dir(root / "results" / "phase_e" / "planning")
    write_json_atomic(root / "config" / "phase_e_noise_model_v0.1-draft.json", snapshot, overwrite=overwrite)
    write_json_atomic(output_dir / "phase_e_noise_model_snapshot.json", snapshot, overwrite=overwrite)
    return snapshot
