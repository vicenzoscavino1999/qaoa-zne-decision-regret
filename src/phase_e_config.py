from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic


PHASE_E_PROTOCOL_VERSION = "phase_e_cobyla_endpoint_depth2_depth3_v0_1_draft"
PHASE_E_CONFIG_SCHEMA_VERSION = "phase_e_config_schema_v0_1"
PHASE_E_OFFICIAL_TOKEN = "accept_phase_e_cobyla_depth2_official"

PHASE_E_PROTOCOL_FILE = "PHASE_E_COBYLA_ENDPOINT_PROTOCOL_v0.1-draft.md"
PHASE_E_COMPUTE_PLAN_FILE = "computacion.md"
PHASE_F_HARDWARE_PLAN_FILE = "PHASE_F_HARDWARE_TARGETED_REPLICATION_PLAN_v0.1-draft.md"
PHASE_E_COBYLA_RUNTIME_OPTIMIZATION_PLAN_FILE = "PHASE_E_COBYLA_RUNTIME_OPTIMIZATION_AND_CHUNKED_EXECUTION_PLAN_v0.1-draft.md"

DEFAULT_NOISE_RATES = (0.001, 0.003, 0.005, 0.01, 0.02)
DEFAULT_SHOT_BUDGETS = (512, 1024, 4096)
DEFAULT_DEPTH2_INSTANCE_COUNT = 6
DEFAULT_SCALE_FACTORS = {
    "linear": (1, 3),
    "quadratic": (1, 3, 5),
    "cubic": (1, 3, 5, 7),
}


@dataclass(frozen=True)
class PhaseENoiseModelConfig:
    model: str = "per_gate_depolarizing"
    model_version: str = "phase_e_per_gate_depolarizing_v0_1"
    p_2q_rule: str = "p_2q = p_dep"
    p_1q_rule: str = "p_1q = p_dep / 10"
    p_1q_ratio: float = 0.1
    readout_noise: bool = False
    idle_decoherence: bool = False
    folding_rule: str = "global_odd_integer_u_u_dagger_u"


@dataclass(frozen=True)
class PhaseEBackendConfig:
    primary_backend: str = "custom_density_matrix_phase_e"
    reference_backend: str = "qiskit_aer_density_matrix"
    state_representation: str = "density_matrix"
    shot_noise_model: str = "exact_probabilities_plus_multinomial_synthetic_shots"
    probability_tolerance_ideal: float = 1e-10
    probability_tolerance_noisy: float = 1e-8
    observable_tolerance: float = 1e-8


@dataclass(frozen=True)
class PhaseEOptimizerConfig:
    primary_optimizer: str = "cobyla"
    secondary_optimizers: tuple[str, ...] = ()
    random_start_candidates: tuple[int, ...] = (20, 50, 100)
    cobyla_evaluation_budget_candidates: tuple[int, ...] = (300, 500, 750, 1000)
    lhs_size_candidates: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    reference_start_candidates: tuple[int, ...] = (50, 100, 200)
    selected_random_starts: int | None = None
    selected_cobyla_evaluation_budget: int | None = None
    selected_lhs_size: int | None = None
    selected_reference_starts: int | None = None


def phase_e_paths(root: Path = PROJECT_ROOT) -> dict[str, str]:
    return {
        "config": str(root / "config" / "phase_e_config_v0.1-draft.json"),
        "instances": str(root / "config" / "phase_e_instances_v0.1-draft.json"),
        "noise_model": str(root / "config" / "phase_e_noise_model_v0.1-draft.json"),
        "results": str(root / "results" / "phase_e"),
        "planning": str(root / "results" / "phase_e" / "planning"),
        "raw": str(root / "data" / "raw" / "phase_e"),
        "logs": str(root / "logs" / "phase_e"),
    }


def _file_hash(root: Path, relative_path: str) -> str | None:
    path = root / relative_path
    if not path.exists():
        return None
    return sha256_file(path)


def phase_e_document_hashes(root: Path = PROJECT_ROOT) -> dict[str, str | None]:
    return {
        PHASE_E_PROTOCOL_FILE: _file_hash(root, PHASE_E_PROTOCOL_FILE),
        PHASE_E_COMPUTE_PLAN_FILE: _file_hash(root, PHASE_E_COMPUTE_PLAN_FILE),
        PHASE_F_HARDWARE_PLAN_FILE: _file_hash(root, PHASE_F_HARDWARE_PLAN_FILE),
        PHASE_E_COBYLA_RUNTIME_OPTIMIZATION_PLAN_FILE: _file_hash(root, PHASE_E_COBYLA_RUNTIME_OPTIMIZATION_PLAN_FILE),
    }


def phase_e_selected_optimizer_values(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    optimizer = asdict(PhaseEOptimizerConfig())
    status_path = root / "results" / "phase_e" / "calibration" / "phase_e_calibration_status.json"
    if not status_path.exists():
        return optimizer
    status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    optimizer["selected_random_starts"] = status.get("selected_random_start_count")
    optimizer["selected_cobyla_evaluation_budget"] = status.get("selected_cobyla_evaluation_budget")
    optimizer["selected_lhs_size"] = status.get("selected_N_LHS")
    optimizer["selected_reference_starts"] = status.get("selected_reference_starts")
    optimizer["selected_reference_strategy"] = status.get("selected_reference_strategy")
    return optimizer


def default_phase_e_config(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    instances_path = root / "config" / "phase_e_instances_v0.1-draft.json"
    selected_instances = None
    if instances_path.exists():
        selected_instances = json.loads(instances_path.read_text(encoding="utf-8-sig")).get("selected_depth2_instances")
    return {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "draft_not_frozen_not_executable",
        "schema_version": PHASE_E_CONFIG_SCHEMA_VERSION,
        "protocol_version": PHASE_E_PROTOCOL_VERSION,
        "official_execution_token_required": PHASE_E_OFFICIAL_TOKEN,
        "official_execution_enabled": False,
        "documents": phase_e_document_hashes(root),
        "paths": phase_e_paths(root),
        "qaoa_depths": {
            "depth1_optimizer_validation": 1,
            "depth2_primary": 2,
            "depth3_gated_supplement": 3,
        },
        "depth2_design": {
            "selected_instance_count": DEFAULT_DEPTH2_INSTANCE_COUNT,
            "selection_rule": "stratified_frozen_before_execution",
            "selected_instances": selected_instances,
            "seed_count": 30,
        },
        "noise_model": asdict(PhaseENoiseModelConfig()),
        "backend": asdict(PhaseEBackendConfig()),
        "optimizer": phase_e_selected_optimizer_values(root),
        "noise_rates": list(DEFAULT_NOISE_RATES),
        "shot_budgets": list(DEFAULT_SHOT_BUDGETS),
        "zne": {
            "primary_method": "richardson",
            "scale_factors": {key: list(value) for key, value in DEFAULT_SCALE_FACTORS.items()},
            "shot_allocation": "proportional_to_abs_weights",
            "secondary_methods": [
                "exponential_requires_at_least_4_scale_factors",
                "polynomial_least_squares_requires_residual_dof",
                "bounded_extrapolation_targeted_check",
                "layerwise_richardson_targeted_check",
            ],
        },
        "thresholds": {
            "tau_R_opt": None,
            "tau_conc": None,
            "tau_trivial": None,
            "variance_dominated_fraction_gate": None,
            "nonfinite_failure_rate_gate": None,
        },
        "runtime_gates": {
            "must_pass_backend_tests": True,
            "must_pass_runtime_gate": True,
            "must_pass_storage_gate": True,
            "must_pass_dry_run": True,
            "official_run_requires_user_token": True,
        },
        "planning_tracks": {
            "toy_model": {
                "script": "scripts/43_phase_e_toy_model.py",
                "summary_output": "results/phase_e/toy_model/phase_e_toy_model_summary.json",
                "status_required_before_freeze": True,
            },
            "noise_model_snapshot": {
                "script": "scripts/46_phase_e_noise_model_snapshot.py",
                "summary_output": "results/phase_e/planning/phase_e_noise_model_snapshot.json",
                "status_required_before_freeze": True,
            },
            "software_stack_snapshot": {
                "script": "scripts/47_phase_e_stack_snapshot.py",
                "summary_output": "results/phase_e/planning/phase_e_stack_snapshot.json",
                "status_required_before_freeze": True,
            },
            "depth1_regression": {
                "script": "scripts/48_phase_e_depth1_optimizer_validation.py",
                "summary_output": "results/phase_e/planning/phase_e_depth1_regression_summary.json",
                "status_required_before_freeze": True,
            },
            "power_sizing": {
                "script": "scripts/44_phase_e_power_sizing.py",
                "summary_output": "results/phase_e/power_sizing/phase_e_power_simulation.json",
                "status_required_before_freeze": True,
            },
            "instance_selection": {
                "script": "scripts/45_phase_e_instance_selection.py",
                "summary_output": "config/phase_e_instances_v0.1-draft.json",
                "status_required_before_freeze": True,
            },
            "unified_depth2_pilot": {
                "script": "scripts/49_phase_e_unified_depth2_pilot.py",
                "summary_output": "results/phase_e/calibration/phase_e_unified_pilot_summary.json",
                "status_required_before_freeze": True,
            },
            "calibration_methodology_status": {
                "script": "scripts/50_phase_e_calibrate_cobyla_k_lhs.py",
                "summary_output": "results/phase_e/calibration/phase_e_calibration_status.json",
                "status_required_before_freeze": True,
            },
            "bias_variance_runtime_gate": {
                "script": "scripts/78_phase_e_bias_variance_runtime_gate.py",
                "summary_output": "results/phase_e/calibration/freeze_gates/phase_e_bias_variance_runtime_gate.json",
                "status_required_before_freeze": True,
            },
            "stage_c0_optimization_plan": {
                "script": "scripts/52_phase_e_stage_c0_optimization_plan.py",
                "summary_output": "results/phase_e/calibration/phase_e_stage_c0_optimization_plan.json",
                "status_required_before_freeze": False,
                "status": "retired_spsa_primary_path",
            },
            "stage_c0_benchmark_smoke": {
                "script": "scripts/53_phase_e_stage_c0_benchmark.py",
                "summary_output": "results/phase_e/calibration/c0_benchmark/phase_e_stage_c0_benchmark_summary.json",
                "status_required_before_freeze": False,
                "status": "historical_c0_diagnostic_not_required_for_cobyla_freeze",
            },
            "stage_c0_jitter_smoke": {
                "script": "scripts/54_phase_e_stage_c0_jitter.py",
                "summary_output": "results/phase_e/calibration/c0_jitter/phase_e_stage_c0_jitter_summary.json",
                "status_required_before_freeze": False,
                "status": "historical_c0_diagnostic_not_required_for_cobyla_freeze",
            },
            "stage_c0_resume_smoke": {
                "script": "scripts/55_phase_e_stage_c0_resume.py",
                "summary_output": "results/phase_e/calibration/c0_resume/phase_e_stage_c0_resume_summary.json",
                "status_required_before_freeze": False,
                "status": "historical_c0_diagnostic_not_required_for_cobyla_freeze",
            },
            "stage_c0_storage_projection": {
                "script": "scripts/56_phase_e_stage_c0_storage.py",
                "summary_output": "results/phase_e/calibration/c0_storage/phase_e_stage_c0_storage_summary.json",
                "status_required_before_freeze": False,
                "status": "historical_c0_diagnostic_not_required_for_cobyla_freeze",
            },
            "stage_c0_shard_balance_projection": {
                "script": "scripts/57_phase_e_stage_c0_shard_balance.py",
                "summary_output": "results/phase_e/calibration/c0_shard_balance/phase_e_stage_c0_shard_balance_summary.json",
                "status_required_before_freeze": False,
                "status": "historical_c0_diagnostic_not_required_for_cobyla_freeze",
            },
            "stage_c0_runtime_projection": {
                "script": "scripts/58_phase_e_stage_c0_runtime.py",
                "summary_output": "results/phase_e/calibration/c0_runtime/phase_e_stage_c0_runtime_summary.json",
                "status_required_before_freeze": False,
                "status": "historical_c0_diagnostic_not_required_for_cobyla_freeze",
            },
            "runtime_scope_amendment": {
                "script": "scripts/59_phase_e_runtime_scope_amendment.py",
                "summary_output": "results/phase_e/calibration/c0_runtime/phase_e_runtime_scope_amendment_summary.json",
                "status_required_before_freeze": False,
                "status": "conditional_only_if_cobyla_runtime_gate_fails",
            },
            "stage_c_authorization_guard": {
                "script": "scripts/60_phase_e_stage_c_authorization.py",
                "summary_output": "results/phase_e/calibration/stage_c_authorization/phase_e_stage_c_authorization_summary.json",
                "status_required_before_freeze": False,
                "status": "conditional_only_after_cobyla_runtime_gate_and_freeze",
            },
            "runtime_optimization_plan": {
                "script": "scripts/61_phase_e_runtime_optimization_plan.py",
                "summary_output": "results/phase_e/calibration/runtime_optimization/phase_e_runtime_optimization_plan_summary.json",
                "status_required_before_freeze": False,
                "status": "conditional_only_if_cobyla_runtime_gate_fails",
            },
            "cobyla_runtime_pre_o1": {
                "script": "scripts/79_phase_e_cobyla_runtime_o0_o0_75.py",
                "summary_output": "results/phase_e/calibration/cobyla_runtime_optimization/post_o1_status.json",
                "status_required_before_freeze": True,
                "status": "o2_p2_rejected_rolled_back_to_o2_p1_o3_o5_not_started",
            },
        },
        "first_safe_implementation_slice": [
            "phase_e_config.py",
            "phase_e_toy_model.py",
            "phase_e_sim_backend.py",
            "phase_e_optimizers.py",
            "phase_e_tests.py",
            "phase_e_noise.py",
            "phase_e_stack_snapshot.py",
            "phase_e_depth1_adapter.py",
            "phase_e_cobyla_runtime_optimization.py",
            "scripts/42_phase_e_config_preflight.py",
            "scripts/43_phase_e_toy_model.py",
            "scripts/46_phase_e_noise_model_snapshot.py",
            "scripts/47_phase_e_stack_snapshot.py",
            "scripts/48_phase_e_depth1_optimizer_validation.py",
            "scripts/51_phase_e_backend_benchmark.py",
            "scripts/79_phase_e_cobyla_runtime_o0_o0_75.py",
        ],
    }


def validate_phase_e_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if config.get("schema_version") != PHASE_E_CONFIG_SCHEMA_VERSION:
        errors.append("schema_version does not match Phase E config schema.")
    if config.get("protocol_version") != PHASE_E_PROTOCOL_VERSION:
        errors.append("protocol_version does not match Phase E protocol version.")
    if config.get("official_execution_enabled") is not False:
        errors.append("official_execution_enabled must be false in the draft config.")

    paths = config.get("paths", {})
    for key in ("results", "raw", "logs"):
        value = str(paths.get(key, ""))
        normalized = value.replace("\\", "/").lower()
        if "/phase_c" in normalized or "/phase_d" in normalized:
            errors.append(f"Phase E write target points at a previous phase directory: {key}={value}")

    noise = config.get("noise_model", {})
    if noise.get("model") != "per_gate_depolarizing":
        errors.append("first safe implementation slice expects per_gate_depolarizing.")
    if float(noise.get("p_1q_ratio", -1.0)) != 0.1:
        errors.append("Phase E draft default expects p_1q_ratio = 0.1.")

    backend = config.get("backend", {})
    if backend.get("primary_backend") != "custom_density_matrix_phase_e":
        errors.append("first safe implementation slice expects custom_density_matrix_phase_e.")
    if backend.get("shot_noise_model") != "exact_probabilities_plus_multinomial_synthetic_shots":
        errors.append("shot noise model must be exact probabilities plus multinomial synthetic shots for this draft slice.")

    optimizer = config.get("optimizer", {})
    if optimizer.get("primary_optimizer") != "cobyla":
        errors.append("amended Phase E protocol expects COBYLA as primary optimizer.")
    if "spsa_evaluation_budget_candidates" in optimizer:
        errors.append("amended Phase E config must not expose SPSA budget candidates as active calibration settings.")
    if "cobyla_evaluation_budget_candidates" not in optimizer:
        errors.append("amended Phase E config requires cobyla_evaluation_budget_candidates.")

    zne = config.get("zne", {})
    scale_factors = zne.get("scale_factors", {})
    if scale_factors.get("linear") != [1, 3]:
        errors.append("linear Richardson scale factors must be [1, 3].")
    if scale_factors.get("cubic") != [1, 3, 5, 7]:
        errors.append("cubic Richardson scale factors must be [1, 3, 5, 7].")

    return errors


def write_phase_e_config_preflight(root: Path = PROJECT_ROOT, overwrite: bool = True) -> dict[str, Any]:
    config = default_phase_e_config(root)
    errors = validate_phase_e_config(config)
    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "config": config,
    }

    paths = phase_e_paths(root)
    ensure_dir(Path(paths["planning"]))
    write_json_atomic(paths["config"], config, overwrite=overwrite)
    write_json_atomic(Path(paths["planning"]) / "phase_e_config_preflight.json", report, overwrite=overwrite)
    return report
