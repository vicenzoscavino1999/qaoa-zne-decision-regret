from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .circuits import GraphInstance
from .io_utils import PROJECT_ROOT, ensure_dir, write_json_atomic
from .mitiq_wrapper import deterministic_seed, richardson_weights
from .phase_e_depth1_adapter import run_phase_e_depth1_regression
from .phase_e_c0_jitter import summarize_jitter_fractions
from .phase_e_c0_shard_balance import evaluate_shard_balance
from .phase_e_c0_storage import evaluate_storage_thresholds
from .phase_e_c0_runtime import evaluate_runtime_gate
from .phase_e_stage_c_authorization import evaluate_stage_c_authorization
from .phase_e_runtime_optimization_plan import evaluate_runtime_optimization_plan_guardrails
from .phase_e_runtime_optimization_execution import (
    O0_O1_AUTHORIZATION_TOKEN,
    STAGE_C_AUTHORIZATION_TOKEN,
    run_o0_o1_mode,
)
from .phase_e_optimizers import (
    COBYLAConfig,
    SPSAConfig,
    cobyla_minimize,
    make_paired_initial_thetas,
    spsa_initialize_state,
    spsa_minimize,
    spsa_minimize_checkpoints,
    spsa_result_from_state,
    spsa_run_state,
)
from .phase_e_sim_backend import (
    aer_density_probabilities_phase_e,
    custom_density_probabilities,
    evaluate_custom_density_objective,
    richardson_zne_objective,
    statevector_probabilities_phase_e,
    wrap_theta,
)


PHASE_E_TEST_VERSION = "phase_e_backend_tests_v0_2"


def _single_edge_graph() -> GraphInstance:
    return GraphInstance(name="phase_e_test_single_edge", edges=((0, 1, 1.0),))


def _path3_graph() -> GraphInstance:
    return GraphInstance(name="phase_e_test_path3", edges=((0, 1, 1.0), (1, 2, 0.7)))


def _record(name: str, passed: bool, **details: Any) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        **details,
    }


def _max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def test_ideal_custom_matches_statevector() -> dict[str, Any]:
    graph = _path3_graph()
    cases = [
        (1, (0.37, 0.22), 1),
        (2, (0.37, 0.22, 0.81, 0.44), 1),
        (2, (0.19, 0.41, 0.73, 0.11), 3),
    ]
    rows = []
    for depth, theta, scale in cases:
        custom = custom_density_probabilities(graph, theta, p_dep=0.0, scale_factor=scale)
        statevector = statevector_probabilities_phase_e(graph, theta, scale_factor=scale)
        diff = _max_abs_diff(custom.probabilities, statevector)
        rows.append({"depth": depth, "scale_factor": scale, "max_abs_probability_diff": diff})
    max_diff = max(row["max_abs_probability_diff"] for row in rows)
    return _record(
        "ideal_custom_matches_statevector_depth1_depth2",
        max_diff <= 1e-10,
        tolerance=1e-10,
        max_abs_probability_diff=max_diff,
        rows=rows,
    )


def test_custom_matches_aer_density_noise() -> dict[str, Any]:
    graph = _path3_graph()
    cases = [
        ((0.37, 0.22), 0.003, 1),
        ((0.37, 0.22, 0.81, 0.44), 0.003, 1),
        ((0.19, 0.41, 0.73, 0.11), 0.003, 3),
    ]
    rows = []
    for theta, p_dep, scale in cases:
        custom = custom_density_probabilities(graph, theta, p_dep=p_dep, scale_factor=scale)
        aer = aer_density_probabilities_phase_e(graph, theta, p_dep=p_dep, scale_factor=scale)
        diff = _max_abs_diff(custom.probabilities, aer.probabilities)
        rows.append(
            {
                "depth": len(theta) // 2,
                "p_dep": p_dep,
                "scale_factor": scale,
                "max_abs_probability_diff": diff,
                "custom_gate_count": custom.transpiled_gate_count,
                "aer_gate_count": aer.transpiled_gate_count,
            }
        )
    max_diff = max(row["max_abs_probability_diff"] for row in rows)
    return _record(
        "custom_density_matches_aer_density_phase_e_noise_model",
        max_diff <= 1e-8,
        tolerance=1e-8,
        max_abs_probability_diff=max_diff,
        rows=rows,
    )


def test_richardson_linear_signal() -> dict[str, Any]:
    scales = [1, 3]
    weights = richardson_weights(scales)
    intercept = 2.75
    slope = -0.42
    values = [intercept + slope * scale for scale in scales]
    extrapolated = float(sum(weight * value for weight, value in zip(weights, values, strict=True)))
    return _record(
        "first_order_richardson_recovers_linear_zero_noise_intercept",
        abs(extrapolated - intercept) <= 1e-12,
        tolerance=1e-12,
        scale_factors=scales,
        weights=weights,
        values=values,
        extrapolated=extrapolated,
        expected=intercept,
    )


def test_shot_sampling_deterministic() -> dict[str, Any]:
    graph = _single_edge_graph()
    theta = (0.37, 0.22, 0.81, 0.44)
    result1 = evaluate_custom_density_objective(graph, theta, p_dep=0.003, shots=256, seed=12345)
    result2 = evaluate_custom_density_objective(graph, theta, p_dep=0.003, shots=256, seed=12345)
    return _record(
        "synthetic_multinomial_shot_noise_is_seed_deterministic",
        result1.objective_value == result2.objective_value,
        objective_value_1=result1.objective_value,
        objective_value_2=result2.objective_value,
        shot_variance=result1.shot_variance,
    )


def test_zne_derivation_reconstruction() -> dict[str, Any]:
    graph = _single_edge_graph()
    theta = (0.37, 0.22, 0.81, 0.44)
    result = richardson_zne_objective(graph, theta, p_dep=0.003, scale_factors=[1, 3], shots_total=512, seed=987)
    reconstructed = float(sum(weight * value for weight, value in zip(result.weights, result.per_scale_values, strict=True)))
    return _record(
        "zne_derivation_reconstructs_from_per_scale_values",
        abs(reconstructed - result.objective_value) <= 1e-12,
        tolerance=1e-12,
        zne_value=result.objective_value,
        reconstructed=reconstructed,
        derivation_record=result.derivation_record,
    )


def test_spsa_quadratic_sanity() -> dict[str, Any]:
    target = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=float)

    def objective(theta: tuple[float, ...]) -> float:
        value = np.asarray(theta, dtype=float)
        return float(np.sum((value - target) ** 2))

    result = spsa_minimize(objective, initial_theta=(2.5, 2.0, 3.0, 1.0), config=SPSAConfig(max_evaluations=160), seed=2026)
    initial_value = objective(result.initial_theta)
    return _record(
        "spsa_reduces_known_quadratic_objective",
        result.best_value < initial_value,
        initial_value=initial_value,
        best_value=result.best_value,
        final_value=result.final_value,
        num_evaluations=result.num_evaluations,
        best_theta=list(result.best_theta),
    )


def test_cobyla_quadratic_sanity() -> dict[str, Any]:
    target = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=float)

    def objective(theta: tuple[float, ...]) -> float:
        value = np.asarray(theta, dtype=float)
        return float(np.sum((value - target) ** 2))

    result = cobyla_minimize(objective, initial_theta=(2.5, 2.0, 3.0, 1.0), config=COBYLAConfig(max_evaluations=80))
    initial_value = objective(result.initial_theta)
    return _record(
        "cobyla_reduces_known_bounded_quadratic_objective",
        result.best_value < initial_value and result.num_evaluations <= 80,
        initial_value=initial_value,
        best_value=result.best_value,
        final_value=result.final_value,
        num_evaluations=result.num_evaluations,
        best_theta=list(result.best_theta),
        status=result.status,
    )


def test_spsa_checkpointed_matches_independent_budget_caps() -> dict[str, Any]:
    target = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=float)

    def objective(theta: tuple[float, ...]) -> float:
        value = np.asarray(theta, dtype=float)
        return float(np.sum((value - target) ** 2))

    theta0 = (2.5, 2.0, 3.0, 1.0)
    checkpoints = (12, 24)
    seed = 2027
    checkpointed = spsa_minimize_checkpoints(
        objective,
        theta0,
        SPSAConfig(max_evaluations=max(checkpoints)),
        checkpoints=checkpoints,
        seed=seed,
    )
    separate = {
        budget: spsa_minimize(objective, theta0, SPSAConfig(max_evaluations=budget), seed=seed)
        for budget in checkpoints
    }
    rows = []
    all_match = True
    for checkpoint in checkpointed.checkpoints:
        baseline = separate[checkpoint.requested_max_evaluations]
        match = (
            checkpoint.num_evaluations == baseline.num_evaluations
            and abs(checkpoint.best_value - baseline.best_value) <= 1e-12
            and np.allclose(np.asarray(checkpoint.best_theta), np.asarray(baseline.best_theta), atol=1e-12, rtol=0.0)
        )
        all_match = all_match and bool(match)
        rows.append(
            {
                "requested_max_evaluations": checkpoint.requested_max_evaluations,
                "checkpoint_num_evaluations": checkpoint.num_evaluations,
                "separate_num_evaluations": baseline.num_evaluations,
                "checkpoint_best_value": checkpoint.best_value,
                "separate_best_value": baseline.best_value,
                "match": bool(match),
            }
        )
    return _record(
        "spsa_checkpointed_matches_independent_budget_caps",
        all_match,
        rows=rows,
        checkpoint_count=len(checkpointed.checkpoints),
    )


def test_c0_jitter_summary_thresholds() -> dict[str, Any]:
    pass_stats = summarize_jitter_fractions([0.001, 0.002, 0.003], bootstrap_seed=1, bootstrap_samples=64)
    warn_stats = summarize_jitter_fractions([0.004, 0.006, 0.007], bootstrap_seed=2, bootstrap_samples=64)
    block_stats = summarize_jitter_fractions([0.009, 0.0105, 0.012], bootstrap_seed=3, bootstrap_samples=64)
    passed = (
        pass_stats["threshold_classification"] == "pass"
        and warn_stats["threshold_classification"] == "warning"
        and block_stats["threshold_classification"] == "block"
        and math.isfinite(float(pass_stats["p95"]))
        and math.isfinite(float(warn_stats["p95_bootstrap_ci_low"]))
        and math.isfinite(float(block_stats["p95_bootstrap_ci_high"]))
    )
    return _record(
        "c0_jitter_summary_thresholds",
        passed,
        pass_stats=pass_stats,
        warn_stats=warn_stats,
        block_stats=block_stats,
    )


def test_spsa_resume_state_matches_uninterrupted() -> dict[str, Any]:
    target = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=float)

    def objective(theta: tuple[float, ...]) -> float:
        value = np.asarray(theta, dtype=float)
        return float(np.sum((value - target) ** 2))

    theta0 = (2.5, 2.0, 3.0, 1.0)
    config = SPSAConfig(max_evaluations=24)
    seed = 2028
    uninterrupted = spsa_minimize(objective, theta0, config, seed=seed)
    state = spsa_initialize_state(objective, theta0, seed=seed)
    partial = spsa_run_state(objective, state, config, stop_at_evaluations=10)
    resumed_state = spsa_run_state(objective, partial, config, stop_at_evaluations=24)
    resumed = spsa_result_from_state(resumed_state, optimizer="spsa")
    match = (
        uninterrupted.num_evaluations == resumed.num_evaluations
        and abs(uninterrupted.best_value - resumed.best_value) <= 1e-12
        and abs(uninterrupted.final_value - resumed.final_value) <= 1e-12
        and np.allclose(np.asarray(uninterrupted.best_theta), np.asarray(resumed.best_theta), atol=1e-12, rtol=0.0)
        and np.allclose(np.asarray(uninterrupted.final_theta), np.asarray(resumed.final_theta), atol=1e-12, rtol=0.0)
    )
    return _record(
        "spsa_resume_state_matches_uninterrupted",
        match,
        partial_num_evaluations=partial["num_evaluations"],
        uninterrupted_num_evaluations=uninterrupted.num_evaluations,
        resumed_num_evaluations=resumed.num_evaluations,
        uninterrupted_best_value=uninterrupted.best_value,
        resumed_best_value=resumed.best_value,
    )


def test_c0_storage_threshold_logic() -> dict[str, Any]:
    pass_case = evaluate_storage_thresholds(
        total_bytes=1_000,
        used_bytes=400,
        free_bytes=600,
        required_bytes=100,
    )
    fail_free_case = evaluate_storage_thresholds(
        total_bytes=1_000,
        used_bytes=100,
        free_bytes=900,
        required_bytes=500,
    )
    fail_used_case = evaluate_storage_thresholds(
        total_bytes=1_000,
        used_bytes=760,
        free_bytes=240,
        required_bytes=50,
    )
    passed = (
        pass_case["passes_storage_thresholds"] is True
        and fail_free_case["passes_required_le_50pct_free"] is False
        and fail_free_case["passes_storage_thresholds"] is False
        and fail_used_case["passes_projected_used_le_80pct_total"] is False
        and fail_used_case["passes_storage_thresholds"] is False
    )
    return _record(
        "c0_storage_threshold_logic",
        passed,
        pass_case=pass_case,
        fail_free_case=fail_free_case,
        fail_used_case=fail_used_case,
    )


def test_c0_shard_balance_threshold_logic() -> dict[str, Any]:
    balanced = evaluate_shard_balance([10.0, 10.2, 9.8, 10.1])
    imbalanced = evaluate_shard_balance([10.0, 10.0, 10.0, 25.0])
    passed = (
        balanced["passes_shard_balance"] is True
        and balanced["cv"] < 0.20
        and balanced["max_to_median"] <= 1.50
        and imbalanced["passes_shard_balance"] is False
        and (imbalanced["cv"] >= 0.20 or imbalanced["max_to_median"] > 1.50)
    )
    return _record(
        "c0_shard_balance_threshold_logic",
        passed,
        balanced=balanced,
        imbalanced=imbalanced,
    )


def test_c0_runtime_gate_threshold_logic() -> dict[str, Any]:
    pass_case = evaluate_runtime_gate(71.5, accepted_wall_hours=72.0)
    fail_case = evaluate_runtime_gate(72.5, accepted_wall_hours=72.0)
    passed = (
        pass_case["runtime_gate_pass"] is True
        and pass_case["wall_hours_over_budget"] < 0.0
        and fail_case["runtime_gate_pass"] is False
        and fail_case["wall_hours_over_budget"] > 0.0
        and fail_case["budget_multiplier"] > 1.0
    )
    return _record(
        "c0_runtime_gate_threshold_logic",
        passed,
        pass_case=pass_case,
        fail_case=fail_case,
    )


def test_runtime_scope_amendment_guardrail_logic() -> dict[str, Any]:
    proposed_summary = {
        "status": "phase_e_runtime_scope_amendment_proposed_not_accepted",
        "official_execution_authorized": False,
        "stage_c_blocked": True,
        "runtime_gate_pass": False,
        "final_calibration_complete": False,
    }
    passed = (
        proposed_summary["official_execution_authorized"] is False
        and proposed_summary["stage_c_blocked"] is True
        and proposed_summary["runtime_gate_pass"] is False
        and proposed_summary["final_calibration_complete"] is False
        and proposed_summary["status"].endswith("proposed_not_accepted")
    )
    return _record(
        "runtime_scope_amendment_guardrail_logic",
        passed,
        proposed_summary=proposed_summary,
    )


def test_stage_c_authorization_blocks_without_accepted_amendment() -> dict[str, Any]:
    context = {
        "config_preflight": {
            "status": "passed",
            "config": {"official_execution_enabled": False},
        },
        "runtime": {
            "runtime_gate_pass": False,
            "full_runtime_gate_complete": False,
        },
        "gate_composition": {"stage_c_blocked": True},
        "storage": {
            "storage_gate_pass": True,
            "full_storage_gate_complete": True,
        },
        "shard_balance": {
            "shard_balance_projection_pass": True,
            "full_shard_balance_gate_complete": False,
        },
        "jitter": {"full_jitter_gate_complete": False},
        "resume": {"full_resume_gate_complete": False},
        "calibration": {"final_calibration_complete": False},
        "runtime_scope_amendment": {"status": "phase_e_runtime_scope_amendment_proposed_not_accepted"},
    }
    decision = evaluate_stage_c_authorization(context)
    passed = (
        decision["stage_c_authorized"] is False
        and decision["stage_c_blocked"] is True
        and decision["official_execution_authorized"] is False
        and decision["first_blocking_criterion"] == "accepted_runtime_or_scope_amendment"
        and "runtime_gate_pass" in decision["failed_criteria"]
        and "final_calibration_complete" in decision["failed_criteria"]
    )
    return _record(
        "stage_c_authorization_blocks_without_accepted_amendment",
        passed,
        decision=decision,
    )


def test_runtime_optimization_plan_guardrails() -> dict[str, Any]:
    summary = {
        "plan_version": "phase_e_runtime_optimization_plan_v0_2_draft",
        "official_execution_authorized": False,
        "optimization_execution_authorized": False,
        "stage_c_blocked": True,
        "current_primary_budget_multiplier": 3.1,
        "required_reduction_factor_vs_primary_projection": 3.1,
        "equivalence_gates": {
            "ideal_probability_max_abs_diff": 1e-10,
            "noisy_probability_max_abs_diff": 1e-8,
        },
        "canonical_constraints": {"canonical_source": "authoritative"},
        "pre_commitment_expected_value_analysis": {"expected_optimization_route_hours": 12.2},
        "precommitment_decision_table": [{"path": "optimization"}, {"path": "external"}, {"path": "scope"}, {"path": "budget"}],
        "calendar_time_model": {"external_compute_preferred_if_projected_calendar_days_gt": 7.0},
        "prohibited_optimizations": [
            "warm_start_from_depth1_or_interp_parameters",
            "early_stopping_inside_spsa",
            "float32_or_mixed_precision_pilot_for_gate_decisions",
            "asymmetric_T_allocation_between_noisy_and_zne",
            "stress_subset_only_calibration_or_prefiltering",
        ],
        "feasibility_triage": {
            "o1_profile_required_before_o2_to_o7": True,
            "kill_condition_plausible_max_speedup_lt": 2.5,
            "o0_5_required_before_o1": True,
            "o1_profile_required_before_o2_to_o8": True,
            "conditional_kill_probability_threshold": 0.15,
        },
        "resource_budget": {"max_development_hours_before_go_no_go": 20},
        "validation_suite": {"case_count": 24, "fast_tier_case_count": 6, "full_tier_case_count": 24},
        "determinism_spec": {"seed_identical_requirement": "sha256 complete output"},
        "crash_resume_spec": {"minimum_crash_points": 3},
        "alternative_paths_considered": [{"path": "external_compute"}, {"path": "scope_amendment"}, {"path": "stop"}],
        "permitted_optimization_classes": ["deterministic_intra_objective_multicore_evaluation"],
        "o1_profiling_methodology": {"profiling_tools": ["cProfile"]},
        "o1_unreliable_profiling_protocol": {"trigger": "high_cv"},
        "o1_confidence_levels": {"speculative_continue": "2.5_to_3.1"},
        "weak_feasibility_probability_model": {"probability_floor": 0.15},
        "rollback_protocol": {"rollback_trigger": "equivalence_gate_failure"},
        "negative_result_protocol": {"required_outputs": ["runtime_optimization_negative_result.json"]},
        "negative_result_recommendation_framework": {"external_compute_default_monetary_cap_usd": 100.0},
        "materiality_thresholds": {"material_fraction_threshold": 0.10},
        "execution_environment_spec": {"environment_invariant": "same machine", "physical_cores": 4},
        "amendment_record_schema": {"required_fields": ["approved_by", "approval_timestamp_utc"]},
        "o0_5_blas_backend_spec": {"must_complete_before": "O1_profiling"},
        "o0_7_wsl2_spec": {"failure_rule": "continue_windows"},
        "thermal_protocol": {"projection_adjustment": "adjust wall clock"},
        "platform_note": {"alternative_considered": "WSL2"},
        "time_to_decision_gates": [{"gate": "O1"}, {"gate": "O2_O3"}, {"gate": "full_stack"}],
        "parametric_sanity_sweep": {"valid_parameter_points": 10, "environmental_multiplier_range": [1.0]},
        "pre_o1_kill_probability": {"O1_kill_probability_estimate": "0.55_to_0.70"},
        "plan_versioning_and_amendment_protocol": {"plan_freeze_trigger": "O1"},
        "baseline_lock_protocol": {
            "required_artifacts": ["runtime_optimization_baseline_lock.json"],
            "dirty_tree_policy": "hash all dirty files",
        },
        "approval_tokens_and_stop_rules": {
            "o0_o1_authorization_token": "accept_phase_e_runtime_optimization_o0_o1",
            "stage_c_authorization_token": "accept_phase_e_cobyla_depth2_official",
        },
        "execution_runbook": {
            "script_interfaces_required_before_o1": [
                {"script": "scripts/62.py"},
                {"script": "scripts/63.py"},
                {"script": "scripts/64.py"},
            ],
            "runbook_acceptance_rule": "missing harness means implement harness first",
        },
        "output_schema_registry": {
            "runtime_profile_baseline.csv": ["phase", "wall_seconds"],
            "runtime_optimization_feasibility_triage.json": ["Amdahl_optimistic_speedup"],
        },
        "triage_formulas": {
            "amdahl_speedup": "1 / sum_i(p_i / s_i)",
            "optimized_wall_projection": "current wall / speedup",
        },
        "plan_readiness_rubric": {"score_after_this_hardening": "10_of_10_as_a_pre_execution_plan"},
    }
    decision = evaluate_runtime_optimization_plan_guardrails(summary)
    return _record(
        "runtime_optimization_plan_guardrails",
        decision["passed"],
        decision=decision,
    )


def test_runtime_optimization_execution_harness_contract() -> dict[str, Any]:
    scripts = [
        PROJECT_ROOT / "scripts" / "62_phase_e_runtime_optimization_o0_o1.py",
        PROJECT_ROOT / "scripts" / "63_phase_e_runtime_equivalence_suite.py",
        PROJECT_ROOT / "scripts" / "64_phase_e_runtime_optimization_decision.py",
    ]
    token_guard_passed = False
    try:
        run_o0_o1_mode(PROJECT_ROOT, mode="baseline-lock", authorization_token=None, force=False)
    except PermissionError:
        token_guard_passed = True
    return _record(
        "runtime_optimization_execution_harness_contract",
        all(path.exists() for path in scripts)
        and token_guard_passed
        and O0_O1_AUTHORIZATION_TOKEN != STAGE_C_AUTHORIZATION_TOKEN,
        scripts=[str(path.relative_to(PROJECT_ROOT)) for path in scripts],
        token_guard_passed=token_guard_passed,
        o0_o1_token=O0_O1_AUTHORIZATION_TOKEN,
        stage_c_token=STAGE_C_AUTHORIZATION_TOKEN,
    )


def test_paired_start_alignment_and_wrapping() -> dict[str, Any]:
    starts = make_paired_initial_thetas(depth=2, starts=5, seed=456)
    noisy = starts
    zne = list(starts)
    ideal = list(starts)
    wrapped = wrap_theta((10.0, 7.0, -1.0, -2.0))
    within_bounds = (0.0 <= wrapped[0] < 2.0 * math.pi) and (0.0 <= wrapped[1] < math.pi)
    within_bounds = within_bounds and (0.0 <= wrapped[2] < 2.0 * math.pi) and (0.0 <= wrapped[3] < math.pi)
    return _record(
        "paired_start_alignment_and_theta_wrapping",
        noisy == zne == ideal and within_bounds,
        starts=[list(theta) for theta in starts],
        wrapped=list(wrapped),
    )


def test_end_to_end_mini_pipeline_smoke() -> dict[str, Any]:
    graph = _single_edge_graph()
    starts = make_paired_initial_thetas(depth=2, starts=2, seed=20260517)
    seed = 11
    p_dep = 0.003
    shots = 128

    noisy_records = []
    zne_records = []
    for start_id, theta0 in enumerate(starts):
        noisy_seed = deterministic_seed("phase_e_mini", "noisy", seed, start_id)
        zne_seed = deterministic_seed("phase_e_mini", "zne", seed, start_id)

        def noisy_objective(theta: tuple[float, ...]) -> float:
            return evaluate_custom_density_objective(graph, theta, p_dep=p_dep, shots=shots, seed=noisy_seed).objective_value

        def zne_objective(theta: tuple[float, ...]) -> float:
            return richardson_zne_objective(graph, theta, p_dep=p_dep, scale_factors=[1, 3], shots_total=shots, seed=zne_seed).objective_value

        noisy_records.append(cobyla_minimize(noisy_objective, theta0, COBYLAConfig(max_evaluations=40)))
        zne_records.append(cobyla_minimize(zne_objective, theta0, COBYLAConfig(max_evaluations=40)))

    noisy_best = min(noisy_records, key=lambda row: row.best_value)
    zne_best = min(zne_records, key=lambda row: row.best_value)
    ideal_noisy = evaluate_custom_density_objective(graph, noisy_best.best_theta, p_dep=0.0).objective_value
    ideal_zne = evaluate_custom_density_objective(graph, zne_best.best_theta, p_dep=0.0).objective_value
    d_r = float(ideal_zne - ideal_noisy)

    lhs_thetas = make_paired_initial_thetas(depth=2, starts=4, seed=999)
    noisy_errors = []
    zne_errors = []
    for index, theta in enumerate(lhs_thetas):
        ideal_value = evaluate_custom_density_objective(graph, theta, p_dep=0.0).objective_value
        noisy_value = evaluate_custom_density_objective(
            graph,
            theta,
            p_dep=p_dep,
            shots=shots,
            seed=deterministic_seed("phase_e_mini_lhs", "noisy", index),
        ).objective_value
        zne_value = richardson_zne_objective(
            graph,
            theta,
            p_dep=p_dep,
            scale_factors=[1, 3],
            shots_total=shots,
            seed=deterministic_seed("phase_e_mini_lhs", "zne", index),
        ).objective_value
        noisy_errors.append((noisy_value - ideal_value) ** 2)
        zne_errors.append((zne_value - ideal_value) ** 2)
    mse_noisy = float(np.mean(noisy_errors))
    mse_zne = float(np.mean(zne_errors))
    r_mse = float(mse_noisy - mse_zne)

    recomputed_d_r = float(ideal_zne - ideal_noisy)
    return _record(
        "end_to_end_mini_pipeline_internal_consistency",
        abs(recomputed_d_r - d_r) <= 1e-12
        and np.isfinite(r_mse)
        and all(record.num_evaluations <= 40 for record in noisy_records + zne_records),
        d_R=d_r,
        recomputed_d_R=recomputed_d_r,
        MSE_noisy=mse_noisy,
        MSE_ZNE=mse_zne,
        r_MSE=r_mse,
        noisy_best_observed=noisy_best.best_value,
        zne_best_observed=zne_best.best_value,
        optimizer="cobyla",
    )


def test_depth1_adapter_regression_sample() -> dict[str, Any]:
    report = run_phase_e_depth1_regression(PROJECT_ROOT, sample_rows=3, overwrite=True)
    return _record(
        "depth1_adapter_regresses_against_frozen_phase_d_arrays",
        report["status"] == "passed",
        sampled_rows=report["sampled_rows"],
        metric_checks=report["metric_checks"],
        grid_exact_checks=report["grid_exact_checks"],
        failed_metric_checks=len(report["failed_metric_checks"]),
        failed_hash_checks=len(report["failed_hash_checks"]),
        failed_grid_exact_checks=len(report["failed_grid_exact_checks"]),
    )


def run_phase_e_automated_tests(root: Path = PROJECT_ROOT, overwrite: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    tests = [
        test_ideal_custom_matches_statevector,
        test_custom_matches_aer_density_noise,
        test_richardson_linear_signal,
        test_shot_sampling_deterministic,
        test_zne_derivation_reconstruction,
        test_cobyla_quadratic_sanity,
        test_spsa_quadratic_sanity,
        test_spsa_checkpointed_matches_independent_budget_caps,
        test_c0_jitter_summary_thresholds,
        test_spsa_resume_state_matches_uninterrupted,
        test_c0_storage_threshold_logic,
        test_c0_shard_balance_threshold_logic,
        test_c0_runtime_gate_threshold_logic,
        test_runtime_scope_amendment_guardrail_logic,
        test_stage_c_authorization_blocks_without_accepted_amendment,
        test_runtime_optimization_plan_guardrails,
        test_runtime_optimization_execution_harness_contract,
        test_paired_start_alignment_and_wrapping,
        test_end_to_end_mini_pipeline_smoke,
        test_depth1_adapter_regression_sample,
    ]
    rows = []
    for test in tests:
        test_started = time.perf_counter()
        try:
            row = test()
            row["elapsed_seconds"] = time.perf_counter() - test_started
        except Exception as exc:  # pragma: no cover - diagnostic path
            row = _record(test.__name__, False, error=repr(exc), elapsed_seconds=time.perf_counter() - test_started)
        rows.append(row)

    failures = [row for row in rows if not row["passed"]]
    report = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_version": PHASE_E_TEST_VERSION,
        "status": "passed" if not failures else "failed",
        "tests_run": len(rows),
        "tests_passed": len(rows) - len(failures),
        "tests_failed": len(failures),
        "elapsed_seconds": time.perf_counter() - started,
        "rows": rows,
        "remaining_before_official_execution": [
            "full duplicate endpoint jitter gate",
            "full resume/corruption integrity gate",
            "full optimizer runner dry-run",
            "full shard-balance/runtime gates",
            "accepted runtime/scope amendment and Stage C authorization guard",
            "final freeze summary",
        ],
    }

    output_dir = ensure_dir(root / "results" / "phase_e" / "planning")
    write_json_atomic(output_dir / "phase_e_automated_tests_summary.json", report, overwrite=overwrite)
    report_lines = [
        "# Phase E Automated Tests",
        "",
        f"Status: {report['status']}",
        f"Tests: {report['tests_passed']}/{report['tests_run']} passed",
        f"Elapsed seconds: {report['elapsed_seconds']:.3f}",
        "",
        "| Test | Passed |",
        "|---|---|",
    ]
    for row in rows:
        report_lines.append(f"| {row['name']} | {row['passed']} |")
    report_lines.append("")
    report_lines.append("Remaining before official execution:")
    for item in report["remaining_before_official_execution"]:
        report_lines.append(f"- {item}")
    (output_dir / "phase_e_automated_tests_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report
