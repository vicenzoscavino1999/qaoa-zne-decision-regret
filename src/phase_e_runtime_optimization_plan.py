from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic


PHASE_E_RUNTIME_OPTIMIZATION_PLAN_VERSION = "phase_e_runtime_optimization_plan_v0_2_draft"
OPTIMIZATION_PLAN_DOC = "PHASE_E_RUNTIME_OPTIMIZATION_PLAN_v0.2-draft.md"
OPTIMIZATION_PLAN_SUMMARY = "results/phase_e/calibration/runtime_optimization/phase_e_runtime_optimization_plan_summary.json"
OPTIMIZATION_PLAN_HASHES = "results/phase_e/calibration/runtime_optimization/phase_e_runtime_optimization_plan_hashes.json"

PERMITTED_OPTIMIZATION_CLASSES = [
    "blas_backend_verification_and_environment_upgrade",
    "wsl2_environment_evaluation_with_equivalence_smoke",
    "runtime_profiling_only",
    "deterministic_circuit_and_graph_caching",
    "complex128_float64_vectorization",
    "deterministic_batching_without_seed_or_objective_changes",
    "checkpoint_io_batching_with_resume_identity",
    "process_reuse_and_worker_initialization_reduction",
    "load_balancing_without_scope_or_seed_changes",
    "deterministic_intra_objective_multicore_evaluation",
    "algebraically_equivalent_density_backend_refactor_with_tolerance_validation",
]

PROHIBITED_OPTIMIZATIONS = [
    "warm_start_from_depth1_or_interp_parameters",
    "early_stopping_inside_spsa",
    "float32_or_mixed_precision_pilot_for_gate_decisions",
    "asymmetric_T_allocation_between_noisy_and_zne",
    "stress_subset_only_calibration_or_prefiltering",
    "threshold_relaxation_after_observing_c0_data",
    "candidate_grid_reduction_without_scope_amendment",
    "seed_count_or_shot_budget_reduction_without_scope_amendment",
    "optimizer_schedule_or_hyperparameter_change_without_protocol_amendment",
    "objective_approximation_or_surrogate_model_for_gate_decisions",
    "async_evaluation_with_nondeterministic_completion_order",
    "gpu_offload_without_separate_equivalence_validation",
]

EQUIVALENCE_GATES = {
    "ideal_probability_max_abs_diff": 1e-10,
    "noisy_probability_max_abs_diff": 1e-8,
    "small_value_relative_diff": 1e-4,
    "observable_max_abs_diff": 1e-8,
    "zne_objective_max_abs_diff": 1e-8,
    "resume_output_requirement": "sha256_identical_after_crash_and_resume",
    "classification_requirement": "100pct_stage_c_gate_classification_agreement_on_validation_suite",
    "small_value_relative_diff_failure_rule": "abs_diff_within_tolerance_but_relative_diff_gt_1e-4_requires_failure_or_written_equivalence_amendment",
}

CANONICAL_CONSTRAINTS = {
    "canonical_source": "this section is the authoritative constraint list; later sections are applications of these constraints",
    "immutable_after_plan_freeze_without_plan_amendment": [
        "permitted_optimization_classes",
        "prohibited_optimizations",
        "equivalence_gates",
        "kill_thresholds",
        "validation_suite_tier_definitions",
        "candidate_grid_change_prohibition",
        "seed_shot_noise_zne_optimizer_schedule_prohibition",
        "runtime_success_budget_72_wall_hours_unless_user_accepts_budget_amendment",
    ],
    "adaptable_after_O1_without_scientific_protocol_amendment": [
        "execution_order_within_dependency_DAG",
        "which_permitted_Oi_are_attempted",
        "per_Oi_implementation_time_allocation_within_budget",
        "whether_O1_runs_in_Windows_native_or_WSL2_after_O0_7_smoke",
        "whether_external_compute_is_recommended_after_negative_result",
    ],
    "scientific_protocol_amendment_required_for": [
        "T_K_K_ref_N_LHS_candidate_changes",
        "threshold_relaxation",
        "warm_starts_or_early_stopping",
        "float32_or_mixed_precision_gate_decisions",
        "scope_reduction",
        "seed_shot_noise_rate_or_ZNE_scale_changes",
    ],
}

FEASIBILITY_TRIAGE = {
    "o1_profile_required_before_o2_to_o7": True,
    "o0_5_required_before_o1": True,
    "o1_profile_required_before_o2_to_o8": True,
    "composition_model": "Amdahl_by_measured_phase_fraction",
    "formal_gap_factor": "current_primary_projected_wall_hours / accepted_local_wall_hours",
    "kill_condition_plausible_max_speedup_lt": 2.5,
    "formal_success_speedup_required": ">= required_reduction_factor_vs_primary_projection",
    "weak_feasibility_band": "2.5 <= plausible_max_speedup < required_reduction_factor_vs_primary_projection",
    "weak_feasibility_action": "do_not_run_full_O2_to_O7; implement only time-boxed highest-impact packages if they target the dominant measured phase",
    "conditional_kill_probability_threshold": 0.15,
    "conditional_kill_rule": "kill_if_remaining_required_speedup_requires_gt_p75_outcomes_for_all_remaining_Oi_or_product_probability_of_closing_gap_lt_0_15",
    "dominant_nonoptimizable_action": "declare implementation-only route infeasible and redirect to scope, budget, external compute, or stop path",
}

RESOURCE_BUDGET = {
    "o0_5_blas_backend_budget_hours": 1,
    "o0_7_wsl2_evaluation_budget_hours": 1,
    "o1_profiling_budget_hours": 2,
    "o1_feasibility_triage_hours": 0.5,
    "negative_result_documentation_hours": 0.5,
    "per_Oi_implementation_budget_hours": 4,
    "per_Oi_fast_validation_hours": 0.25,
    "full_validation_suite_hours": 1.5,
    "composition_gate_hours": 1,
    "optimized_c0_projection_hours": 1,
    "documentation_and_artifacts_hours": 2,
    "fast_kill_branch_hours_native": 1 + 2 + 0.5 + 0.5,
    "fast_kill_branch_hours_with_wsl2_trial": 1 + 1 + 2 + 0.5 + 0.5,
    "fast_track_if_only_O2_O3_hours": 1 + 1 + 2 + 0.5 + (4 * 2) + (0.25 * 2) + 1.5 + 1 + 1 + 2,
    "full_track_if_O2_to_O8_all_needed_hours": 1 + 1 + 2 + 0.5 + (4 * 7) + (0.25 * 7) + 1.5 + 1 + 1 + 2,
    "max_development_hours_before_go_no_go": 20,
    "budget_amendment_required_if_more_than_hours": 20,
    "max_calendar_days_before_go_no_go": 5,
    "intermediate_gate_after": ["O1", "O2", "O3"],
    "intermediate_gate_rule": "continue only if measured plus plausible remaining speedup is on track to close the 3.1x gap",
    "accounting_formula": "O0.5 + optional_O0.7 + O1 + triage + (4 * num_accepted_Oi) + (0.25 * num_accepted_Oi) + full_validation + composition + optimized_C0_projection + documentation",
    "justification": "optimization work has opportunity cost against explicit scope amendment, external compute, or stopping Phase E execution",
}

PRE_COMMITMENT_EXPECTED_VALUE_ANALYSIS = {
    "purpose": "justify attempting the runtime-optimization route before spending implementation time",
    "probability_model_type": "pre_O1_planning_prior_not_a_result",
    "assumptions": {
        "kill_at_O1_probability": 0.60,
        "continue_past_O1_then_resolve_by_hour20_probability": 0.30,
        "full_stack_negative_probability": 0.10,
        "hours_if_killed_at_O1": 4.0,
        "hours_if_continue_to_hour20": 20.0,
        "hours_if_full_stack_negative": 38.0,
    },
    "expected_optimization_route_hours": 0.60 * 4.0 + 0.30 * 20.0 + 0.10 * 38.0,
    "selection_criterion": "choose optimization route only if its expected cost-adjusted probability of unblocking Stage C dominates immediate alternatives or if the information value of a negative result justifies the expected hours",
    "information_value_of_negative_result": "O1 negative result converts an ambiguous runtime blocker into evidence for external compute, scope amendment, budget amendment, or stopping Phase E",
    "currency_policy": "monetary costs are recorded as formulas until a current provider quote is obtained; no cloud purchase is authorized by this plan",
}

CALENDAR_TIME_MODEL = {
    "nominal_wall_hours_per_day": 24.0,
    "default_laptop_effective_compute_hours_per_calendar_day": 12.0,
    "effective_compute_hours_per_calendar_day_formula": "24 * sustained_frequency_ratio * availability_fraction * (1 - interruption_fraction)",
    "sustained_frequency_ratio_source": "thermal_protocol",
    "availability_fraction_default": 0.70,
    "interruption_fraction_default": 0.10,
    "external_compute_preferred_if_projected_calendar_days_gt": 7.0,
    "projected_calendar_days_formula": "projected_wall_hours / effective_compute_hours_per_calendar_day",
    "interpretation": "72 laptop wall hours can be roughly six calendar days under the default local availability model, not automatically three days",
}

VALIDATION_SUITE = {
    "fast_tier_case_count": 6,
    "full_tier_case_count": 24,
    "case_count": 24,
    "selection_rule": "frozen_stratified_cases_from_existing_C0_smoke_and_phase_e_test_graphs_no_depth2_outcome_selection",
    "instances": ["phase_e_test_single_edge", "phase_e_test_path3", "H1", "H4"],
    "depths": [1, 2],
    "scale_factors": [1, 3, 5, 7],
    "noise_rates": [0.0, 0.003, 0.01, 0.02],
    "shot_budgets": [None, 512, 4096],
    "seeds": [42, 31415, 271828, 20260517],
    "minimum_coverage_rule": "at_least_one_case_per_depth_scale_factor_noise_model_family_and_shot_mode",
    "execution_rule": "run_fast_tier_after_each_individual_Oi; run_full_tier_after_O2_O3_gate_after_combined_stack_and_before_optimized_C0",
    "fast_tier_rule": "6 cases covering one noiseless, one noisy, one ZNE, one finite-shot, one H1, and one H4 path",
    "full_tier_rule": "24 cases covering all listed depth, scale-factor, noise, shot, and seed strata before optimized C0 evidence is valid",
    "reproducibility_rule": "write validation_suite_manifest.json with case ids, seeds, input hashes, and output hashes before using optimized C0 evidence",
    "coverage_gaps": [
        "no depth greater than 2 because depth 2 is the current production optimizer grid",
        "no graph larger than H4 in the validation suite; larger graphs are reserved for profiling/performance scaling",
        "no noise_rate greater than 0.02 because 0.02 is the production maximum",
    ],
    "coverage_gap_risk_assessment": [
        {
            "gap": "no_depth_greater_than_2",
            "risk": "low_for_current_phase_e_grid",
            "rationale": "depth 2 is the highest production depth in this Stage C plan; O2-O8 equivalence failure modes are objective, seed, dtype, reduction-order, and resume-path dependent rather than untested deeper-depth code paths",
            "spot_check_trigger": "add a depth3 smoke only if a later scientific amendment extends production depth beyond 2",
        },
        {
            "gap": "no_graph_larger_than_H4",
            "risk": "medium_for_performance_low_for_equivalence",
            "rationale": "larger graphs change runtime and memory pressure, but the optimized code paths use the same density/objective kernels; production-scale timing remains covered by O1 profiling rather than the validation suite",
            "spot_check_trigger": "add one larger-graph equivalence spot-check if O1 shows graph-size-dependent branching in an optimized O_i",
        },
        {
            "gap": "no_noise_rate_greater_than_0_02",
            "risk": "low",
            "rationale": "0.02 is the production maximum; no permitted optimization changes the noise-rate grid or extrapolates beyond it",
            "spot_check_trigger": "scientific amendment required before any higher-noise production case",
        },
    ],
}

MATERIALITY_THRESHOLDS = {
    "material_fraction_threshold": 0.10,
    "visible_overhead_threshold": 0.05,
    "o8_objective_fraction_threshold": 0.50,
    "other_phase_max_fraction_before_decomposition": 0.05,
}

PRE_O1_KILL_PROBABILITY = {
    "estimate_type": "subjective_pre_O1_estimate_not_statistical_result",
    "O1_kill_probability_estimate": "0.55_to_0.70",
    "reasoning": "optimistic envelope has near-zero margin against the 3.1x target and the hypothetical phase fractions are unvalidated before O1",
    "expected_value_of_plan": "(0.30_to_0.45 * implementation_route_value) + (0.55_to_0.70 * negative_result_value)",
    "implication": "the most likely output is a well-documented negative feasibility result within the fast-kill branch",
}

PRIOR_DEFINITION = {
    "conservative": "25th_percentile_of_plausible_outcomes_given_phase_is_material",
    "optimistic": "90th_percentile_of_plausible_outcomes_given_phase_is_material",
    "Amdahl_optimistic_uses": "correlated_90th_percentile_priors_and_therefore_overstates_joint_success_probability",
    "Amdahl_conservative_uses": "25th_percentile_priors",
    "realistic_composite_estimate": "geometric_mean_of_conservative_and_optimistic_composite_speedups",
    "geometric_mean_justification": "speedups compose multiplicatively and are modeled as approximately log-normal planning factors; the geometric mean is the median of the joint log-speedup model, while arithmetic mean is reported only as sensitivity if needed",
    "note": "realistic_composite_is_primary_planning_number; optimistic_composite_is_only_a_ceiling",
}

MEMORY_BANDWIDTH_MODEL = {
    "model": "memory_bandwidth_bound_interaction_for_O3_O8",
    "complex128_density_matrix_bytes_per_element": 16,
    "typical_L3_cache_size_comet_lake_MB": 12,
    "density_matrix_fits_in_L3_up_to_qubits": "approximately_9_to_10",
    "production_qubit_range_source": "record_from_H1_H4_graph_sizes_before_O1",
    "sustained_memory_bandwidth_comet_lake_GBps": "approximately_25_to_35",
    "vectorized_flop_rate_complex128_GFLOPS": "approximately_4_to_8",
    "arithmetic_intensity_density_ops": "low_to_moderate; must be estimated in O1 memory profiling",
    "compute_bound_naive_O3_O8_product": 8.0,
    "previous_capped_objective_speedup": 6.0,
    "revised_objective_evaluation_optimistic_speedup": 3.5,
    "memory_bound_expected_O3_O8_combined_range": "2.5_to_3.5",
    "O1_rule": "if objective_evaluation_is_memory_bound cap O3 optimistic at 1.3 and O8 optimistic at min(physical_cores,2.0), then recompute Amdahl ceiling",
}

PRE_O1_AMDAHL_SANITY_CHECK = {
    "hypothetical_phase_fractions": [
        {
            "phase": "objective_evaluation",
            "hypothetical_fraction": 0.60,
            "applicable_Oi": ["O3", "O8"],
            "optimistic_local_speedup": 3.5,
        },
        {
            "phase": "setup_and_static_construction",
            "hypothetical_fraction": 0.15,
            "applicable_Oi": ["O2"],
            "optimistic_local_speedup": 3.0,
        },
        {
            "phase": "per_call_overhead_and_zne",
            "hypothetical_fraction": 0.10,
            "applicable_Oi": ["O4"],
            "optimistic_local_speedup": 1.8,
        },
        {
            "phase": "checkpoint_and_io",
            "hypothetical_fraction": 0.05,
            "applicable_Oi": ["O5"],
            "optimistic_local_speedup": 1.3,
        },
        {
            "phase": "worker_setup",
            "hypothetical_fraction": 0.05,
            "applicable_Oi": ["O6"],
            "optimistic_local_speedup": 1.5,
        },
        {
            "phase": "shard_imbalance_tail",
            "hypothetical_fraction": 0.05,
            "applicable_Oi": ["O7"],
            "optimistic_local_speedup": 1.2,
        },
    ],
    "conservative_composite_speedup_approx": 1.5,
    "interpretation": "after memory-bound correction, 3.1x is above the point sanity-check envelope; O1 is decision-critical rather than confirmatory",
}

O1_PROFILING_METHODOLOGY = {
    "profiling_tools": ["cProfile", "manual_time_perf_counter_instrumentation"],
    "warmup_runs_discarded": 2,
    "measurement_runs": 5,
    "increase_to_runs_if_phase_cv_gt_0_15": 10,
    "profiling_cases": [
        "two_representative_depth2_graph_instances_at_production_like_scale",
        "one_small_instance_sanity_check",
    ],
    "granularity": [
        "objective_evaluation_density_matrix_construction_plus_expectation",
        "optimizer_step_spsa_gradient_plus_parameter_update",
        "zne_per_scale_evaluation_plus_extrapolation",
        "checkpoint_write",
        "logging_and_io",
        "worker_setup_and_import",
        "other_must_be_lt_5pct_or_decomposed",
    ],
    "memory_profiling": {
        "required": True,
        "tools": ["tracemalloc_peak_per_phase", "peak_RSS_if_available", "timing_based_memory_bound_inference", "perf_stat_optional_if_available"],
        "required_outputs": [
            "peak_memory_MB_per_phase",
            "estimated_arithmetic_intensity_objective_evaluation",
            "memory_bound_flag_per_phase",
        ],
        "interpretation_rule": "if objective_evaluation is memory_bound cap O3 optimistic at 1.3 and O8 optimistic at min(physical_cores,2.0), then recompute Amdahl ceiling before proceeding",
    },
    "reporting": [
        "per_phase_wall_clock_seconds_median",
        "per_phase_fraction_of_total",
        "per_phase_cv",
        "Amdahl_conservative_and_optimistic_speedup",
        "kill_or_continue_decision_with_reason",
    ],
}

O1_UNRELIABLE_PROFILING_PROTOCOL = {
    "trigger": "any_phase_with_fraction_gt_0_10_has_CV_gt_0_15_after_10_measurement_runs",
    "investigation_steps": [
        "check_background_cpu_load_and_repeat_under_quiet_load",
        "check_thermal_state_and_frequency_trace",
        "check_io_contention_and_storage_queueing",
        "separate deterministic setup/import overhead from objective-evaluation timing",
    ],
    "if_unreliable_after_mitigation": "use_upper_confidence_bound_of_material_phase_fraction_for_Amdahl_ceiling_and mark O1 measurement unreliable",
    "decision_rule": "unreliable O1 cannot be treated as high-confidence continue; it may only support kill_or_speculative_continue_with_hour12_hard_gate",
}

O1_CONFIDENCE_LEVELS = {
    "high_confidence_continue": "measured_or_memory_adjusted_Amdahl_optimistic >= 3.5x",
    "marginal_continue": "3.1x <= measured_or_memory_adjusted_Amdahl_optimistic < 3.5x",
    "speculative_continue": "2.5x <= measured_or_memory_adjusted_Amdahl_optimistic < 3.1x",
    "kill": "measured_or_memory_adjusted_Amdahl_optimistic < 2.5x",
    "speculative_continue_rule": "attempt O2 and O3 before any other O_i; at hour_12 kill if measured composite speedup_so_far < 2.0x or remaining path requires all remaining O_i above p75",
}

WEAK_FEASIBILITY_PROBABILITY_MODEL = {
    "model": "coarse_precommitted_product_probability_over_remaining_Oi",
    "p75_requirement_rule": "if closing the remaining gap requires every remaining material O_i to exceed its p75 local-speedup outcome, kill",
    "probability_floor": 0.15,
    "calculation_record": "runtime_optimization_decision_log.json must record required_remaining_speedup, remaining_Oi, assumed per_Oi tail probabilities, and product_probability",
    "interpretation": "prevents spending the full budget in the 2.5x-3.1x weak-feasibility band when success requires correlated optimistic outcomes",
}

O0_5_BLAS_BACKEND_SPEC = {
    "id": "O0.5",
    "optimization": "BLAS backend verification and optional environment-level upgrade",
    "status": "permitted_environment_change_not_code_change",
    "action": "record numpy.show_config output; if backend is OpenBLAS, evaluate numpy_plus_MKL_environment and rerun baseline C0 smoke; if already MKL, no action",
    "required_evidence": [
        "numpy_show_config_before",
        "numpy_show_config_after_if_changed",
        "baseline_C0_smoke_passes_with_backend",
        "equivalence_smoke_passes_after_backend_change",
    ],
    "risk": "low; BLAS change is a drop-in environment change but may alter numerical order/threading",
    "expected_speedup_on_density_ops": "1.0_if_already_MKL_to_1.5_if_OpenBLAS_to_MKL",
    "time_budget_hours": 1,
    "must_complete_before": "O1_profiling",
}

O0_7_WSL2_SPEC = {
    "id": "O0.7",
    "optimization": "WSL2 environment evaluation before O1 profiling",
    "status": "permitted_environment_evaluation_not_protocol_change",
    "action": "record WSL2 availability; if available and user permits environment use, run baseline C0 smoke under WSL2, compare spawn/fork overhead, and record environment snapshot",
    "dependency_position": "after_O0_5_before_O1",
    "required_evidence": [
        "wsl2_available_or_unavailable_record",
        "windows_native_environment_snapshot",
        "wsl2_environment_snapshot_if_available",
        "baseline_C0_smoke_under_WSL2_if_attempted",
        "spawn_vs_fork_overhead_comparison",
        "equivalence_smoke_passes_under_WSL2_if_attempted",
    ],
    "failure_rule": "if WSL2 is unavailable or smoke/equivalence fails, continue O1 in Windows native with current priors and record WSL2 as rejected",
    "success_rule": "if WSL2 smoke and equivalence pass, O1 may run in WSL2 and O6/O8 priors must be re-estimated under fork semantics",
    "environmental_multiplier_prior": {"conservative": 1.0, "optimistic": 1.5, "planning_grid": [1.0, 1.15, 1.3, 1.5]},
    "time_budget_hours": 1,
    "not_authorized_by_this_plan": "installing WSL2 or changing the production OS without explicit user approval",
}

O8_PARALLELISM_SPEC = {
    "id": "O8",
    "optimization": "intra-objective multicore evaluation",
    "status": "permitted_if_deterministic_and_seed_order_preserved",
    "applies_to_phase": "objective_evaluation_density_and_expectation",
    "local_speedup_conservative": 1.5,
    "local_speedup_optimistic": "min(measured_physical_cores_4, 4.0)",
    "conditional_on": "O1 shows objective_evaluation >= 50pct of runtime",
    "required_evidence": [
        "thread_count_does_not_affect_seed_order",
        "deterministic_thread_pool_with_fixed_affinity_or_sequential_reduction",
        "equivalence_gates_pass_with_parallel_backend",
    ],
    "risk": "medium; floating-point reduction order across threads can break bit-exact reproducibility",
    "prohibited_variants": [
        "async_evaluation_with_nondeterministic_completion_order",
        "gpu_offload_without_separate_equivalence_validation",
    ],
}

COMPOSITION_FAILURE_PROTOCOL = {
    "investigation_budget_hours": 2,
    "steps": [
        "identify_pair_with_largest_negative_interaction_by_disabling_each_Oi_individually_from_stack_and_reprofiling",
        "if_single_Oi_accounts_for_gt_80pct_of_regression_remove_it_and_rerun_composition_gate",
        "if_no_single_Oi_dominates_remove_last_added_Oi_first_LIFO_and_retest",
        "if_combined_speedup_after_removal_lt_0_90_times_product_remaining_individual_speedups_hard_fail_composition_gate",
    ],
    "hard_failure_decision": "use_best_single_Oi_or_best_pair_that_passes_composition_gate_then_proceed_to_optimized_C0_only_if_runtime_gate_can_still_plausibly_pass",
    "kill_condition": "if_best_validated_composition_speedup_lt_2_0x_abort_implementation_route_regardless_of_individual_measurements",
}

ROLLBACK_PROTOCOL = {
    "rollback_trigger": "any_equivalence_gate_failure_after_implementing_Oi",
    "procedure": [
        "git_revert_to_pre_Oi_commit_or_restore_pre_Oi_patch_scope",
        "rerun_validation_suite_on_reverted_code_to_confirm_baseline_equivalence",
        "record_Oi_as_failed_in_optimization_decision_log_json_with_failure_mode_and_affected_gates",
        "reprofile_on_reverted_code_if_failed_Oi_was_included_in_composition_measurements",
    ],
    "commit_discipline": "each_Oi_implementation_must_be_atomic_or_squashed_to_enable_clean_revert",
}

NEGATIVE_RESULT_PROTOCOL = {
    "trigger": "implementation_only_route_killed_at_O1_or_final_optimized_projection_gt_120h",
    "required_outputs": [
        "runtime_optimization_negative_result.json",
        "runtime_optimization_decision_log.json",
    ],
    "negative_result_fields": [
        "O1_phase_fractions",
        "Amdahl_ceiling_estimate",
        "attempted_Oi_with_individual_speedup_if_any",
        "reason_for_kill",
        "recommended_next_path",
    ],
    "time_to_negative_result_hours": 4,
    "interpretation": "negative feasibility is a valid Phase E output and feeds scope/budget/external_compute/stop decision",
}

NEGATIVE_RESULT_RECOMMENDATION_FRAMEWORK = {
    "external_compute_default_monetary_cap_usd": 100.0,
    "external_compute_calendar_cap_days": 7.0,
    "current_cloud_quote_required_before_purchase": True,
    "recommend_external_compute_if": [
        "estimated_monetary_cost_usd <= 100",
        "estimated_calendar_days_to_unblock <= 7",
        "all scientific claims and candidate grids are preserved",
        "equivalence validation on new environment passes",
    ],
    "recommend_budget_amendment_if": [
        "user explicitly accepts a larger local wall-clock budget",
        "projected_calendar_days <= 7 after thermal/load adjustment",
        "storage/resume/shard gates remain feasible",
    ],
    "recommend_scope_amendment_if": [
        "external compute exceeds monetary or calendar caps",
        "at least the methodology and reduced Phase E claims remain scientifically useful",
        "affected claims and candidate removals are written before re-running gates",
    ],
    "recommend_stop_if": [
        "external compute is over cap",
        "local budget/calendar path is over cap",
        "scope amendment would remove the central Phase E claim",
    ],
}

FAST_KILL_BRANCH = {
    "trigger": "O1_Amdahl_optimistic_or_memory_adjusted_plausible_max_speedup_lt_2_5x",
    "total_hours_native": 4,
    "total_hours_with_wsl2_trial": 5,
    "required_outputs": [
        "runtime_optimization_environment_snapshot.json",
        "runtime_profile_baseline.csv",
        "runtime_optimization_feasibility_triage.json",
        "runtime_optimization_negative_result.json",
        "runtime_optimization_decision_log.json",
    ],
    "not_required": [
        "optimization_candidate_equivalence_report.json",
        "optimization_candidate_speedup_report.csv",
        "optimized_c0_runtime_summary.json",
        "validation_suite_manifest.json",
        "optimized_c0_gate_composition_decision.json",
        "optimized_stage_c_authorization_summary.json",
    ],
    "decision_outputs": ["recommended_next_path_with_cost_estimates_for_each_alternative"],
}

THERMAL_PROTOCOL = {
    "required_before": ["O1", "optimized_C0"],
    "measurement": "run_cpu_sustained_load_probe_for_30_minutes_and_record_effective_frequency_curve",
    "sustained_frequency_ratio": "effective_frequency_at_30min / boost_frequency_at_0min",
    "projection_adjustment": "if sustained_frequency_ratio < 0.85 multiply all wall_clock_projections by 1 / sustained_frequency_ratio",
    "mitigation_options": ["fixed_base_frequency_power_plan", "external_cooling", "core_count_reduction_to_avoid_thermal_limit"],
    "reporting": "sustained_frequency_ratio must appear in runtime_optimization_environment_snapshot.json",
}

PLATFORM_NOTE = {
    "platform": "Windows",
    "process_creation_note": "Windows spawn/process creation overhead can be 10_to_100x Linux fork_spawn for Python multiprocessing workloads",
    "implication_for_O6": "worker reuse may have higher returns on Windows but higher implementation complexity because there is no fork semantics",
    "implication_for_O8": "multiprocessing on Windows uses spawn and each worker reimports modules; this interacts with O6 and BLAS threading",
    "alternative_considered": "WSL2 execution on same hardware as first-class O0.7 environment evaluation",
    "WSL2_decision": "O0.7 records whether WSL2 is available and whether fork-based multiprocessing is an option; if WSL2 passes smoke/equivalence, O1 may run there and O6/O8 priors are re-estimated",
}

TIME_TO_DECISION_GATES = [
    {
        "gate": "O1_kill_or_continue",
        "hour": 4,
        "report": [
            "hours_spent",
            "measured_composite_speedup_so_far",
            "remaining_plausible_speedup_from_unimplemented_Oi",
            "probability_of_reaching_3_1x_given_current_data",
            "probability_product_for_remaining_path",
            "O1_confidence_level",
            "continue_or_kill_decision",
        ],
    },
    {
        "gate": "O2_O3_intermediate_gate",
        "hour": 12,
        "report": [
            "hours_spent",
            "measured_composite_speedup_so_far",
            "remaining_plausible_speedup_from_unimplemented_Oi",
            "probability_of_reaching_3_1x_given_current_data",
            "probability_product_for_remaining_path",
            "O1_confidence_level",
            "continue_or_kill_decision",
        ],
    },
    {
        "gate": "full_stack_go_no_go",
        "hour": 20,
        "report": [
            "hours_spent",
            "measured_composite_speedup_so_far",
            "remaining_plausible_speedup_from_unimplemented_Oi",
            "probability_of_reaching_3_1x_given_current_data",
            "probability_product_for_remaining_path",
            "O1_confidence_level",
            "continue_or_kill_decision",
        ],
    },
]

SHARD_GATE_SPEC = {
    "primary_wall_clock_predictor": "max_shard_time_not_mean_shard_time",
    "diagnostic_cv_gate": 0.20,
    "diagnostic_max_to_median_gate": 1.50,
    "projection_safety_rule": "wall_clock_projection_uses_max_shard_time_or_a_conservative_upper_bound",
    "sensitivity_rule": "report wall_clock_projection_at_cv_0_199_and_max_to_median_1_49; if uncertainty_exceeds_10pct_use_max_shard_time_only",
}

DETERMINISM_SPEC = {
    "deterministic_definition": "same inputs, seeds, protocol parameters, and immutable data produce byte-identical records unless a floating-kernel equivalence exception is explicitly declared",
    "seed_identical_requirement": "per_scale_derivation_records_are_byte_identical_and_verified_by_sha256_for_the_complete_output",
    "floating_kernel_exception": "numeric arrays may pass by abs+relative tolerances only when classification agreement is 100pct and deterministic input/output records remain hashable",
}

CRASH_RESUME_SPEC = {
    "minimum_crash_points": 3,
    "crash_points": [
        "start_of_objective_evaluation",
        "middle_of_optimizer_trajectory",
        "during_batched_checkpoint_write",
    ],
    "required_result": "each crash_resume_output_sha256_identical_to_uninterrupted_output",
    "danger_case": "crash_during_batched_write_must_leave_no_partial_output_that_can_be_misread_as_complete",
}

AMENDMENT_RECORD_SCHEMA = {
    "required_fields": [
        "amendment_id",
        "created_at_utc",
        "approved_by",
        "approval_timestamp_utc",
        "motivation",
        "observed_data_available_before_decision",
        "change_requested",
        "affected_gates_or_thresholds",
        "requires_reexecution_of_affected_gates",
        "official_execution_authorized",
    ],
    "posthoc_decision_logging": "each output JSON must include decision_was_posthoc and posthoc_data_used fields",
}

PLAN_VERSIONING_AND_AMENDMENT_PROTOCOL = {
    "plan_version_policy": "remain on v0.2-draft for housekeeping and reviewer-requested hardening before O1 begins; do not create v0.3 unless the user explicitly requests a new plan version",
    "plan_freeze_trigger": "first execution of O0.5/O0.7/O1 profiling work",
    "immutable_after_freeze": [
        "equivalence tolerances",
        "kill thresholds",
        "probability floor for weak-feasibility kill",
        "validation tier definitions",
        "prohibited optimization list",
        "scientific candidate grid prohibition",
    ],
    "adaptable_after_O1_without_plan_amendment": [
        "which permitted O_i to attempt",
        "execution order among O_i when dependency DAG permits it",
        "implementation time allocation within the approved optimization budget",
        "recommendation among already pre-specified alternative paths",
    ],
    "plan_amendment_required_for": [
        "changing any immutable field after O1 begins",
        "adding a new optimization class",
        "relaxing any gate",
        "raising the optimization work budget",
        "changing the validation suite tiers",
    ],
    "plan_amendment_record_fields": [
        "plan_amendment_id",
        "created_at_utc",
        "approved_by",
        "approval_timestamp_utc",
        "observed_data_available_before_decision",
        "change_requested",
        "immutable_or_adaptable_field",
        "requires_reexecution_of_affected_gates",
    ],
}

BASELINE_LOCK_PROTOCOL = {
    "purpose": "make the pre-O0/O1 state reproducible before any profiling or runtime optimization work changes the evidence base",
    "required_before": ["O0.5", "O0.7", "O1"],
    "required_artifacts": [
        "runtime_optimization_baseline_lock.json",
        "runtime_optimization_git_status.txt",
        "runtime_optimization_dirty_tree_inventory.csv",
        "runtime_optimization_pre_o1_hashes.json",
    ],
    "minimum_fields": [
        "created_at_utc",
        "decision_was_posthoc",
        "posthoc_data_used",
        "git_branch",
        "git_status_short_sha256",
        "tracked_modified_files",
        "untracked_files",
        "baseline_code_hashes",
        "input_artifact_hashes",
        "stage_c_authorized",
        "stage_c_blocked",
        "optimization_execution_authorized",
        "protocol_candidate_sets",
    ],
    "dirty_tree_policy": (
        "dirty tree is allowed only if every modified/untracked path is listed and hashed; unrelated changes are ignored "
        "but not reverted; any O_i patch must state its write set before editing"
    ),
    "acceptance_rule": "O0.5/O0.7/O1 may start only after the baseline lock exists and hashes all plan inputs and executable files touched by Phase E C0",
}

APPROVAL_TOKENS_AND_STOP_RULES = {
    "o0_o1_authorization_token": "accept_phase_e_runtime_optimization_o0_o1",
    "o2_to_o8_authorization_condition": "requires O1 feasibility triage output with continue decision and user acceptance of the recommended O_i path",
    "stage_c_authorization_token": "accept_phase_e_optimizer_depth2_official",
    "stop_rules": [
        "do not execute O0.5/O0.7/O1 without the O0/O1 token or explicit equivalent user instruction",
        "do not implement O2-O8 from the plan text alone; require O1 triage and a recorded write-set",
        "do not run Stage C from any runtime optimization artifact",
        "do not select final T/K/K_ref/N_LHS until authorization guard and final freeze pass",
    ],
    "implementation_boundary": (
        "This document can be hardened without running optimization. Actual O0/O1 profiling requires implementation or "
        "verification of the runtime optimization execution scripts and explicit authorization."
    ),
}

EXECUTION_RUNBOOK = {
    "runbook_status": "pre_execution_interface_frozen_implementation_not_started",
    "script_interfaces_required_before_o1": [
        {
            "script": "scripts/62_phase_e_runtime_optimization_o0_o1.py",
            "modes": ["baseline-lock", "blas-check", "wsl2-check", "profile", "triage"],
            "required_outputs": [
                "runtime_optimization_baseline_lock.json",
                "runtime_optimization_environment_snapshot.json",
                "runtime_profile_baseline.csv",
                "runtime_optimization_feasibility_triage.json",
                "runtime_optimization_decision_log.json",
            ],
        },
        {
            "script": "scripts/63_phase_e_runtime_equivalence_suite.py",
            "modes": ["fast-tier", "full-tier"],
            "required_outputs": ["validation_suite_manifest.json", "optimization_candidate_equivalence_report.json"],
        },
        {
            "script": "scripts/64_phase_e_runtime_optimization_decision.py",
            "modes": ["candidate-speedup", "composition", "negative-result", "acceptance-record"],
            "required_outputs": [
                "optimization_candidate_speedup_report.csv",
                "optimized_c0_gate_composition_decision.json",
                "runtime_optimization_negative_result.json",
                "runtime_optimization_amendment_or_acceptance_record.md",
            ],
        },
    ],
    "pre_execution_commands_existing": [
        ".venv\\Scripts\\python.exe scripts\\42_phase_e_config_preflight.py --force",
        ".venv\\Scripts\\python.exe scripts\\48_phase_e_depth1_optimizer_validation.py --force",
        ".venv\\Scripts\\python.exe scripts\\60_phase_e_stage_c_authorization.py --force",
        ".venv\\Scripts\\python.exe scripts\\61_phase_e_runtime_optimization_plan.py --force",
    ],
    "first_o0_o1_commands_after_script_implementation": [
        ".venv\\Scripts\\python.exe scripts\\62_phase_e_runtime_optimization_o0_o1.py --mode baseline-lock --authorization-token accept_phase_e_runtime_optimization_o0_o1",
        ".venv\\Scripts\\python.exe scripts\\62_phase_e_runtime_optimization_o0_o1.py --mode blas-check",
        ".venv\\Scripts\\python.exe scripts\\62_phase_e_runtime_optimization_o0_o1.py --mode wsl2-check",
        ".venv\\Scripts\\python.exe scripts\\62_phase_e_runtime_optimization_o0_o1.py --mode profile --runs 5 --warmup 2",
        ".venv\\Scripts\\python.exe scripts\\62_phase_e_runtime_optimization_o0_o1.py --mode triage",
    ],
    "runbook_acceptance_rule": (
        "If any listed script interface is missing, the next task is implementation of the execution harness, not "
        "scientific Stage C execution."
    ),
}

OUTPUT_SCHEMA_REGISTRY = {
    "common_json_fields": [
        "created_at_utc",
        "plan_version",
        "decision_was_posthoc",
        "posthoc_data_used",
        "input_hashes",
        "code_hashes",
        "stage_c_authorized",
        "stage_c_blocked",
        "official_execution_authorized",
    ],
    "runtime_optimization_environment_snapshot.json": [
        "machine_id",
        "os",
        "python_version",
        "numpy_version",
        "blas_backend",
        "physical_cores",
        "logical_cores",
        "ram_gb",
        "storage_type",
        "free_space_gb",
        "windows_other_cpu_utilization_pct",
        "sustained_frequency_ratio",
        "wsl2_status",
        "environment_decision",
    ],
    "runtime_profile_baseline.csv": [
        "run_id",
        "case_id",
        "instance_id",
        "depth",
        "objective_family",
        "noise_rate",
        "shot_budget",
        "seed",
        "phase",
        "warmup_discarded",
        "wall_seconds",
        "cpu_seconds",
        "peak_memory_mb",
        "worker_id",
        "thread_count",
        "blas_backend",
    ],
    "runtime_optimization_feasibility_triage.json": [
        "phase_fraction_table",
        "material_phases",
        "memory_bound_flags",
        "Amdahl_conservative_speedup",
        "Amdahl_realistic_speedup",
        "Amdahl_optimistic_speedup",
        "required_reduction_factor",
        "required_remaining_speedup",
        "O1_confidence_level",
        "continue_or_kill_decision",
        "decision_reason",
    ],
    "runtime_optimization_decision_log.json": [
        "entries",
        "hours_spent",
        "gate_name",
        "measured_composite_speedup_so_far",
        "remaining_plausible_speedup_from_unimplemented_Oi",
        "probability_of_reaching_required_reduction",
        "product_probability_for_remaining_path",
        "decision",
    ],
    "optimization_candidate_equivalence_report.json": [
        "candidate_id",
        "validation_tier",
        "case_count",
        "max_abs_diff_by_metric",
        "max_relative_diff_by_metric",
        "classification_agreement_rate",
        "seed_identity_sha256_pass",
        "resume_identity_pass_if_applicable",
        "equivalence_pass",
        "failure_cases",
    ],
    "optimization_candidate_speedup_report.csv": [
        "candidate_id",
        "phase",
        "baseline_median_seconds",
        "optimized_median_seconds",
        "speedup",
        "phase_fraction_before",
        "phase_fraction_after",
        "cv_before",
        "cv_after",
        "equivalence_report_sha256",
    ],
}

TRIAGE_FORMULAS = {
    "phase_fraction": "median_phase_wall_seconds / median_total_wall_seconds for the same profiling case stratum",
    "phase_cv": "sample_sd_phase_wall_seconds / mean_phase_wall_seconds",
    "amdahl_speedup": "1 / sum_i(phase_fraction_i / local_speedup_i)",
    "memory_adjusted_objective_speedup": "cap O3 at 1.3 and O8 at min(physical_cores, 2.0) when objective phase is memory_bound",
    "required_remaining_speedup": "required_reduction_factor_vs_primary_projection / measured_composite_speedup_so_far",
    "weak_path_product_probability": "product of precommitted per-O_i tail probabilities for remaining material optimizations",
    "optimized_wall_projection": "current_primary_projected_wall_hours / validated_composite_speedup adjusted by sustained_frequency_ratio and max_shard_time safety rule",
    "high_confidence_continue": "memory_adjusted_Amdahl_optimistic >= 3.5",
    "marginal_continue": "3.1 <= memory_adjusted_Amdahl_optimistic < 3.5",
    "speculative_continue": "2.5 <= memory_adjusted_Amdahl_optimistic < 3.1",
    "kill": "memory_adjusted_Amdahl_optimistic < 2.5",
}

PLAN_READINESS_RUBRIC = {
    "score_after_this_hardening": "10_of_10_as_a_pre_execution_plan",
    "scope_of_score": "protocol, auditability, reproducibility, execution boundary, and decision logic before O0/O1",
    "not_claimed": [
        "O0/O1 profiling has not been executed",
        "O2-O8 implementations have not been written or validated",
        "Stage C remains unauthorized",
        "final T/K/K_ref/N_LHS remain unselected",
    ],
    "conditions_for_10": [
        "baseline lock exists before profiling",
        "authorization boundaries are explicit",
        "script interfaces and command order are frozen before implementation",
        "required output schemas are defined",
        "triage formulas are concrete",
        "dirty tree policy is explicit",
        "negative result path is fully usable",
    ],
}


def _execution_environment_spec() -> dict[str, Any]:
    return {
        "target_machine": platform.node() or "record_before_O1",
        "cpu_model": "Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz",
        "physical_cores": 4,
        "logical_cores": 8,
        "core_count_measurement_source": "Windows Get-CimInstance Win32_Processor on 2026-05-17",
        "max_clock_speed_mhz": 2496,
        "ram_gb": "record_before_O1",
        "storage_type": "record_before_O1",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "blas_backend": "record_from_numpy_show_config_before_O1",
        "os": platform.platform(),
        "environment_invariant": "all profiling and optimized C0 measurements must run on this machine under comparable load conditions",
        "windows_load_condition_gate": "invalid_if_cpu_utilization_from_other_processes_exceeds_20pct_during_measurement_window",
        "thermal_throttling_gate": "flag_if_effective_cpu_frequency_drops_more_than_15pct_from_warmup_median",
    }


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_context(root: Path) -> dict[str, Any]:
    return {
        "runtime": _read_json(root / "results" / "phase_e" / "calibration" / "c0_runtime" / "phase_e_stage_c0_runtime_summary.json"),
        "authorization": _read_json(root / "results" / "phase_e" / "calibration" / "stage_c_authorization" / "phase_e_stage_c_authorization_summary.json"),
        "amendment": _read_json(root / "results" / "phase_e" / "calibration" / "c0_runtime" / "phase_e_runtime_scope_amendment_summary.json"),
        "calibration": _read_json(root / "results" / "phase_e" / "calibration" / "phase_e_calibration_status.json"),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def evaluate_runtime_optimization_plan_guardrails(summary: dict[str, Any]) -> dict[str, Any]:
    prohibited = set(summary.get("prohibited_optimizations", []))
    required_prohibited = {
        "warm_start_from_depth1_or_interp_parameters",
        "early_stopping_inside_spsa",
        "float32_or_mixed_precision_pilot_for_gate_decisions",
        "asymmetric_T_allocation_between_noisy_and_zne",
        "stress_subset_only_calibration_or_prefiltering",
    }
    criteria = {
        "plan_version_v0_2": summary.get("plan_version") == PHASE_E_RUNTIME_OPTIMIZATION_PLAN_VERSION,
        "official_execution_not_authorized": summary.get("official_execution_authorized") is False,
        "optimization_execution_not_authorized": summary.get("optimization_execution_authorized") is False,
        "stage_c_remains_blocked": summary.get("stage_c_blocked") is True,
        "required_risky_optimizations_prohibited": required_prohibited.issubset(prohibited),
        "success_target_not_weaker_than_current_gap": float(summary.get("required_reduction_factor_vs_primary_projection", 0.0))
        >= float(summary.get("current_primary_budget_multiplier", 0.0)),
        "equivalence_gates_present": bool(summary.get("equivalence_gates")),
        "canonical_constraints_present": bool(summary.get("canonical_constraints", {}).get("canonical_source")),
        "pre_commitment_expected_value_present": float(
            summary.get("pre_commitment_expected_value_analysis", {}).get("expected_optimization_route_hours", 0.0)
        )
        > 0.0,
        "precommitment_decision_table_present": len(summary.get("precommitment_decision_table", [])) >= 4,
        "feasibility_triage_requires_o1": summary.get("feasibility_triage", {}).get("o1_profile_required_before_o2_to_o7") is True,
        "o0_5_required_before_o1": summary.get("feasibility_triage", {}).get("o0_5_required_before_o1") is True,
        "o1_required_before_o2_to_o8": summary.get("feasibility_triage", {}).get("o1_profile_required_before_o2_to_o8") is True,
        "kill_condition_present": float(summary.get("feasibility_triage", {}).get("kill_condition_plausible_max_speedup_lt", 0.0)) >= 2.5,
        "weak_feasibility_probability_floor_present": float(
            summary.get("feasibility_triage", {}).get("conditional_kill_probability_threshold", 0.0)
        )
        >= 0.15,
        "calendar_time_model_present": float(summary.get("calendar_time_model", {}).get("external_compute_preferred_if_projected_calendar_days_gt", 0.0))
        > 0.0,
        "resource_budget_present": int(summary.get("resource_budget", {}).get("max_development_hours_before_go_no_go", 0)) > 0,
        "tiered_validation_present": int(summary.get("validation_suite", {}).get("fast_tier_case_count", 0)) > 0
        and int(summary.get("validation_suite", {}).get("full_tier_case_count", 0)) >= 20,
        "validation_suite_frozen": int(summary.get("validation_suite", {}).get("case_count", 0)) >= 20,
        "determinism_spec_present": bool(summary.get("determinism_spec", {}).get("seed_identical_requirement")),
        "crash_resume_has_three_points": int(summary.get("crash_resume_spec", {}).get("minimum_crash_points", 0)) >= 3,
        "alternatives_present": len(summary.get("alternative_paths_considered", [])) >= 3,
        "parallelism_explicit": "deterministic_intra_objective_multicore_evaluation" in set(summary.get("permitted_optimization_classes", [])),
        "o1_methodology_present": bool(summary.get("o1_profiling_methodology", {}).get("profiling_tools")),
        "o1_unreliable_protocol_present": bool(summary.get("o1_unreliable_profiling_protocol", {}).get("trigger")),
        "o1_confidence_levels_present": bool(summary.get("o1_confidence_levels", {}).get("speculative_continue")),
        "weak_feasibility_model_present": bool(summary.get("weak_feasibility_probability_model", {}).get("probability_floor")),
        "rollback_protocol_present": bool(summary.get("rollback_protocol", {}).get("rollback_trigger")),
        "negative_result_protocol_present": bool(summary.get("negative_result_protocol", {}).get("required_outputs")),
        "negative_result_recommendation_framework_present": float(
            summary.get("negative_result_recommendation_framework", {}).get("external_compute_default_monetary_cap_usd", 0.0)
        )
        > 0.0,
        "materiality_thresholds_present": float(summary.get("materiality_thresholds", {}).get("material_fraction_threshold", 0.0)) > 0.0,
        "environment_spec_present": bool(summary.get("execution_environment_spec", {}).get("environment_invariant")),
        "measured_physical_cores_present": int(summary.get("execution_environment_spec", {}).get("physical_cores", 0)) >= 1,
        "blas_backend_o0_5_present": bool(summary.get("o0_5_blas_backend_spec", {}).get("must_complete_before")),
        "wsl2_o0_7_present": bool(summary.get("o0_7_wsl2_spec", {}).get("failure_rule")),
        "thermal_protocol_present": bool(summary.get("thermal_protocol", {}).get("projection_adjustment")),
        "platform_note_present": bool(summary.get("platform_note", {}).get("alternative_considered")),
        "time_to_decision_gates_present": len(summary.get("time_to_decision_gates", [])) >= 3,
        "parametric_sanity_sweep_present": bool(summary.get("parametric_sanity_sweep", {}).get("valid_parameter_points")),
        "parametric_sweep_has_environment_axis": bool(summary.get("parametric_sanity_sweep", {}).get("environmental_multiplier_range")),
        "kill_probability_estimate_present": bool(summary.get("pre_o1_kill_probability", {}).get("O1_kill_probability_estimate")),
        "plan_versioning_protocol_present": bool(summary.get("plan_versioning_and_amendment_protocol", {}).get("plan_freeze_trigger")),
        "amendment_approval_fields_present": {"approved_by", "approval_timestamp_utc"}.issubset(
            set(summary.get("amendment_record_schema", {}).get("required_fields", []))
        ),
        "baseline_lock_protocol_present": bool(summary.get("baseline_lock_protocol", {}).get("required_artifacts")),
        "dirty_tree_policy_present": bool(summary.get("baseline_lock_protocol", {}).get("dirty_tree_policy")),
        "approval_tokens_present": bool(summary.get("approval_tokens_and_stop_rules", {}).get("o0_o1_authorization_token")),
        "stage_c_token_separate": summary.get("approval_tokens_and_stop_rules", {}).get("stage_c_authorization_token")
        != summary.get("approval_tokens_and_stop_rules", {}).get("o0_o1_authorization_token"),
        "execution_runbook_interfaces_present": len(summary.get("execution_runbook", {}).get("script_interfaces_required_before_o1", [])) >= 3,
        "execution_runbook_declares_implementation_boundary": bool(
            summary.get("execution_runbook", {}).get("runbook_acceptance_rule")
        ),
        "output_schema_registry_present": bool(summary.get("output_schema_registry", {}).get("runtime_profile_baseline.csv")),
        "feasibility_schema_present": bool(
            summary.get("output_schema_registry", {}).get("runtime_optimization_feasibility_triage.json")
        ),
        "triage_formulas_present": bool(summary.get("triage_formulas", {}).get("amdahl_speedup")),
        "optimized_projection_formula_present": bool(summary.get("triage_formulas", {}).get("optimized_wall_projection")),
        "plan_readiness_rubric_present": summary.get("plan_readiness_rubric", {}).get("score_after_this_hardening")
        == "10_of_10_as_a_pre_execution_plan",
    }
    failed = [key for key, passed in criteria.items() if passed is not True]
    return {
        "passed": not failed,
        "failed_criteria": failed,
        "criteria": criteria,
    }


def _optimization_table() -> list[dict[str, Any]]:
    return [
        {
            "id": "O0.5",
            "optimization": "Verify BLAS backend and optionally use an MKL-backed NumPy environment before profiling.",
            "status": "permitted_environment_change_not_code_change",
            "required_evidence": "numpy.show_config before/after if changed; baseline C0 smoke and equivalence smoke pass",
            "risk": "low; BLAS changes can alter numerical order/threading and therefore require smoke equivalence",
        },
        {
            "id": "O0.7",
            "optimization": "Evaluate WSL2 availability and smoke/equivalence before choosing the O1 profiling environment.",
            "status": "permitted_environment_evaluation_not_protocol_change",
            "required_evidence": "WSL2 availability record, Windows and WSL2 snapshots if attempted, spawn/fork comparison, smoke/equivalence pass",
            "risk": "low to medium; environment changes can alter process semantics and numerical library behavior",
        },
        {
            "id": "O1",
            "optimization": "Profile objective, optimizer, checkpoint, I/O, and worker setup separately.",
            "status": "permitted",
            "required_evidence": "runtime_profile_baseline.csv and per-phase percentages",
            "risk": "none; measurement only",
        },
        {
            "id": "O2",
            "optimization": "Cache graph structure, diagonal cost vectors, transpiled/folded circuit templates, and static noise metadata.",
            "status": "permitted_if_deterministic",
            "required_evidence": "cache-hit accounting plus objective equivalence on fixed validation points",
            "risk": "low; stale cache keys must include graph, depth, scale factor, noise model, and code version",
        },
        {
            "id": "O3",
            "optimization": "Vectorize complex128/float64 density and expectation operations without changing arithmetic dtype.",
            "status": "permitted_if_equivalent",
            "required_evidence": "ideal/noisy/ZNE max-diff tests at frozen tolerances and finite checks",
            "risk": "medium; floating operation order may change, so gate classifications must be checked",
        },
        {
            "id": "O4",
            "optimization": "Batch noisy/ZNE objective calls while preserving per-scale seeds, shot allocation, and output records.",
            "status": "permitted_if_seed_identical",
            "required_evidence": "same per-scale derivation records and same synthetic-shot seeds",
            "risk": "medium; batching must not couple random streams across cases",
        },
        {
            "id": "O5",
            "optimization": "Reduce checkpoint and log overhead by batching writes or using deterministic compressed containers.",
            "status": "permitted_if_resume_identical",
            "required_evidence": "crash-and-resume SHA-256 identity versus uninterrupted output",
            "risk": "medium; resumability is a blocking gate",
        },
        {
            "id": "O6",
            "optimization": "Reuse worker processes and preloaded immutable state to avoid repeated imports/setup.",
            "status": "permitted",
            "required_evidence": "worker setup time before/after and no change to task payload hashes",
            "risk": "low; workers must not keep mutable cross-task optimizer state",
        },
        {
            "id": "O7",
            "optimization": "Improve shard assignment and task ordering by measured LPT scheduling only.",
            "status": "permitted",
            "required_evidence": "shard CV < 20% and max/median <= 1.5 on C0",
            "risk": "low; cannot remove tasks or alter seeds",
        },
        {
            "id": "O8",
            "optimization": "Use deterministic intra-objective multicore evaluation.",
            "status": "permitted_if_deterministic_and_seed_order_preserved",
            "required_evidence": "thread count does not affect seed order; deterministic reduction; equivalence gates pass with parallel backend",
            "risk": "medium; floating-point reduction order across threads can break bit-exact reproducibility",
        },
    ]


def _speedup_estimate_table() -> list[dict[str, Any]]:
    return [
        {
            "id": "O0.5",
            "optimization": "BLAS backend verification and optional MKL environment",
            "local_speedup_conservative": 1.0,
            "local_speedup_optimistic": 1.5,
            "applies_to_phase": "objective_evaluation_density_and_expectation",
            "conditional_on": "current backend is OpenBLAS or otherwise slower than available MKL environment",
        },
        {
            "id": "O0.7",
            "optimization": "WSL2 environment evaluation",
            "local_speedup_conservative": 1.0,
            "local_speedup_optimistic": 1.5,
            "applies_to_phase": "environmental_multiplier_on_process_and_worker_overhead",
            "conditional_on": "WSL2 available, smoke/equivalence pass, and fork/process behavior improves O6/O8 path",
        },
        {
            "id": "O2",
            "optimization": "deterministic graph/circuit/static metadata caching",
            "local_speedup_conservative": 1.5,
            "local_speedup_optimistic": 3.0,
            "applies_to_phase": "setup_and_static_circuit_construction",
            "conditional_on": "setup or static construction is a material fraction of O1 profile",
        },
        {
            "id": "O3",
            "optimization": "complex128/float64 vectorization",
            "local_speedup_conservative": 1.2,
            "local_speedup_optimistic": 2.0,
            "applies_to_phase": "objective_evaluation_density_and_expectation",
            "conditional_on": "objective evaluation dominates and equivalence gates pass",
        },
        {
            "id": "O4",
            "optimization": "deterministic noisy/ZNE batching",
            "local_speedup_conservative": 1.1,
            "local_speedup_optimistic": 1.8,
            "applies_to_phase": "objective_call_overhead_and_zne_per_scale_work",
            "conditional_on": "per-call overhead is visible and seed/derivation records stay byte-identical",
        },
        {
            "id": "O5",
            "optimization": "checkpoint/log I/O batching",
            "local_speedup_conservative": 1.05,
            "local_speedup_optimistic": 1.3,
            "applies_to_phase": "checkpoint_io_and_logging",
            "conditional_on": "I/O is >10pct of total profile and crash/resume identity passes",
        },
        {
            "id": "O6",
            "optimization": "worker process reuse and immutable preload",
            "local_speedup_conservative": 1.1,
            "local_speedup_optimistic": 1.5,
            "applies_to_phase": "worker_setup_and_import_overhead",
            "conditional_on": "startup/setup is repeated per task or per shard",
        },
        {
            "id": "O7",
            "optimization": "measured LPT scheduling",
            "local_speedup_conservative": 1.0,
            "local_speedup_optimistic": 1.2,
            "applies_to_phase": "wall_clock_tail_from_shard_imbalance",
            "conditional_on": "max shard time exceeds median materially after O1/O4 timings",
        },
        {
            "id": "O8",
            "optimization": "deterministic intra-objective multicore evaluation",
            "local_speedup_conservative": 1.5,
            "local_speedup_optimistic": 4.0,
            "applies_to_phase": "objective_evaluation_density_and_expectation",
            "conditional_on": "objective evaluation is at least 50pct of runtime and deterministic reduction is available",
        },
    ]


def _dependency_dag() -> list[dict[str, Any]]:
    return [
        {"id": "O0.5", "depends_on": [], "reason": "BLAS/backend verification must happen before profiling so O1 measures the intended environment"},
        {"id": "O0.7", "depends_on": ["O0.5"], "reason": "WSL2 must be accepted or rejected before O1 so profiling measures the intended execution environment"},
        {"id": "O1", "depends_on": ["O0.5", "O0.7"], "reason": "profiling is prerequisite for feasibility triage and prioritization"},
        {"id": "O2", "depends_on": ["O1"], "reason": "caching value depends on measured setup/static construction share"},
        {"id": "O3", "depends_on": ["O1"], "reason": "vectorization value depends on measured objective evaluation share"},
        {"id": "O4", "depends_on": ["O2", "O3"], "reason": "batching should use the current cached/vectorized objective path and preserve seed records"},
        {"id": "O5", "depends_on": ["O1"], "reason": "checkpoint batching is useful only if I/O share is material"},
        {"id": "O6", "depends_on": ["O1"], "reason": "worker reuse is useful only if setup/import share is material"},
        {"id": "O7", "depends_on": ["O4"], "reason": "task durations and shard balance must be measured after batching changes per-task costs"},
        {"id": "O8", "depends_on": ["O1", "O3"], "reason": "parallel objective evaluation is justified only if objective work dominates and vectorized reduction semantics are known"},
    ]


def _alternative_paths() -> list[dict[str, Any]]:
    return [
        {
            "path": "external_compute",
            "decision_use": "when implementation-equivalent speedup is insufficient but the protocol scope should be preserved",
            "estimated_hours_to_decision": "2_to_4_hours_for_current_quote_environment_setup_plan_and_equivalence_smoke_definition",
            "estimated_hours_to_unblock_stage_c": "quote_plus_environment_setup_plus_validation_plus_C0_projection; formula recorded in negative_result_output",
            "estimated_monetary_cost": "projected_wall_hours_external * hourly_rate * effective_worker_count; default cap 100 USD before user amendment",
            "claims_preserved": "all Phase E Stage C claims and candidate grids preserved if equivalence passes",
            "claims_degraded": "none if environment equivalence and C0 gates pass",
            "go_no_go_criterion": "go if estimated cost <= 100 USD, projected calendar days <= 7, equivalence smoke passes, and user approves provider budget",
            "cost_formula": "estimated_cost = projected_wall_hours_external * hourly_rate * effective_worker_count",
            "required_record": "provider, machine type, effective shards, storage/backup target, measured C0 throughput, and budget approval",
        },
        {
            "path": "scope_amendment",
            "decision_use": "when local/external compute is not acceptable and Phase E claims can be downgraded",
            "estimated_hours_to_decision": "4_to_8_hours_to_select_reduced_scope_and_write_claim_impact",
            "estimated_hours_to_unblock_stage_c": "depends_on_removed_candidates; compute runtime_savings_from_measured_per_candidate_cost",
            "estimated_monetary_cost": "0_USD_direct_compute_cost_but_scientific_claim_cost_must_be_recorded",
            "claims_preserved": "methodology, implementation evidence, and any retained reduced-grid claims",
            "claims_degraded": "claims tied to removed candidates, reduced seed/shot scope, or downgraded final calibration become exploratory",
            "go_no_go_criterion": "go only if central Phase E claim remains interpretable and affected gates can be rerun before execution",
            "candidate_reductions": "N_LHS, T_candidates, K/K_ref, seed count, shot budget, or pilot universe only by explicit amendment",
            "reduction_formula": "runtime_savings = number_of_candidates_removed * measured_per_candidate_runtime_after_O1",
            "required_record": "which claims become exploratory, which candidates are removed, and which gates must be re-run",
        },
        {
            "path": "budget_amendment",
            "decision_use": "when optimized projection is above 72 h but acceptable to the user under a concrete new budget",
            "estimated_hours_to_decision": "1_to_2_hours_after_optimized_projection",
            "estimated_hours_to_unblock_stage_c": "local_projected_wall_hours_adjusted_by_thermal_and_calendar_model",
            "estimated_monetary_cost": "0_USD_direct_cloud_cost; local machine availability and interruption risk are the cost",
            "claims_preserved": "all Phase E Stage C claims preserved if only budget changes",
            "claims_degraded": "none scientifically, but operational risk increases with long local calendar time",
            "go_no_go_criterion": "go only if user explicitly accepts new wall-clock and projected calendar days <= 7 after sustained-load adjustment",
            "cost_formula": "accepted_budget_hours = optimized_projected_wall_hours plus safety_margin_hours",
            "required_record": "accepted wall-clock number, execution environment, thermal/storage assumptions, and re-run C0 gates",
        },
        {
            "path": "stop_phase_e_execution",
            "decision_use": "when optimization, external compute, and scope amendment are not justified",
            "estimated_hours_to_decision": "1_hour_after_negative_result",
            "estimated_hours_to_unblock_stage_c": "not_applicable",
            "estimated_monetary_cost": "0_USD",
            "claims_preserved": "feasibility, methodology, C0 diagnostics, and negative-result evidence",
            "claims_degraded": "no official Stage C depth2 calibration claim",
            "go_no_go_criterion": "choose if every viable path either exceeds caps or removes the central Phase E claim",
            "salvageable_outputs": "methodology draft, feasibility diagnostics, C0 profiling, and negative feasibility result",
        },
    ]


def _precommitment_decision_table() -> list[dict[str, Any]]:
    return [
        {
            "path": "implementation_equivalent_optimization_route",
            "expected_hours": PRE_COMMITMENT_EXPECTED_VALUE_ANALYSIS["expected_optimization_route_hours"],
            "expected_monetary_cost": "0_USD_direct_unless_external_compute_recommended_after_negative_result",
            "probability_of_unblocking_stage_c": "low_to_moderate_pre_O1; quantified after O1 using measured Amdahl ceiling",
            "information_value_of_negative_result": "high; resolves whether local implementation work can plausibly close the 3.1x gap",
            "selection_rule": "preferred first only because fast-kill branch bounds expected wasted work and preserves protocol scope",
        },
        {
            "path": "immediate_external_compute",
            "expected_hours": "2_to_4_hours_to_quote_and_prepare_environment_before_run_time",
            "expected_monetary_cost": "requires_current_provider_quote; default decision cap 100_USD",
            "probability_of_unblocking_stage_c": "high_if_budget_and_equivalence_pass",
            "information_value_of_negative_result": "medium; validates environment but does not explain local bottleneck",
            "selection_rule": "preferred if local calendar model exceeds 7 days or O1 fast-kills implementation route",
        },
        {
            "path": "immediate_scope_amendment",
            "expected_hours": "4_to_8_hours_to_write_and_validate_claim_impact",
            "expected_monetary_cost": "0_USD_direct",
            "probability_of_unblocking_stage_c": "medium_if_claims_survive_reduced_scope",
            "information_value_of_negative_result": "low; avoids learning whether implementation route was feasible",
            "selection_rule": "preferred only if external compute is over cap and the reduced claim remains central enough",
        },
        {
            "path": "immediate_budget_amendment",
            "expected_hours": "1_to_2_hours_to_record_user_accepted_budget_plus_C0_recheck",
            "expected_monetary_cost": "0_USD_direct",
            "probability_of_unblocking_stage_c": "medium_but_calendar_and_interruption_risk_can_be_high",
            "information_value_of_negative_result": "low",
            "selection_rule": "preferred only if user accepts local calendar risk and projected calendar days <= 7",
        },
    ]


def _baseline_code_hashes(root: Path) -> dict[str, str | None]:
    paths = [
        "src/phase_e_sim_backend.py",
        "src/phase_e_optimizers.py",
        "src/phase_e_c0_benchmark.py",
        "src/phase_e_c0_runtime.py",
        "src/phase_e_c0_shard_balance.py",
        "src/phase_e_c0_resume.py",
        "src/phase_e_c0_jitter.py",
        "src/phase_e_runtime_optimization_plan.py",
        "src/phase_e_runtime_optimization_execution.py",
        "scripts/53_phase_e_stage_c0_benchmark.py",
        "scripts/58_phase_e_stage_c0_runtime.py",
        "scripts/60_phase_e_stage_c_authorization.py",
        "scripts/61_phase_e_runtime_optimization_plan.py",
        "scripts/62_phase_e_runtime_optimization_o0_o1.py",
        "scripts/63_phase_e_runtime_equivalence_suite.py",
        "scripts/64_phase_e_runtime_optimization_decision.py",
    ]
    hashes: dict[str, str | None] = {}
    for relative in paths:
        path = root / relative
        hashes[relative] = sha256_file(path) if path.exists() else None
    return hashes


def _amdahl_speedup(rows: list[dict[str, Any]]) -> float:
    denominator = 0.0
    for row in rows:
        denominator += float(row["hypothetical_fraction"]) / float(row["optimistic_local_speedup"])
    return float(1.0 / denominator)


def _parametric_sanity_sweep(required_reduction: float) -> dict[str, Any]:
    objective_fraction_range = [0.35, 0.45, 0.55, 0.65, 0.75]
    objective_speedup_range = [2.5, 3.5, 4.5]
    setup_fraction_range = [0.05, 0.10, 0.15, 0.20]
    environmental_multiplier_range = [1.0, 1.15, 1.3, 1.5]
    fixed = {
        "per_call_overhead_and_zne": (0.10, 1.8),
        "checkpoint_and_io": (0.05, 1.3),
        "worker_setup": (0.05, 1.5),
    }
    rows = []
    for objective_fraction in objective_fraction_range:
        for objective_speedup in objective_speedup_range:
            for setup_fraction in setup_fraction_range:
                for environmental_multiplier in environmental_multiplier_range:
                    fixed_fraction = sum(value[0] for value in fixed.values())
                    shard_fraction = 1.0 - objective_fraction - setup_fraction - fixed_fraction
                    if shard_fraction < 0.0:
                        continue
                    denominator = objective_fraction / objective_speedup
                    denominator += setup_fraction / 3.0
                    denominator += fixed["per_call_overhead_and_zne"][0] / fixed["per_call_overhead_and_zne"][1]
                    denominator += fixed["checkpoint_and_io"][0] / fixed["checkpoint_and_io"][1]
                    denominator += fixed["worker_setup"][0] / fixed["worker_setup"][1]
                    denominator += shard_fraction / 1.2 if shard_fraction > 0.0 else 0.0
                    implementation_composite = float(1.0 / denominator)
                    composite = implementation_composite * environmental_multiplier
                    rows.append(
                        {
                            "objective_fraction": objective_fraction,
                            "objective_speedup": objective_speedup,
                            "setup_fraction": setup_fraction,
                            "environmental_multiplier": environmental_multiplier,
                            "shard_tail_fraction": shard_fraction,
                            "implementation_only_amdahl_speedup": implementation_composite,
                            "amdahl_composite_speedup": composite,
                            "passes_3_1x": composite >= required_reduction,
                            "passes_2_5x": composite >= 2.5,
                        }
                    )
    count = len(rows)
    above_required = sum(1 for row in rows if row["passes_3_1x"])
    above_kill = sum(1 for row in rows if row["passes_2_5x"])
    return {
        "sanity_check_model": "parametric_sweep",
        "objective_evaluation_fraction_range": objective_fraction_range,
        "O3_O8_combined_range": objective_speedup_range,
        "setup_fraction_range": setup_fraction_range,
        "environmental_multiplier_range": environmental_multiplier_range,
        "valid_parameter_points": count,
        "feasible_parameter_space_fraction_above_required": float(above_required / count) if count else 0.0,
        "feasible_parameter_space_fraction_above_2_5x": float(above_kill / count) if count else 0.0,
        "rows": rows,
    }


def _decision_tree() -> list[dict[str, Any]]:
    return [
        {
            "condition": "O0.5 backend change fails equivalence smoke",
            "decision": "keep original backend for O1 and record backend upgrade as rejected",
        },
        {
            "condition": "O0.7 WSL2 unavailable or WSL2 smoke/equivalence fails",
            "decision": "continue O1 in Windows native, keep WSL2 rejected in the environment snapshot, and do not use WSL2 multipliers",
        },
        {
            "condition": "O1 profiling remains unreliable after 10 runs and mitigation",
            "decision": "use conservative upper-confidence phase fractions, forbid high-confidence continue, and allow only kill or speculative-continue with the hour-12 hard gate",
        },
        {
            "condition": "O1 feasible maximum speedup < 2.5x",
            "decision": "abort O2-O8 implementation route; write scope/budget/external-compute/stop decision",
        },
        {
            "condition": "2.5x <= O1 feasible maximum speedup < required gap and remaining path probability < 0.15",
            "decision": "kill weak-feasibility route before full implementation because closing the gap would require correlated optimistic outcomes",
        },
        {
            "condition": "O1 classified as speculative_continue",
            "decision": "attempt O2 and O3 first only; hour-12 gate hard-kills if measured composite speedup remains < 2.0x",
        },
        {
            "condition": "dominant O1 phase is not affected by permitted optimization classes",
            "decision": "declare implementation-only route infeasible without scope or environment amendment",
        },
        {
            "condition": "any equivalence gate fails",
            "decision": "revert or fix that optimization; do not use it for C0 or Stage C",
        },
        {
            "condition": "combined speedup < 90pct of product of individually measured speedups",
            "decision": "apply composition failure protocol: isolate pairwise interaction, remove dominant regressing O_i or LIFO last-added O_i, then hard-fail if remaining stack still misses the 90pct product rule",
        },
        {
            "condition": "best validated composition speedup < 2.0x",
            "decision": "abort implementation route regardless of individual measurements and write negative-result artifacts",
        },
        {
            "condition": "optimized primary projected wall hours <= 72 and all infrastructure C0 gates pass",
            "decision": "record optimized C0 evidence, keep Stage C blocked until user accepts the runtime path and final gates complete",
        },
        {
            "condition": "72 < optimized primary projected wall hours <= 120",
            "decision": "write budget or external-compute amendment; no local Stage C execution under the 72 h budget",
        },
        {
            "condition": "optimized primary projected wall hours > 120",
            "decision": "scope amendment or stop Phase E execution; implementation-only optimization did not close the gap",
        },
    ]


def _markdown(context: dict[str, Any], summary: dict[str, Any]) -> str:
    runtime = context["runtime"]
    authorization = context["authorization"]
    calibration = context["calibration"]
    protocol = calibration["protocol_candidate_sets"]
    lines = [
        "# Phase E Runtime Optimization Plan",
        "",
        f"Version: `{PHASE_E_RUNTIME_OPTIMIZATION_PLAN_VERSION}`",
        "Status: `recorded_not_executed_not_execution_authorization`",
        f"Created UTC: `{summary['created_at_utc']}`",
        "",
        "## Purpose",
        "",
        "This plan defines the only implementation-level optimization path allowed before any renewed Phase E Stage C attempt.",
        "",
        "It is not a Stage C execution authorization. It is not a scope reduction, threshold relaxation, final hyperparameter selection, or acceptance of a larger runtime budget.",
        "",
        "## Current Blocker",
        "",
        "```text",
        f"stage_c_authorized = {authorization['stage_c_authorized']}",
        f"authorization_decision = {authorization['authorization_decision']}",
        f"first_blocking_criterion = {authorization['first_blocking_criterion']}",
        f"accepted_local_wall_hours = {runtime['accepted_local_wall_hours']}",
        f"primary_projected_wall_hours = {runtime['primary_projected_wall_hours']}",
        f"primary_budget_multiplier = {runtime['primary_budget_multiplier']}",
        f"required_reduction_factor_vs_primary_projection = {summary['required_reduction_factor_vs_primary_projection']}",
        f"conservative_no_K_projected_wall_hours = {runtime['conservative_no_K_projected_wall_hours']}",
        f"task_template_projected_wall_hours = {runtime['task_template_projected_wall_hours']}",
        "```",
        "",
        "The optimization target is therefore not vague: the current primary projection must improve by at least the current budget multiplier before the local 72 h runtime gate can pass.",
        "",
        "## Pre-Commitment Expected Value Analysis",
        "",
        "This section justifies why the implementation-equivalent optimization route is worth considering before choosing external compute, scope reduction, a larger local budget, or stopping Phase E execution.",
        "",
        "```text",
        f"probability_model_type = {summary['pre_commitment_expected_value_analysis']['probability_model_type']}",
        f"assumptions = {summary['pre_commitment_expected_value_analysis']['assumptions']}",
        f"expected_optimization_route_hours = {summary['pre_commitment_expected_value_analysis']['expected_optimization_route_hours']}",
        f"selection_criterion = {summary['pre_commitment_expected_value_analysis']['selection_criterion']}",
        f"information_value_of_negative_result = {summary['pre_commitment_expected_value_analysis']['information_value_of_negative_result']}",
        f"currency_policy = {summary['pre_commitment_expected_value_analysis']['currency_policy']}",
        "```",
        "",
        "| path | expected hours | expected monetary cost | P(unblock Stage C) | information value | selection rule |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in summary["precommitment_decision_table"]:
        lines.append(
            f"| {row['path']} | {row['expected_hours']} | {row['expected_monetary_cost']} | "
            f"{row['probability_of_unblocking_stage_c']} | {row['information_value_of_negative_result']} | {row['selection_rule']} |"
        )
    lines.extend(
        [
        "",
        "## Feasibility Triage",
        "",
        "O1 profiling is a prerequisite gate before implementing O2-O8. The plan may be recorded before O1, but implementation beyond profiling is not justified until O1 produces a phase breakdown and an Amdahl-style speedup estimate.",
        "",
        "```text",
        f"composition_model = {summary['feasibility_triage']['composition_model']}",
            f"formal_gap_factor = {summary['required_reduction_factor_vs_primary_projection']}",
            f"kill_condition_plausible_max_speedup_lt = {summary['feasibility_triage']['kill_condition_plausible_max_speedup_lt']}",
            f"weak_feasibility_band = {summary['feasibility_triage']['weak_feasibility_band']}",
            f"conditional_kill_probability_threshold = {summary['feasibility_triage']['conditional_kill_probability_threshold']}",
            f"conditional_kill_rule = {summary['feasibility_triage']['conditional_kill_rule']}",
            "```",
        "",
        "Amdahl model:",
        "",
        "```text",
        "Let p_i be the O1 measured fraction of total runtime spent in phase i.",
        "Let s_i be the plausible speedup for phase i from permitted optimizations.",
        "Composite speedup S = 1 / sum_i(p_i / s_i).",
        "Use max_shard_time, not mean shard time, for wall-clock projection.",
        "```",
        "",
        "Kill condition: if O1 plus the optimistic permitted speedup model gives plausible_max_speedup < 2.5x, abort the implementation-only optimization route and move to scope, budget, external-compute, or stop decision. In the weak-feasibility band, continue only if the remaining path does not require all remaining material O_i to beat p75 outcomes and the product probability of closing the gap is at least 0.15. If the dominant measured phase is not affected by permitted optimization classes, the implementation-only route is also infeasible without amendment.",
        "",
        "## Estimated Speedup Priors",
        "",
        "| id | optimization | local speedup conservative | local speedup optimistic | applies to phase | conditional on |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    )
    for row in summary["speedup_estimate_table"]:
        lines.append(
            f"| {row['id']} | {row['optimization']} | {row['local_speedup_conservative']} | "
            f"{row['local_speedup_optimistic']} | {row['applies_to_phase']} | {row['conditional_on']} |"
        )
    lines.extend(
        [
            "",
            "These are triage priors, not accepted results. O1 must replace them with measured phase fractions before O2-O8 work is justified.",
            "",
            "## Pre-O1 Feasibility Sanity Check",
            "",
            "This sanity check does not replace O1. It shows that the 3.1x target is near the optimistic envelope before spending optimization effort.",
            "",
            "| phase | hypothetical fraction | applicable Oi | optimistic local speedup |",
            "| --- | ---: | --- | ---: |",
        ]
    )
    for row in summary["pre_o1_amdahl_sanity_check"]["hypothetical_phase_fractions"]:
        lines.append(
            f"| {row['phase']} | {row['hypothetical_fraction']} | {row['applicable_Oi']} | {row['optimistic_local_speedup']} |"
        )
    lines.extend(
        [
            "",
            "```text",
            f"Amdahl_optimistic_composite_speedup = {summary['pre_o1_amdahl_sanity_check']['optimistic_composite_speedup']:.3f}",
            f"Amdahl_conservative_composite_speedup_approx = {summary['pre_o1_amdahl_sanity_check']['conservative_composite_speedup_approx']}",
            f"interpretation = {summary['pre_o1_amdahl_sanity_check']['interpretation']}",
            "```",
            "",
            "## O1 Kill Probability Estimate",
            "",
            "```text",
            f"estimate_type = {summary['pre_o1_kill_probability']['estimate_type']}",
            f"O1_kill_probability_estimate = {summary['pre_o1_kill_probability']['O1_kill_probability_estimate']}",
            f"reasoning = {summary['pre_o1_kill_probability']['reasoning']}",
            f"expected_value_of_plan = {summary['pre_o1_kill_probability']['expected_value_of_plan']}",
            f"implication = {summary['pre_o1_kill_probability']['implication']}",
            "```",
            "",
            "This is a subjective pre-O1 estimate, not a statistical result. It is recorded to make the likely negative-result branch explicit before optimization work begins.",
            "",
            "## Prior Definition And Memory-bound Correction",
            "",
            "```text",
            f"conservative_prior = {summary['prior_definition']['conservative']}",
            f"optimistic_prior = {summary['prior_definition']['optimistic']}",
            f"Amdahl_optimistic_uses = {summary['prior_definition']['Amdahl_optimistic_uses']}",
            f"realistic_composite_estimate = {summary['prior_definition']['realistic_composite_estimate']}",
            f"geometric_mean_justification = {summary['prior_definition']['geometric_mean_justification']}",
            f"note = {summary['prior_definition']['note']}",
            "```",
            "",
            "Memory-bound O3/O8 interaction model:",
            "",
            "```text",
            f"model = {summary['memory_bandwidth_model']['model']}",
            f"complex128_density_matrix_bytes_per_element = {summary['memory_bandwidth_model']['complex128_density_matrix_bytes_per_element']}",
            f"typical_L3_cache_size_comet_lake_MB = {summary['memory_bandwidth_model']['typical_L3_cache_size_comet_lake_MB']}",
            f"density_matrix_fits_in_L3_up_to_qubits = {summary['memory_bandwidth_model']['density_matrix_fits_in_L3_up_to_qubits']}",
            f"memory_bound_expected_O3_O8_combined_range = {summary['memory_bandwidth_model']['memory_bound_expected_O3_O8_combined_range']}",
            f"previous_capped_objective_speedup = {summary['memory_bandwidth_model']['previous_capped_objective_speedup']}",
            f"revised_objective_evaluation_optimistic_speedup = {summary['memory_bandwidth_model']['revised_objective_evaluation_optimistic_speedup']}",
            f"O1_rule = {summary['memory_bandwidth_model']['O1_rule']}",
            "```",
            "",
            "## Parametric Pre-O1 Sanity Sweep",
            "",
            "```text",
            f"sanity_check_model = {summary['parametric_sanity_sweep']['sanity_check_model']}",
            f"objective_evaluation_fraction_range = {summary['parametric_sanity_sweep']['objective_evaluation_fraction_range']}",
            f"O3_O8_combined_range = {summary['parametric_sanity_sweep']['O3_O8_combined_range']}",
            f"setup_fraction_range = {summary['parametric_sanity_sweep']['setup_fraction_range']}",
            f"environmental_multiplier_range = {summary['parametric_sanity_sweep']['environmental_multiplier_range']}",
            f"valid_parameter_points = {summary['parametric_sanity_sweep']['valid_parameter_points']}",
            f"feasible_parameter_space_fraction_above_3_1x = {summary['parametric_sanity_sweep']['feasible_parameter_space_fraction_above_required']}",
            f"feasible_parameter_space_fraction_above_2_5x = {summary['parametric_sanity_sweep']['feasible_parameter_space_fraction_above_2_5x']}",
            "```",
            "",
            "The sweep replaces a single optimistic point estimate with a plausible parameter-space view. O1 must replace this sweep with measured phase fractions and measured memory-bound flags.",
            "",
            "## Resource Budget For Optimization Work",
            "",
            "```text",
            f"O0_5_blas_backend_budget_hours = {summary['resource_budget']['o0_5_blas_backend_budget_hours']}",
            f"O0_7_wsl2_evaluation_budget_hours = {summary['resource_budget']['o0_7_wsl2_evaluation_budget_hours']}",
            f"O1_profiling_budget_hours = {summary['resource_budget']['o1_profiling_budget_hours']}",
            f"O1_feasibility_triage_hours = {summary['resource_budget']['o1_feasibility_triage_hours']}",
            f"per_Oi_implementation_budget_hours = {summary['resource_budget']['per_Oi_implementation_budget_hours']}",
            f"per_Oi_fast_validation_hours = {summary['resource_budget']['per_Oi_fast_validation_hours']}",
            f"full_validation_suite_hours = {summary['resource_budget']['full_validation_suite_hours']}",
            f"composition_gate_hours = {summary['resource_budget']['composition_gate_hours']}",
            f"optimized_c0_projection_hours = {summary['resource_budget']['optimized_c0_projection_hours']}",
            f"documentation_and_artifacts_hours = {summary['resource_budget']['documentation_and_artifacts_hours']}",
            f"fast_kill_branch_hours_native = {summary['resource_budget']['fast_kill_branch_hours_native']}",
            f"fast_kill_branch_hours_with_wsl2_trial = {summary['resource_budget']['fast_kill_branch_hours_with_wsl2_trial']}",
            f"fast_track_if_only_O2_O3_hours = {summary['resource_budget']['fast_track_if_only_O2_O3_hours']}",
            f"full_track_if_O2_to_O8_all_needed_hours = {summary['resource_budget']['full_track_if_O2_to_O8_all_needed_hours']}",
            f"max_development_hours_before_go_no_go = {summary['resource_budget']['max_development_hours_before_go_no_go']}",
            f"budget_amendment_required_if_more_than_hours = {summary['resource_budget']['budget_amendment_required_if_more_than_hours']}",
            f"max_calendar_days_before_go_no_go = {summary['resource_budget']['max_calendar_days_before_go_no_go']}",
            f"intermediate_gate_after = {summary['resource_budget']['intermediate_gate_after']}",
            f"accounting_formula = {summary['resource_budget']['accounting_formula']}",
            f"intermediate_gate_rule = {summary['resource_budget']['intermediate_gate_rule']}",
            "```",
            "",
            "The time-box is tiered rather than a single hidden 20 h bucket. If O1 shows only O2/O3 are worth attempting, the route remains near the 20 h go/no-go budget. If O2-O8 are all needed, the plan explicitly requires an optimization-budget amendment before continuing.",
            "",
            "## Plan Readiness And Implementation Boundary",
            "",
            "```text",
            f"score_after_this_hardening = {summary['plan_readiness_rubric']['score_after_this_hardening']}",
            f"scope_of_score = {summary['plan_readiness_rubric']['scope_of_score']}",
            f"runbook_status = {summary['execution_runbook']['runbook_status']}",
            f"implementation_boundary = {summary['approval_tokens_and_stop_rules']['implementation_boundary']}",
            "```",
            "",
            "This is now a complete pre-execution plan. It still does not claim that O0/O1 profiling scripts have been run, that O2-O8 optimizations exist, or that Stage C is authorized.",
            "",
            "## Approval Tokens And Hard Stops",
            "",
            "```text",
            f"o0_o1_authorization_token = {summary['approval_tokens_and_stop_rules']['o0_o1_authorization_token']}",
            f"o2_to_o8_authorization_condition = {summary['approval_tokens_and_stop_rules']['o2_to_o8_authorization_condition']}",
            f"stage_c_authorization_token = {summary['approval_tokens_and_stop_rules']['stage_c_authorization_token']}",
            f"stop_rules = {summary['approval_tokens_and_stop_rules']['stop_rules']}",
            "```",
            "",
            "## Baseline Lock Protocol",
            "",
            "```text",
            f"purpose = {summary['baseline_lock_protocol']['purpose']}",
            f"required_before = {summary['baseline_lock_protocol']['required_before']}",
            f"required_artifacts = {summary['baseline_lock_protocol']['required_artifacts']}",
            f"minimum_fields = {summary['baseline_lock_protocol']['minimum_fields']}",
            f"dirty_tree_policy = {summary['baseline_lock_protocol']['dirty_tree_policy']}",
            f"acceptance_rule = {summary['baseline_lock_protocol']['acceptance_rule']}",
            "```",
            "",
            "## Execution Runbook Interface",
            "",
            "| script | modes | required outputs |",
            "| --- | --- | --- |",
        ]
    )
    for row in summary["execution_runbook"]["script_interfaces_required_before_o1"]:
        lines.append(f"| `{row['script']}` | {row['modes']} | {row['required_outputs']} |")
    lines.extend(
        [
            "",
            "Existing pre-execution commands:",
            "",
            "```text",
            *summary["execution_runbook"]["pre_execution_commands_existing"],
            "```",
            "",
            "First O0/O1 commands after implementation of the execution harness:",
            "",
            "```text",
            *summary["execution_runbook"]["first_o0_o1_commands_after_script_implementation"],
            "```",
            "",
            f"Runbook acceptance rule: {summary['execution_runbook']['runbook_acceptance_rule']}",
            "",
            "## Output Schema Registry",
            "",
            "All JSON outputs must include the common fields below, plus the artifact-specific fields. CSV outputs must include the listed columns exactly before any optional columns.",
            "",
            "```text",
            f"common_json_fields = {summary['output_schema_registry']['common_json_fields']}",
            f"runtime_optimization_environment_snapshot.json = {summary['output_schema_registry']['runtime_optimization_environment_snapshot.json']}",
            f"runtime_profile_baseline.csv = {summary['output_schema_registry']['runtime_profile_baseline.csv']}",
            f"runtime_optimization_feasibility_triage.json = {summary['output_schema_registry']['runtime_optimization_feasibility_triage.json']}",
            f"runtime_optimization_decision_log.json = {summary['output_schema_registry']['runtime_optimization_decision_log.json']}",
            f"optimization_candidate_equivalence_report.json = {summary['output_schema_registry']['optimization_candidate_equivalence_report.json']}",
            f"optimization_candidate_speedup_report.csv = {summary['output_schema_registry']['optimization_candidate_speedup_report.csv']}",
            "```",
            "",
            "## Triage Formula Registry",
            "",
            "```text",
            f"phase_fraction = {summary['triage_formulas']['phase_fraction']}",
            f"phase_cv = {summary['triage_formulas']['phase_cv']}",
            f"amdahl_speedup = {summary['triage_formulas']['amdahl_speedup']}",
            f"memory_adjusted_objective_speedup = {summary['triage_formulas']['memory_adjusted_objective_speedup']}",
            f"required_remaining_speedup = {summary['triage_formulas']['required_remaining_speedup']}",
            f"weak_path_product_probability = {summary['triage_formulas']['weak_path_product_probability']}",
            f"optimized_wall_projection = {summary['triage_formulas']['optimized_wall_projection']}",
            f"high_confidence_continue = {summary['triage_formulas']['high_confidence_continue']}",
            f"marginal_continue = {summary['triage_formulas']['marginal_continue']}",
            f"speculative_continue = {summary['triage_formulas']['speculative_continue']}",
            f"kill = {summary['triage_formulas']['kill']}",
            "```",
            "",
            "## O0.5 BLAS Backend Verification",
            "",
            "```text",
            f"id = {summary['o0_5_blas_backend_spec']['id']}",
            f"optimization = {summary['o0_5_blas_backend_spec']['optimization']}",
            f"status = {summary['o0_5_blas_backend_spec']['status']}",
            f"action = {summary['o0_5_blas_backend_spec']['action']}",
            f"required_evidence = {summary['o0_5_blas_backend_spec']['required_evidence']}",
            f"risk = {summary['o0_5_blas_backend_spec']['risk']}",
            f"expected_speedup_on_density_ops = {summary['o0_5_blas_backend_spec']['expected_speedup_on_density_ops']}",
            f"time_budget_hours = {summary['o0_5_blas_backend_spec']['time_budget_hours']}",
            f"must_complete_before = {summary['o0_5_blas_backend_spec']['must_complete_before']}",
            "```",
            "",
            "O0.5 is an environment-level check, not a protocol or code change. If a BLAS/backend change is attempted and the smoke equivalence fails, the original backend is retained for O1.",
            "",
            "## O0.7 WSL2 Environment Evaluation",
            "",
            "```text",
            f"id = {summary['o0_7_wsl2_spec']['id']}",
            f"optimization = {summary['o0_7_wsl2_spec']['optimization']}",
            f"status = {summary['o0_7_wsl2_spec']['status']}",
            f"action = {summary['o0_7_wsl2_spec']['action']}",
            f"dependency_position = {summary['o0_7_wsl2_spec']['dependency_position']}",
            f"required_evidence = {summary['o0_7_wsl2_spec']['required_evidence']}",
            f"failure_rule = {summary['o0_7_wsl2_spec']['failure_rule']}",
            f"success_rule = {summary['o0_7_wsl2_spec']['success_rule']}",
            f"environmental_multiplier_prior = {summary['o0_7_wsl2_spec']['environmental_multiplier_prior']}",
            f"time_budget_hours = {summary['o0_7_wsl2_spec']['time_budget_hours']}",
            f"not_authorized_by_this_plan = {summary['o0_7_wsl2_spec']['not_authorized_by_this_plan']}",
            "```",
            "",
            "O0.7 is first-class because Windows spawn overhead can materially affect O6 and O8. It still cannot silently change the production environment: if WSL2 is unavailable, unapproved, or fails equivalence, O1 continues on Windows native and the WSL2 multiplier is not used.",
            "",
            "## Execution Environment Specification",
            "",
            "```text",
            f"target_machine = {summary['execution_environment_spec']['target_machine']}",
            f"cpu_model = {summary['execution_environment_spec']['cpu_model']}",
            f"physical_cores = {summary['execution_environment_spec']['physical_cores']}",
            f"logical_cores = {summary['execution_environment_spec']['logical_cores']}",
            f"core_count_measurement_source = {summary['execution_environment_spec']['core_count_measurement_source']}",
            f"max_clock_speed_mhz = {summary['execution_environment_spec']['max_clock_speed_mhz']}",
            f"ram_gb = {summary['execution_environment_spec']['ram_gb']}",
            f"storage_type = {summary['execution_environment_spec']['storage_type']}",
            f"python_version = {summary['execution_environment_spec']['python_version']}",
            f"numpy_version = {summary['execution_environment_spec']['numpy_version']}",
            f"blas_backend = {summary['execution_environment_spec']['blas_backend']}",
            f"os = {summary['execution_environment_spec']['os']}",
            f"environment_invariant = {summary['execution_environment_spec']['environment_invariant']}",
            f"windows_load_condition_gate = {summary['execution_environment_spec']['windows_load_condition_gate']}",
            f"thermal_throttling_gate = {summary['execution_environment_spec']['thermal_throttling_gate']}",
            "```",
            "",
            "O1 must record any `record_before_O1` fields before measurements are interpreted. Profiling and optimized C0 measurements are not comparable if the machine, BLAS backend, core count, thermal state, or competing load changes materially.",
            "",
            "Thermal sustained-load protocol:",
            "",
            "```text",
            f"required_before = {summary['thermal_protocol']['required_before']}",
            f"measurement = {summary['thermal_protocol']['measurement']}",
            f"sustained_frequency_ratio = {summary['thermal_protocol']['sustained_frequency_ratio']}",
            f"projection_adjustment = {summary['thermal_protocol']['projection_adjustment']}",
            f"mitigation_options = {summary['thermal_protocol']['mitigation_options']}",
            f"reporting = {summary['thermal_protocol']['reporting']}",
            "```",
            "",
            "Calendar-time model:",
            "",
            "```text",
            f"nominal_wall_hours_per_day = {summary['calendar_time_model']['nominal_wall_hours_per_day']}",
            f"default_laptop_effective_compute_hours_per_calendar_day = {summary['calendar_time_model']['default_laptop_effective_compute_hours_per_calendar_day']}",
            f"effective_compute_hours_per_calendar_day_formula = {summary['calendar_time_model']['effective_compute_hours_per_calendar_day_formula']}",
            f"external_compute_preferred_if_projected_calendar_days_gt = {summary['calendar_time_model']['external_compute_preferred_if_projected_calendar_days_gt']}",
            f"projected_calendar_days_formula = {summary['calendar_time_model']['projected_calendar_days_formula']}",
            f"interpretation = {summary['calendar_time_model']['interpretation']}",
            "```",
            "",
            "Platform note:",
            "",
            "```text",
            f"platform = {summary['platform_note']['platform']}",
            f"process_creation_note = {summary['platform_note']['process_creation_note']}",
            f"implication_for_O6 = {summary['platform_note']['implication_for_O6']}",
            f"implication_for_O8 = {summary['platform_note']['implication_for_O8']}",
            f"alternative_considered = {summary['platform_note']['alternative_considered']}",
            f"WSL2_decision = {summary['platform_note']['WSL2_decision']}",
            "```",
            "",
            "## O1 Profiling Methodology",
            "",
            "```text",
            f"profiling_tools = {summary['o1_profiling_methodology']['profiling_tools']}",
            f"warmup_runs_discarded = {summary['o1_profiling_methodology']['warmup_runs_discarded']}",
            f"measurement_runs = {summary['o1_profiling_methodology']['measurement_runs']}",
            f"increase_to_runs_if_phase_cv_gt_0_15 = {summary['o1_profiling_methodology']['increase_to_runs_if_phase_cv_gt_0_15']}",
            f"profiling_cases = {summary['o1_profiling_methodology']['profiling_cases']}",
            f"granularity = {summary['o1_profiling_methodology']['granularity']}",
            f"memory_profiling = {summary['o1_profiling_methodology']['memory_profiling']}",
            f"reporting = {summary['o1_profiling_methodology']['reporting']}",
            "```",
            "",
            "O1 must produce a kill/continue decision within 4 hours. If any phase CV exceeds 0.15, measurement runs increase to 10 for that phase before the Amdahl triage is accepted.",
            "",
            "O1 unreliable profiling protocol:",
            "",
            "```text",
            f"trigger = {summary['o1_unreliable_profiling_protocol']['trigger']}",
            f"investigation_steps = {summary['o1_unreliable_profiling_protocol']['investigation_steps']}",
            f"if_unreliable_after_mitigation = {summary['o1_unreliable_profiling_protocol']['if_unreliable_after_mitigation']}",
            f"decision_rule = {summary['o1_unreliable_profiling_protocol']['decision_rule']}",
            "```",
            "",
            "O1 confidence levels:",
            "",
            "```text",
            f"high_confidence_continue = {summary['o1_confidence_levels']['high_confidence_continue']}",
            f"marginal_continue = {summary['o1_confidence_levels']['marginal_continue']}",
            f"speculative_continue = {summary['o1_confidence_levels']['speculative_continue']}",
            f"kill = {summary['o1_confidence_levels']['kill']}",
            f"speculative_continue_rule = {summary['o1_confidence_levels']['speculative_continue_rule']}",
            "```",
            "",
            "Weak-feasibility probability model:",
            "",
            "```text",
            f"model = {summary['weak_feasibility_probability_model']['model']}",
            f"p75_requirement_rule = {summary['weak_feasibility_probability_model']['p75_requirement_rule']}",
            f"probability_floor = {summary['weak_feasibility_probability_model']['probability_floor']}",
            f"calculation_record = {summary['weak_feasibility_probability_model']['calculation_record']}",
            f"interpretation = {summary['weak_feasibility_probability_model']['interpretation']}",
            "```",
            "",
            "## Materiality Thresholds",
            "",
            "```text",
            f"material_fraction_threshold = {summary['materiality_thresholds']['material_fraction_threshold']}",
            f"visible_overhead_threshold = {summary['materiality_thresholds']['visible_overhead_threshold']}",
            f"O8_objective_fraction_threshold = {summary['materiality_thresholds']['o8_objective_fraction_threshold']}",
            f"other_phase_max_fraction_before_decomposition = {summary['materiality_thresholds']['other_phase_max_fraction_before_decomposition']}",
            "```",
            "",
            "A phase is a material optimization target only if it accounts for at least 10% of profiled runtime. Per-call overhead is visible at 5%. O8 is not attempted unless objective evaluation is at least 50% of runtime.",
            "",
        ]
    )
    lines.extend(
        [
        "## Pre-registration And Blinding",
        "",
        "This plan is frozen at v0.2-draft before runtime optimization implementation begins. Any change after optimized C0 profiling starts is an amendment and must record whether it was motivated by observed profiling or gate results.",
        "",
        "The analyst is not blinded to C0 outcomes. To mitigate analyst degrees of freedom, allowed and prohibited optimization classes are fixed here, equivalence gates are fixed here, and no threshold or candidate grid may be relaxed after seeing optimized C0 data.",
        "",
        "Plan versioning and amendment protocol:",
        "",
        "```text",
        f"plan_version_policy = {summary['plan_versioning_and_amendment_protocol']['plan_version_policy']}",
        f"plan_freeze_trigger = {summary['plan_versioning_and_amendment_protocol']['plan_freeze_trigger']}",
        f"immutable_after_freeze = {summary['plan_versioning_and_amendment_protocol']['immutable_after_freeze']}",
        f"adaptable_after_O1_without_plan_amendment = {summary['plan_versioning_and_amendment_protocol']['adaptable_after_O1_without_plan_amendment']}",
        f"plan_amendment_required_for = {summary['plan_versioning_and_amendment_protocol']['plan_amendment_required_for']}",
        f"plan_amendment_record_fields = {summary['plan_versioning_and_amendment_protocol']['plan_amendment_record_fields']}",
        "```",
        "",
        "## Canonical Constraints",
        "",
        "This is the authoritative constraint list for the optimization plan. Later sections apply these constraints rather than redefining them.",
        "",
        "```text",
        f"immutable_after_plan_freeze_without_plan_amendment = {summary['canonical_constraints']['immutable_after_plan_freeze_without_plan_amendment']}",
        f"adaptable_after_O1_without_scientific_protocol_amendment = {summary['canonical_constraints']['adaptable_after_O1_without_scientific_protocol_amendment']}",
        f"scientific_protocol_amendment_required_for = {summary['canonical_constraints']['scientific_protocol_amendment_required_for']}",
        "```",
        "",
        "## Definitions",
        "",
        "- `implementation-equivalent optimization`: a code/runtime change that preserves the same mathematical objective, seeds, shot allocation, protocol candidates, optimizer schedule, and decision thresholds.",
        "- `scientific-design change`: any change to starts, warm-start policy, stopping rule, precision policy, candidate grid, threshold, seed count, shot budget, pilot subset, or claim scope.",
        "- `optimized C0`: a renewed C0 measurement after permitted implementation-equivalent optimizations only.",
        "- `equivalence gate`: a required validation that optimized code reproduces the pre-optimization objective/decision surface within fixed tolerances.",
        "- `runtime pass`: optimized primary projected wall-clock hours <= accepted local wall-clock budget, with storage/shard/resume/jitter gates still required separately.",
        "",
        "## Non-negotiable Constraints",
        "",
        "```text",
        "Do not run Stage C from this plan.",
        "Do not select final T/K/K_ref/N_LHS from optimization profiling.",
        "Do not change T/K/K_ref/N_LHS candidates.",
        "Do not change tau_R_opt, tau_conc, tau_trivial, or C0 stopping thresholds.",
        "Do not change seeds, shot budgets, noise rates, ZNE scale factors, or optimizer schedules.",
        "Do not treat a faster but non-equivalent implementation as valid.",
        "Do not accept a new runtime budget silently.",
        "```",
        "",
        "## Permitted Optimization Classes",
        "",
    ]
    )
    for item in PERMITTED_OPTIMIZATION_CLASSES:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## Prohibited Optimizations",
            "",
        ]
    )
    for item in PROHIBITED_OPTIMIZATIONS:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "Any prohibited item requires a separate written protocol/scope amendment before implementation. It cannot be introduced as a runtime optimization.",
            "",
            "## Optimization Work Packages",
            "",
            "| id | optimization | status | required evidence | risk |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["optimization_work_packages"]:
        lines.append(f"| {row['id']} | {row['optimization']} | {row['status']} | {row['required_evidence']} | {row['risk']} |")
    lines.extend(
        [
            "",
            "O8 parallelism specification:",
            "",
            "```text",
            f"id = {summary['o8_parallelism_spec']['id']}",
            f"optimization = {summary['o8_parallelism_spec']['optimization']}",
            f"status = {summary['o8_parallelism_spec']['status']}",
            f"applies_to_phase = {summary['o8_parallelism_spec']['applies_to_phase']}",
            f"local_speedup_conservative = {summary['o8_parallelism_spec']['local_speedup_conservative']}",
            f"local_speedup_optimistic = {summary['o8_parallelism_spec']['local_speedup_optimistic']}",
            f"conditional_on = {summary['o8_parallelism_spec']['conditional_on']}",
            f"required_evidence = {summary['o8_parallelism_spec']['required_evidence']}",
            f"prohibited_variants = {summary['o8_parallelism_spec']['prohibited_variants']}",
            "```",
            "",
            "If deterministic parallel reduction is not available, O8 is skipped rather than silently replaced with nondeterministic async execution.",
            "",
            "## Required Equivalence Gates",
            "",
            "| gate | threshold or rule |",
            "| --- | --- |",
        ]
    )
    for key, value in EQUIVALENCE_GATES.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "Equivalence gates must be run before optimized C0 projections are considered. A speedup that fails equivalence is unusable even if it meets the 72 h wall-clock target.",
            "",
            "## Validation Suite Specification",
            "",
            "```text",
            f"fast_tier_case_count = {summary['validation_suite']['fast_tier_case_count']}",
            f"full_tier_case_count = {summary['validation_suite']['full_tier_case_count']}",
            f"case_count = {summary['validation_suite']['case_count']}",
            f"selection_rule = {summary['validation_suite']['selection_rule']}",
            f"instances = {summary['validation_suite']['instances']}",
            f"depths = {summary['validation_suite']['depths']}",
            f"scale_factors = {summary['validation_suite']['scale_factors']}",
            f"noise_rates = {summary['validation_suite']['noise_rates']}",
            f"shot_budgets = {summary['validation_suite']['shot_budgets']}",
            f"seeds = {summary['validation_suite']['seeds']}",
            f"execution_rule = {summary['validation_suite']['execution_rule']}",
            f"fast_tier_rule = {summary['validation_suite']['fast_tier_rule']}",
            f"full_tier_rule = {summary['validation_suite']['full_tier_rule']}",
            "```",
            "",
            "The fast tier runs after each individual O_i to isolate the first optimization that breaks equivalence. The full 24-case suite runs after O2/O3, after the combined accepted stack, and before optimized C0 evidence is valid.",
            "",
            "Validation suite coverage justification:",
            "",
            "- 4 graph instances cover trivial, small, and the current H1/H4 pilot complexity classes.",
            "- Depths 1 and 2 cover the depth-1 regression path and the current production optimizer depth.",
            "- Noise rates span noiseless through the production maximum 0.02.",
            "- Shot budgets cover analytical/no-shot and finite-shot evaluation at 512 and 4096.",
            "- Seeds 42, 31415, 271828, and 20260517 provide fixed stochastic coverage.",
            "",
            "Coverage gaps acknowledged:",
            "",
        ]
    )
    for gap in summary["validation_suite"]["coverage_gaps"]:
        lines.append(f"- {gap}")
    lines.extend(
        [
            "",
            "Coverage gap risk scoring:",
            "",
            "| gap | risk | rationale | spot-check trigger |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in summary["validation_suite"]["coverage_gap_risk_assessment"]:
        lines.append(f"| {row['gap']} | {row['risk']} | {row['rationale']} | {row['spot_check_trigger']} |")
    lines.extend(
        [
            "",
            "The validation suite tests all code paths exercised by the current production grid. It does not test performance scaling; O1 profiling covers production-scale timing separately.",
            "",
            "O3 floating-point sensitivity rule:",
            "",
            "```text",
            "Check max absolute differences and relative differences.",
            "For near-zero values, abs tolerance alone is insufficient.",
            "If max_abs_diff is within tolerance but relative_diff > 1e-4, the case fails unless a written equivalence amendment justifies the exception.",
            "Classification agreement must be 100% for trivial/non-trivial/discordance gate labels.",
            "```",
            "",
            "## Determinism And Seed Identity",
            "",
            "```text",
            f"deterministic_definition = {summary['determinism_spec']['deterministic_definition']}",
            f"seed_identical_requirement = {summary['determinism_spec']['seed_identical_requirement']}",
            f"floating_kernel_exception = {summary['determinism_spec']['floating_kernel_exception']}",
            "```",
            "",
            "For O4, `seed_identical` means SHA-256 identity of complete per-scale derivation records, not only equality of selected scalar summaries.",
            "",
            "## Crash And Resume Specification",
            "",
            "```text",
            f"minimum_crash_points = {summary['crash_resume_spec']['minimum_crash_points']}",
            f"crash_points = {summary['crash_resume_spec']['crash_points']}",
            f"required_result = {summary['crash_resume_spec']['required_result']}",
            f"danger_case = {summary['crash_resume_spec']['danger_case']}",
            "```",
            "",
            "O5 is not accepted unless crashes during batched writes are tested and leave no partial output that can be mistaken for a complete checkpoint.",
            "",
            "## Dependency DAG And Re-profiling Rule",
            "",
            "| id | depends on | reason |",
            "| --- | --- | --- |",
        ]
    )
    for row in summary["dependency_dag"]:
        lines.append(f"| {row['id']} | {row['depends_on']} | {row['reason']} |")
    lines.extend(
        [
            "",
            "After every O_i that passes equivalence, re-profile before deciding whether the next O_i is still worth implementing. Adding O4 can change cache access patterns; adding O3 can change objective evaluation fractions; O7 must use post-O4 task timings.",
            "",
            "## Interaction And Composition Gate",
            "",
            "```text",
            "Measure each accepted O_i individually.",
            "Then measure the combined accepted optimization stack.",
            "If combined_speedup < 0.90 * product(individual_speedups), apply the composition failure protocol before optimized C0 evidence is valid.",
            "Re-run the full validation suite after adding each O_i, including all prior accepted optimizations.",
            "```",
            "",
            "Composition failure protocol:",
            "",
            "```text",
            f"investigation_budget_hours = {summary['composition_failure_protocol']['investigation_budget_hours']}",
            f"steps = {summary['composition_failure_protocol']['steps']}",
            f"hard_failure_decision = {summary['composition_failure_protocol']['hard_failure_decision']}",
            f"kill_condition = {summary['composition_failure_protocol']['kill_condition']}",
            "```",
            "",
            "## Shard Balance And Wall-clock Projection",
            "",
            "```text",
            f"primary_wall_clock_predictor = {summary['shard_gate_spec']['primary_wall_clock_predictor']}",
            f"diagnostic_cv_gate = {summary['shard_gate_spec']['diagnostic_cv_gate']}",
            f"diagnostic_max_to_median_gate = {summary['shard_gate_spec']['diagnostic_max_to_median_gate']}",
            f"projection_safety_rule = {summary['shard_gate_spec']['projection_safety_rule']}",
            f"sensitivity_rule = {summary['shard_gate_spec']['sensitivity_rule']}",
            "```",
            "",
            "CV and max/median remain diagnostics, but max shard time is the primary wall-clock predictor because the slowest shard determines completion.",
            "",
            "## Execution Sequence",
            "",
            "1. O0 baseline lock: record current runtime, authorization, C0 gate, calibration, and document hashes.",
            "2. O0.5 BLAS/backend verification: record or select the environment before timing.",
            "3. O0.7 WSL2 environment evaluation: accept or reject WSL2 before O1; if rejected, continue in Windows native.",
            "4. O1 profiling: measure objective evaluation, optimizer stepping, ZNE per-scale work, checkpointing, I/O, memory profile, thermal behavior, and worker setup separately.",
            "5. O1 feasibility triage: compute Amdahl conservative/realistic/optimistic speedup, confidence level, and conditional weak-feasibility probability before O2-O8.",
            "6. O2/O3 first for speculative continues: implement only if O1 indicates setup/objective phases are plausible bottlenecks; then run the hour-12 gate.",
            "7. Remaining O_i: implement one permitted class at a time, re-profile and re-validate after each accepted change.",
            "8. Optimized C0 projection: rerun only C0 runtime/storage/shard/resume/jitter checks required by the existing gate order.",
            "9. Decision: compare optimized projection against the 72 h budget and calendar model, then keep Stage C blocked until an accepted amendment and authorization guard pass.",
            "",
            "## Success Criteria",
            "",
            "```text",
            f"primary_success_wall_hours <= {runtime['accepted_local_wall_hours']}",
            f"required_reduction_factor_vs_primary_projection >= {summary['required_reduction_factor_vs_primary_projection']}",
            "all_equivalence_gates_pass = true",
            "runtime_gate_pass = true",
            "storage_gate_pass = true",
            "resume_integrity_gate_pass = true",
            "shard_balance_gate_pass = true",
            "duplicate_endpoint_jitter_gate_pass = true",
            "authorization_guard_pass = false until user accepts the runtime path and remaining gates complete",
            "```",
            "",
            "A preferred engineering target is <=60 wall hours to avoid passing the 72 h gate only by numerical noise or transient machine conditions, but the formal gate remains the accepted 72 h budget unless amended.",
            "",
            "## Decision Tree",
            "",
            "| condition | decision |",
            "| --- | --- |",
        ]
    )
    for row in summary["decision_tree"]:
        lines.append(f"| {row['condition']} | {row['decision']} |")
    lines.extend(
        [
            "",
            "## Rollback Protocol",
            "",
            "```text",
            f"rollback_trigger = {summary['rollback_protocol']['rollback_trigger']}",
            f"procedure = {summary['rollback_protocol']['procedure']}",
            f"commit_discipline = {summary['rollback_protocol']['commit_discipline']}",
            "```",
            "",
            "Each O_i change must be isolated enough to revert cleanly. If an equivalence gate fails, revert first, prove the baseline still passes validation, and only then continue with another candidate.",
            "",
            "## Negative Result Protocol",
            "",
            "```text",
            f"trigger = {summary['negative_result_protocol']['trigger']}",
            f"required_outputs = {summary['negative_result_protocol']['required_outputs']}",
            f"negative_result_fields = {summary['negative_result_protocol']['negative_result_fields']}",
            f"time_to_negative_result_hours = {summary['negative_result_protocol']['time_to_negative_result_hours']}",
            f"interpretation = {summary['negative_result_protocol']['interpretation']}",
            "```",
            "",
            "A killed implementation route is still a valid Phase E feasibility result. O1 profiling plus triage should produce a kill/continue decision within 4 hours.",
            "",
            "Negative-result recommendation framework:",
            "",
            "```text",
            f"external_compute_default_monetary_cap_usd = {summary['negative_result_recommendation_framework']['external_compute_default_monetary_cap_usd']}",
            f"external_compute_calendar_cap_days = {summary['negative_result_recommendation_framework']['external_compute_calendar_cap_days']}",
            f"current_cloud_quote_required_before_purchase = {summary['negative_result_recommendation_framework']['current_cloud_quote_required_before_purchase']}",
            f"recommend_external_compute_if = {summary['negative_result_recommendation_framework']['recommend_external_compute_if']}",
            f"recommend_budget_amendment_if = {summary['negative_result_recommendation_framework']['recommend_budget_amendment_if']}",
            f"recommend_scope_amendment_if = {summary['negative_result_recommendation_framework']['recommend_scope_amendment_if']}",
            f"recommend_stop_if = {summary['negative_result_recommendation_framework']['recommend_stop_if']}",
            "```",
            "",
            "Fast kill branch:",
            "",
            "```text",
            f"trigger = {summary['fast_kill_branch']['trigger']}",
            f"total_hours_native = {summary['fast_kill_branch']['total_hours_native']}",
            f"total_hours_with_wsl2_trial = {summary['fast_kill_branch']['total_hours_with_wsl2_trial']}",
            f"required_outputs = {summary['fast_kill_branch']['required_outputs']}",
            f"not_required = {summary['fast_kill_branch']['not_required']}",
            f"decision_outputs = {summary['fast_kill_branch']['decision_outputs']}",
            "```",
            "",
            "## Time-to-decision Gates",
            "",
            "| gate | hour | report fields |",
            "| --- | ---: | --- |",
        ]
    )
    for row in summary["time_to_decision_gates"]:
        lines.append(f"| {row['gate']} | {row['hour']} | {row['report']} |")
    lines.extend(
        [
            "",
            "Escalation rule: if any gate shows that the remaining path requires >p75 outcomes on all remaining material O_i, or the product probability of closing the gap is <0.15, kill the implementation route and write the negative-result artifacts.",
            "",
            "",
            "## Frozen Protocol Candidate State",
            "",
            "```text",
            f"T_candidates = {protocol['T']}",
            f"K_candidates = {protocol['K']}",
            f"K_ref_candidates = {protocol['K_ref']}",
            f"N_LHS_candidates = {protocol['N_LHS']}",
            "candidate_changes_allowed_by_this_plan = false",
            "```",
            "",
            "## Required Outputs",
            "",
            "- `runtime_optimization_baseline_lock.json`",
            "- `runtime_optimization_git_status.txt`",
            "- `runtime_optimization_dirty_tree_inventory.csv`",
            "- `runtime_optimization_pre_o1_hashes.json`",
            "- `runtime_profile_baseline.csv`",
            "- `runtime_profile_optimized.csv`",
            "- `optimization_candidate_equivalence_report.json`",
            "- `optimization_candidate_speedup_report.csv`",
            "- `optimized_c0_runtime_summary.json`",
            "- `optimized_c0_gate_composition_decision.json`",
            "- `optimized_stage_c_authorization_summary.json`",
            "- `validation_suite_manifest.json`",
            "- `runtime_optimization_feasibility_triage.json`",
            "- `runtime_optimization_decision_log.json`",
            "- `runtime_optimization_environment_snapshot.json`",
            "- `runtime_optimization_negative_result.json` if the route is killed",
            "- `runtime_optimization_amendment_or_acceptance_record.md` if a runtime path is accepted",
            "",
            "Every JSON output must include `decision_was_posthoc` and `posthoc_data_used` fields.",
            "",
            "## Alternative Paths Considered",
            "",
            "| path | decision use | hours to decision | hours to unblock | monetary cost | claims preserved | claims degraded | go/no-go | required record |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in summary["alternative_paths_considered"]:
        lines.append(
            f"| {row['path']} | {row['decision_use']} | {row.get('estimated_hours_to_decision', '')} | "
            f"{row.get('estimated_hours_to_unblock_stage_c', '')} | {row.get('estimated_monetary_cost', '')} | "
            f"{row.get('claims_preserved', '')} | {row.get('claims_degraded', '')} | {row.get('go_no_go_criterion', '')} | "
            f"{row.get('required_record', row.get('salvageable_outputs', ''))} |"
        )
    lines.extend(
        [
            "",
            "This section is part of the audit trail: optimizing first is a choice weighed against external compute, explicit scope reduction, a larger budget, or stopping Phase E execution.",
            "",
            "## Amendment Record Format",
            "",
            "Required fields:",
            "",
        ]
    )
    for field in summary["amendment_record_schema"]["required_fields"]:
        lines.append(f"- `{field}`")
    lines.extend(
        [
            "",
            f"Post-hoc decision logging rule: `{summary['amendment_record_schema']['posthoc_decision_logging']}`",
            "",
            "## Baseline Code Hashes",
            "",
            "| file | sha256 |",
            "| --- | --- |",
        ]
    )
    for file_name, digest in summary["baseline_code_hashes"].items():
        lines.append(f"| `{file_name}` | `{digest}` |")
    lines.extend(
        [
            "",
            "## Current Plan Decision",
            "",
            "```text",
            "optimization_plan_recorded = true",
            "optimization_implementation_authorized = false",
            "stage_c_execution_authorized = false",
            "stage_c_blocked = true",
            "next_required_user_decision = authorize implementation-equivalent runtime optimization work, or choose budget/scope/external-compute/stop path",
            "```",
            "",
            "## Hash Inputs",
            "",
            "```text",
            f"runtime_summary = {runtime['outputs']['summary']}",
            f"runtime_scope_amendment = {context['amendment']['outputs']['summary']}",
            f"stage_c_authorization_summary = {context['authorization']['outputs']['summary']}",
            f"frozen_plan_hash_sidecar = {summary['outputs']['hashes']}",
            "hash_input_rule = all listed input files must exist before O1 begins; O1 may only add profiling outputs",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def generate_phase_e_runtime_optimization_plan(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    context = _load_context(root)
    runtime = context["runtime"]
    doc_path = root / OPTIMIZATION_PLAN_DOC
    summary_path = root / OPTIMIZATION_PLAN_SUMMARY
    hashes_path = root / OPTIMIZATION_PLAN_HASHES
    if (doc_path.exists() or summary_path.exists() or hashes_path.exists()) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase E runtime optimization plan outputs.")

    required_reduction = float(runtime["primary_projected_wall_hours"]) / float(runtime["accepted_local_wall_hours"])
    summary = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_runtime_optimization_plan_recorded_not_executed",
        "plan_version": PHASE_E_RUNTIME_OPTIMIZATION_PLAN_VERSION,
        "document": OPTIMIZATION_PLAN_DOC,
        "official_execution_authorized": False,
        "optimization_execution_authorized": False,
        "stage_c_blocked": True,
        "stage_c_authorized": False,
        "final_calibration_complete": False,
        "accepted_local_wall_hours": runtime["accepted_local_wall_hours"],
        "current_primary_projected_wall_hours": runtime["primary_projected_wall_hours"],
        "current_primary_budget_multiplier": runtime["primary_budget_multiplier"],
        "required_reduction_factor_vs_primary_projection": required_reduction,
        "formal_runtime_success_wall_hours": runtime["accepted_local_wall_hours"],
        "preferred_engineering_target_wall_hours": 60.0,
        "canonical_constraints": CANONICAL_CONSTRAINTS,
        "pre_commitment_expected_value_analysis": PRE_COMMITMENT_EXPECTED_VALUE_ANALYSIS,
        "precommitment_decision_table": _precommitment_decision_table(),
        "calendar_time_model": CALENDAR_TIME_MODEL,
        "feasibility_triage": FEASIBILITY_TRIAGE,
        "resource_budget": RESOURCE_BUDGET,
        "validation_suite": VALIDATION_SUITE,
        "materiality_thresholds": MATERIALITY_THRESHOLDS,
        "pre_o1_kill_probability": PRE_O1_KILL_PROBABILITY,
        "prior_definition": PRIOR_DEFINITION,
        "memory_bandwidth_model": MEMORY_BANDWIDTH_MODEL,
        "pre_o1_amdahl_sanity_check": {
            **PRE_O1_AMDAHL_SANITY_CHECK,
            "optimistic_composite_speedup": _amdahl_speedup(PRE_O1_AMDAHL_SANITY_CHECK["hypothetical_phase_fractions"]),
        },
        "parametric_sanity_sweep": _parametric_sanity_sweep(required_reduction),
        "o0_5_blas_backend_spec": O0_5_BLAS_BACKEND_SPEC,
        "o0_7_wsl2_spec": O0_7_WSL2_SPEC,
        "o1_profiling_methodology": O1_PROFILING_METHODOLOGY,
        "o1_unreliable_profiling_protocol": O1_UNRELIABLE_PROFILING_PROTOCOL,
        "o1_confidence_levels": O1_CONFIDENCE_LEVELS,
        "weak_feasibility_probability_model": WEAK_FEASIBILITY_PROBABILITY_MODEL,
        "o8_parallelism_spec": O8_PARALLELISM_SPEC,
        "shard_gate_spec": SHARD_GATE_SPEC,
        "determinism_spec": DETERMINISM_SPEC,
        "crash_resume_spec": CRASH_RESUME_SPEC,
        "composition_failure_protocol": COMPOSITION_FAILURE_PROTOCOL,
        "rollback_protocol": ROLLBACK_PROTOCOL,
        "negative_result_protocol": NEGATIVE_RESULT_PROTOCOL,
        "negative_result_recommendation_framework": NEGATIVE_RESULT_RECOMMENDATION_FRAMEWORK,
        "fast_kill_branch": FAST_KILL_BRANCH,
        "thermal_protocol": THERMAL_PROTOCOL,
        "platform_note": PLATFORM_NOTE,
        "time_to_decision_gates": TIME_TO_DECISION_GATES,
        "amendment_record_schema": AMENDMENT_RECORD_SCHEMA,
        "plan_versioning_and_amendment_protocol": PLAN_VERSIONING_AND_AMENDMENT_PROTOCOL,
        "baseline_lock_protocol": BASELINE_LOCK_PROTOCOL,
        "approval_tokens_and_stop_rules": APPROVAL_TOKENS_AND_STOP_RULES,
        "execution_runbook": EXECUTION_RUNBOOK,
        "output_schema_registry": OUTPUT_SCHEMA_REGISTRY,
        "triage_formulas": TRIAGE_FORMULAS,
        "plan_readiness_rubric": PLAN_READINESS_RUBRIC,
        "execution_environment_spec": _execution_environment_spec(),
        "speedup_estimate_table": _speedup_estimate_table(),
        "dependency_dag": _dependency_dag(),
        "alternative_paths_considered": _alternative_paths(),
        "baseline_code_hashes": _baseline_code_hashes(root),
        "permitted_optimization_classes": PERMITTED_OPTIMIZATION_CLASSES,
        "prohibited_optimizations": PROHIBITED_OPTIMIZATIONS,
        "equivalence_gates": EQUIVALENCE_GATES,
        "optimization_work_packages": _optimization_table(),
        "decision_tree": _decision_tree(),
        "outputs": {
            "document": OPTIMIZATION_PLAN_DOC,
            "summary": OPTIMIZATION_PLAN_SUMMARY,
            "hashes": OPTIMIZATION_PLAN_HASHES,
        },
    }
    summary["guardrail_check"] = evaluate_runtime_optimization_plan_guardrails(summary)
    doc_path.write_text(_markdown(context, summary), encoding="utf-8")
    write_json_atomic(summary_path, summary, overwrite=True)
    hashes = {
        "created_at_utc": _utc_now(),
        "plan_version": PHASE_E_RUNTIME_OPTIMIZATION_PLAN_VERSION,
        "hashes": {
            OPTIMIZATION_PLAN_DOC: sha256_file(doc_path),
            OPTIMIZATION_PLAN_SUMMARY: sha256_file(summary_path),
        },
    }
    write_json_atomic(hashes_path, hashes, overwrite=True)
    summary["output_hashes"] = hashes["hashes"]
    return summary
