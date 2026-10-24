from __future__ import annotations

import json
import math
import time
import csv
from pathlib import Path
from typing import Any

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .phase_e_config import PhaseEOptimizerConfig


PHASE_E_STAGE_C0_PLAN_VERSION = "phase_e_stage_c0_optimization_plan_v0_3_draft"
ACCEPTED_LOCAL_WALL_HOURS = 72.0
SHARD_RUNTIME_CV_GATE = 0.20
SHARD_MAX_TO_MEDIAN_GATE = 1.50
DISK_POST_RUN_USAGE_GATE_FRACTION = 0.80
DISK_FREE_SPACE_USE_GATE_FRACTION = 0.50
JITTER_WARNING_FRACTION_OF_RANGE = 0.005
JITTER_BLOCK_FRACTION_OF_RANGE = 0.010
INSTANCE_VARIANCE_TRIGGER_CV = 0.50
INSTANCE_VARIANCE_TRIGGER_MAX_TO_MIN = 2.0


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _max_changes_strict_less_than(fraction: float, n_cases: int) -> int:
    """Maximum integer count satisfying count / n_cases < fraction."""
    return max(0, int(math.ceil(float(fraction) * int(n_cases)) - 1))


def _binom_cdf(k: int, n: int, p: float) -> float:
    return float(sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(0, int(k) + 1)))


def _binom_sf_at_least(k: int, n: int, p: float) -> float:
    return float(sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(int(k), int(n) + 1)))


def _percentile_rank(values: list[float], value: float) -> float:
    if not values:
        return math.nan
    less_or_equal = sum(1 for item in values if item <= value)
    return float(less_or_equal / len(values))


def _runtime_accounting(calibration_status: dict[str, Any]) -> dict[str, Any]:
    optimizer = PhaseEOptimizerConfig()
    t_candidates = tuple(int(value) for value in optimizer.spsa_evaluation_budget_candidates)
    k_candidates = tuple(int(value) for value in optimizer.random_start_candidates)
    baseline_stage_c_hours = float(
        calibration_status["runtime_projection_summary"]["protocol_T_candidate_total_serial_hours_no_reuse"]
    )
    checkpoint_factor_exact = float(sum(t_candidates) / max(t_candidates))
    checkpointed_t_only_hours = float(baseline_stage_c_hours / checkpoint_factor_exact)
    k_nested_factor_exact = float(sum(k_candidates) / max(k_candidates))
    external_checkpoint_factor = 4.0
    external_k_factor = 1.8
    effective_shards = 7.0
    external_target_serial_hours = float(baseline_stage_c_hours / (external_checkpoint_factor * external_k_factor))
    external_target_wall_hours = float(external_target_serial_hours / effective_shards)
    strict_target_wall_hours = float(checkpointed_t_only_hours / effective_shards)
    local_pass_checkpoint_factor = float(baseline_stage_c_hours / (ACCEPTED_LOCAL_WALL_HOURS * k_nested_factor_exact * effective_shards))
    external_pass_checkpoint_factor = float(
        baseline_stage_c_hours / (external_target_wall_hours * k_nested_factor_exact * effective_shards)
    )
    def projected_wall_hours(checkpoint_factor: float) -> float:
        return float(baseline_stage_c_hours / (float(checkpoint_factor) * k_nested_factor_exact * effective_shards))

    checkpoint_decision_table = [
        {
            "measured_checkpoint_factor_range": f"<= {checkpoint_factor_exact:.4f}",
            "projected_wall_hours_at_range_edge": projected_wall_hours(checkpoint_factor_exact),
            "decision": "blocked; strict checkpointing alone is insufficient for the 72h local budget",
        },
        {
            "measured_checkpoint_factor_range": f"{checkpoint_factor_exact:.4f} to 3.0",
            "projected_wall_hours_at_range_edge": projected_wall_hours(3.0),
            "decision": "blocked; continue C0 optimization or write amendment before Stage C",
        },
        {
            "measured_checkpoint_factor_range": f"3.0 to {local_pass_checkpoint_factor:.4f}",
            "projected_wall_hours_at_range_edge": ACCEPTED_LOCAL_WALL_HOURS,
            "decision": "blocked until measured factor reaches the local runtime budget threshold",
        },
        {
            "measured_checkpoint_factor_range": f"{local_pass_checkpoint_factor:.4f} to 4.0",
            "projected_wall_hours_at_range_edge": projected_wall_hours(4.0),
            "decision": "runtime gate may pass local 72h budget; report gap versus aggressive external target",
        },
        {
            "measured_checkpoint_factor_range": "4.0 to 5.0",
            "projected_wall_hours_at_range_edge": projected_wall_hours(5.0),
            "decision": "runtime gate passes local budget if all other gates pass; external target depends on measured nested-K factor and shard efficiency",
        },
        {
            "measured_checkpoint_factor_range": "> 5.0",
            "projected_wall_hours_at_range_edge": projected_wall_hours(5.0),
            "decision": "runtime gate passes local budget if all other gates pass; record measured factor and proceed only through strict gate composition",
        },
    ]
    return {
        "baseline_source": "results/phase_e/calibration/phase_e_calibration_status.json",
        "baseline_stage_c_T_only_serial_hours": baseline_stage_c_hours,
        "T_candidates": list(t_candidates),
        "diagnostic_T_checkpoints_not_selection_candidates": [3000],
        "K_candidates": list(k_candidates),
        "exact_checkpoint_factor_sum_T_over_max_T": checkpoint_factor_exact,
        "checkpointed_T_only_serial_hours_exact_accounting": checkpointed_t_only_hours,
        "exact_K_nested_factor_sum_K_over_max_K": k_nested_factor_exact,
        "strict_T_only_wall_hours_at_7_effective_shards": strict_target_wall_hours,
        "accepted_local_wall_hours": ACCEPTED_LOCAL_WALL_HOURS,
        "local_pass_checkpoint_factor_with_exact_K_and_7_shards": local_pass_checkpoint_factor,
        "external_target_checkpoint_factor_with_exact_K_and_7_shards": external_pass_checkpoint_factor,
        "checkpoint_factor_decision_table": checkpoint_decision_table,
        "external_feedback_target": {
            "checkpoint_factor_assumed": external_checkpoint_factor,
            "K_nested_factor_assumed": external_k_factor,
            "effective_shards_assumed": effective_shards,
            "target_serial_hours": external_target_serial_hours,
            "target_wall_hours": external_target_wall_hours,
            "interpretation": (
                "Useful as an aggressive C0 target, but not the strict reduction from the current Stage C T-only baseline "
                "unless the baseline includes additional repeated or independent trajectory work beyond the current projection."
            ),
        },
        "accounting_caveats": [
            "The current 3352h baseline is Stage C T-only with K=20; K nesting does not reduce that baseline directly.",
            "K nesting is relevant when Stage D K calibration is combined with or compared against separate K runs.",
            "For the current Stage C T-only baseline, checkpointing from independent T runs to one T=max(T) trajectory gives a strict factor of sum(T)/max(T)=1.9375.",
            "Any 4x checkpoint factor must be justified by a different baseline, additional endpoint-doubling duplication, batching, or measured C0 speedups.",
            "T=3000 may be recorded as a diagnostic checkpoint only; it is not a final selection candidate unless a protocol amendment adds it.",
        ],
    }


def _pilot_design() -> dict[str, Any]:
    optimizer_cases = 2 * 2 * 2 * 30
    lhs_cases = 2 * 2 * 2 * 20
    return {
        "stage_c_optimizer_pilot_cases": optimizer_cases,
        "stage_c_optimizer_case_definition": "one (pilot_instance, noise_rate, shot_budget, seed) unit for T/K endpoint comparison",
        "stage_c_optimizer_design": {
            "depth2_instances": 2,
            "noise_rates": 2,
            "shot_budgets": 2,
            "seeds": 30,
            "pilot_cases": optimizer_cases,
        },
        "stage_c_lhs_pilot_cases": lhs_cases,
        "stage_c_lhs_case_definition": "one (pilot_instance, noise_rate, shot_budget, independent_lhs_design) unit for N_LHS comparison",
        "stage_c_lhs_design": {
            "depth2_instances": 2,
            "noise_rates": 2,
            "shot_budgets": 2,
            "independent_lhs_designs": 20,
            "pilot_cases": lhs_cases,
        },
        "minimum_n_for_5pct_change_to_represent_at_least_two_cases": 40,
        "optimizer_5pct_max_changed_cases": _max_changes_strict_less_than(0.05, optimizer_cases),
        "lhs_2pct_max_changed_cases": _max_changes_strict_less_than(0.02, lhs_cases),
        "instance_selection": (
            "Use the already frozen Stage B pilot instances H1 and H4 from the Phase E stratified depth-2 selection. "
            "Do not select instances using depth-2 outcomes."
        ),
        "pilot_case_reuse_policy": (
            "Pilot cases may be reused in Stage C only as calibration cases, not as an independent holdout. "
            "If an independent confirmation of the gate is required, add a holdout validation subset by amendment before Stage C."
        ),
        "c0_runtime_probe_design": {
            "purpose": "runtime/resume/storage benchmark only, not final threshold acceptance",
            "instances": ["H1", "H4"],
            "noise_rates": [0.003, 0.01],
            "shot_budgets": [512, 4096],
            "seeds": [42, 31415],
            "minimum_probe_cases": 16,
        },
    }


def _power_analysis_table(pilot: dict[str, Any]) -> list[dict[str, Any]]:
    optimizer_n = int(pilot["stage_c_optimizer_pilot_cases"])
    optimizer_threshold = int(math.ceil(0.80 * optimizer_n))
    lhs_n = int(pilot["stage_c_lhs_pilot_cases"])
    sign_threshold = int(math.ceil(0.95 * lhs_n))
    change_threshold = _max_changes_strict_less_than(0.02, lhs_n)
    rows: list[dict[str, Any]] = []
    for true_rate in (0.80, 0.85, 0.90, 0.95):
        p_pass = _binom_sf_at_least(optimizer_threshold, optimizer_n, true_rate)
        rows.append(
            {
                "gate": "T_or_K_80pct_pass_rate",
                "n": optimizer_n,
                "threshold": f">={optimizer_threshold} stable cases",
                "true_rate": true_rate,
                "P_pass": p_pass,
                "P_fail": 1.0 - p_pass,
            }
        )
    for true_rate in (0.93, 0.95, 0.96, 0.98, 0.99):
        p_pass = _binom_sf_at_least(sign_threshold, lhs_n, true_rate)
        rows.append(
            {
                "gate": "N_LHS_95pct_sign_stability",
                "n": lhs_n,
                "threshold": f">={sign_threshold} stable signs",
                "true_rate": true_rate,
                "P_pass": p_pass,
                "P_fail": 1.0 - p_pass,
            }
        )
    for true_change_rate in (0.005, 0.01, 0.02, 0.03, 0.05):
        p_pass = _binom_cdf(change_threshold, lhs_n, true_change_rate)
        rows.append(
            {
                "gate": "N_LHS_lt2pct_classification_change",
                "n": lhs_n,
                "threshold": f"<2% changes, <={change_threshold} changed cases",
                "true_rate": true_change_rate,
                "P_pass": p_pass,
                "P_fail": 1.0 - p_pass,
            }
        )
    return rows


def _k_spacing_policy() -> dict[str, Any]:
    base_threshold = 0.01
    k_candidates = PhaseEOptimizerConfig().random_start_candidates
    transitions = []
    reference_ratio = 2.0
    for left, right in zip(k_candidates[:-1], k_candidates[1:]):
        ratio = float(right / left)
        multiplier = float(math.log(ratio) / math.log(reference_ratio))
        transitions.append(
            {
                "transition": f"{left}->{right}",
                "multiplicative_ratio": ratio,
                "log_ratio_multiplier_vs_doubling": multiplier,
                "regret_change_threshold_fraction_of_ideal_range": base_threshold * multiplier,
                "regret_change_threshold_percent_of_ideal_range": 100.0 * base_threshold * multiplier,
            }
        )
    return {
        "K_candidates_frozen": list(k_candidates),
        "policy": (
            "Keep the frozen non-uniform K grid {20,50,100}. To avoid penalizing the larger 20->50 multiplicative step, "
            "scale the per-transition regret-change threshold by log(K2/K1)/log(2), with 1% as the doubling baseline."
        ),
        "transitions": transitions,
        "sensitivity": "Also report the unscaled 1% threshold for both K transitions.",
        "grid_change_requires_amendment": True,
    }


def _duplicate_jitter_protocol(calibration_status: dict[str, Any]) -> dict[str, Any]:
    pilot_metrics = calibration_status.get("pilot_metrics", {})
    noisy_seconds = float(pilot_metrics.get("mean_noisy_objective_seconds") or 0.09163467498729005)
    zne_seconds = float(pilot_metrics.get("mean_zne_objective_seconds") or 0.2327884999976959)
    max_t = max(PhaseEOptimizerConfig().spsa_evaluation_budget_candidates)
    duplicate_cases = 30
    duplicate_seeds_per_case = 3
    duplicate_runs = duplicate_cases * duplicate_seeds_per_case
    estimated_serial_hours = float(duplicate_runs * max_t * (noisy_seconds + zne_seconds) / 3600.0)
    return {
        "definition": (
            "A duplicate endpoint is a full optimizer execution for the same instance, noise rate, shot budget, objective family, "
            "and T checkpoint, using a different optimizer seed/start schedule."
        ),
        "ordering": "Run before any T/K/N_LHS gate; it is a prerequisite, not a parallel optional diagnostic.",
        "design": {
            "representative_optimizer_cases": duplicate_cases,
            "duplicate_seeds_per_case": duplicate_seeds_per_case,
            "additional_duplicate_runs": duplicate_runs,
            "selection": "mechanically stratified from the optimizer pilot cases before inspecting C0 outcomes",
        },
        "metric": "p95 absolute ideal-regret difference between duplicate endpoints, expressed as a fraction of ideal objective range",
        "gate": {
            "pass": f"p95 jitter < {JITTER_WARNING_FRACTION_OF_RANGE:.3f} of ideal objective range",
            "warning": f"{JITTER_WARNING_FRACTION_OF_RANGE:.3f} <= p95 jitter < {JITTER_BLOCK_FRACTION_OF_RANGE:.3f}",
            "block": f"p95 jitter >= {JITTER_BLOCK_FRACTION_OF_RANGE:.3f}",
        },
        "p95_uncertainty_note": (
            "With 30 representative cases and 3 duplicate endpoints each, C0 must bootstrap pairwise duplicate differences "
            "and report an order-statistic/bootstrap interval for the p95 jitter estimate."
        ),
        "runtime_cost_estimate": {
            "mean_noisy_objective_seconds": noisy_seconds,
            "mean_zne_objective_seconds": zne_seconds,
            "max_T_for_costing": max_t,
            "additional_duplicate_runs": duplicate_runs,
            "estimated_serial_hours": estimated_serial_hours,
            "estimated_wall_hours_at_7_effective_shards": estimated_serial_hours / 7.0,
        },
    }


def _instance_coverage_justification(root: Path) -> dict[str, Any]:
    structural_path = root / "results" / "phase_e" / "planning" / "phase_e_structural_diagnostics.csv"
    rows = _read_csv_dicts(structural_path)
    pilot_labels = {"H1", "H4"}
    metrics = ("n_nodes", "n_edges", "density", "degree_mean", "degree_std", "weight_std")
    universe: dict[str, list[float]] = {metric: [] for metric in metrics}
    pilot_rows = []
    for row in rows:
        label = row.get("selected_label", "")
        parsed = {"selected_label": label, "stratum_id": row.get("stratum_id", ""), "stratum_name": row.get("stratum_name", ""), "weighted_flag": row.get("weighted_flag", "")}
        for metric in metrics:
            try:
                value = float(row.get(metric, "nan"))
            except ValueError:
                value = math.nan
            if math.isfinite(value):
                universe[metric].append(value)
                parsed[metric] = value
        if label in pilot_labels:
            pilot_rows.append(parsed)
    pilot_percentiles = []
    for row in pilot_rows:
        out = {"selected_label": row["selected_label"], "stratum_id": row["stratum_id"], "stratum_name": row["stratum_name"], "weighted_flag": row["weighted_flag"]}
        for metric in metrics:
            if metric in row:
                out[f"{metric}_value"] = row[metric]
                out[f"{metric}_percentile_rank"] = _percentile_rank(universe[metric], row[metric])
        pilot_percentiles.append(out)
    return {
        "source": "results/phase_e/planning/phase_e_structural_diagnostics.csv",
        "pilot_instances": ["H1", "H4"],
        "current_justification": (
            "H1 and H4 were frozen before depth-2 outcomes as one unweighted and one weighted pilot instance. "
            "They cover different strata, qubit counts, edge counts, density regimes, and weight heterogeneity in the available structural diagnostic universe."
        ),
        "pilot_structural_percentiles": pilot_percentiles,
        "limitation": "Two instances cannot estimate inter-instance variance robustly.",
        "amendment_trigger": (
            f"If any C0 gate metric has pilot-instance CV > {INSTANCE_VARIANCE_TRIGGER_CV:.0%} or max/min > {INSTANCE_VARIANCE_TRIGGER_MAX_TO_MIN}, "
            "add a third pilot instance by amendment before Stage C."
        ),
    }


def _gate_composition_logic() -> list[dict[str, str]]:
    return [
        {"order": "1", "gate": "duplicate_endpoint_jitter", "fail_action": "Stage C blocked; amendment required", "pass_action": "continue"},
        {"order": "2", "gate": "resume_integrity", "fail_action": "Stage C blocked; fix resume path", "pass_action": "continue"},
        {"order": "3", "gate": "storage", "fail_action": "Stage C blocked; free space or reduce scope by amendment", "pass_action": "continue"},
        {"order": "4", "gate": "shard_balance", "fail_action": "Stage C blocked; rebalance shards", "pass_action": "continue"},
        {"order": "5", "gate": "runtime", "fail_action": "Stage C blocked; amendment for budget or scope", "pass_action": "continue"},
        {"order": "6", "gate": "T_selection", "fail_action": "If all T fail, amendment required", "pass_action": "freeze T_star"},
        {"order": "7", "gate": "K_selection", "fail_action": "If all K fail, amendment required", "pass_action": "freeze K_star"},
        {"order": "8", "gate": "N_LHS_selection", "fail_action": "If all N_LHS fail, amendment required", "pass_action": "freeze N_LHS_star"},
    ]


def _threshold_table(pilot: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "parameter": "single_case_regret_change",
            "value": "<= 1% of ideal objective range",
            "justification": (
                "Operational stability threshold. C0 must measure duplicate-endpoint jitter; if the p95 duplicate jitter exceeds this "
                "threshold, Stage C is blocked for amendment rather than auto-relaxing the rule."
            ),
            "sensitivity": "Report pass rates at 0.5%, 1%, and 2%.",
        },
        {
            "parameter": "mean_additional_regret_improvement",
            "value": "<= 2%",
            "justification": (
                "Must be larger than the measured endpoint noise floor and stricter than a visually flat convergence tail. "
                "If endpoint noise is comparable to 2%, Stage C requires amendment or a larger pilot."
            ),
            "sensitivity": "Report sensitivity at 1%, 2%, and 3%.",
        },
        {
            "parameter": "T_or_K_discordance_classification_change",
            "value": "< 5% of optimizer pilot cases",
            "justification": (
                f"With n={pilot['stage_c_optimizer_pilot_cases']} optimizer pilot cases, strict <5% allows at most "
                f"{pilot['optimizer_5pct_max_changed_cases']} changed cases. This avoids fractional case thresholds and exceeds the n>=40 minimum."
            ),
            "sensitivity": "Report changed-case counts and exact binomial confidence intervals.",
        },
        {
            "parameter": "pass_rate_for_T_or_K_stability",
            "value": ">= 80% of optimizer pilot cases",
            "justification": (
                "Balances SPSA endpoint noise against rejecting broad instability. 90% is treated as conservative sensitivity; "
                "70% is treated as permissive sensitivity, not the primary gate."
            ),
            "sensitivity": "Report pass rates at 70%, 80%, and 90%; report exact binomial confidence intervals.",
        },
        {
            "parameter": "N_LHS_relative_monte_carlo_error",
            "value": "< 5%",
            "justification": "Standard precision target for the relative r_MSE estimator across independent LHS designs.",
            "sensitivity": "Report estimates at 2.5%, 5%, and 10%.",
        },
        {
            "parameter": "N_LHS_r_MSE_sign_stability",
            "value": ">= 95% of independent LHS designs",
            "justification": "The mitigation/no-mitigation MSE sign must be stable before selecting N_LHS.",
            "sensitivity": "Report sign stability at each N_LHS and each pilot cell.",
        },
        {
            "parameter": "N_LHS_mean_doubling_change",
            "value": "< 0.01 absolute and < 5% relative",
            "justification": "The conjunction prevents either absolute or relative tolerance from masking instability in the other scale.",
            "sensitivity": "Report both criteria separately plus the combined pass indicator.",
        },
        {
            "parameter": "N_LHS_discordance_classification_change",
            "value": "< 2% of LHS pilot cases",
            "justification": (
                f"Uses a fixed value rather than a range. With n={pilot['stage_c_lhs_pilot_cases']} LHS pilot cases, "
                f"strict <2% allows at most {pilot['lhs_2pct_max_changed_cases']} changed cases."
            ),
            "sensitivity": "Report the 1% gate as conservative sensitivity only.",
        },
    ]


def _definitions() -> dict[str, str]:
    return {
        "ideal_regret": "ideal objective value at an optimizer endpoint minus the best available ideal reference value for the same cell",
        "ideal_objective_range": "max ideal objective minus min ideal objective over the validation/reference set for the same instance/cell",
        "discordance_classification": "the pre-specified sign/threshold label comparing noisy and ZNE endpoint or MSE behavior",
        "pilot_case": "one pre-specified unit used by a gate; optimizer cases are instance/noise/shot/seed units and LHS cases are instance/noise/shot/design units",
        "checkpoint_factor": "separate-T evaluation cost divided by checkpointed max-T trajectory cost under the same objective family",
        "nested_K_factor": "separate-K start cost divided by one max-K start set with deterministic prefix aggregation",
        "shard_efficiency": "effective parallel speedup divided by nominal shard count",
        "effective_shards": "nominal shards multiplied by measured shard efficiency",
        "duplicate_endpoint_jitter": "endpoint variability under duplicated optimizer executions for the same case with different optimizer seeds",
        "pre_registration": "freezing this C0 plan before C0 execution begins; later changes are recorded as amendments",
    }


def _gate_summary_table(pilot: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "gate": "T stability",
            "criterion": "single-case regret change, mean improvement, classification changes, nonfinite failures",
            "threshold": "1% range, 2% mean, <5% classification, zero nonfinite",
            "justification": "convergence stability above endpoint noise floor",
        },
        {
            "gate": "K stability",
            "criterion": "best-of-K regret change and classification changes under nested starts",
            "threshold": "log-ratio-scaled regret threshold, <5% classification, >=80% stable cases",
            "justification": f"n={pilot['stage_c_optimizer_pilot_cases']} optimizer pilot cases supports integer changed-case threshold",
        },
        {
            "gate": "duplicate endpoint jitter",
            "criterion": "p95 duplicate endpoint ideal-regret difference",
            "threshold": f"warn at {JITTER_WARNING_FRACTION_OF_RANGE:.3f}; block at {JITTER_BLOCK_FRACTION_OF_RANGE:.3f} of ideal range",
            "justification": "decision thresholds must exceed intrinsic endpoint jitter",
        },
        {
            "gate": "N_LHS stability",
            "criterion": "MC error, sign stability, doubling change, classification changes",
            "threshold": "<5% MC error, >=95% sign, <0.01 and <5%, <2% classification",
            "justification": "replicable MSE validation across independent LHS designs",
        },
        {
            "gate": "runtime",
            "criterion": "projected wall-clock after C0 benchmark",
            "threshold": f"<= {ACCEPTED_LOCAL_WALL_HOURS} hours accepted local wall-clock budget",
            "justification": "avoid launching protocol-scale calibration if local runtime is not feasible",
        },
        {
            "gate": "storage",
            "criterion": "checkpoint/output bytes and available disk",
            "threshold": f"post-run disk use <= {DISK_POST_RUN_USAGE_GATE_FRACTION:.0%}; required bytes <= {DISK_FREE_SPACE_USE_GATE_FRACTION:.0%} free space",
            "justification": "avoid partial calibration failure from disk exhaustion",
        },
        {
            "gate": "resume integrity",
            "criterion": "crash/resume output equals uninterrupted output",
            "threshold": "SHA-256 identical for deterministic C0 subset",
            "justification": "checkpointed Stage C must be resumable without changing outputs",
        },
        {
            "gate": "shard balance",
            "criterion": "runtime CV and max/median shard runtime",
            "threshold": f"CV < {SHARD_RUNTIME_CV_GATE:.0%}; max/median <= {SHARD_MAX_TO_MEDIAN_GATE}",
            "justification": "wall-clock projection invalid if one shard dominates",
        },
    ]


def _build_plan(calibration_status: dict[str, Any], root: Path) -> dict[str, Any]:
    accounting = _runtime_accounting(calibration_status)
    pilot = _pilot_design()
    power = _power_analysis_table(pilot)
    return {
        "created_at_utc": _utc_now(),
        "status": "phase_e_stage_c0_optimization_plan_recorded_not_executed",
        "version": PHASE_E_STAGE_C0_PLAN_VERSION,
        "changelog": [
            {
                "version": "v0_1_draft",
                "changes": "Initial C0 plan with checkpointed SPSA, nested K, runtime accounting, and basic stopping rules.",
            },
            {
                "version": "v0_2_draft",
                "changes": (
                    "Adds threshold justification, exact pilot-case counts, diagnostic-only T=3000 handling, fixed N_LHS rules, "
                    "storage/resume/shard gates, diagnostic outputs, definitions, and final gate summary."
                ),
            },
            {
                "version": "v0_3_draft",
                "changes": (
                    "Adds binomial pilot sample-size calculations, corrected checkpoint-factor decision tree, K spacing policy, duplicate endpoint jitter protocol, "
                    "instance coverage justification, strict gate composition, explicit T-grid freeze, pre-registration, and no-blinding statement."
                ),
            },
        ],
        "official_execution_authorized": False,
        "purpose": "Optimize and gate protocol-scale calibration before running Stage C.",
        "pre_registration_statement": (
            f"This plan is frozen at version {PHASE_E_STAGE_C0_PLAN_VERSION}. Any change after C0 execution begins constitutes an amendment "
            "and must be recorded with justification, timestamp, and whether it was motivated by C0 results. Changes motivated by C0 results require re-execution of affected gates."
        ),
        "blinding_statement": (
            "The analyst is not blinded to C0 outcomes. To mitigate analyst degrees of freedom, all thresholds are frozen before C0 execution, "
            "all amendments require written justification, and no threshold may be relaxed after observing C0 data."
        ),
        "definitions": _definitions(),
        "statistical_interpretation": {
            "checkpointed_trajectories_are_dependent": True,
            "declared_use": (
                "Checkpointed SPSA endpoints at T={250,500,1000,2000,4000} are treated as a paired convergence curve "
                "along the same stochastic trajectory, not as independent statistical comparisons between budgets."
            ),
            "allowed_for_calibration": True,
            "independent_test_claims_allowed": False,
            "amendment_required_if_protocol_requires_independent_T_comparisons": True,
            "required_amendment_if_independent_T_comparisons_are_required": (
                "Add independent trajectory families for each T budget before proceeding, with seed schedules and start sets "
                "frozen before inspecting depth-2 calibration outcomes."
            ),
        },
        "pilot_design": pilot,
        "pilot_sample_size_justification": {
            "binomial_power_table": power,
            "interpretation": [
                "For n=240 and an 80% pass gate, the gate has high probability to pass when true stability is at least 85%, but only about coin-flip probability at exactly 80%.",
                "For n=160 and 95% sign stability, true 96% sign stability gives about 80% pass probability; true 93% is unlikely to pass.",
                "For n=160 and <2% classification change, a true 1% change rate usually passes, while true 5% almost always fails.",
            ],
        },
        "instance_coverage_justification": _instance_coverage_justification(root),
        "threshold_calibration": {
            "thresholds_are_final_for_C0_plan": False,
            "threshold_noise_floor_rule": (
                "C0 must estimate duplicate endpoint variability. If the measured endpoint noise floor is not below the 1%/2% "
                "thresholds, Stage C is blocked for amendment rather than changing thresholds after seeing Stage C data."
            ),
            "threshold_table": _threshold_table(pilot),
        },
        "runtime_reduction_accounting": accounting,
        "frozen_T_grid": {
            "T_candidates_frozen": list(PhaseEOptimizerConfig().spsa_evaluation_budget_candidates),
            "T_transitions_evaluated": ["250->500", "500->1000", "1000->2000", "2000->4000"],
            "T3000_status": "diagnostic_checkpoint_only_not_selection_candidate_without_amendment",
            "T8000_status": "not_pre_specified; if 2000->4000 fails, amendment required",
            "grid_change_requires_amendment": True,
        },
        "K_spacing_policy": _k_spacing_policy(),
        "duplicate_endpoint_jitter_protocol": _duplicate_jitter_protocol(calibration_status),
        "pre_specified_stopping_rules": {
            "T_selection": [
                "For each paired transition T -> 2T, compute endpoint best-of-K ideal regret at both checkpoints on the same trajectory family.",
                "A pilot case is T-stable if absolute ideal-regret change is <= 1% of the ideal objective range for that instance/cell.",
                "Also require mean additional regret improvement after doubling to be <= 2%.",
                "Also require discordance classification changes after doubling in < 5% of optimizer pilot cases.",
                "Select the smallest T whose T -> 2T transition satisfies all criteria in at least 80% of optimizer pilot cases and has no nonfinite optimizer failures.",
                "If multiple T values pass, select the smallest passing T.",
                "Report absolute ideal-regret-change percentiles p50/p90/p95 for each passing and failing T transition.",
                "T=3000 may be collected as a diagnostic checkpoint, but it is not a final candidate unless an amendment adds it.",
                "If 2000 -> 4000 fails, do not select 4000 by default; trigger runtime/scope amendment because no 8000 checkpoint is pre-specified.",
            ],
            "K_selection": [
                "Use nested starts: run up to max K once and evaluate K={20,50,100} as deterministic prefixes of the same start set.",
                "A pilot case is K-stable if best-of-K ideal regret changes by the transition-specific log-ratio-scaled threshold when K increases to the next candidate.",
                "Require classification changes after K increase in < 5% of optimizer pilot cases.",
                "Select the smallest K satisfying stability in at least 80% of optimizer pilot cases with no nonfinite optimizer failures.",
                "Report regret-change distributions by K transition, including p50/p90/p95.",
                "If no K passes, trigger amendment; extending K to 200 is not automatic.",
            ],
            "N_LHS_selection": [
                "Use 20 independent LHS designs per pilot cell for protocol-scale N_LHS calibration.",
                "For each N_LHS, compute r_MSE within each same-size independent design against ideal values on the same design points.",
                "Do not use a larger reference design for the primary N_LHS gate; an N_ref diagnostic may be added only as non-primary evidence if the runtime gate approves it.",
                "Require relative Monte Carlo error of r_MSE < 5%.",
                "Require r_MSE sign stability in >= 95% of independent designs.",
                "Require mean change after doubling N_LHS < 0.01 absolute and < 5% relative.",
                "Require discordance classification changes after doubling N_LHS in < 2% of LHS pilot cases.",
                "Select the smallest N_LHS satisfying all criteria.",
            ],
            "C0_runtime_gate": [
                "C0 must benchmark checkpointed SPSA and nested K on the fixed C0 runtime probe design before Stage C.",
                "C0 must report measured effective checkpoint factor, nested-K factor, shard efficiency, and projected wall-clock hours.",
                f"Proceed to Stage C only if projected wall-clock <= {ACCEPTED_LOCAL_WALL_HOURS} hours or another budget is explicitly accepted before execution.",
                "If C0 cannot reach the external-feedback target near 66 wall hours or the accepted local runtime budget, write an amendment before Stage C.",
                f"Shard runtime CV must be < {SHARD_RUNTIME_CV_GATE:.0%} and max/median shard runtime must be <= {SHARD_MAX_TO_MEDIAN_GATE}.",
            ],
            "jitter_gate": [
                "Run duplicate endpoint jitter protocol before any T/K/N_LHS gate.",
                f"Block Stage C if p95 duplicate endpoint jitter >= {JITTER_BLOCK_FRACTION_OF_RANGE:.3f} of ideal objective range.",
                f"Warn if p95 duplicate endpoint jitter is between {JITTER_WARNING_FRACTION_OF_RANGE:.3f} and {JITTER_BLOCK_FRACTION_OF_RANGE:.3f}.",
            ],
            "storage_gate": [
                "C0 must estimate checkpoint bytes per shard, total checkpoint bytes, output bytes, logs, hashes, and backup overhead.",
                f"Pass only if projected post-run disk usage stays <= {DISK_POST_RUN_USAGE_GATE_FRACTION:.0%}.",
                f"Pass only if required bytes are <= {DISK_FREE_SPACE_USE_GATE_FRACTION:.0%} of currently available free space.",
                "No Stage C run may proceed if the storage estimate omits checkpoints, logs, hashes, or backups.",
            ],
            "resume_integrity_gate": [
                "C0 must run a deterministic crash-and-resume simulation on a fixed subset.",
                "The post-resume output must be SHA-256 identical to the uninterrupted output for the same subset.",
                "If bit identity fails, checkpointed Stage C is blocked until the resume path is fixed.",
            ],
        },
        "C0_diagnostic_outputs": [
            "ideal_regret_summary_by_cell.csv with p50/p90/p95 and finite counts",
            "checkpoint_factor_by_instance.csv and scatter-ready data",
            "nested_K_factor_by_instance.csv and scatter-ready data",
            "spsa_convergence_curves.csv with regret versus evaluation/checkpoint",
            "discordance_classification_transition_table.csv for each hyperparameter transition",
            "runtime_profile_by_phase.csv split into optimizer, objective evaluation, I/O, checkpointing, and aggregation",
            "shard_runtime_balance.csv with mean, sd, CV, median, max, and max/median",
            "storage_projection.json with checkpoint bytes per shard and total required bytes",
            "resume_integrity_report.json with uninterrupted/resumed hashes",
            "threshold_sensitivity_report.csv for 70/80/90 pass rates and threshold sensitivity values",
            "pilot_sample_size_power_table.csv with binomial pass/fail probabilities",
            "duplicate_endpoint_jitter_report.csv with p95 jitter and bootstrap/order-statistic interval",
            "gate_composition_decision.json with strict pass/fail status for each ordered gate",
        ],
        "C0_execution_plan": [
            "Implement checkpointed SPSA runner that records endpoints at all protocol T checkpoints during one max-T trajectory.",
            "Record that checkpoint endpoints are paired/dependent convergence checkpoints.",
            "Optionally record T=3000 as diagnostic-only evidence; do not use it for final selection without amendment.",
            "Implement nested-K aggregation from one max-K start set.",
            "Benchmark the fixed C0 subset without official outputs and compare measured speedups against both strict and aggressive targets.",
            "Run storage and crash/resume gates before any Stage C calibration.",
            "Only after C0, decide whether Stage C runs, is parallelized further, or is amended.",
        ],
        "gate_composition_logic": {
            "strict_AND": True,
            "no_soft_pass": True,
            "no_conditional_proceed": True,
            "ordering_is_mandatory": True,
            "rule": "If any gate fails, the only allowed path is a written amendment before re-execution.",
            "ordered_gates": _gate_composition_logic(),
        },
        "gate_summary_table": _gate_summary_table(pilot),
        "not_included_due_to_protocol_risk": [
            "warm-start from depth-1 parameters",
            "early stopping SPSA",
            "asymmetric T allocation between noisy and ZNE",
            "pre-filtered stress subset replacing the protocol pilot",
            "float32 calibration in place of float64 without equivalence validation",
        ],
        "next_decision": "implement_C0_benchmark_runner_not_protocol_scale_stage_C",
    }


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            values.append(str(value).replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _markdown(plan: dict[str, Any]) -> str:
    accounting = plan["runtime_reduction_accounting"]
    external = accounting["external_feedback_target"]
    pilot = plan["pilot_design"]
    lines = [
        "# Phase E Stage C0 Optimization Plan",
        "",
        f"Version: `{plan['version']}`",
        f"Created UTC: {plan['created_at_utc']}",
        f"Status: `{plan['status']}`",
        "",
        "This plan records how Stage C calibration will be optimized and gated before any protocol-scale run.",
        "",
        "Pre-registration statement:",
        "",
        plan["pre_registration_statement"],
        "",
        "No-blinding statement:",
        "",
        plan["blinding_statement"],
        "",
        "## Changelog",
        "",
    ]
    lines.extend(f"- `{row['version']}`: {row['changes']}" for row in plan["changelog"])
    lines.extend(["", "## Definitions", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in plan["definitions"].items())
    lines.extend(
        [
            "",
            "## Statistical Interpretation",
            "",
            "- Checkpointed SPSA endpoints are dependent paired checkpoints on the same trajectory.",
            "- They are valid for calibration as a convergence curve.",
            "- They are not independent statistical tests between T values.",
            "- If the frozen protocol requires independent comparisons across T values, the paired-checkpoint design is insufficient and an amendment adding independent trajectory families per T is required before proceeding.",
            "",
            "## Calibration Of Decision Thresholds",
            "",
        ]
    )
    lines.extend(_markdown_table(plan["threshold_calibration"]["threshold_table"], ["parameter", "value", "justification", "sensitivity"]))
    lines.extend(
        [
            "",
            f"Threshold noise-floor rule: {plan['threshold_calibration']['threshold_noise_floor_rule']}",
            "",
            "## Pilot Design",
            "",
            f"- Optimizer pilot cases: `{pilot['stage_c_optimizer_pilot_cases']}`",
            f"- Optimizer case definition: {pilot['stage_c_optimizer_case_definition']}",
            f"- LHS pilot cases: `{pilot['stage_c_lhs_pilot_cases']}`",
            f"- LHS case definition: {pilot['stage_c_lhs_case_definition']}",
            f"- Minimum n for 5% to represent at least two cases: `{pilot['minimum_n_for_5pct_change_to_represent_at_least_two_cases']}`",
            f"- Optimizer <5% max changed cases: `{pilot['optimizer_5pct_max_changed_cases']}`",
            f"- LHS <2% max changed cases: `{pilot['lhs_2pct_max_changed_cases']}`",
            f"- Instance selection: {pilot['instance_selection']}",
            f"- Pilot reuse policy: {pilot['pilot_case_reuse_policy']}",
            "",
            "## Pilot Sample Size Justification",
            "",
        ]
    )
    lines.extend(_markdown_table(plan["pilot_sample_size_justification"]["binomial_power_table"], ["gate", "n", "threshold", "true_rate", "P_pass", "P_fail"]))
    lines.append("")
    lines.extend(f"- {item}" for item in plan["pilot_sample_size_justification"]["interpretation"])
    coverage = plan["instance_coverage_justification"]
    lines.extend(
        [
            "",
            "## Instance Coverage Justification",
            "",
            f"- Source: `{coverage['source']}`",
            f"- Current justification: {coverage['current_justification']}",
            f"- Limitation: {coverage['limitation']}",
            f"- Amendment trigger: {coverage['amendment_trigger']}",
            "",
            "Pilot structural percentiles:",
        ]
    )
    lines.extend(
        _markdown_table(
            coverage["pilot_structural_percentiles"],
            [
                "selected_label",
                "stratum_id",
                "stratum_name",
                "weighted_flag",
                "n_nodes_value",
                "n_nodes_percentile_rank",
                "n_edges_value",
                "n_edges_percentile_rank",
                "density_value",
                "density_percentile_rank",
                "weight_std_value",
                "weight_std_percentile_rank",
            ],
        )
    )
    frozen_t = plan["frozen_T_grid"]
    lines.extend(
        [
            "",
            "## Frozen T Grid",
            "",
            f"- T candidates frozen: `{frozen_t['T_candidates_frozen']}`",
            f"- T transitions evaluated: `{frozen_t['T_transitions_evaluated']}`",
            f"- T=3000 status: `{frozen_t['T3000_status']}`",
            f"- T=8000 status: `{frozen_t['T8000_status']}`",
            f"- Grid change requires amendment: `{frozen_t['grid_change_requires_amendment']}`",
            "",
            "## Runtime Accounting",
            "",
            f"- Baseline Stage C T-only serial hours: `{accounting['baseline_stage_c_T_only_serial_hours']}`",
            f"- Exact checkpoint factor from current T baseline: `{accounting['exact_checkpoint_factor_sum_T_over_max_T']}`",
            f"- Checkpointed T-only serial hours, strict accounting: `{accounting['checkpointed_T_only_serial_hours_exact_accounting']}`",
            f"- Exact nested-K factor: `{accounting['exact_K_nested_factor_sum_K_over_max_K']}`",
            f"- Strict T-only wall hours at 7 effective shards: `{accounting['strict_T_only_wall_hours_at_7_effective_shards']}`",
            f"- Accepted local runtime budget: `{accounting['accepted_local_wall_hours']}` wall hours",
            "",
            "External-feedback target:",
            "",
            f"- Assumed checkpoint factor: `{external['checkpoint_factor_assumed']}`",
            f"- Assumed K factor: `{external['K_nested_factor_assumed']}`",
            f"- Effective shards: `{external['effective_shards_assumed']}`",
            f"- Target serial hours: `{external['target_serial_hours']}`",
            f"- Target wall hours: `{external['target_wall_hours']}`",
            "",
            "Important caveat: the 4x checkpoint factor is an aggressive target, not the strict factor from the current Stage C T-only baseline.",
            "",
            "Checkpoint-factor decision table:",
            "",
        ]
    )
    lines.extend(_markdown_table(accounting["checkpoint_factor_decision_table"], ["measured_checkpoint_factor_range", "projected_wall_hours_at_range_edge", "decision"]))
    k_policy = plan["K_spacing_policy"]
    lines.extend(
        [
            "",
            "## K Spacing Policy",
            "",
            f"- K candidates frozen: `{k_policy['K_candidates_frozen']}`",
            f"- Policy: {k_policy['policy']}",
            f"- Sensitivity: {k_policy['sensitivity']}",
            f"- Grid change requires amendment: `{k_policy['grid_change_requires_amendment']}`",
            "",
            "K transition thresholds:",
            "",
        ]
    )
    lines.extend(_markdown_table(k_policy["transitions"], ["transition", "multiplicative_ratio", "log_ratio_multiplier_vs_doubling", "regret_change_threshold_percent_of_ideal_range"]))
    jitter = plan["duplicate_endpoint_jitter_protocol"]
    lines.extend(
        [
            "",
            "## Duplicate Endpoint Jitter Protocol",
            "",
            f"- Definition: {jitter['definition']}",
            f"- Ordering: {jitter['ordering']}",
            f"- Representative optimizer cases: `{jitter['design']['representative_optimizer_cases']}`",
            f"- Duplicate seeds per case: `{jitter['design']['duplicate_seeds_per_case']}`",
            f"- Additional duplicate runs: `{jitter['design']['additional_duplicate_runs']}`",
            f"- Metric: {jitter['metric']}",
            f"- Pass: `{jitter['gate']['pass']}`",
            f"- Warning: `{jitter['gate']['warning']}`",
            f"- Block: `{jitter['gate']['block']}`",
            f"- p95 uncertainty: {jitter['p95_uncertainty_note']}",
            f"- Estimated serial hours: `{jitter['runtime_cost_estimate']['estimated_serial_hours']}`",
            f"- Estimated wall hours at 7 effective shards: `{jitter['runtime_cost_estimate']['estimated_wall_hours_at_7_effective_shards']}`",
            "",
            "## Stopping Rules",
            "",
            "T selection:",
        ]
    )
    lines.extend(f"- {item}" for item in plan["pre_specified_stopping_rules"]["T_selection"])
    lines.append("")
    lines.append("K selection:")
    lines.extend(f"- {item}" for item in plan["pre_specified_stopping_rules"]["K_selection"])
    lines.append("")
    lines.append("N_LHS selection:")
    lines.extend(f"- {item}" for item in plan["pre_specified_stopping_rules"]["N_LHS_selection"])
    lines.append("")
    lines.append("C0 runtime gate:")
    lines.extend(f"- {item}" for item in plan["pre_specified_stopping_rules"]["C0_runtime_gate"])
    lines.append("")
    lines.append("Jitter gate:")
    lines.extend(f"- {item}" for item in plan["pre_specified_stopping_rules"]["jitter_gate"])
    lines.append("")
    lines.append("Storage gate:")
    lines.extend(f"- {item}" for item in plan["pre_specified_stopping_rules"]["storage_gate"])
    lines.append("")
    lines.append("Resume/integrity gate:")
    lines.extend(f"- {item}" for item in plan["pre_specified_stopping_rules"]["resume_integrity_gate"])
    lines.extend(["", "## C0 Diagnostic Outputs", ""])
    lines.extend(f"- `{item}`" for item in plan["C0_diagnostic_outputs"])
    composition = plan["gate_composition_logic"]
    lines.extend(
        [
            "",
            "## Gate Composition Logic",
            "",
            f"- Strict AND: `{composition['strict_AND']}`",
            f"- No soft pass: `{composition['no_soft_pass']}`",
            f"- No conditional proceed: `{composition['no_conditional_proceed']}`",
            f"- Ordering mandatory: `{composition['ordering_is_mandatory']}`",
            f"- Rule: {composition['rule']}",
            "",
        ]
    )
    lines.extend(_markdown_table(composition["ordered_gates"], ["order", "gate", "fail_action", "pass_action"]))
    lines.extend(["", "## Not Included Due To Protocol Risk", ""])
    lines.extend(f"- {item}" for item in plan["not_included_due_to_protocol_risk"])
    lines.extend(["", "## Gate Summary", ""])
    lines.extend(_markdown_table(plan["gate_summary_table"], ["gate", "criterion", "threshold", "justification"]))
    lines.extend(["", f"Next decision: `{plan['next_decision']}`", ""])
    return "\n".join(lines)


def write_phase_e_stage_c0_plan(root: Path = PROJECT_ROOT, overwrite: bool = True) -> dict[str, Any]:
    calibration_dir = ensure_dir(root / "results" / "phase_e" / "calibration")
    status_path = calibration_dir / "phase_e_calibration_status.json"
    output_json = calibration_dir / "phase_e_stage_c0_optimization_plan.json"
    output_md = calibration_dir / "phase_e_stage_c0_optimization_plan.md"
    output_hashes = calibration_dir / "phase_e_stage_c0_optimization_plan_hashes.json"
    if any(path.exists() for path in (output_json, output_md, output_hashes)) and not overwrite:
        raise FileExistsError("Refusing to overwrite existing Phase E Stage C0 plan outputs.")
    if not status_path.exists():
        raise FileNotFoundError("Phase E calibration status is required before writing Stage C0 plan.")
    calibration_status = _read_json(status_path)
    plan = _build_plan(calibration_status, root)
    write_json_atomic(output_json, plan, overwrite=overwrite)
    output_md.write_text(_markdown(plan), encoding="utf-8")
    hashes = {
        "created_at_utc": _utc_now(),
        "version": PHASE_E_STAGE_C0_PLAN_VERSION,
        "hashes": {
            "phase_e_stage_c0_optimization_plan_json": sha256_file(output_json),
            "phase_e_stage_c0_optimization_plan_md": sha256_file(output_md),
        },
    }
    write_json_atomic(output_hashes, hashes, overwrite=overwrite)
    plan["output_hashes"] = hashes["hashes"]
    return plan
