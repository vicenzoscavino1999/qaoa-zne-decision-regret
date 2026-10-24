from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .phase_e_config import PhaseEOptimizerConfig


PHASE_E_CALIBRATION_VERSION = "phase_e_calibration_status_v0_3_draft"
E_BUDGET_DECISION_FILENAME = "phase_e_cobyla_e_budget_decision.json"
K_RANDOM_START_DECISION_FILENAME = "phase_e_k_random_start_decision.json"
K_REF_REFERENCE_DECISION_FILENAME = "phase_e_k_ref_reference_decision.json"
N_LHS_OPERATIONAL_DECISION_FILENAME = "phase_e_n_lhs_operational_decision.json"
N_LHS_FIRST_SCREEN_ANALYSIS_REL = Path(
    "results/phase_e/calibration/lhs_sanity_e300_k100_n256_512_designs0_4/phase_e_lhs_sanity_analysis.json"
)

CALIBRATION_PROTOCOL_PILOT_DESIGN = {
    "depth2_instances": 2,
    "noise_rates": 2,
    "shot_budgets": 2,
    "richardson_orders": 1,
    "seeds": 30,
    "random_starts_for_calibration": 20,
    "lhs_independent_designs": 20,
}

SCALED_METHODOLOGY_VALIDATION_DESIGN = {
    "depth2_instances": 2,
    "noise_rates": 2,
    "shot_budgets": 2,
    "richardson_orders": 1,
    "seeds": 3,
    "E_COBYLA": (60, 120, 240),
    "K": (4, 8),
    "N_LHS": (16, 32, 64),
    "lhs_independent_designs": 3,
}

N_LHS_SEQUENTIAL_FIRST_SCREEN_DESIGN = {
    "depth2_instances": 2,
    "noise_rates": 2,
    "shot_budgets": 2,
    "richardson_orders": 1,
    "N_LHS": (256, 512),
    "lhs_independent_designs": 5,
    "escalation_N_LHS": (1024, 2048, 4096),
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _as_float(value: Any) -> float:
    return float(value)


def _as_int(value: Any) -> int:
    return int(float(value))


def _mean_by(rows: list[dict[str, str]], key: str, value_key: str, absolute: bool = False) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        value = _as_float(row[value_key])
        if absolute:
            value = abs(value)
        grouped.setdefault(str(row[key]), []).append(value)
    return {key_value: float(np.mean(values)) for key_value, values in sorted(grouped.items(), key=lambda item: float(item[0]))}


def _delta_last(means: dict[str, float]) -> float | None:
    keys = sorted(means, key=lambda value: float(value))
    if len(keys) < 2:
        return None
    return float(means[keys[-1]] - means[keys[-2]])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _positive_float(value: Any, default: float) -> float:
    try:
        candidate = float(value)
    except (TypeError, ValueError):
        return default
    if candidate <= 0.0 or not np.isfinite(candidate):
        return default
    return candidate


def _calibration_cells(design: dict[str, Any]) -> int:
    return int(design["depth2_instances"]) * int(design["noise_rates"]) * int(design["shot_budgets"]) * int(design["richardson_orders"])


def _optimizer_grid_hours(
    cells: int,
    seeds: int,
    e_candidates: tuple[int, ...],
    k_candidates: tuple[int, ...],
    noisy_objective_seconds: float,
    zne_objective_seconds: float,
) -> float:
    objective_pair_seconds = noisy_objective_seconds + zne_objective_seconds
    evaluation_units = sum(int(e) * int(k) for e in e_candidates for k in k_candidates)
    return float(cells * seeds * evaluation_units * objective_pair_seconds / 3600.0)


def _optimizer_single_hours(
    cells: int,
    seeds: int,
    e_budget: int,
    k_starts: int,
    noisy_objective_seconds: float,
    zne_objective_seconds: float,
) -> float:
    return _optimizer_grid_hours(cells, seeds, (int(e_budget),), (int(k_starts),), noisy_objective_seconds, zne_objective_seconds)


def _lhs_grid_hours(
    cells: int,
    lhs_candidates: tuple[int, ...],
    independent_designs: int,
    noisy_objective_seconds: float,
    zne_objective_seconds: float,
) -> float:
    # One LHS point needs an ideal, noisy, and ZNE objective. The current pilot
    # does not time ideal separately, so noisy timing is a conservative proxy.
    seconds_per_point = noisy_objective_seconds + noisy_objective_seconds + zne_objective_seconds
    return float(cells * independent_designs * sum(int(n) for n in lhs_candidates) * seconds_per_point / 3600.0)


def _build_runtime_projection(summary: dict[str, Any], optimizer_config: PhaseEOptimizerConfig) -> dict[str, Any]:
    decision = summary.get("calibration_decision", {})
    noisy_seconds = _positive_float(decision.get("mean_noisy_objective_seconds"), 0.07701855000050273)
    zne_seconds = _positive_float(decision.get("mean_zne_objective_seconds"), 0.174854074997711)
    protocol_cells = _calibration_cells(CALIBRATION_PROTOCOL_PILOT_DESIGN)
    scaled_cells = _calibration_cells(SCALED_METHODOLOGY_VALIDATION_DESIGN)
    protocol_seeds = int(CALIBRATION_PROTOCOL_PILOT_DESIGN["seeds"])
    protocol_k = int(CALIBRATION_PROTOCOL_PILOT_DESIGN["random_starts_for_calibration"])
    protocol_e_candidates = tuple(int(value) for value in optimizer_config.cobyla_evaluation_budget_candidates)
    protocol_lhs_candidates = tuple(int(value) for value in optimizer_config.lhs_size_candidates)
    e_projection = [
        {
            "E_COBYLA": e_budget,
            "cells": protocol_cells,
            "seeds": protocol_seeds,
            "K": protocol_k,
            "estimated_hours": _optimizer_single_hours(protocol_cells, protocol_seeds, e_budget, protocol_k, noisy_seconds, zne_seconds),
        }
        for e_budget in protocol_e_candidates
    ]
    lhs_projection = [
        {
            "N_LHS": n_lhs,
            "cells": protocol_cells,
            "independent_designs": int(CALIBRATION_PROTOCOL_PILOT_DESIGN["lhs_independent_designs"]),
            "estimated_hours": _lhs_grid_hours(
                protocol_cells,
                (n_lhs,),
                int(CALIBRATION_PROTOCOL_PILOT_DESIGN["lhs_independent_designs"]),
                noisy_seconds,
                zne_seconds,
            ),
        }
        for n_lhs in protocol_lhs_candidates
    ]
    return {
        "basis": "measured_objective_seconds_from_unified_depth2_technical_pilot",
        "caveats": [
            "Estimates use E_COBYLA as a max-evaluation cap and assume no trajectory reuse across candidate budgets.",
            "Estimates are serial wall-clock equivalents before parallelism, batching, or backend optimization.",
            "The scaled validation stage is a computational stepping stone; it is not sufficient to freeze final E_COBYLA, K, K_ref, or N_LHS.",
        ],
        "timing_inputs": {
            "mean_noisy_objective_seconds": noisy_seconds,
            "mean_zne_objective_seconds": zne_seconds,
            "objective_pair_seconds_noisy_plus_zne": noisy_seconds + zne_seconds,
            "lhs_point_seconds_ideal_proxy_plus_noisy_plus_zne": noisy_seconds + noisy_seconds + zne_seconds,
        },
        "protocol_pilot_design": CALIBRATION_PROTOCOL_PILOT_DESIGN,
        "scaled_methodology_validation_design": SCALED_METHODOLOGY_VALIDATION_DESIGN,
        "scaled_methodology_validation_estimated_hours": {
            "optimizer_grid": _optimizer_grid_hours(
                scaled_cells,
                int(SCALED_METHODOLOGY_VALIDATION_DESIGN["seeds"]),
                tuple(int(value) for value in SCALED_METHODOLOGY_VALIDATION_DESIGN["E_COBYLA"]),
                tuple(int(value) for value in SCALED_METHODOLOGY_VALIDATION_DESIGN["K"]),
                noisy_seconds,
                zne_seconds,
            ),
            "lhs_grid": _lhs_grid_hours(
                scaled_cells,
                tuple(int(value) for value in SCALED_METHODOLOGY_VALIDATION_DESIGN["N_LHS"]),
                int(SCALED_METHODOLOGY_VALIDATION_DESIGN["lhs_independent_designs"]),
                noisy_seconds,
                zne_seconds,
            ),
        },
        "protocol_E_COBYLA_candidate_serial_hours": e_projection,
        "protocol_E_COBYLA_candidate_total_serial_hours_no_reuse": float(sum(row["estimated_hours"] for row in e_projection)),
        "protocol_N_LHS_candidate_serial_hours": lhs_projection,
        "protocol_N_LHS_candidate_total_serial_hours": float(sum(row["estimated_hours"] for row in lhs_projection)),
        "sequential_N_LHS_first_screen": {
            "N_LHS": list(N_LHS_SEQUENTIAL_FIRST_SCREEN_DESIGN["N_LHS"]),
            "independent_designs": int(N_LHS_SEQUENTIAL_FIRST_SCREEN_DESIGN["lhs_independent_designs"]),
            "cells": protocol_cells,
            "estimated_hours": _lhs_grid_hours(
                protocol_cells,
                tuple(int(value) for value in N_LHS_SEQUENTIAL_FIRST_SCREEN_DESIGN["N_LHS"]),
                int(N_LHS_SEQUENTIAL_FIRST_SCREEN_DESIGN["lhs_independent_designs"]),
                noisy_seconds,
                zne_seconds,
            ),
            "status": "first_screen_only_not_final_N_LHS_decision",
            "escalation_N_LHS": list(N_LHS_SEQUENTIAL_FIRST_SCREEN_DESIGN["escalation_N_LHS"]),
        },
        "full_depth2_runtime_gate_example": {
            "depth2_cells_from_compute_plan": 270,
            "example_E_COBYLA": 500,
            "example_K_primary": 50,
            "example_K_ref": 100,
            "seeds": 30,
            "noisy_plus_zne_serial_hours": _optimizer_single_hours(270, 30, 500, 50, noisy_seconds, zne_seconds),
            "reference_serial_hours_ideal_proxy": float(270 * 30 * 100 * 500 * noisy_seconds / 3600.0),
            "interpretation": "Back-of-envelope only; final runtime gate must use selected E_COBYLA/K/K_ref, batching, parallelism, and storage estimates.",
        },
    }


def _stage_b_completed(summary: dict[str, Any]) -> bool:
    observed_e = {int(value) for value in summary.get("e_cobyla_candidates", [])}
    observed_k = {int(value) for value in summary.get("k_candidates", [])}
    observed_lhs = {int(value) for value in summary.get("lhs_candidates", [])}
    seed_count = len(summary.get("seeds", []))
    lhs_designs = int(summary.get("lhs_independent_designs", 1))
    required_e = {int(value) for value in SCALED_METHODOLOGY_VALIDATION_DESIGN["E_COBYLA"]}
    required_k = {int(value) for value in SCALED_METHODOLOGY_VALIDATION_DESIGN["K"]}
    required_lhs = {int(value) for value in SCALED_METHODOLOGY_VALIDATION_DESIGN["N_LHS"]}
    return bool(
        required_e.issubset(observed_e)
        and required_k.issubset(observed_k)
        and required_lhs.issubset(observed_lhs)
        and seed_count >= int(SCALED_METHODOLOGY_VALIDATION_DESIGN["seeds"])
        and lhs_designs >= int(SCALED_METHODOLOGY_VALIDATION_DESIGN["lhs_independent_designs"])
    )


def _protocol_e_candidate_started(summary: dict[str, Any]) -> bool:
    observed_e = {int(value) for value in summary.get("e_cobyla_candidates", [])}
    protocol_e = {int(value) for value in PhaseEOptimizerConfig().cobyla_evaluation_budget_candidates}
    scope = str(summary.get("scope", ""))
    return bool(observed_e.intersection(protocol_e) or "official_cap" in scope or "protocol_scale" in scope)


def _endpoint_budget_comparison_valid(summary: dict[str, Any]) -> bool:
    note = summary.get("implementation_note", {})
    if isinstance(note, dict) and note.get("endpoint_budget_comparison_valid") is False:
        return False
    return True


def _valid_e_budget_decision(
    decision: dict[str, Any] | None,
    observed_e_for_endpoint: list[int],
) -> dict[str, Any] | None:
    if decision is None:
        return None
    if decision.get("stage_c_E_budget_closed") is not True:
        return None
    if decision.get("official_execution_authorized") is not False:
        return None
    selected = decision.get("selected_E_COBYLA")
    if selected is None or int(selected) not in {int(value) for value in observed_e_for_endpoint}:
        return None
    criteria = decision.get("criteria", {})
    if not isinstance(criteria, dict) or not all(bool(value) for value in criteria.values()):
        return None
    return decision


def _valid_k_random_start_decision(
    decision: dict[str, Any] | None,
    observed_k: list[int],
    e_budget_decision: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if decision is None:
        return None
    if decision.get("stage_d_K_calibration_closed") is not True:
        return None
    if decision.get("official_execution_authorized") is not False:
        return None
    if decision.get("final_freeze_complete") is not False:
        return None
    selected = decision.get("selected_K")
    if selected is None or int(selected) not in {int(value) for value in observed_k}:
        return None
    if e_budget_decision is not None and int(decision.get("selected_E_COBYLA")) != int(e_budget_decision["selected_E_COBYLA"]):
        return None
    criteria = decision.get("criteria", {})
    required_true = (
        "protocol_k_coverage_valid",
        "selected_K_is_largest_prespecified_candidate",
        "selected_K_has_lowest_mean_abs_d_R",
        "endpoint_stability_not_satisfied_and_acknowledged",
        "random_start_limitation_acknowledged",
        "official_execution_not_authorized",
        "final_freeze_not_complete",
    )
    if not isinstance(criteria, dict) or not all(bool(criteria.get(key)) for key in required_true):
        return None
    return decision


def _valid_k_ref_reference_decision(
    decision: dict[str, Any] | None,
    k_random_start_decision: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if decision is None:
        return None
    if k_random_start_decision is None:
        return None
    if decision.get("stage_e_K_ref_reference_strategy_closed") is not True:
        return None
    if decision.get("official_execution_authorized") is not False:
        return None
    if decision.get("final_freeze_complete") is not False:
        return None
    criteria = decision.get("criteria", {})
    required_true = (
        "k_random_start_decision_recorded",
        "all_l_bfgs_b_and_cobyla_runs_finite",
        "deterministic_best_improves_sobol_audit_min_for_all_pilot_instances",
        "random_start_K_ref_not_required",
        "official_execution_not_authorized",
        "final_freeze_not_complete",
    )
    if not isinstance(criteria, dict) or not all(bool(criteria.get(key)) for key in required_true):
        return None
    return decision


def _valid_n_lhs_sanity_analysis(analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    if analysis is None:
        return None
    if analysis.get("official_execution_authorized") is not False:
        return None
    if int(analysis.get("selected_E_COBYLA") or -1) != 300:
        return None
    if int(analysis.get("selected_K") or -1) != 100:
        return None
    if analysis.get("status") not in {
        "lhs_5design_screen_stable_not_final_extend_to_20_designs_before_decision",
        "lhs_5design_screen_not_stable_extend_candidates_or_designs",
    }:
        return None
    return analysis


def _valid_n_lhs_operational_decision(
    decision: dict[str, Any] | None,
    n_lhs_sanity_analysis: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if decision is None:
        return None
    if decision.get("status") != "selected_N_LHS_2048_operational_with_quantitative_magnitude_limitation_not_final_freeze":
        return None
    selected = int(decision.get("selected_N_LHS", -1))
    allowed = set(int(value) for value in N_LHS_SEQUENTIAL_FIRST_SCREEN_DESIGN["N_LHS"])
    allowed.update(int(value) for value in N_LHS_SEQUENTIAL_FIRST_SCREEN_DESIGN["escalation_N_LHS"])
    if selected not in allowed:
        return None
    if n_lhs_sanity_analysis is not None:
        observed = {int(value) for value in n_lhs_sanity_analysis.get("observed_N_LHS_candidates", [])}
        if selected not in observed:
            return None
    return decision


def _latest_n_lhs_sanity_analysis(root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    calibration_dir = root / "results" / "phase_e" / "calibration"
    candidates: list[tuple[int, int, float, Path, dict[str, Any]]] = []
    for path in calibration_dir.glob("lhs_sanity*/phase_e_lhs_sanity_analysis.json"):
        analysis = _valid_n_lhs_sanity_analysis(_read_json_optional(path))
        if analysis is None:
            continue
        observed = [int(value) for value in analysis.get("observed_N_LHS_candidates", [])]
        if not observed:
            continue
        candidates.append((max(observed), len(observed), path.stat().st_mtime, path, analysis))
    if not candidates:
        return None, None
    _, _, _, path, analysis = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    return path, analysis


def _relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _next_n_lhs_candidate(observed: list[int]) -> int | None:
    observed_set = {int(value) for value in observed}
    for candidate in N_LHS_SEQUENTIAL_FIRST_SCREEN_DESIGN["escalation_N_LHS"]:
        if int(candidate) not in observed_set:
            return int(candidate)
    return None


def _build_stage_plan(
    projection: dict[str, Any],
    stage_b_completed: bool = False,
    protocol_e_candidate_started: bool = False,
    stage_b_actual_hours: float | None = None,
    endpoint_budget_comparison_valid: bool = True,
    e_budget_decision: dict[str, Any] | None = None,
    k_random_start_decision: dict[str, Any] | None = None,
    k_ref_reference_decision: dict[str, Any] | None = None,
    n_lhs_sanity_analysis: dict[str, Any] | None = None,
    n_lhs_operational_decision: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    scaled = projection["scaled_methodology_validation_estimated_hours"]
    stage_b_hours = stage_b_actual_hours if stage_b_completed and stage_b_actual_hours is not None else float(scaled["optimizer_grid"] + scaled["lhs_grid"])
    if e_budget_decision is not None:
        selected_e = int(e_budget_decision["selected_E_COBYLA"])
        stage_c_status = "completed"
        stage_c_design = f"selected E_COBYLA={selected_e} by patched official-cap sanity over E={{300,500,750,1000}}; final freeze still pending"
        stage_c_gate = "E_COBYLA_decision_recorded; proceed_to_K_calibration"
        stage_c_freeze_allowed = True
        if k_random_start_decision is not None:
            selected_k = int(k_random_start_decision["selected_K"])
            stage_d_status = "completed"
            stage_d_design = f"selected K={selected_k} by 30-seed calibration with random-start-limited label; final freeze still pending"
            stage_d_gate = "K_random_start_decision_recorded; proceed_to_K_ref_reference_strategy"
        else:
            stage_d_status = "ready_for_protocol_scale_K_calibration"
            stage_d_design = f"K={{20,50,100}}; E_COBYLA={selected_e}; seeds=30; same pilot cells"
            stage_d_gate = "run_K_calibration_or_write_scope_amendment_before_freeze"
    elif protocol_e_candidate_started and not endpoint_budget_comparison_valid:
        stage_c_status = "rerun_required_prepatch_endpoint_budget_comparison_invalid"
        stage_c_design = "patched rerun E_COBYLA={300,500}; K=20; same 2x2x2 pilot cells and seeds before extending to E={750,1000}"
        stage_c_gate = "rerun patched E_COBYLA={300,500} with same-objective seeds"
        stage_c_freeze_allowed = False
        stage_d_status = "blocked_until_E_COBYLA_strategy_selected"
        stage_d_design = "K={20,50,100}; E_COBYLA=selected_E_COBYLA; seeds=30; same pilot cells"
        stage_d_gate = "requires selected or bounded E_COBYLA plus runtime gate"
    elif protocol_e_candidate_started:
        stage_c_status = "in_progress_protocol_candidate_coverage_incomplete"
        stage_c_design = "E_COBYLA={300,500,750,1000}; K=20; seeds=30; same 2x2x2 pilot cells"
        stage_c_gate = "continue_missing_protocol_candidates_and_seed_count_or_write_scope_amendment"
        stage_c_freeze_allowed = True
        stage_d_status = "blocked_until_E_COBYLA_strategy_selected"
        stage_d_design = "K={20,50,100}; E_COBYLA=selected_E_COBYLA; seeds=30; same pilot cells"
        stage_d_gate = "requires selected or bounded E_COBYLA plus runtime gate"
    else:
        stage_c_status = "blocked_by_runtime_gate"
        stage_c_design = "E_COBYLA={300,500,750,1000}; K=20; seeds=30; same 2x2x2 pilot cells"
        stage_c_gate = "runtime_gate_or_scope_amendment_required"
        stage_c_freeze_allowed = True
        stage_d_status = "blocked_until_E_COBYLA_strategy_selected"
        stage_d_design = "K={20,50,100}; E_COBYLA=selected_E_COBYLA; seeds=30; same pilot cells"
        stage_d_gate = "requires selected or bounded E_COBYLA plus runtime gate"
    if k_ref_reference_decision is not None:
        stage_e_status = "completed"
        if n_lhs_operational_decision is not None:
            selected_n_lhs = int(n_lhs_operational_decision["selected_N_LHS"])
            stage_f_status = "completed_operational_N_LHS_2048_with_quantitative_limitation"
            stage_f_design = (
                f"selected N_LHS={selected_n_lhs} as operational validation budget; "
                "r_MSE magnitude convergence remains a documented limitation, while sign/classification stability is accepted"
            )
            stage_f_hours = None
            stage_f_gate = "N_LHS_operational_decision_recorded; proceed_to_remaining_freeze_gates"
        elif n_lhs_sanity_analysis is None:
            stage_f_status = "ready_for_sequential_N_LHS_first_screen_after_K_ref_decision"
            stage_f_design = "first screen N_LHS={256,512}; 5 independent LHS designs; same pilot cells; if unstable add next pre-specified N_LHS by combined batches; if stable extend active pair to 20 designs before decision"
            stage_f_hours = float(projection["sequential_N_LHS_first_screen"]["estimated_hours"])
            stage_f_gate = "user approval for first N_LHS sanity; no official execution; do not run full 20-design grid by default"
        elif n_lhs_sanity_analysis["status"] == "lhs_5design_screen_not_stable_extend_candidates_or_designs":
            observed_n = [int(value) for value in n_lhs_sanity_analysis.get("observed_N_LHS_candidates", [])]
            design_ids = [int(value) for value in n_lhs_sanity_analysis.get("observed_design_ids", [])]
            next_candidate = _next_n_lhs_candidate(observed_n)
            if next_candidate is None:
                stage_f_status = "lhs_5design_screen_not_stable_manual_review_required"
                stage_f_design = "all pre-specified N_LHS escalation candidates have been observed; review active adjacent comparison before any freeze"
                stage_f_hours = None
                stage_f_gate = "manual review of N_LHS sanity analysis"
            else:
                stage_f_status = f"lhs_5design_screen_not_stable_add_N_LHS_{next_candidate}"
                stage_f_design = f"add N_LHS={next_candidate} with design ids={design_ids}; combine with observed N_LHS={observed_n}; rerun report using expected N_LHS={observed_n + [next_candidate]}"
                stage_f_hours = _lhs_grid_hours(
                    int(projection["sequential_N_LHS_first_screen"]["cells"]),
                    (next_candidate,),
                    int(n_lhs_sanity_analysis.get("required_designs", 5)),
                    float(projection["timing_inputs"]["mean_noisy_objective_seconds"]),
                    float(projection["timing_inputs"]["mean_zne_objective_seconds"]),
                )
                stage_f_gate = "user approval for next sequential N_LHS batch; no official execution"
        else:
            active_pair = [int(value) for value in n_lhs_sanity_analysis.get("screen_metrics", {}).get("target_adjacent_pair", [256, 512])]
            required_designs = int(n_lhs_sanity_analysis.get("required_designs", 5))
            additional_designs = max(20 - required_designs, 0)
            stage_f_status = "lhs_5design_screen_stable_extend_active_pair_to_20_designs"
            stage_f_design = f"extend active adjacent pair N_LHS={active_pair} from {required_designs} to 20 independent designs before writing an N_LHS decision"
            stage_f_hours = _lhs_grid_hours(
                int(projection["sequential_N_LHS_first_screen"]["cells"]),
                tuple(active_pair),
                additional_designs,
                float(projection["timing_inputs"]["mean_noisy_objective_seconds"]),
                float(projection["timing_inputs"]["mean_zne_objective_seconds"]),
            )
            stage_f_gate = "user approval for 20-design confirmation; no final N_LHS freeze from 5 designs alone"
        stage_f_freeze_allowed = n_lhs_operational_decision is not None
    else:
        stage_e_status = "ready_for_K_ref_reference_strategy" if k_random_start_decision is not None else "blocked_until_K_decision_recorded"
        stage_f_status = "blocked_until_K_ref_reference_strategy"
        stage_f_design = "N_LHS={256,512,1024,2048,4096}; 20 independent LHS designs; same pilot cells"
        stage_f_hours = float(projection["protocol_N_LHS_candidate_total_serial_hours"])
        stage_f_gate = "confirm repeated LHS design implementation and pass runtime/storage feasibility"
        stage_f_freeze_allowed = True
    return [
        {
            "stage_id": "A",
            "stage_name": "technical_smoke_and_expanded_tiny_pilot",
            "status": "completed",
            "purpose": "validate code path and obtain objective timings",
            "design": "observed E_COBYLA={30}; K={1}; N_LHS={4}; seeds=1",
            "estimated_or_actual_hours": None,
            "gate_before_running": "already_run_no_official_outputs",
            "freeze_decision_allowed": False,
        },
        {
            "stage_id": "B",
            "stage_name": "scaled_methodology_validation",
            "status": "completed" if (stage_b_completed or protocol_e_candidate_started) else "next_recommended_not_run",
            "purpose": "check scaling, repeated LHS mechanics, and COBYLA evaluation-count semantics before protocol-scale calibration",
            "design": "E_COBYLA={60,120,240}; K={4,8}; N_LHS={16,32,64}; seeds=3; LHS_designs=3 on same 2x2x2 pilot cells",
            "estimated_or_actual_hours": float(stage_b_hours),
            "gate_before_running": "completed_or_superseded_by_protocol_candidate_sanity" if (stage_b_completed or protocol_e_candidate_started) else "user_or_runtime_budget_acknowledgement_if_local_runtime_is_too_high",
            "freeze_decision_allowed": False,
        },
        {
            "stage_id": "C",
            "stage_name": "protocol_scale_COBYLA_E_calibration",
            "status": stage_c_status,
            "purpose": "select smallest E_COBYLA from protocol candidates using endpoint-budget criteria",
            "design": stage_c_design,
            "estimated_or_actual_hours": float(projection["protocol_E_COBYLA_candidate_total_serial_hours_no_reuse"]),
            "gate_before_running": stage_c_gate,
            "freeze_decision_allowed": stage_c_freeze_allowed,
        },
        {
            "stage_id": "D",
            "stage_name": "protocol_scale_K_calibration",
            "status": stage_d_status,
            "purpose": "select K from {20,50,100} by best-of-K stability and classification stability",
            "design": stage_d_design,
            "estimated_or_actual_hours": None,
            "gate_before_running": stage_d_gate,
            "freeze_decision_allowed": True,
        },
        {
            "stage_id": "E",
            "stage_name": "K_ref_reference_strategy",
            "status": stage_e_status,
            "purpose": "avoid defaulting to K_ref=200 random-start reference unless deterministic reference path fails",
            "design": "test deterministic L-BFGS-B/COBYLA first; if needed calibrate K_ref={50,100,200}",
            "estimated_or_actual_hours": None,
            "gate_before_running": "reference optimizer correctness tests and runtime gate",
            "freeze_decision_allowed": True,
        },
        {
            "stage_id": "F",
            "stage_name": "protocol_scale_N_LHS_calibration",
            "status": stage_f_status,
            "purpose": "select N_LHS from protocol candidates with independent LHS designs",
            "design": stage_f_design,
            "estimated_or_actual_hours": stage_f_hours,
            "gate_before_running": stage_f_gate,
            "freeze_decision_allowed": stage_f_freeze_allowed,
        },
    ]


def _next_stage(stage_plan: list[dict[str, Any]]) -> dict[str, Any]:
    for row in stage_plan:
        if row["status"] != "completed":
            return row
    return stage_plan[-1]


def _recommended_next_step(
    stage_b_completed: bool,
    protocol_e_candidate_started: bool = False,
    endpoint_budget_comparison_valid: bool = True,
    e_budget_decision: dict[str, Any] | None = None,
    k_random_start_decision: dict[str, Any] | None = None,
    k_ref_reference_decision: dict[str, Any] | None = None,
    n_lhs_sanity_analysis: dict[str, Any] | None = None,
    n_lhs_operational_decision: dict[str, Any] | None = None,
) -> str:
    if k_ref_reference_decision is not None:
        if n_lhs_operational_decision is not None:
            return "continue remaining Phase E freeze gates with N_LHS=2048 recorded as an operational budget; do not run N_LHS=4096 unless a later amendment reopens the quantitative r_MSE target."
        if n_lhs_sanity_analysis is not None:
            if n_lhs_sanity_analysis["status"] == "lhs_5design_screen_not_stable_extend_candidates_or_designs":
                next_candidate = _next_n_lhs_candidate([int(value) for value in n_lhs_sanity_analysis.get("observed_N_LHS_candidates", [])])
                return f"add N_LHS={next_candidate} with the same 5 design ids, combine batches, and rerun the active adjacent-pair report."
            active_pair = n_lhs_sanity_analysis.get("screen_metrics", {}).get("target_adjacent_pair", [256, 512])
            return f"extend active adjacent pair N_LHS={active_pair} to 20 independent designs before any N_LHS decision."
        return "run the first sequential N_LHS screen N_LHS={256,512} with 5 independent designs after user approval; do not freeze N_LHS from this screen alone."
    if k_random_start_decision is not None:
        return "test deterministic L-BFGS-B/COBYLA reference strategy first; calibrate K_ref={50,100,200} only if deterministic reference is insufficient."
    if e_budget_decision is not None:
        return f"calibrate K={{20,50,100}} using selected E_COBYLA={int(e_budget_decision['selected_E_COBYLA'])}, while keeping final freeze and official execution blocked."
    if protocol_e_candidate_started and not endpoint_budget_comparison_valid:
        return "rerun patched E_COBYLA={300,500} with same-objective seeds before extending protocol E coverage or selecting an endpoint budget."
    if protocol_e_candidate_started:
        return "continue protocol-candidate E_COBYLA coverage with patched same-objective seeds, or write an explicit scope amendment before treating partial coverage as final."
    if stage_b_completed:
        return "run protocol-scale E_COBYLA calibration only after confirming runtime gate assumptions or writing an explicit scope amendment."
    return "run Stage B scaled methodology validation only if the local runtime budget is acceptable; otherwise amend the protocol scope before protocol-scale calibration."


def _build_methodology(
    report: dict[str, Any],
    projection: dict[str, Any],
    stage_plan: list[dict[str, Any]],
    stage_b_completed: bool,
) -> dict[str, Any]:
    protocol_e_started = bool(report.get("protocol_e_candidate_started", False))
    endpoint_budget_valid = bool(report.get("endpoint_budget_comparison_valid", True))
    e_budget_decision = report.get("e_budget_decision") if report.get("e_budget_decision_recorded") else None
    k_random_start_decision = report.get("k_random_start_decision") if report.get("k_random_start_decision_recorded") else None
    k_ref_reference_decision = report.get("k_ref_reference_decision") if report.get("k_ref_reference_decision_recorded") else None
    n_lhs_sanity_analysis = report.get("n_lhs_sanity_screen") if report.get("n_lhs_sanity_screen_recorded") else None
    n_lhs_operational_decision = report.get("n_lhs_operational_decision") if report.get("n_lhs_operational_decision_recorded") else None
    if stage_b_completed:
        missing_before_freeze = [
            "COBYLA evaluation-count semantics were recorded in Stage B, but final protocol must either accept max_evaluations-cap semantics or adjust requested E_COBYLA to exact evaluation counts.",
            "Protocol-scale 20 independent LHS designs have not been run; Stage B only validates the repeated-design machinery with 3 designs.",
            "tau_R_opt, tau_conc, tau_trivial, VIF/variance, nonfinite failure-rate, pause, memory, and storage thresholds are not frozen.",
            "K_ref reference optimizer path has not been tested.",
            "Full runtime/storage gate has not passed.",
        ]
    else:
        missing_before_freeze = [
            "Exact COBYLA evaluation-count semantics must be recorded; current E_COBYLA is a max_evaluations cap.",
            "Sequential N_LHS scripts now support repeated independent LHS designs, but the first 5-design screen and required 20-design confirmation have not been run.",
            "tau_R_opt, tau_conc, tau_trivial, VIF/variance, nonfinite failure-rate, pause, memory, and storage thresholds are not frozen.",
            "K_ref reference optimizer path has not been tested.",
            "Full runtime/storage gate has not passed.",
        ]
    if not endpoint_budget_valid:
        missing_before_freeze.insert(
            0,
            "The current E_COBYLA endpoint-budget comparison came from a pre-patch run with E-dependent objective seeds; rerun patched E_COBYLA candidates before using endpoint movement to select a budget.",
        )
    if e_budget_decision is not None:
        missing_before_freeze.insert(
            0,
            f"E_COBYLA={int(e_budget_decision['selected_E_COBYLA'])} is selected for the next calibration stages by 5-seed official-cap sanity; final freeze must explicitly accept this decision or supersede it with an amendment.",
        )
    if k_random_start_decision is not None:
        missing_before_freeze.insert(
            1 if e_budget_decision is not None else 0,
            f"K={int(k_random_start_decision['selected_K'])} is selected as the largest pre-specified random-start budget with a random-start-limited label; final freeze must explicitly accept this decision or supersede it with an amendment.",
        )
    if k_ref_reference_decision is not None:
        missing_before_freeze = [
            item
            for item in missing_before_freeze
            if item != "K_ref reference optimizer path has not been tested."
        ]
        insert_index = 0
        if e_budget_decision is not None:
            insert_index += 1
        if k_random_start_decision is not None:
            insert_index += 1
        missing_before_freeze.insert(
            insert_index,
            f"K_ref random-start calibration is not required because deterministic reference strategy `{k_ref_reference_decision['selected_reference_strategy']}` passed sanity; final freeze must explicitly accept this decision or supersede it with an amendment.",
        )
    if n_lhs_operational_decision is not None:
        missing_before_freeze = [
            item
            for item in missing_before_freeze
            if not item.startswith("Sequential N_LHS scripts now support repeated independent LHS designs")
        ]
        missing_before_freeze.insert(
            0,
            f"N_LHS={int(n_lhs_operational_decision['selected_N_LHS'])} is selected as an operational MSE-validation budget with an explicit limitation: r_MSE magnitude did not meet the strict quantitative convergence rule, so r_MSE magnitude is diagnostic while sign/classification stability is the robust claim.",
        )
    elif n_lhs_sanity_analysis is not None:
        missing_before_freeze = [
            item
            for item in missing_before_freeze
            if not item.startswith("Sequential N_LHS scripts now support repeated independent LHS designs")
        ]
        if n_lhs_sanity_analysis["status"] == "lhs_5design_screen_not_stable_extend_candidates_or_designs":
            missing_before_freeze.insert(
                0,
                "The current 5-design N_LHS active-pair screen was not stable; add the next pre-specified N_LHS candidate and rerun the combined active-pair report before any N_LHS decision.",
            )
        else:
            missing_before_freeze.insert(
                0,
                "The first 5-design N_LHS screen was stable but cannot freeze N_LHS; extend the active pair to 20 independent designs before any N_LHS decision.",
            )
    return {
        "created_at_utc": report["created_at_utc"],
        "status": "phase_e_calibration_methodology_recorded_stage_b_complete" if stage_b_completed else "phase_e_calibration_methodology_recorded_not_executed",
        "calibration_version": PHASE_E_CALIBRATION_VERSION,
        "official_execution_authorized": False,
        "methodology_source": {
            "protocol_document": "PHASE_E_COBYLA_ENDPOINT_PROTOCOL_v0.1-draft.md",
            "compute_plan_document": "computacion.md",
        },
        "ordered_methodology": [
            "technical benchmark of depth-2 objective evaluations",
            "COBYLA evaluation-budget calibration",
            "K random-start stability calibration",
            "K_ref reference strategy decision",
            "Latin-hypercube MSE calibration",
            "runtime and storage gates before any official execution",
        ],
        "selection_rules_to_freeze": {
            "E_COBYLA": [
                "Compare endpoints at E_COBYLA and the next larger candidate when possible using same initial points, seeds, and objective definitions.",
                "Require median ideal-regret change below 1% of the ideal objective range.",
                "Require discordance-classification changes below 5% of pilot cases.",
                "Require mean additional regret improvement below 2%.",
                "Require projected full-experiment runtime to remain feasible.",
            ],
            "K": [
                "Best-of-K endpoint regret should change little when K doubles.",
                "Discordance classification changes must stay below the pre-specified threshold.",
                "Repeated random-start sets must be stable enough for the pilot cells.",
                "If the largest pre-specified K still shows endpoint movement, K may only be carried forward with an explicit random-start-limited label or a protocol amendment expanding K.",
            ],
            "N_LHS": [
                "Use a sequential first screen N_LHS={256,512} with 5 independent designs before committing to the full 20-design candidate grid.",
                "If the first screen is unstable, add the next pre-specified candidate with the same design ids and combine batches before re-reporting.",
                "If the active adjacent pair is stable, extend that pair to 20 independent designs before writing any N_LHS decision.",
                "Use multiple independent LHS designs, planned as 20 protocol designs.",
                "Require relative Monte Carlo error of r_MSE below 5%.",
                "Require r_MSE sign stability in at least 95% of independent designs.",
                "Require mean change after doubling below 0.01 absolute or below 5% relative.",
                "Require classification changes below 1-2% after doubling N_LHS.",
            ],
            "K_ref": [
                "Test deterministic L-BFGS-B/COBYLA reference first.",
                "Only retain random-start K_ref candidates {50,100,200} if deterministic reference tests fail or are insufficient.",
            ],
        },
        "runtime_projection": projection,
        "stage_plan": stage_plan,
        "stage_b_completed": bool(stage_b_completed),
        "protocol_e_candidate_started": protocol_e_started,
        "e_budget_decision": e_budget_decision,
        "missing_before_freeze": missing_before_freeze,
        "do_not_run_full_candidate_grid_without_gate": True,
        "recommended_next_executable_step": _recommended_next_step(
            stage_b_completed,
            protocol_e_started,
            endpoint_budget_valid,
            e_budget_decision,
            k_random_start_decision,
            k_ref_reference_decision,
            n_lhs_sanity_analysis,
            n_lhs_operational_decision,
        ),
    }


def _methodology_markdown(methodology: dict[str, Any]) -> str:
    projection = methodology["runtime_projection"]
    scaled = projection["scaled_methodology_validation_estimated_hours"]
    lines = [
        "# Phase E Calibration Methodology",
        "",
        f"Created UTC: {methodology['created_at_utc']}",
        f"Status: `{methodology['status']}`",
        "",
        "This artifact records the calibration order before any protocol-scale compute is launched. It does not authorize official Phase E execution.",
        "",
        "## Required Order",
        "",
    ]
    lines.extend(f"{index}. {item}" for index, item in enumerate(methodology["ordered_methodology"], start=1))
    lines.extend(
        [
            "",
            "## Runtime Basis",
            "",
            f"- Mean noisy objective seconds: `{projection['timing_inputs']['mean_noisy_objective_seconds']}`",
            f"- Mean ZNE objective seconds: `{projection['timing_inputs']['mean_zne_objective_seconds']}`",
            f"- Noisy+ZNE objective-pair seconds: `{projection['timing_inputs']['objective_pair_seconds_noisy_plus_zne']}`",
            "",
            "## Stage Plan",
            "",
            "| Stage | Status | Gate | Est./actual serial hours |",
            "|---|---|---|---:|",
        ]
    )
    for row in methodology["stage_plan"]:
        hours = "" if row["estimated_or_actual_hours"] is None else f"{row['estimated_or_actual_hours']:.3f}"
        lines.append(f"| {row['stage_id']} {row['stage_name']} | `{row['status']}` | `{row['gate_before_running']}` | {hours} |")
    lines.extend(
        [
            "",
            "## Key Projection",
            "",
            f"- Stage B scaled validation estimate: optimizer `{scaled['optimizer_grid']:.3f}` h + LHS `{scaled['lhs_grid']:.3f}` h.",
            f"- Protocol E_COBYLA calibration no-reuse serial estimate: `{projection['protocol_E_COBYLA_candidate_total_serial_hours_no_reuse']:.3f}` h.",
            f"- Protocol N_LHS calibration serial estimate: `{projection['protocol_N_LHS_candidate_total_serial_hours']:.3f}` h.",
            f"- Sequential N_LHS first-screen estimate: `{projection['sequential_N_LHS_first_screen']['estimated_hours']:.3f}` h for N_LHS={projection['sequential_N_LHS_first_screen']['N_LHS']} and `{projection['sequential_N_LHS_first_screen']['independent_designs']}` designs.",
            "",
            "## Freeze Blockers",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in methodology["missing_before_freeze"])
    lines.extend(
        [
            "",
            f"Recommended next executable step: `{methodology['recommended_next_executable_step']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Phase E Calibration Status",
        "",
        f"Created UTC: {report['created_at_utc']}",
        f"Status: `{report['status']}`",
        "",
        "This status file summarizes calibration progress from the current technical pilot. It is not a freeze decision.",
        "",
        "## Protocol Candidate Sets",
        "",
        f"- E_COBYLA candidates: {report['protocol_candidate_sets']['E_COBYLA']}",
        f"- K candidates: {report['protocol_candidate_sets']['K']}",
        f"- K_ref candidates: {report['protocol_candidate_sets']['K_ref']}",
        f"- N_LHS candidates: {report['protocol_candidate_sets']['N_LHS']}",
        "",
        "## Current Pilot Coverage",
        "",
        f"- Observed E_COBYLA candidates: {report['observed_candidate_sets']['E_COBYLA']}",
        f"- Valid endpoint E_COBYLA candidates: {report.get('valid_endpoint_candidate_sets', {}).get('E_COBYLA', [])}",
        f"- Observed K candidates: {report['observed_candidate_sets']['K']}",
        f"- Observed N_LHS candidates: {report['observed_candidate_sets']['N_LHS']}",
        f"- K_ref calibrated: `{report['k_ref_calibrated']}`",
        "",
        "## Calibration Readout",
        "",
        f"- Mean abs d_R by E_COBYLA: {report['pilot_metrics']['mean_abs_d_R_by_E_COBYLA']}",
        f"- E_COBYLA metrics valid for budget selection: `{report.get('pilot_metrics_validity', {}).get('mean_abs_d_R_by_E_COBYLA_valid_for_budget_selection', True)}`",
        f"- Mean abs d_R by K: {report['pilot_metrics']['mean_abs_d_R_by_K']}",
        f"- Mean r_MSE by N_LHS: {report['pilot_metrics']['mean_r_MSE_by_N_LHS']}",
        "",
        "## E_COBYLA Budget Decision",
        "",
        f"- Decision recorded: `{report.get('e_budget_decision_recorded', False)}`",
        f"- Selected E_COBYLA for next calibration stages: `{report.get('selected_cobyla_evaluation_budget')}`",
        f"- Decision status: `{(report.get('e_budget_decision') or {}).get('status')}`",
        f"- Evidence level: `{(report.get('e_budget_decision') or {}).get('evidence_level')}`",
        "",
        "## K Random-Start Decision",
        "",
        f"- Decision recorded: `{report.get('k_random_start_decision_recorded', False)}`",
        f"- Selected K for next calibration stages: `{report.get('selected_random_start_count')}`",
        f"- Random-start limitation label: `{report.get('random_start_limitation_label')}`",
        f"- Decision status: `{(report.get('k_random_start_decision') or {}).get('status')}`",
        f"- Evidence level: `{(report.get('k_random_start_decision') or {}).get('evidence_level')}`",
        "",
        "## K_ref Reference Decision",
        "",
        f"- Decision recorded: `{report.get('k_ref_reference_decision_recorded', False)}`",
        f"- Selected reference strategy: `{report.get('selected_reference_strategy')}`",
        f"- Random-start K_ref required: `{report.get('random_start_K_ref_required')}`",
        f"- Selected K_ref: `{report.get('selected_reference_starts')}`",
        f"- Decision status: `{(report.get('k_ref_reference_decision') or {}).get('status')}`",
        "",
        "## N_LHS Sanity Screen",
        "",
        f"- Screen recorded: `{report.get('n_lhs_sanity_screen_recorded', False)}`",
        f"- Screen status: `{(report.get('n_lhs_sanity_screen') or {}).get('status')}`",
        f"- Observed N_LHS candidates: `{(report.get('n_lhs_sanity_screen') or {}).get('observed_N_LHS_candidates')}`",
        f"- Observed design ids: `{(report.get('n_lhs_sanity_screen') or {}).get('observed_design_ids')}`",
        f"- Target adjacent pair: `{(report.get('n_lhs_sanity_screen') or {}).get('target_adjacent_pair')}`",
        f"- Target mean abs delta r_MSE: `{(report.get('n_lhs_sanity_screen') or {}).get('target_mean_abs_delta_r_MSE_cell_mean')}`",
        f"- Target relative SEM: `{(report.get('n_lhs_sanity_screen') or {}).get('target_cell_relative_sem')}`",
        "",
        "## N_LHS Operational Decision",
        "",
        f"- Decision recorded: `{report.get('n_lhs_operational_decision_recorded', False)}`",
        f"- Selected N_LHS: `{report.get('selected_N_LHS')}`",
        f"- Decision status: `{(report.get('n_lhs_operational_decision') or {}).get('status')}`",
        f"- Quantitative limitation: `{(report.get('n_lhs_operational_decision') or {}).get('quantitative_magnitude_limitation')}`",
        "",
        "## Methodology Gate",
        "",
        f"- Methodology recorded: `{report['methodology_recorded']}`",
        f"- Next stage: `{report['next_stage']['stage_id']} {report['next_stage']['stage_name']}`",
        f"- Next stage status: `{report['next_stage']['status']}`",
        f"- Do not run full candidate grid without gate: `{report['do_not_run_full_candidate_grid_without_gate']}`",
        "",
        "## Freeze Status",
        "",
        f"- Final calibration complete: `{report['final_calibration_complete']}`",
        f"- Recommendation: `{report['recommendation']}`",
        "",
    ]
    return "\n".join(lines)


def write_phase_e_calibration_status(root: Path = PROJECT_ROOT, overwrite: bool = True) -> dict[str, Any]:
    calibration_dir = ensure_dir(root / "results" / "phase_e" / "calibration")
    cobyla_path = calibration_dir / "cobyla_budget_calibration.csv"
    lhs_path = calibration_dir / "lhs_size_calibration.csv"
    summary_path = calibration_dir / "phase_e_unified_pilot_summary.json"
    e_budget_decision_path = calibration_dir / E_BUDGET_DECISION_FILENAME
    k_random_start_decision_path = calibration_dir / K_RANDOM_START_DECISION_FILENAME
    k_ref_reference_decision_path = calibration_dir / K_REF_REFERENCE_DECISION_FILENAME
    n_lhs_operational_decision_path = calibration_dir / N_LHS_OPERATIONAL_DECISION_FILENAME
    n_lhs_sanity_analysis_path, n_lhs_sanity_analysis = _latest_n_lhs_sanity_analysis(root)
    output_json = calibration_dir / "phase_e_calibration_status.json"
    output_md = calibration_dir / "phase_e_calibration_status.md"
    output_hashes = calibration_dir / "phase_e_calibration_status_hashes.json"
    methodology_json = calibration_dir / "phase_e_calibration_methodology.json"
    methodology_md = calibration_dir / "phase_e_calibration_methodology.md"
    stage_plan_csv = calibration_dir / "phase_e_calibration_stage_plan.csv"
    all_outputs = (output_json, output_md, output_hashes, methodology_json, methodology_md, stage_plan_csv)
    if any(path.exists() for path in all_outputs) and not overwrite:
        raise FileExistsError("Refusing to overwrite existing Phase E calibration status outputs.")
    if not (cobyla_path.exists() and lhs_path.exists() and summary_path.exists()):
        raise FileNotFoundError("Unified pilot outputs are required before writing Phase E calibration status.")

    cobyla_rows = _read_csv(cobyla_path)
    lhs_rows = _read_csv(lhs_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    optimizer_config = PhaseEOptimizerConfig()
    endpoint_budget_valid = _endpoint_budget_comparison_valid(summary)
    observed_e_raw = sorted({_as_int(row["E_COBYLA"]) for row in cobyla_rows})
    observed_e_for_endpoint = observed_e_raw if endpoint_budget_valid else []
    e_budget_decision = _valid_e_budget_decision(_read_json_optional(e_budget_decision_path), observed_e_for_endpoint)
    observed_k = sorted({_as_int(row["K"]) for row in cobyla_rows})
    k_random_start_decision = _valid_k_random_start_decision(
        _read_json_optional(k_random_start_decision_path),
        observed_k,
        e_budget_decision,
    )
    k_ref_reference_decision = _valid_k_ref_reference_decision(
        _read_json_optional(k_ref_reference_decision_path),
        k_random_start_decision,
    )
    n_lhs_operational_decision = _valid_n_lhs_operational_decision(
        _read_json_optional(n_lhs_operational_decision_path),
        n_lhs_sanity_analysis,
    )
    observed_lhs = sorted({_as_int(row["N_LHS"]) for row in lhs_rows})
    mean_abs_d_by_e = _mean_by(cobyla_rows, "E_COBYLA", "d_R", absolute=True)
    mean_abs_d_by_k = _mean_by(cobyla_rows, "K", "d_R", absolute=True)
    mean_r_by_lhs = _mean_by(lhs_rows, "N_LHS", "r_MSE_Richardson", absolute=False)
    projection = _build_runtime_projection(summary, optimizer_config)
    stage_b_is_complete = _stage_b_completed(summary)
    protocol_e_started = _protocol_e_candidate_started(summary)
    stage_b_actual_hours = float(summary.get("elapsed_seconds", 0.0)) / 3600.0 if stage_b_is_complete else None
    stage_plan = _build_stage_plan(
        projection,
        stage_b_completed=stage_b_is_complete,
        protocol_e_candidate_started=protocol_e_started,
        stage_b_actual_hours=stage_b_actual_hours,
        endpoint_budget_comparison_valid=endpoint_budget_valid,
        e_budget_decision=e_budget_decision,
        k_random_start_decision=k_random_start_decision,
        k_ref_reference_decision=k_ref_reference_decision,
        n_lhs_sanity_analysis=n_lhs_sanity_analysis,
        n_lhs_operational_decision=n_lhs_operational_decision,
    )
    next_stage = _next_stage(stage_plan)
    missing_e_candidates = (
        []
        if e_budget_decision is not None
        else [value for value in optimizer_config.cobyla_evaluation_budget_candidates if value not in observed_e_for_endpoint]
    )
    missing_k_ref_candidates = [] if k_ref_reference_decision is not None else list(optimizer_config.reference_start_candidates)
    missing_protocol = {
        "E_COBYLA": missing_e_candidates,
        "K": [value for value in optimizer_config.random_start_candidates if value not in observed_k],
        "K_ref": missing_k_ref_candidates,
        "N_LHS": []
        if n_lhs_operational_decision is not None
        else [value for value in optimizer_config.lhs_size_candidates if value not in observed_lhs],
    }
    report = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_calibration_started_not_final",
        "calibration_version": PHASE_E_CALIBRATION_VERSION,
        "official_execution_authorized": False,
        "protocol_candidate_sets": {
            "E_COBYLA": list(optimizer_config.cobyla_evaluation_budget_candidates),
            "K": list(optimizer_config.random_start_candidates),
            "K_ref": list(optimizer_config.reference_start_candidates),
            "N_LHS": list(optimizer_config.lhs_size_candidates),
        },
        "observed_candidate_sets": {
            "E_COBYLA": observed_e_raw,
            "K": observed_k,
            "N_LHS": observed_lhs,
            "seeds": summary.get("seeds", []),
            "lhs_independent_designs": int(summary.get("lhs_independent_designs", 1)),
        },
        "valid_endpoint_candidate_sets": {
            "E_COBYLA": observed_e_for_endpoint,
            "reason": "raw E_COBYLA observations are excluded when endpoint_budget_comparison_valid is false",
        },
        "endpoint_budget_comparison_valid": endpoint_budget_valid,
        "pilot_metrics": {
            "mean_abs_d_R_by_E_COBYLA": mean_abs_d_by_e,
            "mean_abs_d_R_by_K": mean_abs_d_by_k,
            "mean_r_MSE_by_N_LHS": mean_r_by_lhs,
            "last_step_delta_abs_d_R_E_COBYLA": _delta_last(mean_abs_d_by_e),
            "last_step_delta_abs_d_R_K": _delta_last(mean_abs_d_by_k),
            "last_step_delta_r_MSE_N_LHS": _delta_last(mean_r_by_lhs),
            "mean_noisy_objective_seconds": summary["calibration_decision"].get("mean_noisy_objective_seconds"),
            "mean_zne_objective_seconds": summary["calibration_decision"].get("mean_zne_objective_seconds"),
        },
        "pilot_metrics_validity": {
            "mean_abs_d_R_by_E_COBYLA_valid_for_budget_selection": endpoint_budget_valid,
            "raw_E_COBYLA_metrics_retained_for_audit_only": not endpoint_budget_valid,
        },
        "missing_protocol_candidate_coverage": missing_protocol,
        "k_ref_calibrated": k_ref_reference_decision is not None,
        "methodology_recorded": True,
        "stage_b_completed": bool(stage_b_is_complete),
        "protocol_e_candidate_started": bool(protocol_e_started),
        "e_budget_decision_recorded": e_budget_decision is not None,
        "selected_cobyla_evaluation_budget": (
            int(e_budget_decision["selected_E_COBYLA"]) if e_budget_decision is not None else None
        ),
        "e_budget_decision": (
            {
                "status": e_budget_decision["status"],
                "selected_E_COBYLA": int(e_budget_decision["selected_E_COBYLA"]),
                "evidence_level": e_budget_decision["evidence_level"],
                "stage_c_E_budget_closed": bool(e_budget_decision["stage_c_E_budget_closed"]),
                "source": f"results/phase_e/calibration/{E_BUDGET_DECISION_FILENAME}",
                "input_hash": sha256_file(e_budget_decision_path),
            }
            if e_budget_decision is not None
            else None
        ),
        "k_random_start_decision_recorded": k_random_start_decision is not None,
        "selected_random_start_count": (
            int(k_random_start_decision["selected_K"]) if k_random_start_decision is not None else None
        ),
        "random_start_limitation_label": (
            k_random_start_decision["random_start_limitation_label"] if k_random_start_decision is not None else None
        ),
        "k_random_start_decision": (
            {
                "status": k_random_start_decision["status"],
                "selected_K": int(k_random_start_decision["selected_K"]),
                "selected_E_COBYLA": int(k_random_start_decision["selected_E_COBYLA"]),
                "evidence_level": k_random_start_decision["evidence_level"],
                "random_start_limitation_label": k_random_start_decision["random_start_limitation_label"],
                "stage_d_K_calibration_closed": bool(k_random_start_decision["stage_d_K_calibration_closed"]),
                "source": f"results/phase_e/calibration/{K_RANDOM_START_DECISION_FILENAME}",
                "input_hash": sha256_file(k_random_start_decision_path),
            }
            if k_random_start_decision is not None
            else None
        ),
        "k_ref_reference_decision_recorded": k_ref_reference_decision is not None,
        "selected_reference_strategy": (
            k_ref_reference_decision["selected_reference_strategy"] if k_ref_reference_decision is not None else None
        ),
        "random_start_K_ref_required": (
            bool(k_ref_reference_decision["random_start_K_ref_required"]) if k_ref_reference_decision is not None else None
        ),
        "selected_reference_starts": (
            k_ref_reference_decision["selected_K_ref"] if k_ref_reference_decision is not None else None
        ),
        "k_ref_reference_decision": (
            {
                "status": k_ref_reference_decision["status"],
                "selected_reference_strategy": k_ref_reference_decision["selected_reference_strategy"],
                "random_start_K_ref_required": bool(k_ref_reference_decision["random_start_K_ref_required"]),
                "selected_K_ref": k_ref_reference_decision["selected_K_ref"],
                "stage_e_K_ref_reference_strategy_closed": bool(k_ref_reference_decision["stage_e_K_ref_reference_strategy_closed"]),
                "source": f"results/phase_e/calibration/{K_REF_REFERENCE_DECISION_FILENAME}",
                "input_hash": sha256_file(k_ref_reference_decision_path),
            }
            if k_ref_reference_decision is not None
            else None
        ),
        "n_lhs_operational_decision_recorded": n_lhs_operational_decision is not None,
        "selected_N_LHS": (
            int(n_lhs_operational_decision["selected_N_LHS"]) if n_lhs_operational_decision is not None else None
        ),
        "n_lhs_operational_decision": (
            {
                "status": n_lhs_operational_decision["status"],
                "selected_N_LHS": int(n_lhs_operational_decision["selected_N_LHS"]),
                "decision_type": n_lhs_operational_decision["decision_type"],
                "evidence_level": n_lhs_operational_decision["evidence_level"],
                "quantitative_magnitude_limitation": n_lhs_operational_decision["quantitative_magnitude_limitation"],
                "source": f"results/phase_e/calibration/{N_LHS_OPERATIONAL_DECISION_FILENAME}",
                "input_hash": sha256_file(n_lhs_operational_decision_path),
            }
            if n_lhs_operational_decision is not None
            else None
        ),
        "n_lhs_sanity_screen_recorded": n_lhs_sanity_analysis is not None,
        "n_lhs_sanity_screen": (
            {
                "status": n_lhs_sanity_analysis["status"],
                "source": _relative_path(root, n_lhs_sanity_analysis_path),
                "input_hash": sha256_file(n_lhs_sanity_analysis_path),
                "observed_N_LHS_candidates": n_lhs_sanity_analysis["observed_N_LHS_candidates"],
                "observed_design_ids": n_lhs_sanity_analysis["observed_design_ids"],
                "lhs_rows": int(n_lhs_sanity_analysis["lhs_rows"]),
                "target_adjacent_pair": n_lhs_sanity_analysis["screen_metrics"]["target_adjacent_pair"],
                "target_mean_abs_delta_r_MSE_cell_mean": n_lhs_sanity_analysis["screen_metrics"]["target_mean_abs_delta_r_MSE_cell_mean"],
                "target_r_MSE_sign_change_fraction_cell_mean": n_lhs_sanity_analysis["screen_metrics"]["target_r_MSE_sign_change_fraction_cell_mean"],
                "target_sign_only_discordance_classification_change_fraction": n_lhs_sanity_analysis["screen_metrics"]["target_sign_only_discordance_classification_change_fraction"],
                "target_cell_relative_sem": n_lhs_sanity_analysis["screen_metrics"]["target_cell_relative_sem"],
                "consequence": n_lhs_sanity_analysis["consequence"],
            }
            if n_lhs_sanity_analysis is not None
            else None
        ),
        "next_stage": next_stage,
        "do_not_run_full_candidate_grid_without_gate": True,
        "runtime_projection_summary": {
            "scaled_methodology_validation_estimated_hours": projection["scaled_methodology_validation_estimated_hours"],
            "protocol_E_COBYLA_candidate_total_serial_hours_no_reuse": projection["protocol_E_COBYLA_candidate_total_serial_hours_no_reuse"],
            "protocol_N_LHS_candidate_total_serial_hours": projection["protocol_N_LHS_candidate_total_serial_hours"],
            "sequential_N_LHS_first_screen_estimated_hours": projection["sequential_N_LHS_first_screen"]["estimated_hours"],
        },
        "final_calibration_complete": False,
        "recommendation": (
            "continue_remaining_phase_e_freeze_gates_with_N_LHS_2048_operational_limitation"
            if k_ref_reference_decision is not None and n_lhs_operational_decision is not None
            else
            "run_sequential_N_LHS_first_screen_256_512_designs0_4_after_user_approval"
            if k_ref_reference_decision is not None and n_lhs_sanity_analysis is None
            else
            f"add_N_LHS_{_next_n_lhs_candidate([int(value) for value in n_lhs_sanity_analysis.get('observed_N_LHS_candidates', [])])}_same_5_designs_then_combine"
            if k_ref_reference_decision is not None and n_lhs_sanity_analysis is not None and n_lhs_sanity_analysis["status"] == "lhs_5design_screen_not_stable_extend_candidates_or_designs"
            else
            "extend_stable_N_LHS_active_pair_to_20_designs_before_decision"
            if k_ref_reference_decision is not None and n_lhs_sanity_analysis is not None
            else
            "proceed_to_K_ref_reference_strategy_after_K100_random_start_limited_decision"
            if k_random_start_decision is not None
            else
            f"calibrate_K_random_starts_with_selected_E_COBYLA_{int(e_budget_decision['selected_E_COBYLA'])}"
            if e_budget_decision is not None
            else
            "rerun_patched_E_COBYLA_300_500_before_extending_protocol_E_coverage"
            if protocol_e_started and not endpoint_budget_valid
            else
            "continue_protocol_candidate_E_COBYLA_coverage_with_patched_same_objective_seeds"
            if protocol_e_started
            else "runtime_gate_or_scope_amendment_before_protocol_scale_calibration"
            if stage_b_is_complete
            else "continue_scaled_pilot_or_amend_scope_before_runtime_gate"
        ),
        "inputs": {
            "phase_e_unified_pilot_summary": "results/phase_e/calibration/phase_e_unified_pilot_summary.json",
            "cobyla_budget_calibration": "results/phase_e/calibration/cobyla_budget_calibration.csv",
            "lhs_size_calibration": "results/phase_e/calibration/lhs_size_calibration.csv",
            **(
                {"phase_e_k_random_start_decision": f"results/phase_e/calibration/{K_RANDOM_START_DECISION_FILENAME}"}
                if k_random_start_decision is not None
                else {}
            ),
            **(
                {"phase_e_k_ref_reference_decision": f"results/phase_e/calibration/{K_REF_REFERENCE_DECISION_FILENAME}"}
                if k_ref_reference_decision is not None
                else {}
            ),
            **(
                {"phase_e_n_lhs_operational_decision": f"results/phase_e/calibration/{N_LHS_OPERATIONAL_DECISION_FILENAME}"}
                if n_lhs_operational_decision is not None
                else {}
            ),
            **(
                {"phase_e_lhs_sanity_analysis": _relative_path(root, n_lhs_sanity_analysis_path)}
                if n_lhs_sanity_analysis is not None
                else {}
            ),
        },
        "outputs": {
            "phase_e_calibration_methodology": "results/phase_e/calibration/phase_e_calibration_methodology.json",
            "phase_e_calibration_methodology_md": "results/phase_e/calibration/phase_e_calibration_methodology.md",
            "phase_e_calibration_stage_plan": "results/phase_e/calibration/phase_e_calibration_stage_plan.csv",
        },
        "input_hashes": {
            "phase_e_unified_pilot_summary": sha256_file(summary_path),
            "cobyla_budget_calibration": sha256_file(cobyla_path),
            "lhs_size_calibration": sha256_file(lhs_path),
            **(
                {"phase_e_cobyla_e_budget_decision": sha256_file(e_budget_decision_path)}
                if e_budget_decision is not None
                else {}
            ),
            **(
                {"phase_e_k_random_start_decision": sha256_file(k_random_start_decision_path)}
                if k_random_start_decision is not None
                else {}
            ),
            **(
                {"phase_e_k_ref_reference_decision": sha256_file(k_ref_reference_decision_path)}
                if k_ref_reference_decision is not None
                else {}
            ),
            **(
                {"phase_e_n_lhs_operational_decision": sha256_file(n_lhs_operational_decision_path)}
                if n_lhs_operational_decision is not None
                else {}
            ),
            **(
                {"phase_e_lhs_sanity_analysis": sha256_file(n_lhs_sanity_analysis_path)}
                if n_lhs_sanity_analysis is not None
                else {}
            ),
        },
        "limitations": [
            "Current observed candidates are scaled-down technical pilot values, not the full protocol candidate grid.",
            *(
                [
                    f"K={int(k_random_start_decision['selected_K'])} is selected as a conservative finite-budget random-start count with a random-start-limited label.",
                    "Endpoint claims must be interpreted as optimizer-induced finite-budget decisions, not global QAOA optima.",
                ]
                if k_random_start_decision is not None
                else []
            ),
            *(
                [
                    f"K_ref random-start calibration is not required because deterministic strategy `{k_ref_reference_decision['selected_reference_strategy']}` passed sanity.",
                    "Reference endpoints are best-found deterministic noiseless optimizer endpoints, not guaranteed global optima.",
                ]
                if k_ref_reference_decision is not None
                else ["K_ref has not been calibrated."]
            ),
            *(
                [
                    f"N_LHS={int(n_lhs_operational_decision['selected_N_LHS'])} is selected as an operational validation budget with an explicit r_MSE magnitude-convergence limitation.",
                    "N_LHS=4096 was not run because the accepted claim is sign/classification robustness rather than strict quantitative r_MSE convergence.",
                ]
                if n_lhs_operational_decision is not None
                else [
                    "N_LHS protocol candidates include 4096; this has not been run because runtime/storage gates are not complete.",
                ]
            ),
            *(
                [
                    "The first 5-design N_LHS screen was not stable; the next action is adding the next pre-specified candidate with the same 5 designs.",
                ]
                if n_lhs_operational_decision is None
                and n_lhs_sanity_analysis is not None
                and n_lhs_sanity_analysis["status"] == "lhs_5design_screen_not_stable_extend_candidates_or_designs"
                else [
                    "The next N_LHS action is only a 5-design sequential first screen, not a final N_LHS freeze.",
                ]
                if n_lhs_operational_decision is None
                and n_lhs_sanity_analysis is None
                else [
                    "The first 5-design N_LHS screen cannot freeze N_LHS; the active stable pair must be extended to 20 designs.",
                ]
                if n_lhs_operational_decision is None
                else []
            ),
            "No official execution is authorized by this status file.",
        ],
    }
    methodology = _build_methodology(report, projection, stage_plan, stage_b_is_complete)
    write_json_atomic(output_json, report, overwrite=overwrite)
    output_md.write_text(_markdown(report), encoding="utf-8")
    write_json_atomic(methodology_json, methodology, overwrite=overwrite)
    methodology_md.write_text(_methodology_markdown(methodology), encoding="utf-8")
    _write_csv(stage_plan_csv, stage_plan)
    hashes = {
        "created_at_utc": _utc_now(),
        "calibration_version": PHASE_E_CALIBRATION_VERSION,
        "hashes": {
            "phase_e_calibration_status_json": sha256_file(output_json),
            "phase_e_calibration_status_md": sha256_file(output_md),
            "phase_e_calibration_methodology_json": sha256_file(methodology_json),
            "phase_e_calibration_methodology_md": sha256_file(methodology_md),
            "phase_e_calibration_stage_plan_csv": sha256_file(stage_plan_csv),
        },
    }
    write_json_atomic(output_hashes, hashes, overwrite=overwrite)
    report["output_hashes"] = hashes["hashes"]
    return report
