from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic


PHASE_E_RUNTIME_SCOPE_AMENDMENT_VERSION = "phase_e_runtime_scope_amendment_v0_1_draft"
AMENDMENT_DOC = "PHASE_E_RUNTIME_SCOPE_AMENDMENT_v0.1-draft.md"
PHASE_E_RUNTIME_BUDGET_AMENDMENT_VERSION = "phase_e_runtime_budget_amendment_v0_1_accepted"
BUDGET_AMENDMENT_DOC = "PHASE_E_RUNTIME_BUDGET_AMENDMENT_v0.1-accepted.md"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_context(root: Path) -> dict[str, Any]:
    runtime = _read_json(root / "results" / "phase_e" / "calibration" / "c0_runtime" / "phase_e_stage_c0_runtime_summary.json")
    gate = _read_json(root / "results" / "phase_e" / "calibration" / "c0_runtime" / "gate_composition_decision.json")
    storage = _read_json(root / "results" / "phase_e" / "calibration" / "c0_storage" / "phase_e_stage_c0_storage_summary.json")
    shard = _read_json(root / "results" / "phase_e" / "calibration" / "c0_shard_balance" / "phase_e_stage_c0_shard_balance_summary.json")
    jitter = _read_json(root / "results" / "phase_e" / "calibration" / "c0_jitter" / "phase_e_stage_c0_jitter_summary.json")
    resume = _read_json(root / "results" / "phase_e" / "calibration" / "c0_resume" / "phase_e_stage_c0_resume_summary.json")
    calibration = _read_json(root / "results" / "phase_e" / "calibration" / "phase_e_calibration_status.json")
    return {
        "runtime": runtime,
        "gate": gate,
        "storage": storage,
        "shard": shard,
        "jitter": jitter,
        "resume": resume,
        "calibration": calibration,
    }


def _markdown(context: dict[str, Any]) -> str:
    runtime = context["runtime"]
    gate = context["gate"]
    storage = context["storage"]
    shard = context["shard"]
    jitter = context["jitter"]
    resume = context["resume"]
    calibration = context["calibration"]
    protocol_candidates = calibration["protocol_candidate_sets"]
    lines = [
        "# Phase E Runtime/Scope Amendment",
        "",
        "Document version: `v0.1-draft`",
        "Status: `proposed_not_accepted_not_execution_authorization`",
        f"Created UTC: `{_utc_now()}`",
        "",
        "## Purpose",
        "",
        "This amendment records that Phase E Stage C is blocked under the current local runtime budget and cannot proceed from the existing C0 evidence.",
        "",
        "It does not authorize any protocol-scale optimizer run. It does not select final `T`, `K`, `K_ref`, `N_LHS`, `tau_R_opt`, `tau_conc`, or `tau_trivial`.",
        "",
        "## Trigger",
        "",
        "The C0 runtime projection failed the accepted local wall-clock budget.",
        "",
        "```text",
        f"accepted_local_wall_hours = {runtime['accepted_local_wall_hours']}",
        f"primary_projected_wall_hours = {runtime['primary_projected_wall_hours']}",
        f"primary_wall_hours_over_budget = {runtime['primary_wall_hours_over_budget']}",
        f"primary_budget_multiplier = {runtime['primary_budget_multiplier']}",
        f"conservative_no_K_projected_wall_hours = {runtime['conservative_no_K_projected_wall_hours']}",
        f"task_template_projected_wall_hours = {runtime['task_template_projected_wall_hours']}",
        f"measured_checkpoint_factor_mean = {runtime['measured_checkpoint_factor_mean']}",
        f"measured_checkpoint_factor_min = {runtime['measured_checkpoint_factor_min']}",
        f"measured_checkpoint_factor_max = {runtime['measured_checkpoint_factor_max']}",
        "```",
        "",
        "## Gate Status",
        "",
        "```text",
        f"runtime_gate_pass = {runtime['runtime_gate_pass']}",
        f"full_runtime_gate_complete = {runtime['full_runtime_gate_complete']}",
        f"storage_gate_pass = {storage['storage_gate_pass']}",
        f"full_storage_gate_complete = {storage['full_storage_gate_complete']}",
        f"shard_balance_projection_pass = {shard['shard_balance_projection_pass']}",
        f"full_shard_balance_gate_complete = {shard['full_shard_balance_gate_complete']}",
        f"full_jitter_gate_complete = {jitter['full_jitter_gate_complete']}",
        f"full_resume_gate_complete = {resume['full_resume_gate_complete']}",
        f"stage_c_blocked = {gate['stage_c_blocked']}",
        f"first_blocking_gate = {gate['first_blocking_gate']}",
        "```",
        "",
        "The storage projection passed on the project disk, and the shard-balance projection is balanced. These do not override the runtime failure or the incomplete full jitter/resume/shard gates.",
        "",
        "## Amendment Decision",
        "",
        "```text",
        "Stage C protocol-scale calibration is blocked.",
        "No official Phase E optimizer execution is authorized.",
        "No final optimizer or LHS hyperparameter is frozen.",
        "Any Stage C execution requires either a new accepted budget or a written scope/runtime amendment before re-execution.",
        "```",
        "",
        "## Permitted Next Paths",
        "",
        "Path A - Budget amendment:",
        "",
        "```text",
        "Accept a wall-clock budget above the current 72 h limit, with a concrete number and execution environment.",
        "Then re-run the ordered full C0 gates before Stage C.",
        "```",
        "",
        "Path B - Scope amendment:",
        "",
        "```text",
        "Reduce the calibration scope before inspecting protocol-scale Stage C outcomes.",
        "Any reduction must specify which claims are downgraded, which candidate grids are removed, and which outputs become exploratory.",
        "```",
        "",
        "Path C - External compute amendment:",
        "",
        "```text",
        "Move C0/Stage C to a different execution environment with measured effective shards and storage/backups.",
        "The same ordered gates still apply.",
        "```",
        "",
        "Path D - Stop Phase E execution:",
        "",
        "```text",
        "Keep Phase E as a draft methodological extension and do not execute Stage C.",
        "Report no depth-2 optimizer result beyond existing pilot/smoke feasibility diagnostics.",
        "```",
        "",
        "## Not Permitted Without New Amendment",
        "",
        "```text",
        "Do not run Stage C with the current C0 evidence.",
        "Do not relax the 72 h budget after seeing C0 results without recording a budget amendment.",
        "Do not treat smoke jitter/resume/shard outputs as full gates.",
        "Do not select T/K/N_LHS/K_ref from current smoke data.",
        "Do not add warm-start, early stopping, float32, asymmetric T, or stress-only calibration as silent optimizations.",
        "```",
        "",
        "## Current Calibration State",
        "",
        "```text",
        f"stage_B_completed = {calibration['stage_b_completed']}",
        f"final_calibration_complete = {calibration['final_calibration_complete']}",
        f"next_stage_status = {calibration['next_stage']['status']}",
        f"protocol_T_candidates = {protocol_candidates['T']}",
        f"protocol_K_candidates = {protocol_candidates['K']}",
        f"protocol_K_ref_candidates = {protocol_candidates['K_ref']}",
        f"protocol_N_LHS_candidates = {protocol_candidates['N_LHS']}",
        "```",
        "",
        "## Required Before Any Stage C Restart",
        "",
        "1. User accepts one of the permitted next paths in writing.",
        "2. Any changed budget/scope is recorded in a new amendment version.",
        "3. Ordered C0 gates are re-run or completed under the accepted amendment.",
        "4. Gate composition must pass strict AND before Stage C.",
        "5. Final selected hyperparameters are frozen only after valid calibration evidence exists.",
        "",
        "## Hash Inputs",
        "",
        "```text",
        f"runtime_summary = {runtime['outputs']['summary']}",
        f"gate_composition = {runtime['outputs']['gate_composition']}",
        f"storage_summary = {storage['outputs']['summary']}",
        f"shard_balance_summary = {shard['outputs']['summary']}",
        f"jitter_summary = {jitter['outputs']['summary']}",
        f"resume_summary = {resume['outputs']['summary']}",
        "```",
        "",
    ]
    return "\n".join(lines)


def generate_phase_e_runtime_scope_amendment(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    context = _load_context(root)
    doc_path = root / AMENDMENT_DOC
    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration" / "c0_runtime")
    summary_path = output_dir / "phase_e_runtime_scope_amendment_summary.json"
    hashes_path = output_dir / "phase_e_runtime_scope_amendment_hashes.json"
    if (doc_path.exists() or summary_path.exists() or hashes_path.exists()) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase E runtime/scope amendment outputs.")

    doc_path.write_text(_markdown(context), encoding="utf-8")
    runtime = context["runtime"]
    gate = context["gate"]
    summary = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_runtime_scope_amendment_proposed_not_accepted",
        "amendment_version": PHASE_E_RUNTIME_SCOPE_AMENDMENT_VERSION,
        "document": AMENDMENT_DOC,
        "official_execution_authorized": False,
        "stage_c_blocked": True,
        "final_calibration_complete": False,
        "accepted_local_wall_hours": runtime["accepted_local_wall_hours"],
        "primary_projected_wall_hours": runtime["primary_projected_wall_hours"],
        "primary_budget_multiplier": runtime["primary_budget_multiplier"],
        "runtime_gate_pass": runtime["runtime_gate_pass"],
        "full_runtime_gate_complete": runtime["full_runtime_gate_complete"],
        "gate_first_blocking_gate": gate["first_blocking_gate"],
        "permitted_next_paths": [
            "budget_amendment",
            "scope_amendment",
            "external_compute_amendment",
            "stop_phase_e_execution",
        ],
        "not_permitted_without_new_amendment": [
            "run_stage_c_with_current_c0_evidence",
            "relax_thresholds_or_budget_posthoc_without_recording",
            "treat_smoke_outputs_as_full_gates",
            "select_final_T_K_Kref_NLHS_from_smoke_data",
        ],
        "outputs": {
            "document": AMENDMENT_DOC,
            "summary": str(summary_path.relative_to(root)).replace("\\", "/"),
            "hashes": str(hashes_path.relative_to(root)).replace("\\", "/"),
        },
    }
    write_json_atomic(summary_path, summary, overwrite=True)
    hashes = {
        "created_at_utc": _utc_now(),
        "amendment_version": PHASE_E_RUNTIME_SCOPE_AMENDMENT_VERSION,
        "hashes": {
            AMENDMENT_DOC: sha256_file(doc_path),
            str(summary_path.relative_to(root)).replace("\\", "/"): sha256_file(summary_path),
        },
    }
    write_json_atomic(hashes_path, hashes, overwrite=True)
    summary["output_hashes"] = hashes["hashes"]
    return summary


def _budget_markdown(
    context: dict[str, Any],
    accepted_wall_hours: float,
    accepted_calendar_days: float,
    approval_text: str,
    approved_by: str,
) -> str:
    runtime = context["runtime"]
    calibration = context["calibration"]
    protocol_candidates = calibration["protocol_candidate_sets"]
    projected = float(runtime["primary_projected_wall_hours"])
    previous_budget = float(runtime["accepted_local_wall_hours"])
    pass_under_budget = projected <= float(accepted_wall_hours)
    lines = [
        "# Phase E Runtime Budget Amendment",
        "",
        "Document version: `v0.1-accepted`",
        "Status: `accepted_budget_not_execution_authorization`",
        f"Created UTC: `{_utc_now()}`",
        "",
        "## Approval Record",
        "",
        "```text",
        f"approved_by = {approved_by}",
        f"approval_text = {approval_text}",
        f"accepted_local_wall_hours = {accepted_wall_hours}",
        f"accepted_calendar_days = {accepted_calendar_days}",
        "amendment_type = budget_amendment",
        "official_execution_authorized = false",
        "stage_c_authorized = false",
        "```",
        "",
        "## Runtime Budget Change",
        "",
        "```text",
        f"previous_accepted_local_wall_hours = {previous_budget}",
        f"new_accepted_local_wall_hours = {accepted_wall_hours}",
        f"primary_projected_wall_hours = {projected}",
        f"budget_margin_wall_hours = {accepted_wall_hours - projected}",
        f"runtime_projection_pass_under_accepted_budget = {pass_under_budget}",
        f"primary_budget_multiplier_vs_previous_budget = {runtime['primary_budget_multiplier']}",
        "```",
        "",
        "This amendment accepts a larger local wall-clock budget only. It does not change the scientific protocol, candidate grids, seeds, shot budgets, thresholds, optimizer schedules, or claim scope.",
        "",
        "## Scientific Scope",
        "",
        "```text",
        "scientific_scope_changed = false",
        "candidate_grid_changed = false",
        "thresholds_changed = false",
        f"protocol_T_candidates = {protocol_candidates['T']}",
        f"protocol_K_candidates = {protocol_candidates['K']}",
        f"protocol_K_ref_candidates = {protocol_candidates['K_ref']}",
        f"protocol_N_LHS_candidates = {protocol_candidates['N_LHS']}",
        "final_calibration_complete = false",
        "```",
        "",
        "## Remaining Blocks",
        "",
        "```text",
        "Stage C remains blocked.",
        "Full duplicate-endpoint jitter gate remains required.",
        "Full resume/corruption integrity gate remains required.",
        "Full shard-balance/runtime gates remain required under the accepted budget.",
        "Stage C authorization guard remains required.",
        "Final freeze remains required before protocol-scale execution.",
        "```",
        "",
        "## Required Next Step",
        "",
        "Re-run or complete the ordered C0 gates under this accepted budget before any Stage C authorization check can pass.",
        "",
    ]
    return "\n".join(lines)


def accept_phase_e_budget_amendment(
    root: Path = PROJECT_ROOT,
    accepted_wall_hours: float = 240.0,
    accepted_calendar_days: float = 20.0,
    approval_text: str = "Acepto budget_amendment local de 240 h y hasta 20 dias calendario para Phase E.",
    approved_by: str = "user",
    force: bool = False,
) -> dict[str, Any]:
    context = _load_context(root)
    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration" / "c0_runtime")
    doc_path = root / BUDGET_AMENDMENT_DOC
    summary_path = output_dir / "phase_e_runtime_scope_amendment_summary.json"
    budget_summary_path = output_dir / "phase_e_runtime_budget_amendment_summary.json"
    hashes_path = output_dir / "phase_e_runtime_scope_amendment_hashes.json"
    if (doc_path.exists() or budget_summary_path.exists()) and not force:
        raise FileExistsError("Refusing to overwrite existing accepted Phase E budget amendment outputs.")

    accepted_wall_hours = float(accepted_wall_hours)
    accepted_calendar_days = float(accepted_calendar_days)
    runtime = context["runtime"]
    projected = float(runtime["primary_projected_wall_hours"])
    previous_budget = float(runtime["accepted_local_wall_hours"])
    runtime_projection_pass = bool(projected <= accepted_wall_hours)

    doc_path.write_text(
        _budget_markdown(context, accepted_wall_hours, accepted_calendar_days, approval_text, approved_by),
        encoding="utf-8",
    )
    summary = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_runtime_scope_amendment_accepted_budget",
        "amendment_version": PHASE_E_RUNTIME_BUDGET_AMENDMENT_VERSION,
        "document": BUDGET_AMENDMENT_DOC,
        "approved_by": approved_by,
        "approval_text": approval_text,
        "approval_timestamp_utc": _utc_now(),
        "amendment_type": "budget_amendment",
        "official_execution_authorized": False,
        "stage_c_authorized": False,
        "stage_c_blocked": True,
        "final_calibration_complete": False,
        "scientific_scope_changed": False,
        "candidate_grid_changed": False,
        "thresholds_changed": False,
        "previous_accepted_local_wall_hours": previous_budget,
        "accepted_local_wall_hours": accepted_wall_hours,
        "accepted_calendar_days": accepted_calendar_days,
        "primary_projected_wall_hours": projected,
        "budget_margin_wall_hours": float(accepted_wall_hours - projected),
        "runtime_projection_pass_under_accepted_budget": runtime_projection_pass,
        "runtime_gate_pass": runtime_projection_pass,
        "full_runtime_gate_complete": False,
        "full_runtime_gate_required_before_stage_c": True,
        "requires_reexecution_of_affected_gates": True,
        "affected_gates_or_thresholds": [
            "runtime_gate_budget_threshold",
            "stage_c_authorization_guard",
        ],
        "remaining_before_stage_c": [
            "full duplicate endpoint jitter gate",
            "full resume/corruption integrity gate",
            "full optimizer runner dry-run",
            "full shard-balance/runtime gates under accepted 240 h budget",
            "Stage C authorization guard",
            "final freeze summary",
        ],
        "permitted_next_paths": [
            "complete_full_c0_gates_under_accepted_budget",
            "external_compute_amendment_if_local_calendar_risk_becomes_unacceptable",
            "scope_amendment_if_full_gates_fail",
            "stop_phase_e_execution",
        ],
        "outputs": {
            "document": BUDGET_AMENDMENT_DOC,
            "summary": str(summary_path.relative_to(root)).replace("\\", "/"),
            "budget_summary": str(budget_summary_path.relative_to(root)).replace("\\", "/"),
            "hashes": str(hashes_path.relative_to(root)).replace("\\", "/"),
        },
    }
    write_json_atomic(summary_path, summary, overwrite=True)
    write_json_atomic(budget_summary_path, summary, overwrite=True)
    hashes = {
        "created_at_utc": _utc_now(),
        "amendment_version": PHASE_E_RUNTIME_BUDGET_AMENDMENT_VERSION,
        "hashes": {
            BUDGET_AMENDMENT_DOC: sha256_file(doc_path),
            str(summary_path.relative_to(root)).replace("\\", "/"): sha256_file(summary_path),
            str(budget_summary_path.relative_to(root)).replace("\\", "/"): sha256_file(budget_summary_path),
        },
    }
    write_json_atomic(hashes_path, hashes, overwrite=True)
    summary["output_hashes"] = hashes["hashes"]
    return summary
