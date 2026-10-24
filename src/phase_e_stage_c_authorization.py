from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic


PHASE_E_STAGE_C_AUTHORIZATION_VERSION = "phase_e_stage_c_authorization_v0_1_draft"
AUTHORIZATION_SUMMARY = "results/phase_e/calibration/stage_c_authorization/phase_e_stage_c_authorization_summary.json"
AUTHORIZATION_HASHES = "results/phase_e/calibration/stage_c_authorization/phase_e_stage_c_authorization_hashes.json"

ACCEPTED_RUNTIME_SCOPE_AMENDMENT_STATUSES = {
    "phase_e_runtime_scope_amendment_accepted_budget",
    "phase_e_runtime_scope_amendment_accepted_scope",
    "phase_e_runtime_scope_amendment_accepted_external_compute",
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_stage_c_authorization_context(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    return {
        "config_preflight": _read_json_optional(root / "results" / "phase_e" / "planning" / "phase_e_config_preflight.json"),
        "runtime": _read_json_optional(root / "results" / "phase_e" / "calibration" / "c0_runtime" / "phase_e_stage_c0_runtime_summary.json"),
        "gate_composition": _read_json_optional(root / "results" / "phase_e" / "calibration" / "c0_runtime" / "gate_composition_decision.json"),
        "storage": _read_json_optional(root / "results" / "phase_e" / "calibration" / "c0_storage" / "phase_e_stage_c0_storage_summary.json"),
        "shard_balance": _read_json_optional(root / "results" / "phase_e" / "calibration" / "c0_shard_balance" / "phase_e_stage_c0_shard_balance_summary.json"),
        "jitter": _read_json_optional(root / "results" / "phase_e" / "calibration" / "c0_jitter" / "phase_e_stage_c0_jitter_summary.json"),
        "resume": _read_json_optional(root / "results" / "phase_e" / "calibration" / "c0_resume" / "phase_e_stage_c0_resume_summary.json"),
        "calibration": _read_json_optional(root / "results" / "phase_e" / "calibration" / "phase_e_calibration_status.json"),
        "runtime_scope_amendment": _read_json_optional(root / "results" / "phase_e" / "calibration" / "c0_runtime" / "phase_e_runtime_scope_amendment_summary.json"),
    }


def _value(context: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    block = context.get(section)
    if not isinstance(block, dict):
        return default
    return block.get(key, default)


def evaluate_stage_c_authorization(context: dict[str, Any]) -> dict[str, Any]:
    amendment_status = _value(context, "runtime_scope_amendment", "status")
    accepted_budget_runtime_pass = bool(
        amendment_status == "phase_e_runtime_scope_amendment_accepted_budget"
        and _value(context, "runtime_scope_amendment", "runtime_gate_pass") is True
    )
    criteria = {
        "accepted_runtime_or_scope_amendment": amendment_status in ACCEPTED_RUNTIME_SCOPE_AMENDMENT_STATUSES,
        "config_preflight_passed": _value(context, "config_preflight", "status") == "passed",
        "config_official_execution_enabled": _value(context, "config_preflight", "config", {}).get("official_execution_enabled") is True,
        "runtime_gate_pass": _value(context, "runtime", "runtime_gate_pass") is True or accepted_budget_runtime_pass,
        "full_runtime_gate_complete": _value(context, "runtime", "full_runtime_gate_complete") is True,
        "gate_composition_allows_stage_c": _value(context, "gate_composition", "stage_c_blocked") is False,
        "storage_gate_pass": _value(context, "storage", "storage_gate_pass") is True,
        "full_storage_gate_complete": _value(context, "storage", "full_storage_gate_complete") is True,
        "shard_balance_gate_pass": _value(context, "shard_balance", "shard_balance_projection_pass") is True,
        "full_shard_balance_gate_complete": _value(context, "shard_balance", "full_shard_balance_gate_complete") is True,
        "full_jitter_gate_complete": _value(context, "jitter", "full_jitter_gate_complete") is True,
        "full_resume_gate_complete": _value(context, "resume", "full_resume_gate_complete") is True,
        "final_calibration_complete": _value(context, "calibration", "final_calibration_complete") is True,
    }
    failed = [key for key, passed in criteria.items() if passed is not True]
    stage_c_authorized = not failed
    return {
        "authorization_version": PHASE_E_STAGE_C_AUTHORIZATION_VERSION,
        "authorization_decision": "authorized" if stage_c_authorized else "blocked",
        "official_execution_authorized": stage_c_authorized,
        "stage_c_authorized": stage_c_authorized,
        "stage_c_blocked": not stage_c_authorized,
        "first_blocking_criterion": None if stage_c_authorized else failed[0],
        "failed_criteria": failed,
        "criteria": criteria,
        "accepted_runtime_scope_amendment_statuses": sorted(ACCEPTED_RUNTIME_SCOPE_AMENDMENT_STATUSES),
    }


def write_stage_c_authorization_check(root: Path = PROJECT_ROOT, overwrite: bool = True) -> dict[str, Any]:
    context = load_stage_c_authorization_context(root)
    decision = evaluate_stage_c_authorization(context)
    amendment_status = _value(context, "runtime_scope_amendment", "status")
    accepted_budget_runtime_pass = bool(
        amendment_status == "phase_e_runtime_scope_amendment_accepted_budget"
        and _value(context, "runtime_scope_amendment", "runtime_gate_pass") is True
    )
    output_dir = ensure_dir(root / "results" / "phase_e" / "calibration" / "stage_c_authorization")
    summary_path = root / AUTHORIZATION_SUMMARY
    hashes_path = root / AUTHORIZATION_HASHES
    summary = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_stage_c_authorization_check_complete",
        "scope": "authorization_guard_no_protocol_scale_execution",
        **decision,
        "input_statuses": {
            "runtime_scope_amendment": _value(context, "runtime_scope_amendment", "status"),
            "runtime_gate_pass_original_72h": _value(context, "runtime", "runtime_gate_pass"),
            "runtime_gate_pass_accepted_budget": accepted_budget_runtime_pass,
            "runtime_gate_pass_effective": decision["criteria"]["runtime_gate_pass"],
            "full_runtime_gate_complete": _value(context, "runtime", "full_runtime_gate_complete"),
            "gate_stage_c_blocked": _value(context, "gate_composition", "stage_c_blocked"),
            "calibration_final_complete": _value(context, "calibration", "final_calibration_complete"),
        },
        "outputs": {
            "summary": AUTHORIZATION_SUMMARY,
            "hashes": AUTHORIZATION_HASHES,
        },
    }
    write_json_atomic(summary_path, summary, overwrite=overwrite)
    hashes = {
        "created_at_utc": _utc_now(),
        "authorization_version": PHASE_E_STAGE_C_AUTHORIZATION_VERSION,
        "hashes": {
            AUTHORIZATION_SUMMARY: sha256_file(summary_path),
        },
    }
    write_json_atomic(hashes_path, hashes, overwrite=overwrite)
    summary["output_hashes"] = hashes["hashes"]
    return summary
