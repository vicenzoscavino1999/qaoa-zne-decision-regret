from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, sha256_file, write_json_atomic
from .phase_c import _selected_seeds
from .phase_c_120_custom_kernel import CUSTOM_KERNEL_VERSION
from .phase_c_120_preflight import (
    PROTOCOL_VERSION,
    TARGET_GRID_SIZE,
    generate_execution_environment,
    generate_source_snapshot,
)


FINAL_FREEZE_VERSION = "phase_c_120_final_freeze_v1"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_c_120_final_freeze"
ANALYSIS_FILES = (
    "scripts/04_aggregate_metrics.py",
    "scripts/05_make_figures.py",
    "src/metrics.py",
    "src/bootstrap.py",
    "src/figures.py",
    "src/phase_c.py",
    "config/config_official_pilot_v1.6.yaml",
    "richardson_weights_v1.6.json",
)


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _official_data_state(root: Path) -> dict[str, Any]:
    raw_dir = root / "data" / "raw" / "phase_c_120_kernel_official"
    arrays = sorted(raw_dir.rglob("*.npy")) if raw_dir.exists() else []
    done = sorted(raw_dir.rglob("*.done.json")) if raw_dir.exists() else []
    return {
        "official_raw_dir_exists": raw_dir.exists(),
        "official_npy_count": len(arrays),
        "official_done_count": len(done),
        "official_data_available_at_freeze": bool(arrays or done),
    }


def _analysis_hashes(root: Path) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for rel in ANALYSIS_FILES:
        path = root / rel
        out[rel] = sha256_file(path) if path.is_file() else None
    return out


def _preregistration_snapshot(root: Path, cfg: dict[str, Any], analysis_hashes: dict[str, str | None]) -> dict[str, Any]:
    seed_count = int(cfg["phase_c"]["seed_count"])
    seeds = _selected_seeds(root, seed_count)
    return {
        "created_at_utc": _now_utc(),
        "protocol_version": PROTOCOL_VERSION,
        "target_grid_size": TARGET_GRID_SIZE,
        "grid_choice": {
            "official_grid": "120x120",
            "reason": "80x80 failed decision stability and 120x120 passed strict grid adequacy.",
            "reference_artifacts": [
                "results/grid_convergence/grid_convergence_report.json",
                "results/grid_decision_stability/decision_stability_report.json",
                "results/phase_c_120_reference_validation/reference_validation_report.json",
            ],
        },
        "tau_R": cfg["phase_c"]["tau_R"],
        "tau_R_source": cfg["phase_c"]["tau_R_source"],
        "instances": cfg["phase_c"]["instances"],
        "noise_values": cfg["noise"]["p_values"],
        "shot_budgets": cfg["shots"]["total_budget_primary"],
        "seed_count": seed_count,
        "seeds": seeds,
        "zne_methods": cfg["zne"]["methods"],
        "oob": cfg["oob"],
        "statistics": cfg["statistics"],
        "go_no_go": cfg["go_no_go"],
        "metrics": cfg["metrics"],
        "analysis_hashes": analysis_hashes,
        "official_120_results_visible_at_freeze": _official_data_state(root)["official_data_available_at_freeze"],
        "known_pre_freeze_inputs": {
            "grid_convergence_observed": True,
            "decision_stability_80_vs_120_observed": True,
            "kernel_v2_timed_probe_observed": True,
        },
        "primary_analysis_rule": "Run the frozen analysis scripts over validated official 120x120 data once; later changed scripts are sensitivity analyses or amendments.",
    }


def run_phase_c_120_final_freeze(overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    result_dir = ensure_dir(root / "results" / "phase_c_120_final_freeze")
    summary_path = result_dir / "final_freeze_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing final freeze: {summary_path}")

    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    environment = generate_execution_environment(root)
    write_json_atomic(result_dir / "environment_report.json", environment, overwrite=True)
    (result_dir / "requirements_phase_c.txt").write_text("\n".join(environment.get("pip_freeze", [])) + "\n", encoding="utf-8")
    source_snapshot = generate_source_snapshot(root, result_dir)
    analysis_hashes = _analysis_hashes(root)
    prereg = _preregistration_snapshot(root, cfg, analysis_hashes)
    prereg["preregistration_hash"] = _json_hash({k: v for k, v in prereg.items() if k != "preregistration_hash"})
    write_json_atomic(result_dir / "pre_registration_snapshot.json", prereg, overwrite=True)

    amendment = _load_json(root / "results" / "phase_c_120_amendments" / "phase_c_120_kernel_v2_pre_official.json")
    reference = _load_json(root / "results" / "phase_c_120_reference_validation" / "reference_validation_report.json")
    dry_run = _load_json(root / "results" / "phase_c_120_operational_dry_run" / "operational_dry_run_summary.json")
    soak_test = _load_json(root / "results" / "phase_c_120_operational_soak_test" / "operational_soak_test_summary.json")
    official_state = _official_data_state(root)
    analysis_hashes_present = all(value for value in analysis_hashes.values())
    status = "phase_c_120_final_freeze_passed" if (
        amendment.get("status") == "phase_c_120_kernel_v2_amendment_accepted"
        and reference.get("status") == "phase_c_120_reference_validation_passed"
        and dry_run.get("status") == "phase_c_120_operational_dry_run_passed"
        and soak_test.get("status") == "phase_c_120_operational_soak_test_passed"
        and analysis_hashes_present
        and not official_state["official_data_available_at_freeze"]
        and CUSTOM_KERNEL_VERSION == "phase_c_120_custom_kernel_v2"
    ) else "phase_c_120_final_freeze_failed"

    summary = {
        "created_at_utc": _now_utc(),
        "final_freeze_version": FINAL_FREEZE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "custom_kernel_version": CUSTOM_KERNEL_VERSION,
        "status": status,
        "environment_hash": environment["environment_hash"],
        "source_tree_hash": source_snapshot["source_tree_hash"],
        "source_snapshot_sha256": source_snapshot["source_snapshot_sha256"],
        "analysis_hashes": analysis_hashes,
        "analysis_hashes_present": analysis_hashes_present,
        "pre_registration_hash": prereg["preregistration_hash"],
        "official_data_state": official_state,
        "required_gates": {
            "kernel_v2_amendment": amendment.get("status"),
            "reference_validation": reference.get("status"),
            "operational_dry_run": dry_run.get("status"),
            "operational_soak_test": soak_test.get("status"),
        },
        "outputs": {
            "environment_report": "environment_report.json",
            "requirements_phase_c": "requirements_phase_c.txt",
            "source_tree_manifest": "source_tree_manifest.json",
            "source_snapshot": "source_snapshot_phase_c_120.zip",
            "pre_registration_snapshot": "pre_registration_snapshot.json",
        },
        "decision": "ready_for_user_phase_9_authorization" if status.endswith("passed") else "do_not_start_phase_9_until_freeze_gate_is_resolved",
    }
    write_json_atomic(summary_path, summary, overwrite=True)
    return summary
