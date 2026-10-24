from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, sha256_file, write_json_atomic


SCOPE_SNAPSHOT_VERSION = "phase_d_scope_snapshot_v3"
SNAPSHOT_STATUS = "draft_pre_freeze_incomplete"
PLANNING_DIR = PROJECT_ROOT / "results" / "phase_d" / "planning"
PLAN_PATH = PROJECT_ROOT / "PHASE_D_120_EXTERNAL_VALIDITY_EXTENSION_PLAN_v0.2-draft.md"
AMENDMENT_PATH = PROJECT_ROOT / "PHASE_C_D_TWO_STAGE_DESIGN_AMENDMENT.md"
CFG_PATH = PROJECT_ROOT / "config" / "config_official_pilot_v1.6.yaml"

PHASE_C_PRIMARY_OUTPUTS = (
    "results/phase_c_120_analysis/bootstrap_by_cell.csv",
    "results/phase_c_120_analysis/classification_matrix.csv",
    "results/phase_c_120_analysis/paper_go_no_go_summary.json",
    "results/phase_c_120_analysis/statistics_summary.json",
)
PHASE_C_ROBUSTNESS_OUTPUTS = (
    "results/phase_c_120_robustness/robustness_summary.json",
    "results/phase_c_120_holdout_replication/holdout_replication_summary.json",
)
ANALYSIS_FILES = (
    "src/phase_d_generator.py",
    "src/phase_d_scope_snapshot.py",
    "src/phase_d_grid_adequacy.py",
    "src/phase_d_fdr_power_gate.py",
    "src/phase_d_runtime_gate.py",
    "src/phase_d_completion.py",
    "src/phase_d_analysis.py",
    "src/phase_d_robustness.py",
    "src/phase_d_convergence.py",
    "src/phase_d_final_freeze.py",
    "scripts/28_phase_d_generator.py",
    "scripts/29_phase_d_scope_snapshot.py",
    "scripts/30_phase_d_grid_adequacy.py",
    "scripts/31_phase_d_fdr_power_gate.py",
    "scripts/32_phase_d_runtime_gate.py",
    "scripts/35_phase_d_completion.py",
    "scripts/36_phase_d_analysis.py",
    "scripts/37_phase_d_robustness.py",
    "scripts/38_phase_d_convergence.py",
    "scripts/39_phase_d_final_freeze.py",
    "src/phase_c_120_analysis.py",
    "src/metrics.py",
    "src/bootstrap.py",
    "config/phase_d_instances_v1.0.json",
    "config/config_official_pilot_v1.6.yaml",
    "richardson_weights_v1.6.json",
)
RUNNER_FILES = (
    "src/phase_d_kernel_runner.py",
    "src/phase_d_runner_dry_run.py",
    "scripts/33_phase_d_kernel_runner.py",
    "scripts/34_phase_d_runner_dry_run.py",
    "src/phase_c_120_kernel_runner.py",
    "scripts/17_phase_c_120_kernel_runner.py",
)
POST_EXECUTION_PREFLIGHT_OUTPUTS = (
    "results/phase_d/planning/phase_d_completion_preflight.json",
    "results/phase_d/planning/phase_d_analysis_preflight.json",
    "results/phase_d/planning/phase_d_robustness_preflight.json",
    "results/phase_d/planning/phase_d_convergence_preflight.json",
)
SOURCE_FILES = (
    *ANALYSIS_FILES,
    *RUNNER_FILES,
    "PHASE_D_120_EXTERNAL_VALIDITY_EXTENSION_PLAN_v0.2-draft.md",
    "PHASE_C_D_TWO_STAGE_DESIGN_AMENDMENT.md",
)
PACKAGE_NAMES = (
    "numpy",
    "scipy",
    "qiskit",
    "qiskit-aer",
    "mitiq",
    "PyYAML",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def _hashes(root: Path, rels: tuple[str, ...]) -> dict[str, str | None]:
    return {
        rel: sha256_file(root / rel) if (root / rel).is_file() else None
        for rel in rels
    }


def _csv_headers(path: Path) -> list[str]:
    if not path.is_file():
        return []
    import csv

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return next(reader, [])


def _candidate_queue_schema_state(root: Path) -> dict[str, Any]:
    required = [
        "candidate_id",
        "stratum_id",
        "attempt_index",
        "stratum_queue_rank",
        "instance_name",
        "n_nodes",
        "n_edges",
        "weighted",
        "weight_distribution",
        "density_class",
        "graph_density",
        "degree_mean",
        "degree_std",
        "max_degree",
        "edge_list_json",
        "topology_signature",
        "instance_signature",
        "master_seed",
        "stratum_seed",
        "candidate_status",
        "rejection_reason",
        "replacement_for",
        "grid_adequacy_status",
    ]
    headers = _csv_headers(root / "results" / "phase_d" / "planning" / "phase_d_candidate_queue.csv")
    missing = [field for field in required if field not in headers]
    return {
        "candidate_queue_schema_complete": not missing,
        "required_columns": required,
        "missing_columns": missing,
        "header_count": len(headers),
    }


def _git_value(root: Path, args: list[str]) -> str | None:
    try:
        proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, timeout=10, check=False)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in PACKAGE_NAMES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def _selected_by_stratum(selected: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for label, payload in sorted(selected.items()):
        out.setdefault(str(payload["stratum_id"]), []).append(label)
    return {key: sorted(value, key=lambda item: int(item[1:])) for key, value in sorted(out.items())}


def _stratum_queue_hashes(stratum_queues: dict[str, Any]) -> dict[str, str]:
    return {
        stratum_id: _json_hash(payload)
        for stratum_id, payload in sorted(stratum_queues.items())
    }


def _phase_c_output_state(root: Path) -> dict[str, Any]:
    primary_hashes = _hashes(root, PHASE_C_PRIMARY_OUTPUTS)
    robustness_hashes = _hashes(root, PHASE_C_ROBUSTNESS_OUTPUTS)
    robustness_status = "available" if all(robustness_hashes.values()) else "partial_or_missing"
    return {
        "phase_c_primary_output_hashes": primary_hashes,
        "phase_c_robustness_output_hashes": robustness_hashes,
        "phase_c_robustness_status_if_available": robustness_status,
    }


def _grid_adequacy_state(root: Path, planning_dir: Path) -> dict[str, Any]:
    rels = (
        "results/phase_d/planning/phase_d_grid_adequacy_summary.json",
        "results/phase_d/planning/phase_d_grid_adequacy_by_instance.csv",
        "results/phase_d/planning/phase_d_grid_rejection_audit.csv",
        "results/phase_d/planning/phase_d_selected_vs_rejected_topology_summary.csv",
        "results/phase_d/planning/phase_d_grid_failure_modes.json",
        "results/phase_d/planning/grid_adequacy_limitations.md",
        "results/phase_d/planning/phase_d_selected_instances_final.json",
        "config/phase_d_instances_v1.0.json",
    )
    summary = _load_json(planning_dir / "phase_d_grid_adequacy_summary.json")
    final_selected = _load_json(planning_dir / "phase_d_selected_instances_final.json")
    hashes = _hashes(root, rels)
    resolved_statuses = {"pass", "pass_with_mechanical_replacement"}
    gate_complete = bool(
        summary.get("mode") == "full"
        and summary.get("status") in resolved_statuses
        and summary.get("final_selection_complete") is True
        and len(final_selected) == 12
    )
    return {
        "grid_adequacy_results": {
            "status": summary.get("status", "pending_not_run"),
            "mode": summary.get("mode", "not_run"),
            "gate_complete": gate_complete,
            "selected_evaluated": summary.get("selected_evaluated"),
            "reserves_evaluated": summary.get("reserves_evaluated"),
            "grid_pass_count": summary.get("grid_pass_count"),
            "grid_borderline_acceptable_count": summary.get("grid_borderline_acceptable_count"),
            "grid_fail_count": summary.get("grid_fail_count"),
            "final_selection_complete": summary.get("final_selection_complete", False),
            "replacement_events": summary.get("replacement_events", []),
            "final_exclusions": summary.get("final_exclusions", []),
            "failure_modes": summary.get("failure_modes", {}),
            "limitations": summary.get("limitations", []),
            "output_hashes": hashes,
            "grid_adequacy_limitations_present": hashes["results/phase_d/planning/grid_adequacy_limitations.md"] is not None,
            "phase_d_instance_config_hash": hashes["config/phase_d_instances_v1.0.json"],
        },
        "selected_instances_final_after_grid_gate": final_selected,
        "grid_adequacy_gate_complete": gate_complete,
    }


def _fdr_power_state(root: Path, planning_dir: Path) -> dict[str, Any]:
    rels = (
        "results/phase_d/planning/phase_d_fdr_power_simulation.json",
        "results/phase_d/planning/phase_d_fdr_power_simulation.csv",
        "results/phase_d/planning/phase_d_fdr_power_gate_decision.json",
        "results/phase_d/planning/phase_d_fdr_power_interpretation.md",
        "results/phase_d/planning/phase_d_fdr_power_output_hashes.json",
    )
    summary = _load_json(planning_dir / "phase_d_fdr_power_simulation.json")
    decision = _load_json(planning_dir / "phase_d_fdr_power_gate_decision.json")
    hashes = _hashes(root, rels)
    gate_complete = bool(summary.get("status") == "complete" and decision.get("power_gate_amend_required") is False)
    return {
        "fdr_power_results": {
            "status": summary.get("status", "pending_not_run"),
            "gate_complete": gate_complete,
            "simulations_per_scenario": summary.get("simulations_per_scenario"),
            "rng_seed": summary.get("rng_seed"),
            "primary_q": summary.get("primary_q"),
            "continuity_q": summary.get("continuity_q"),
            "power_gate_confirmatory_pass": decision.get("power_gate_confirmatory_pass", False),
            "power_gate_amend_required": decision.get("power_gate_amend_required", True),
            "power_gate_conservative_warning": decision.get("power_gate_conservative_warning", False),
            "co_primary_endpoint_power_warning": decision.get("co_primary_endpoint_power_warning", False),
            "recommendation": decision.get("recommendation", "pending_not_run"),
            "central_metrics": decision.get("central_metrics", {}),
            "conservative_metrics": decision.get("conservative_metrics", {}),
            "output_hashes": hashes,
        },
        "fdr_power_gate_complete": gate_complete,
        "fdr_power_summary_hash": hashes["results/phase_d/planning/phase_d_fdr_power_simulation.json"],
    }


def _runtime_gate_state(root: Path, planning_dir: Path) -> dict[str, Any]:
    rels = (
        "results/phase_d/planning/phase_d_runtime_gate_summary.json",
        "results/phase_d/planning/phase_d_runtime_gate_decision.json",
        "results/phase_d/planning/phase_d_runtime_probe_measurements.csv",
        "results/phase_d/planning/phase_d_runtime_gate_interpretation.md",
        "results/phase_d/planning/phase_d_runtime_gate_output_hashes.json",
    )
    summary = _load_json(planning_dir / "phase_d_runtime_gate_summary.json")
    decision = _load_json(planning_dir / "phase_d_runtime_gate_decision.json")
    hashes = _hashes(root, rels)
    gate_complete = bool(summary.get("status") == "complete" and decision.get("runtime_gate_pass") is True)
    runtime_estimate = summary.get("runtime_estimate", {})
    storage_estimate = summary.get("storage_estimate", {})
    memory_estimate = summary.get("memory_estimate", {})
    return {
        "runtime_gate_results": {
            "status": summary.get("status", "pending_not_run"),
            "gate_complete": gate_complete,
            "recommendation": decision.get("recommendation", "pending_not_run"),
            "runtime_gate_pass": decision.get("runtime_gate_pass", False),
            "runtime_pass": decision.get("runtime_pass", False),
            "runtime_requires_explicit_authorization": decision.get("runtime_requires_explicit_authorization", True),
            "runtime_stop_required": decision.get("runtime_stop_required", True),
            "primary_disk_pass": decision.get("primary_disk_pass", False),
            "backup_disk_pass": decision.get("backup_disk_pass", False),
            "memory_pass": decision.get("memory_pass", False),
            "thermal_pass": decision.get("thermal_pass", False),
            "probe_probabilities_valid": decision.get("probe_probabilities_valid", False),
            "probe_instance_label": summary.get("phase_d_probe", {}).get("probe_instance_label"),
            "probe_candidate_id": summary.get("phase_d_probe", {}).get("probe_candidate_id"),
            "probe_rows": summary.get("phase_d_probe", {}).get("measurement_summary", {}).get("rows"),
            "expected_hours_per_block": runtime_estimate.get("expected_hours_per_block"),
            "expected_total_hours_H1_H12": runtime_estimate.get("expected_total_hours_H1_H12"),
            "target_local_hours": runtime_estimate.get("target_local_hours"),
            "planning_buffer_hours": runtime_estimate.get("planning_buffer_hours"),
            "estimated_primary_storage_bytes": storage_estimate.get("estimated_primary_storage_bytes"),
            "estimated_backup_storage_bytes": storage_estimate.get("estimated_backup_storage_bytes"),
            "estimated_peak_memory_fraction": memory_estimate.get("estimated_peak_memory_fraction"),
            "output_hashes": hashes,
        },
        "runtime_gate_complete": gate_complete,
        "runtime_gate_summary_hash": hashes["results/phase_d/planning/phase_d_runtime_gate_summary.json"],
    }


def _runner_dry_run_state(root: Path) -> dict[str, Any]:
    rels = (
        "results/phase_d_120_operational_dry_run/phase_d_runner_dry_run_summary.json",
        "results/phase_d_120_operational_dry_run/phase_d_runner_dry_run_output_hashes.json",
        "results/phase_d_120_operational_dry_run/phase_d_runner_dry_run_completion_report.json",
        "results/phase_d_120_operational_dry_run/phase_d_runner_dry_run_completion_report.md",
        "results/phase_d_120_operational_dry_run/dry_run_manifest.json",
        "results/phase_d_120_operational_dry_run/runner_summary.json",
        "results/phase_d_120_operational_dry_run/plan.json",
        "results/phase_d_kernel_runner_selftest/runner_summary.json",
        "results/phase_d_kernel_runner_selftest/plan.json",
        "results/phase_d_120_kernel_official/plan.json",
        "results/phase_d_120_kernel_official/runtime_estimate.json",
    )
    summary = _load_json(root / "results" / "phase_d_120_operational_dry_run" / "phase_d_runner_dry_run_summary.json")
    runner_summary = _load_json(root / "results" / "phase_d_120_operational_dry_run" / "runner_summary.json")
    output_hashes = _load_json(root / "results" / "phase_d_120_operational_dry_run" / "phase_d_runner_dry_run_output_hashes.json")
    hashes = _hashes(root, rels)
    checks = summary.get("checks", {})
    gate_complete = bool(
        summary.get("status") == "phase_d_runner_dry_run_passed"
        and summary.get("decision") == "proceed_to_final_scope_snapshot"
        and runner_summary.get("status") == "complete"
        and all(checks.get(key) is True for key in (
            "official_plan_written_without_execution",
            "self_test_tiny_grid_passed",
            "controlled_interrupt_created_pending_tasks",
            "resume_completed_all_tasks",
            "all_subprocesses_returned_zero",
            "duplicate_detection_passed",
            "duplicate_detection_negative_control_passed",
            "missing_task_detection_passed",
            "corrupted_hash_detection_passed",
            "block_filter_execution_check_passed",
            "restore_read_verified",
            "post_merge_completion_passed",
            "progress_files_match_shard_count",
            "backup_sha256_verified",
            "memory_gate_passed",
        ))
    )
    return {
        "runner_dry_run_results": {
            "status": summary.get("status", "pending_not_run"),
            "gate_complete": gate_complete,
            "decision": summary.get("decision", "pending_not_run"),
            "run_name": summary.get("run_name"),
            "shard_count": summary.get("shard_count"),
            "official_plan_written_not_executed": summary.get("official_plan_written_not_executed", {}),
            "partial_summary": {
                "status": summary.get("partial_summary", {}).get("status"),
                "tasks_completed": summary.get("partial_summary", {}).get("tasks_completed"),
                "tasks_pending": summary.get("partial_summary", {}).get("tasks_pending"),
            },
            "final_summary": {
                "status": runner_summary.get("status"),
                "tasks_total": runner_summary.get("tasks_total"),
                "tasks_completed": runner_summary.get("tasks_completed"),
                "tasks_pending": runner_summary.get("tasks_pending"),
                "corrupt_tasks": runner_summary.get("corrupt_tasks"),
            },
            "checks": checks,
            "self_test": summary.get("self_test", {}),
            "backup": summary.get("backup", {}),
            "restore_read": summary.get("restore_read", {}),
            "runner_negative_controls": summary.get("runner_negative_controls", {}),
            "block_filter_execution_check": summary.get("block_filter_execution_check", {}),
            "post_merge_completion": summary.get("post_merge_completion", {}),
            "memory": summary.get("memory", {}),
            "thermal_status": summary.get("thermal", {}).get("status"),
            "output_hash_manifest": output_hashes.get("hashes", {}),
            "output_hashes": hashes,
        },
        "runner_dry_run_gate_complete": gate_complete,
        "runner_dry_run_summary_hash": hashes["results/phase_d_120_operational_dry_run/phase_d_runner_dry_run_summary.json"],
    }


def _post_execution_script_readiness_state(root: Path) -> dict[str, Any]:
    hashes = _hashes(root, POST_EXECUTION_PREFLIGHT_OUTPUTS)
    completion = _load_json(root / "results" / "phase_d" / "planning" / "phase_d_completion_preflight.json")
    analysis = _load_json(root / "results" / "phase_d" / "planning" / "phase_d_analysis_preflight.json")
    robustness = _load_json(root / "results" / "phase_d" / "planning" / "phase_d_robustness_preflight.json")
    convergence = _load_json(root / "results" / "phase_d" / "planning" / "phase_d_convergence_preflight.json")
    checks = {
        "completion_preflight_passed": completion.get("status") == "phase_d_completion_preflight_passed",
        "analysis_preflight_passed": analysis.get("status") == "phase_d_analysis_preflight_passed",
        "robustness_preflight_passed": robustness.get("status") == "phase_d_robustness_preflight_passed",
        "convergence_preflight_passed": convergence.get("status") == "phase_d_convergence_preflight_passed",
    }
    return {
        "post_execution_script_readiness": {
            "gate_complete": all(checks.values()),
            "checks": checks,
            "completion_preflight": completion,
            "analysis_preflight": analysis,
            "robustness_preflight": robustness,
            "convergence_preflight": convergence,
            "output_hashes": hashes,
        },
        "post_execution_script_readiness_complete": all(checks.values()),
    }


def _environment_report(root: Path) -> dict[str, Any]:
    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "dependency_versions": _dependency_versions(),
    }
    payload["environment_hash"] = _json_hash(payload)
    return payload


def build_phase_d_scope_snapshot(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    cfg = read_yaml(root / CFG_PATH.relative_to(PROJECT_ROOT))
    output_dir = root / "results" / "phase_d" / "planning"
    generator_config = _load_json(output_dir / "phase_d_generator_config.json")
    generator_summary = _load_json(output_dir / "phase_d_generator_summary.json")
    reproducibility = _load_json(output_dir / "phase_d_generator_reproducibility.json")
    selected_instances = _load_json(output_dir / "phase_d_selected_instances.json")
    stratum_queues = _load_json(output_dir / "phase_d_stratum_queues.json")
    weight_manifest = _load_json(output_dir / "phase_d_weight_generation_manifest.json")
    grid_state = _grid_adequacy_state(root, output_dir)
    fdr_power_state = _fdr_power_state(root, output_dir)
    runtime_gate_state = _runtime_gate_state(root, output_dir)
    runner_dry_run_state = _runner_dry_run_state(root)
    post_execution_state = _post_execution_script_readiness_state(root)
    candidate_queue_schema_state = _candidate_queue_schema_state(root)
    final_selected = grid_state["selected_instances_final_after_grid_gate"]
    selected_for_execution = final_selected if grid_state["grid_adequacy_gate_complete"] else selected_instances
    env = _environment_report(root)
    source_hashes = _hashes(root, SOURCE_FILES)
    source_tree_hash = _json_hash(source_hashes)
    analysis_hashes = _hashes(root, ANALYSIS_FILES)
    runner_hashes = _hashes(root, RUNNER_FILES)
    phase_c_state = _phase_c_output_state(root)
    zne_methods = cfg["zne"]["methods"]
    pending_required = (
        ([] if candidate_queue_schema_state["candidate_queue_schema_complete"] else [
            "candidate_queue_schema_8_8_columns",
        ])
        + ([] if grid_state["grid_adequacy_gate_complete"] else [
            "grid_adequacy_ideal_only_gate",
            "grid_rejection_audit",
        ])
        + ([] if grid_state["grid_adequacy_results"]["grid_adequacy_limitations_present"] else [
            "grid_adequacy_limitations_md",
        ])
        + ([] if grid_state["grid_adequacy_results"]["phase_d_instance_config_hash"] else [
            "phase_d_instances_source_config",
        ])
        + ([] if fdr_power_state["fdr_power_gate_complete"] else [
            "fdr_power_gate",
        ])
        + ([] if runtime_gate_state["runtime_gate_complete"] else [
            "runtime_gate",
        ])
        + ([] if runner_dry_run_state["runner_dry_run_gate_complete"] else [
            "phase_d_runner_dry_run",
            "final_scope_snapshot_after_pending_gates",
        ])
        + ([] if post_execution_state["post_execution_script_readiness_complete"] else [
            "phase_d_completion_preflight",
            "phase_d_analysis_preflight",
            "phase_d_robustness_preflight",
            "phase_d_convergence_preflight",
        ])
    )
    snapshot_status = (
        "draft_pre_freeze_complete_ready_for_v1_0_freeze"
        if not pending_required
        else SNAPSHOT_STATUS
    )
    snapshot = {
        "created_at_utc": _utc_now(),
        "scope_snapshot_version": SCOPE_SNAPSHOT_VERSION,
        "phase_d_plan_version": "v1.0-frozen_candidate",
        "snapshot_status": snapshot_status,
        "freeze_status": "not_frozen",
        "official_execution_authorized": False,
        "official_execution_token_required": "accept_phase_d_120_independent_replication_official",
        "phase_d_plan_sha256": sha256_file(root / PLAN_PATH.relative_to(PROJECT_ROOT)),
        "phase_c_d_amendment_sha256": sha256_file(root / AMENDMENT_PATH.relative_to(PROJECT_ROOT)),
        "phase_c_reference_commit_hash": _git_value(root, ["rev-parse", "HEAD"]),
        "git_status_short": _git_value(root, ["status", "--short"]),
        **phase_c_state,
        "generator_seed": generator_config.get("master_seed"),
        "generator_config_hash": generator_summary.get("config_hash"),
        "generator_config_file_hash": generator_summary.get("generator_config_file_hash"),
        "candidate_queue_hash": generator_summary.get("candidate_queue_hash"),
        "candidate_queue_schema": candidate_queue_schema_state,
        "stratum_config_hash": _json_hash(generator_config.get("strata", [])),
        "stratum_queue_hashes": _stratum_queue_hashes(stratum_queues),
        "selected_instances_initial_H1_to_H12": selected_instances,
        "selected_instances_H1_to_H12": selected_for_execution,
        "selected_instances_by_stratum": _selected_by_stratum(selected_for_execution),
        "reserve_instances_by_stratum": generator_summary.get("reserves", {}),
        "structural_constraints": generator_config.get("envelope", {}),
        "weight_generation_manifest_hash": generator_summary.get("weight_manifest_hash"),
        "weight_generation_manifest": weight_manifest,
        "grid_adequacy_results": grid_state["grid_adequacy_results"],
        "fdr_power_simulation_hash": fdr_power_state["fdr_power_summary_hash"],
        "fdr_power_results": fdr_power_state["fdr_power_results"],
        "fdr_power_gate_status": fdr_power_state["fdr_power_results"]["recommendation"],
        "runtime_gate_summary_hash": runtime_gate_state["runtime_gate_summary_hash"],
        "runtime_gate_results": runtime_gate_state["runtime_gate_results"],
        "runtime_gate_status": runtime_gate_state["runtime_gate_results"]["recommendation"],
        "runner_dry_run_summary_hash": runner_dry_run_state["runner_dry_run_summary_hash"],
        "runner_dry_run_results": runner_dry_run_state["runner_dry_run_results"],
        "runner_dry_run_status": runner_dry_run_state["runner_dry_run_results"]["decision"],
        "post_execution_script_readiness": post_execution_state["post_execution_script_readiness"],
        "production_grid": "120x120",
        "N_seeds": int(cfg["phase_c"]["seed_count"]),
        "noise_values": cfg["noise"]["p_values"],
        "shot_budgets": cfg["shots"]["total_budget_primary"],
        "ZNE_orders": sorted(zne_methods),
        "scale_factors": {key: value for key, value in sorted(zne_methods.items())},
        "tau_R": cfg["phase_c"]["tau_R"],
        "tau_R_sensitivity": {
            "0.5x": float(cfg["phase_c"]["tau_R"]) * 0.5,
            "1.0x": float(cfg["phase_c"]["tau_R"]),
            "1.5x": float(cfg["phase_c"]["tau_R"]) * 1.5,
            "phase_d_null_diagnostic": "pending_post_execution_from_existing_seed_replicas",
        },
        "bootstrap_method": cfg["statistics"]["bootstrap"],
        "FDR_method": {
            "primary": {"method": "benjamini_hochberg", "q": 0.05, "family": "phase_d_540_cells"},
            "continuity_sensitivity": {"method": "benjamini_hochberg", "q": 0.10, "family": "phase_d_540_cells"},
            "conservative_sensitivity": ["Holm", "Benjamini-Yekutieli"],
        },
        "classification_rules": {
            "source": "PHASE_D_120_INDEPENDENT_REPLICATION_PLAN_v1.0",
            "cell_level_primary_endpoint": "BH q=0.05 over 540 cells preserves criterion-2-style support",
            "instance_level_primary_endpoint": ">=6/12 instance_supportive_q05 and >=8/12 instance_positive_direction across >=3/4 blocks",
        },
        "analysis_script_hashes": analysis_hashes,
        "runner_script_hashes": runner_hashes,
        "source_tree_hash": source_tree_hash,
        "source_hashes": source_hashes,
        "environment_hash": env["environment_hash"],
        "dependency_versions": env["dependency_versions"],
        "generator_reproducibility_verified": reproducibility.get("status") == "pass",
        "generator_reproducibility_hash_pairs": reproducibility.get("hash_pairs", {}),
        "regenerated_candidate_queue_hash": reproducibility.get("hash_pairs", {}).get("candidate_queue_file", {}).get("regenerated_hash"),
        "pending_required_before_v1_0_freeze": pending_required,
    }
    snapshot["scope_snapshot_hash"] = _json_hash({k: v for k, v in snapshot.items() if k != "scope_snapshot_hash"})
    return snapshot


def write_phase_d_scope_snapshot(root: Path = PROJECT_ROOT, force: bool = False) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_d" / "planning")
    path = output_dir / "phase_d_scope_snapshot.json"
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing scope snapshot: {path}")
    snapshot = build_phase_d_scope_snapshot(root)
    write_json_atomic(path, snapshot, overwrite=True)
    return snapshot
