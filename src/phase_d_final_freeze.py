from __future__ import annotations

import hashlib
import json
import time
import zipfile
from pathlib import Path
from typing import Any

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .phase_c_120_preflight import PROTOCOL_VERSION, generate_execution_environment
from .phase_d_kernel_runner import OFFICIAL_RUN_NAME, PHASE_D_BLOCK_IDS
from .phase_d_scope_snapshot import SOURCE_FILES, build_phase_d_scope_snapshot


PHASE_D_FINAL_FREEZE_VERSION = "phase_d_v1_0_final_freeze_v1"
RESULT_DIR = PROJECT_ROOT / "results" / "phase_d" / "final_freeze"
FROZEN_SCOPE_STATUS = "phase_d_v1_0_frozen_not_executed"


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def _hashes(root: Path, rels: tuple[str, ...] | list[str]) -> dict[str, str | None]:
    return {rel: sha256_file(root / rel) if (root / rel).is_file() else None for rel in rels}


def _official_raw_state(root: Path) -> dict[str, Any]:
    raw_dir = root / "data" / "raw" / "phase_d"
    arrays = sorted(raw_dir.rglob("*.npy")) if raw_dir.exists() else []
    done = sorted(raw_dir.rglob("*.done.json")) if raw_dir.exists() else []
    return {
        "official_raw_dir_exists": raw_dir.exists(),
        "official_npy_count": len(arrays),
        "official_done_count": len(done),
        "official_data_available_at_freeze": bool(arrays or done),
    }


def _phase_d_source_snapshot(root: Path, output_dir: Path) -> dict[str, Any]:
    files = [root / rel for rel in SOURCE_FILES if (root / rel).is_file()]
    file_hashes = {path.relative_to(root).as_posix(): sha256_file(path) for path in files}
    manifest = {
        "created_at_utc": _now_utc(),
        "protocol_version": PROTOCOL_VERSION,
        "method": "phase_d_source_files_sorted_path_sha256_canonical_json",
        "files": dict(sorted(file_hashes.items())),
    }
    manifest["source_tree_hash"] = _json_hash({"files": manifest["files"], "method": manifest["method"]})
    manifest_path = output_dir / "source_tree_manifest.json"
    write_json_atomic(manifest_path, manifest, overwrite=True)
    zip_path = output_dir / "source_snapshot_phase_d_v1_0.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
            zf.write(path, path.relative_to(root).as_posix())
    manifest["source_snapshot_path"] = zip_path.relative_to(root).as_posix()
    manifest["source_snapshot_sha256"] = sha256_file(zip_path)
    write_json_atomic(manifest_path, manifest, overwrite=True)
    return manifest


def _frozen_scope_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    frozen = dict(snapshot)
    frozen["freeze_status"] = "frozen_v1_0_pre_official_execution"
    frozen["snapshot_status"] = FROZEN_SCOPE_STATUS
    frozen["phase_d_plan_version"] = "v1.0-frozen"
    frozen["official_execution_authorized"] = False
    frozen["frozen_at_utc"] = _now_utc()
    frozen["pending_required_before_v1_0_freeze"] = []
    frozen["scope_snapshot_hash"] = _json_hash({k: v for k, v in frozen.items() if k != "scope_snapshot_hash"})
    return frozen


def _pre_registration_snapshot(root: Path, frozen_scope: dict[str, Any], source_snapshot: dict[str, Any]) -> dict[str, Any]:
    official_plan = _load_json(root / "results" / OFFICIAL_RUN_NAME / "plan.json")
    payload = {
        "created_at_utc": _now_utc(),
        "phase_d_final_freeze_version": PHASE_D_FINAL_FREEZE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "phase_d_plan_version": "v1.0-frozen",
        "scope_snapshot_hash": frozen_scope["scope_snapshot_hash"],
        "source_tree_hash": source_snapshot["source_tree_hash"],
        "generator_seed": frozen_scope.get("generator_seed"),
        "generator_config_hash": frozen_scope.get("generator_config_hash"),
        "candidate_queue_hash": frozen_scope.get("candidate_queue_hash"),
        "stratum_queue_hashes": frozen_scope.get("stratum_queue_hashes"),
        "selected_instances_H1_to_H12": frozen_scope.get("selected_instances_H1_to_H12"),
        "reserve_instances_by_stratum": frozen_scope.get("reserve_instances_by_stratum"),
        "grid_adequacy_results": frozen_scope.get("grid_adequacy_results"),
        "fdr_power_results": frozen_scope.get("fdr_power_results"),
        "runtime_gate_results": frozen_scope.get("runtime_gate_results"),
        "runner_dry_run_results": frozen_scope.get("runner_dry_run_results"),
        "post_execution_script_readiness": frozen_scope.get("post_execution_script_readiness"),
        "official_plan": {
            "path": f"results/{OFFICIAL_RUN_NAME}/plan.json",
            "tasks_total": official_plan.get("tasks_total"),
            "block_task_counts": official_plan.get("block_task_counts"),
            "official_execution_order": official_plan.get("official_execution_order"),
            "official_block_scoped_execution_required": official_plan.get("official_block_scoped_execution_required"),
            "planning_context": official_plan.get("planning_context"),
        },
        "official_execution_token_required": frozen_scope.get("official_execution_token_required"),
        "official_execution_authorized": False,
        "official_raw_data_visible_at_freeze": _official_raw_state(root)["official_data_available_at_freeze"],
        "primary_analysis_rule": "Phase D global 540-cell family; BH q=0.05 primary, q=0.10 continuity sensitivity, no combined Phase C + Phase D FDR.",
    }
    payload["pre_registration_hash"] = _json_hash({k: v for k, v in payload.items() if k != "pre_registration_hash"})
    return payload


def run_phase_d_final_freeze(overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    result_dir = ensure_dir(root / "results" / "phase_d" / "final_freeze")
    summary_path = result_dir / "final_freeze_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Phase D final freeze: {summary_path}")

    snapshot = build_phase_d_scope_snapshot(root)
    frozen_scope = _frozen_scope_snapshot(snapshot)
    environment = generate_execution_environment(root)
    source_snapshot = _phase_d_source_snapshot(root, result_dir)
    prereg = _pre_registration_snapshot(root, frozen_scope, source_snapshot)
    official_plan = _load_json(root / "results" / OFFICIAL_RUN_NAME / "plan.json")
    raw_state = _official_raw_state(root)

    required_gate_checks = {
        "scope_pending_empty": snapshot.get("pending_required_before_v1_0_freeze") == [],
        "scope_ready": snapshot.get("snapshot_status") == "draft_pre_freeze_complete_ready_for_v1_0_freeze",
        "generator_reproducibility_verified": snapshot.get("generator_reproducibility_verified") is True,
        "candidate_queue_hash_matches_regenerated": snapshot.get("candidate_queue_hash") == snapshot.get("regenerated_candidate_queue_hash"),
        "grid_adequacy_gate_complete": snapshot.get("grid_adequacy_results", {}).get("gate_complete") is True,
        "fdr_power_gate_complete": snapshot.get("fdr_power_results", {}).get("gate_complete") is True,
        "runtime_gate_complete": snapshot.get("runtime_gate_results", {}).get("gate_complete") is True,
        "runner_dry_run_complete": snapshot.get("runner_dry_run_results", {}).get("gate_complete") is True,
        "post_execution_script_readiness_complete": snapshot.get("post_execution_script_readiness", {}).get("gate_complete") is True,
        "official_plan_exists": bool(official_plan),
        "official_plan_not_execution": official_plan.get("mode") == "phase_d_official_120x120_plan_not_executed_by_default",
        "official_plan_task_count": int(official_plan.get("tasks_total", -1)) == 21612,
        "official_block_task_counts": official_plan.get("block_task_counts") == {block_id: 5403 for block_id in PHASE_D_BLOCK_IDS},
        "official_execution_order": official_plan.get("official_execution_order") == list(PHASE_D_BLOCK_IDS),
        "official_block_scoped_execution_required": official_plan.get("official_block_scoped_execution_required") is True,
        "official_plan_resolve_at_execution_hashes": all(
            official_plan.get("planning_context", {}).get(key) == "resolved_at_execution_from_frozen_scope_snapshot"
            for key in ("phase_d_scope_snapshot_hash", "source_tree_hash", "environment_hash")
        ),
        "no_official_raw_data_at_freeze": raw_state["official_data_available_at_freeze"] is False,
        "source_hashes_present": all(value for value in _hashes(root, SOURCE_FILES).values()),
    }
    status = "phase_d_v1_0_final_freeze_passed" if all(required_gate_checks.values()) else "phase_d_v1_0_final_freeze_failed"

    write_json_atomic(result_dir / "environment_report.json", environment, overwrite=True)
    (result_dir / "requirements_phase_d.txt").write_text("\n".join(environment.get("pip_freeze", [])) + "\n", encoding="utf-8")
    write_json_atomic(result_dir / "phase_d_v1_0_scope_snapshot.json", frozen_scope, overwrite=True)
    write_json_atomic(root / "results" / "phase_d" / "planning" / "phase_d_scope_snapshot.json", frozen_scope, overwrite=True)
    write_json_atomic(result_dir / "pre_registration_snapshot.json", prereg, overwrite=True)

    outputs = {
        "environment_report": "results/phase_d/final_freeze/environment_report.json",
        "requirements_phase_d": "results/phase_d/final_freeze/requirements_phase_d.txt",
        "source_tree_manifest": "results/phase_d/final_freeze/source_tree_manifest.json",
        "source_snapshot": source_snapshot["source_snapshot_path"],
        "pre_registration_snapshot": "results/phase_d/final_freeze/pre_registration_snapshot.json",
        "phase_d_v1_0_scope_snapshot": "results/phase_d/final_freeze/phase_d_v1_0_scope_snapshot.json",
        "planning_scope_snapshot": "results/phase_d/planning/phase_d_scope_snapshot.json",
    }
    summary = {
        "created_at_utc": _now_utc(),
        "phase_d_final_freeze_version": PHASE_D_FINAL_FREEZE_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "decision": "ready_for_user_D1_authorization" if status.endswith("passed") else "do_not_start_D1_until_final_freeze_gate_is_resolved",
        "freeze_status": frozen_scope["freeze_status"],
        "snapshot_status": frozen_scope["snapshot_status"],
        "scope_snapshot_hash": frozen_scope["scope_snapshot_hash"],
        "environment_hash": environment["environment_hash"],
        "source_tree_hash": source_snapshot["source_tree_hash"],
        "source_snapshot_sha256": source_snapshot["source_snapshot_sha256"],
        "pre_registration_hash": prereg["pre_registration_hash"],
        "official_raw_state": raw_state,
        "required_gate_checks": required_gate_checks,
        "outputs": outputs,
        "output_hashes": _hashes(root, list(outputs.values())),
    }
    write_json_atomic(summary_path, summary, overwrite=True)
    return summary
