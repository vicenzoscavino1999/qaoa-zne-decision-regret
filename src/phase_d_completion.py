from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .phase_c_120_preflight import PROTOCOL_VERSION
from .phase_c_120_custom_kernel import CUSTOM_KERNEL_VERSION
from .phase_d_kernel_runner import (
    OFFICIAL_RUN_NAME,
    PHASE_D_BLOCK_IDS,
    PHASE_D_KERNEL_RUNNER_VERSION,
    TARGET_GRID_SIZE,
    summarize_phase_d_kernel_runner,
    write_phase_d_official_kernel_plan,
)


PHASE_D_COMPLETION_VERSION = "phase_d_completion_v1"
BLOCK_REPORT_NAME = "completion_report.json"
BLOCK_REPORT_MD_NAME = "completion_report.md"
FINAL_MANIFEST_NAME = "manifest_final.json"
GLOBAL_RESULT_DIR = "global_D1_D2_D3_D4"
EXPECTED_BLOCK_TASKS = 5403
EXPECTED_GLOBAL_TASKS = 21612
FREEZE_PLACEHOLDER = "resolved_at_execution_from_frozen_scope_snapshot"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load_json(path: Path, default: Any | None = None) -> Any:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    if default is not None:
        return default
    raise FileNotFoundError(path)


def _official_result_dir(root: Path) -> Path:
    return root / "results" / OFFICIAL_RUN_NAME


def _phase_d_result_dir(root: Path, block_id: str | None = None) -> Path:
    if block_id is None:
        return ensure_dir(root / "results" / "phase_d" / GLOBAL_RESULT_DIR)
    return ensure_dir(root / "results" / "phase_d" / block_id)


def _load_official_plan(root: Path) -> dict[str, Any]:
    plan_path = _official_result_dir(root) / "plan.json"
    if not plan_path.is_file():
        write_phase_d_official_kernel_plan(root)
    return _load_json(plan_path)


def _tasks_for_block(plan: dict[str, Any], block_id: str) -> list[dict[str, Any]]:
    if block_id not in PHASE_D_BLOCK_IDS:
        raise ValueError(f"block_id must be one of {PHASE_D_BLOCK_IDS}, got {block_id!r}.")
    return [task for task in plan.get("tasks", []) if task.get("block_id") == block_id]


def _expected_context(root: Path, plan: dict[str, Any]) -> dict[str, Any]:
    context = dict(plan.get("planning_context", {}))
    if not any(value == FREEZE_PLACEHOLDER for value in context.values()):
        return context

    snapshot = _load_json(root / "results" / "phase_d" / "planning" / "phase_d_scope_snapshot.json")
    resolved = dict(context)
    placeholder_sources = {
        "phase_d_scope_snapshot_hash": "scope_snapshot_hash",
        "source_tree_hash": "source_tree_hash",
        "environment_hash": "environment_hash",
    }
    for context_key, snapshot_key in placeholder_sources.items():
        if resolved.get(context_key) == FREEZE_PLACEHOLDER:
            resolved[context_key] = snapshot.get(snapshot_key)
    resolved["context_resolution_policy"] = "official_plan_placeholders_resolved_from_frozen_scope_snapshot"
    return resolved


def _task_condition_key(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        task.get("kind"),
        task.get("block_id"),
        task.get("instance"),
        task.get("grid_size"),
        task.get("p_noise"),
        task.get("shots_total"),
        task.get("seed"),
        task.get("zne_order"),
        tuple(task.get("scale_factors", [])),
    )


def _check_task(root: Path, task: dict[str, Any], context: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    output_rel = str(task["output_path"])
    done_rel = str(task["done_path"])
    output_path = root / output_rel
    done_path = root / done_rel
    record: dict[str, Any] = {
        "task_id": task.get("task_id"),
        "task_index": task.get("index"),
        "condition_key": list(_task_condition_key(task)),
        "kind": task.get("kind"),
        "block_id": task.get("block_id"),
        "instance": task.get("instance"),
        "grid_size": task.get("grid_size"),
        "p_noise": task.get("p_noise"),
        "shots_total": task.get("shots_total"),
        "seed": task.get("seed"),
        "zne_order": task.get("zne_order"),
        "scale_factors": list(task.get("scale_factors", [])),
        "output_path": output_rel,
        "done_path": done_rel,
    }
    if not output_path.is_file():
        failures.append("missing_array")
        return record, failures
    if not done_path.is_file():
        failures.append("missing_done")
        return record, failures

    array_sha256 = sha256_file(output_path)
    done_sha256 = sha256_file(done_path)
    record["array_sha256"] = array_sha256
    record["done_sha256"] = done_sha256
    done = _load_json(done_path)
    if done.get("status") != "complete":
        failures.append("done_status_not_complete")
    for key in ("task_id", "block_id", "instance", "kind"):
        if done.get(key) != task.get(key):
            failures.append(f"{key}_mismatch")
    if done.get("runner_version") != PHASE_D_KERNEL_RUNNER_VERSION:
        failures.append("runner_version_mismatch")
    if done.get("protocol_version") != PROTOCOL_VERSION:
        failures.append("protocol_version_mismatch")
    if done.get("custom_kernel_version") != CUSTOM_KERNEL_VERSION:
        failures.append("custom_kernel_version_mismatch")
    if done.get("array_sha256") != array_sha256:
        failures.append("array_sha256_mismatch")
    for key in (
        "phase_d_plan_sha256",
        "phase_d_scope_snapshot_hash",
        "generator_config_hash",
        "candidate_queue_hash",
        "source_tree_hash",
        "environment_hash",
    ):
        if context.get(key) is not None and done.get(key) != context.get(key):
            failures.append(f"{key}_mismatch")

    array = np.load(output_path, allow_pickle=False, mmap_mode="r")
    expected_shape = (int(task["grid_size"]), int(task["grid_size"]))
    record["shape"] = list(array.shape)
    record["dtype"] = str(array.dtype)
    record["array_min"] = float(np.min(array))
    record["array_max"] = float(np.max(array))
    record["array_mean"] = float(np.mean(array))
    if array.shape != expected_shape:
        failures.append("shape_mismatch")
    if str(array.dtype) != "float64":
        failures.append("dtype_not_float64")
    if not bool(np.isfinite(array).all()):
        failures.append("array_not_finite")
    if done.get("shape") != list(array.shape):
        failures.append("done_shape_mismatch")
    return record, failures


def _progress_rows(root: Path, block_id: str | None) -> list[dict[str, Any]]:
    result_dir = _official_result_dir(root)
    if block_id is None:
        paths = sorted(set(result_dir.glob("progress_shard_*.json")) | set(result_dir.glob("progress_*_shard_*.json")))
    else:
        paths = sorted(result_dir.glob(f"progress_{block_id}_shard_*.json"))
    return [_load_json(path) for path in paths]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# {report['title']}",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        f"- Tasks: `{report['summary']['tasks_completed']}` / `{report['summary']['tasks_total']}`",
        f"- Corrupt tasks: `{report['summary']['corrupt_tasks']}`",
        f"- Task validation failures: `{report['checks']['task_validation_failure_count']}`",
        f"- Duplicate task ids: `{report['checks']['duplicate_task_ids']}`",
        f"- Duplicate condition keys: `{report['checks']['duplicate_condition_keys']}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _completion_payload(root: Path, tasks: list[dict[str, Any]], block_id: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _load_official_plan(root)
    context = _expected_context(root, plan)
    task_records: list[dict[str, Any]] = []
    task_failures: dict[str, list[str]] = {}
    for task in tasks:
        record, failures = _check_task(root, task, context)
        task_records.append(record)
        if failures:
            task_failures[str(task.get("task_id"))] = failures

    task_ids = [str(task.get("task_id")) for task in tasks]
    condition_keys = [_task_condition_key(task) for task in tasks]
    duplicate_task_ids = sorted(key for key, count in Counter(task_ids).items() if count > 1)
    duplicate_condition_keys = [
        "|".join(str(part) for part in key)
        for key, count in Counter(condition_keys).items()
        if count > 1
    ]
    expected_tasks = EXPECTED_GLOBAL_TASKS if block_id is None else EXPECTED_BLOCK_TASKS
    summary = summarize_phase_d_kernel_runner(OFFICIAL_RUN_NAME, root=root, block_id=block_id)
    progress = _progress_rows(root, block_id)
    checks = {
        "runner_complete": summary.get("status") == "complete",
        "tasks_completed_match_plan": int(summary.get("tasks_completed", -1)) == len(tasks),
        "tasks_pending_zero": int(summary.get("tasks_pending", -1)) == 0,
        "corrupt_tasks_empty": summary.get("corrupt_tasks") == [],
        "expected_task_count": len(tasks) == expected_tasks,
        "grid_size_120": all(int(task.get("grid_size", -1)) == TARGET_GRID_SIZE for task in tasks),
        "runner_version_match": plan.get("runner_version") == PHASE_D_KERNEL_RUNNER_VERSION,
        "protocol_version_match": plan.get("protocol_version") == PROTOCOL_VERSION,
        "custom_kernel_version_match": plan.get("custom_kernel_version") == CUSTOM_KERNEL_VERSION,
        "progress_files_present": bool(progress),
        "progress_files_complete": bool(progress) and all(row.get("status") == "complete" for row in progress),
        "progress_files_block_scoped": block_id is None or all(row.get("block_id") == block_id for row in progress),
        "duplicate_task_ids": duplicate_task_ids,
        "duplicate_condition_keys": duplicate_condition_keys,
        "task_validation_failure_count": len(task_failures),
    }
    status_pass = (
        all(value is True for value in checks.values() if isinstance(value, bool))
        and not duplicate_task_ids
        and not duplicate_condition_keys
        and not task_failures
    )
    status = "phase_d_global_post_merge_complete" if block_id is None and status_pass else (
        f"phase_d_{block_id}_post_merge_complete" if status_pass else f"phase_d_{block_id or 'global'}_post_merge_incomplete"
    )
    decision = "ready_for_phase_d_global_analysis" if block_id is None and status_pass else (
        f"ready_to_continue_after_{block_id}" if status_pass else "block_next_phase_d_step_until_post_merge_issues_are_resolved"
    )
    manifest = {
        "completion_version": PHASE_D_COMPLETION_VERSION,
        "created_at_utc": _utc_now(),
        "status": status,
        "run_name": OFFICIAL_RUN_NAME,
        "block_id": block_id,
        "summary": {
            "tasks_total": len(tasks),
            "tasks_completed": int(summary.get("tasks_completed", 0)),
            "tasks_pending": int(summary.get("tasks_pending", -1)),
            "corrupt_tasks": summary.get("corrupt_tasks", []),
            "grid_size": TARGET_GRID_SIZE,
            "kind_counts": dict(sorted(Counter(str(task.get("kind")) for task in tasks).items())),
            "instance_counts": dict(sorted(Counter(str(task.get("instance")) for task in tasks).items())),
            "progress_files": [row for row in progress],
        },
        "engine": {
            "runner_version": PHASE_D_KERNEL_RUNNER_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "custom_kernel_version": CUSTOM_KERNEL_VERSION,
            "planning_context": context,
        },
        "checks": checks,
        "task_validation_failures": task_failures,
        "tasks": task_records,
        "supporting_artifacts": {
            "plan": f"results/{OFFICIAL_RUN_NAME}/plan.json",
            "runner_summary": f"results/{OFFICIAL_RUN_NAME}/runner_summary.json",
        },
    }
    report = {
        "completion_version": PHASE_D_COMPLETION_VERSION,
        "created_at_utc": manifest["created_at_utc"],
        "title": "Phase D Global Completion Report" if block_id is None else f"Phase D {block_id} Completion Report",
        "status": status,
        "decision": decision,
        "run_name": OFFICIAL_RUN_NAME,
        "block_id": block_id,
        "summary": manifest["summary"],
        "checks": checks,
        "task_validation_failures": task_failures,
        "final_manifest": FINAL_MANIFEST_NAME,
    }
    return manifest, report


def run_phase_d_block_completion(block_id: str, overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    plan = _load_official_plan(root)
    tasks = _tasks_for_block(plan, block_id)
    result_dir = _phase_d_result_dir(root, block_id)
    manifest_path = result_dir / FINAL_MANIFEST_NAME
    report_path = result_dir / BLOCK_REPORT_NAME
    md_path = result_dir / BLOCK_REPORT_MD_NAME
    if not overwrite:
        for path in (manifest_path, report_path, md_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing Phase D completion output: {path}")
    manifest, report = _completion_payload(root, tasks, block_id)
    write_json_atomic(manifest_path, manifest, overwrite=True)
    write_json_atomic(report_path, report, overwrite=True)
    _write_markdown(md_path, report)
    return report


def run_phase_d_global_completion(overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    plan = _load_official_plan(root)
    tasks = list(plan.get("tasks", []))
    result_dir = _phase_d_result_dir(root, None)
    manifest_path = result_dir / FINAL_MANIFEST_NAME
    report_path = result_dir / BLOCK_REPORT_NAME
    md_path = result_dir / BLOCK_REPORT_MD_NAME
    if not overwrite:
        for path in (manifest_path, report_path, md_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing Phase D completion output: {path}")
    block_reports = {
        block_id: _load_json(root / "results" / "phase_d" / block_id / BLOCK_REPORT_NAME, default={})
        for block_id in PHASE_D_BLOCK_IDS
    }
    if not all(report.get("status") == f"phase_d_{block_id}_post_merge_complete" for block_id, report in block_reports.items()):
        raise RuntimeError("All Phase D block completion reports must be complete before global completion.")
    manifest, report = _completion_payload(root, tasks, None)
    report["block_reports"] = {
        block_id: f"results/phase_d/{block_id}/{BLOCK_REPORT_NAME}"
        for block_id in PHASE_D_BLOCK_IDS
    }
    manifest["block_reports"] = report["block_reports"]
    write_json_atomic(manifest_path, manifest, overwrite=True)
    write_json_atomic(report_path, report, overwrite=True)
    _write_markdown(md_path, report)
    return report


def write_phase_d_completion_preflight(overwrite: bool = True, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    plan = _load_official_plan(root)
    if (
        plan.get("official_block_scoped_execution_required") is not True
        or plan.get("official_execution_order") != list(PHASE_D_BLOCK_IDS)
        or int(plan.get("tasks_total", -1)) != EXPECTED_GLOBAL_TASKS
    ):
        plan = write_phase_d_official_kernel_plan(root)["plan"]
    block_counts = {block_id: len(_tasks_for_block(plan, block_id)) for block_id in PHASE_D_BLOCK_IDS}
    checks = {
        "official_plan_exists": (_official_result_dir(root) / "plan.json").is_file(),
        "official_plan_not_execution": plan.get("mode") == "phase_d_official_120x120_plan_not_executed_by_default",
        "global_task_count_21612": int(plan.get("tasks_total", -1)) == EXPECTED_GLOBAL_TASKS,
        "block_task_counts_5403": all(count == EXPECTED_BLOCK_TASKS for count in block_counts.values()),
        "block_result_dirs_declared": all(f"results/phase_d/{block_id}" in plan.get("block_result_dirs", {}).values() for block_id in PHASE_D_BLOCK_IDS),
        "block_scoped_execution_required": plan.get("official_block_scoped_execution_required") is True,
        "official_execution_order_d1_to_d4": plan.get("official_execution_order") == list(PHASE_D_BLOCK_IDS),
    }
    status = "phase_d_completion_preflight_passed" if all(checks.values()) else "phase_d_completion_preflight_failed"
    report = {
        "completion_version": PHASE_D_COMPLETION_VERSION,
        "created_at_utc": _utc_now(),
        "status": status,
        "decision": "completion_scripts_ready_for_phase_d_v1_0_freeze" if status.endswith("passed") else "block_phase_d_v1_0_freeze_until_completion_preflight_passes",
        "scope": "pre_official_execution_completion_script_readiness",
        "block_task_counts": block_counts,
        "checks": checks,
        "post_execution_outputs": {
            "block_manifest_template": "results/phase_d/{D1,D2,D3,D4}/manifest_final.json",
            "block_completion_report_template": "results/phase_d/{D1,D2,D3,D4}/completion_report.json",
            "global_manifest": f"results/phase_d/{GLOBAL_RESULT_DIR}/manifest_final.json",
        },
    }
    output = ensure_dir(root / "results" / "phase_d" / "planning") / "phase_d_completion_preflight.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing completion preflight: {output}")
    write_json_atomic(output, report, overwrite=True)
    return report
