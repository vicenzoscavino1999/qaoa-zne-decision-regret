from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PROJECT_ROOT, sha256_file, write_json_atomic
from .phase_c_120_kernel_runner import (
    CUSTOM_KERNEL_VERSION,
    KERNEL_RUNNER_VERSION,
    OFFICIAL_RUN_NAME,
    PROTOCOL_VERSION,
    TARGET_GRID_SIZE,
    summarize_kernel_runner,
)


COMPLETION_VERSION = "phase_c_120_completion_v1"
REPORT_NAME = "phase_c_120_completion_report.json"
REPORT_MD_NAME = "phase_c_120_completion_report.md"
FINAL_MANIFEST_NAME = "manifest_final.json"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def _load_json_any(path: Path, default: Any) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else default


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _duration_seconds(progress_rows: list[dict[str, Any]]) -> float | None:
    starts = [_parse_iso(str(row.get("started_at_utc", ""))) for row in progress_rows]
    updates = [_parse_iso(str(row.get("updated_at_utc", ""))) for row in progress_rows]
    starts = [item for item in starts if item is not None]
    updates = [item for item in updates if item is not None]
    if not starts or not updates:
        return None
    return (max(updates) - min(starts)).total_seconds()


def _task_condition_key(task: dict[str, Any]) -> tuple[Any, ...]:
    return (
        task.get("kind"),
        task.get("instance"),
        task.get("grid_size"),
        task.get("p_noise"),
        task.get("shots_total"),
        task.get("seed"),
        task.get("zne_order"),
        tuple(task.get("scale_factors", [])),
    )


def _check_task(root: Path, task: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    output_rel = str(task["output_path"])
    done_rel = str(task["done_path"])
    output_path = root / output_rel
    done_path = root / done_rel
    record: dict[str, Any] = {
        "task_id": task["task_id"],
        "task_index": task["index"],
        "condition_key": list(_task_condition_key(task)),
        "kind": task["kind"],
        "instance": task["instance"],
        "grid_size": task["grid_size"],
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
    if done.get("task_id") != task.get("task_id"):
        failures.append("task_id_mismatch")
    if done.get("runner_version") != KERNEL_RUNNER_VERSION:
        failures.append("runner_version_mismatch")
    if done.get("protocol_version") != PROTOCOL_VERSION:
        failures.append("protocol_version_mismatch")
    if done.get("custom_kernel_version") != CUSTOM_KERNEL_VERSION:
        failures.append("custom_kernel_version_mismatch")
    if done.get("array_sha256") != array_sha256:
        failures.append("array_sha256_mismatch")

    array = np.load(output_path, allow_pickle=False)
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


def _summarize_logs(root: Path) -> dict[str, Any]:
    log_dir = root / "logs"
    stderr_paths = sorted(log_dir.glob("phase_c_120_official_shard_*_stderr.log"))
    stdout_paths = sorted(log_dir.glob("phase_c_120_official_shard_*_stdout.log"))
    return {
        "stderr_files": [path.relative_to(root).as_posix() for path in stderr_paths],
        "stdout_files": [path.relative_to(root).as_posix() for path in stdout_paths],
        "stderr_nonempty": [
            path.relative_to(root).as_posix()
            for path in stderr_paths
            if path.stat().st_size > 0
        ],
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    runtime = report["runtime"]
    checks = report["checks"]
    backup = report["backup"]
    integrity = report["integrity"]
    lines = [
        "# Phase C 120x120 Completion Report",
        "",
        f"- Status: `{report['status']}`",
        f"- Decision: `{report['decision']}`",
        f"- Generated at UTC: `{report['created_at_utc']}`",
        f"- Tasks: `{summary['tasks_completed']}/{summary['tasks_total']}`",
        f"- Corrupt tasks: `{summary['corrupt_tasks']}`",
        f"- Duplicate task ids: `{checks['duplicate_task_ids']}`",
        f"- Duplicate condition keys: `{checks['duplicate_condition_keys']}`",
        f"- Task validation failures: `{checks['task_validation_failure_count']}`",
        f"- Backup verified: `{backup.get('verified')}`",
        f"- Integrity overall pass: `{integrity.get('overall_pass')}`",
    ]
    if runtime.get("official_elapsed_hours") is not None:
        lines.append(f"- Official elapsed hours from progress files: `{runtime['official_elapsed_hours']:.4f}`")
    lines.extend(
        [
            "",
            "## Output Artifacts",
            "",
            f"- Final manifest: `{FINAL_MANIFEST_NAME}`",
            f"- JSON report: `{REPORT_NAME}`",
            "",
            "Phase 11 official analysis is allowed only if this report status remains "
            "`phase_c_120_post_merge_complete` and strict integrity stays green.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_phase_c_120_completion(
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    result_dir = root / "results" / OFFICIAL_RUN_NAME
    plan_path = result_dir / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing official plan: {plan_path}")

    report_path = result_dir / REPORT_NAME
    manifest_path = result_dir / FINAL_MANIFEST_NAME
    md_path = result_dir / REPORT_MD_NAME
    if not overwrite:
        for path in [report_path, manifest_path, md_path]:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing completion output: {path}")

    runner_summary = summarize_kernel_runner(OFFICIAL_RUN_NAME, root)
    plan = _load_json(plan_path)
    backup = _load_json(result_dir / "backup_report.json")
    runtime_estimate = _load_json(result_dir / "runtime_estimate.json")
    final_freeze = _load_json(root / "results" / "phase_c_120_final_freeze" / "final_freeze_summary.json")
    integrity = _load_json(root / "results" / "integrity_report.json")
    official_processes = _load_json_any(result_dir / "official_processes.json", [])

    progress_paths = sorted(result_dir.glob("progress_shard_*.json"))
    progress_rows = [_load_json(path) for path in progress_paths]
    task_records: list[dict[str, Any]] = []
    task_failures: dict[str, list[str]] = {}
    for task in plan.get("tasks", []):
        record, failures = _check_task(root, task)
        task_records.append(record)
        if failures:
            task_failures[str(task.get("task_id"))] = failures

    task_ids = [str(task.get("task_id")) for task in plan.get("tasks", [])]
    condition_keys = [_task_condition_key(task) for task in plan.get("tasks", [])]
    duplicate_task_ids = sorted(key for key, count in Counter(task_ids).items() if count > 1)
    duplicate_condition_keys = [
        "|".join(str(part) for part in key)
        for key, count in Counter(condition_keys).items()
        if count > 1
    ]
    kind_counts = dict(sorted(Counter(str(task.get("kind")) for task in plan.get("tasks", [])).items()))
    instance_counts = dict(sorted(Counter(str(task.get("instance")) for task in plan.get("tasks", [])).items()))

    expected_cache = plan.get("probability_cache_model", {})
    actual_cache = runner_summary.get("cache_totals", {})
    elapsed_seconds = _duration_seconds(progress_rows)
    logs = _summarize_logs(root)

    checks = {
        "runner_complete": runner_summary.get("status") == "complete",
        "tasks_completed_match_plan": int(runner_summary.get("tasks_completed", -1)) == int(plan.get("tasks_total", -2)),
        "tasks_pending_zero": int(runner_summary.get("tasks_pending", -1)) == 0,
        "corrupt_tasks_empty": runner_summary.get("corrupt_tasks") == [],
        "grid_size_120": int(plan.get("grid_size", -1)) == TARGET_GRID_SIZE,
        "runner_version_match": plan.get("runner_version") == KERNEL_RUNNER_VERSION,
        "protocol_version_match": plan.get("protocol_version") == PROTOCOL_VERSION,
        "custom_kernel_version_match": plan.get("custom_kernel_version") == CUSTOM_KERNEL_VERSION,
        "progress_files_complete": all(row.get("status") == "complete" for row in progress_rows) and bool(progress_rows),
        "progress_task_sum_matches_plan": sum(int(row.get("tasks_completed_in_shard", 0)) for row in progress_rows) == int(plan.get("tasks_total", -1)),
        "cache_hits_match_expected": int(actual_cache.get("hits", -1)) == int(expected_cache.get("expected_hits", -2)),
        "cache_misses_match_expected": int(actual_cache.get("misses", -1)) == int(expected_cache.get("expected_misses", -2)),
        "cache_requests_match_expected": int(actual_cache.get("requests", -1)) == int(expected_cache.get("expected_requests", -2)),
        "backup_verified": backup.get("verified") is True and backup.get("missing") == [] and backup.get("mismatches") == [],
        "integrity_overall_pass": integrity.get("overall_pass") is True,
        "final_freeze_passed": final_freeze.get("status") == "phase_c_120_final_freeze_passed",
        "stderr_logs_empty": logs["stderr_nonempty"] == [],
        "duplicate_task_ids": duplicate_task_ids,
        "duplicate_condition_keys": duplicate_condition_keys,
        "task_validation_failure_count": len(task_failures),
    }
    boolean_checks_pass = all(value is True for value in checks.values() if isinstance(value, bool))
    list_checks_pass = not duplicate_task_ids and not duplicate_condition_keys
    task_checks_pass = not task_failures
    status = "phase_c_120_post_merge_complete" if boolean_checks_pass and list_checks_pass and task_checks_pass else "phase_c_120_post_merge_incomplete"
    decision = "ready_for_phase_11_official_analysis" if status == "phase_c_120_post_merge_complete" else "block_phase_11_until_post_merge_issues_are_resolved"

    summary = {
        "tasks_total": int(plan.get("tasks_total", 0)),
        "tasks_completed": int(runner_summary.get("tasks_completed", 0)),
        "tasks_pending": int(runner_summary.get("tasks_pending", -1)),
        "corrupt_tasks": runner_summary.get("corrupt_tasks", []),
        "grid_size": int(plan.get("grid_size", -1)),
        "kind_counts": kind_counts,
        "instance_counts": instance_counts,
        "progress_files": [path.name for path in progress_paths],
        "official_process_count": len(official_processes),
    }
    runtime = {
        "official_elapsed_seconds": elapsed_seconds,
        "official_elapsed_hours": None if elapsed_seconds is None else elapsed_seconds / 3600.0,
        "runtime_estimate": runtime_estimate,
        "progress_started_at_utc_min": min((str(row.get("started_at_utc")) for row in progress_rows), default=None),
        "progress_updated_at_utc_max": max((str(row.get("updated_at_utc")) for row in progress_rows), default=None),
    }
    manifest = {
        "completion_version": COMPLETION_VERSION,
        "created_at_utc": _utc_now(),
        "status": status,
        "run_name": OFFICIAL_RUN_NAME,
        "summary": summary,
        "runtime": runtime,
        "engine": {
            "runner_version": KERNEL_RUNNER_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "custom_kernel_version": CUSTOM_KERNEL_VERSION,
            "environment_hash": final_freeze.get("environment_hash"),
            "source_tree_hash": final_freeze.get("source_tree_hash"),
            "source_snapshot_sha256": final_freeze.get("source_snapshot_sha256"),
            "pre_registration_hash": final_freeze.get("pre_registration_hash"),
            "analysis_hashes": final_freeze.get("analysis_hashes", {}),
        },
        "checks": checks,
        "task_validation_failures": task_failures,
        "tasks": task_records,
        "supporting_artifacts": {
            "plan": "results/phase_c_120_kernel_official/plan.json",
            "runner_summary": "results/phase_c_120_kernel_official/runner_summary.json",
            "backup_report": "results/phase_c_120_kernel_official/backup_report.json",
            "integrity_report": "results/integrity_report.json",
            "final_freeze_summary": "results/phase_c_120_final_freeze/final_freeze_summary.json",
        },
    }
    report = {
        "completion_version": COMPLETION_VERSION,
        "created_at_utc": manifest["created_at_utc"],
        "status": status,
        "decision": decision,
        "run_name": OFFICIAL_RUN_NAME,
        "summary": summary,
        "runtime": runtime,
        "checks": checks,
        "task_validation_failures": task_failures,
        "backup": backup,
        "integrity": {
            "overall_pass": integrity.get("overall_pass"),
            "counts": integrity.get("counts"),
        },
        "logs": logs,
        "final_manifest": FINAL_MANIFEST_NAME,
    }

    write_json_atomic(manifest_path, manifest, overwrite=True)
    write_json_atomic(report_path, report, overwrite=True)
    _write_markdown(md_path, report)
    return report
