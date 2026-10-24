from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .phase_c_120_operational_dry_run import _process_memory_sample, _thermal_snapshot
from .phase_c_120_preflight import PROTOCOL_VERSION
from .phase_d_kernel_runner import (
    DRY_RUN_NAME,
    OFFICIAL_RUN_NAME,
    PHASE_D_KERNEL_RUNNER_VERSION,
    SELF_TEST_RUN_NAME,
    write_phase_d_official_kernel_plan,
)


PHASE_D_RUNNER_DRY_RUN_VERSION = "phase_d_runner_dry_run_v1"
DEFAULT_SHARD_COUNT = min(8, max(1, os.cpu_count() or 1))
POLL_INTERVAL_SECONDS = 1.0
DEFAULT_BACKUP_DIR = Path("C:/phase_d_120_backups/phase_d_120_operational_dry_run")


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_reset(path: Path, root: Path) -> None:
    resolved = path.resolve()
    allowed = {
        (root / "data" / "raw" / DRY_RUN_NAME).resolve(),
        (root / "results" / DRY_RUN_NAME).resolve(),
        (root / "data" / "raw" / SELF_TEST_RUN_NAME).resolve(),
        (root / "results" / SELF_TEST_RUN_NAME).resolve(),
        DEFAULT_BACKUP_DIR.resolve(),
    }
    if resolved not in allowed:
        raise RuntimeError(f"Refusing to remove non-Phase-D-dry-run path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _run_text(args: list[str], root: Path, timeout: int = 120) -> str:
    try:
        proc = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:
        return f"ERROR: {exc!r}"
    return (proc.stdout or proc.stderr or "").strip()


def _run_shards_parallel(
    root: Path,
    shard_count: int,
    stop_after_tasks: int | None,
    label: str,
    mode: str = "dry_run",
    block_id: str | None = None,
) -> dict[str, Any]:
    script = root / "scripts" / "33_phase_d_kernel_runner.py"
    python = root / ".venv" / "Scripts" / "python.exe"
    procs: list[tuple[int, subprocess.Popen[str]]] = []
    started = time.perf_counter()
    for shard_index in range(shard_count):
        args = [
            str(python),
            str(script),
            "--mode",
            mode,
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(shard_count),
        ]
        if block_id is not None:
            args.extend(["--block-id", block_id])
        if stop_after_tasks is not None:
            args.extend(["--stop-after-tasks", str(stop_after_tasks)])
        proc = subprocess.Popen(args, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        procs.append((shard_index, proc))

    memory_samples: list[dict[str, Any]] = []
    while any(proc.poll() is None for _, proc in procs):
        pids = [proc.pid for _, proc in procs if proc.poll() is None]
        for sample in _process_memory_sample(pids, root):
            sample["phase"] = label
            sample["sampled_at_utc"] = _now_utc()
            memory_samples.append(sample)
        time.sleep(POLL_INTERVAL_SECONDS)

    outputs = []
    for shard_index, proc in procs:
        stdout, stderr = proc.communicate(timeout=30)
        outputs.append({
            "shard_index": shard_index,
            "returncode": proc.returncode,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        })
    elapsed = time.perf_counter() - started
    peak_working_set = max((int(row.get("PeakWorkingSet64") or 0) for row in memory_samples), default=0)
    return {
        "label": label,
        "block_id": block_id,
        "block_scoped": block_id is not None,
        "stop_after_tasks": stop_after_tasks,
        "elapsed_seconds": elapsed,
        "all_returncodes_zero": all(row["returncode"] == 0 for row in outputs),
        "outputs": outputs,
        "memory_samples": memory_samples,
        "peak_working_set_bytes": peak_working_set,
    }


def _run_self_test(root: Path) -> dict[str, Any]:
    _safe_reset(root / "data" / "raw" / SELF_TEST_RUN_NAME, root)
    _safe_reset(root / "results" / SELF_TEST_RUN_NAME, root)
    run = _run_shards_parallel(root, shard_count=2, stop_after_tasks=None, label="self_test_tiny_grid", mode="self_test")
    summary_text = _run_text([
        str(root / ".venv" / "Scripts" / "python.exe"),
        str(root / "scripts" / "33_phase_d_kernel_runner.py"),
        "--summary",
        "self_test",
    ], root)
    summary = json.loads(summary_text) if summary_text.startswith("{") else {"raw": summary_text}
    return {
        "run": run,
        "summary": summary,
        "passed": bool(run["all_returncodes_zero"] and summary.get("status") == "complete" and int(summary.get("tasks_pending", -1)) == 0),
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def _stable_files(base: Path) -> list[Path]:
    if not base.exists():
        return []
    return [
        path
        for path in sorted(base.rglob("*"))
        if path.is_file()
        and not path.name.startswith(".")
        and ".tmp." not in path.name
    ]


def _collect_file_hashes(paths: list[Path], root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for base in paths:
        for path in _stable_files(base):
            out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def _copy_and_verify_backup(root: Path, backup_dir: Path) -> dict[str, Any]:
    raw_dir = root / "data" / "raw" / DRY_RUN_NAME
    result_dir = root / "results" / DRY_RUN_NAME
    _safe_reset(backup_dir, root)
    ensure_dir(backup_dir)
    source_hashes = _collect_file_hashes([raw_dir, result_dir], root)
    copied: dict[str, str] = {}
    for rel in sorted(source_hashes):
        source = root / rel
        dest = backup_dir / rel
        ensure_dir(dest.parent)
        shutil.copy2(source, dest)
        copied[rel] = sha256_file(dest)
    mismatches = [rel for rel, digest in source_hashes.items() if copied.get(rel) != digest]
    return {
        "backup_dir": str(backup_dir),
        "files_copied": len(copied),
        "source_files": len(source_hashes),
        "mismatches": mismatches,
        "verified": bool(source_hashes) and not mismatches and len(copied) == len(source_hashes),
    }


def _summarize_plan_integrity(root: Path) -> dict[str, Any]:
    result_dir = root / "results" / DRY_RUN_NAME
    plan = _load_json(result_dir / "plan.json")
    tasks = plan.get("tasks", [])
    task_ids = [row.get("task_id") for row in tasks]
    outputs = [row.get("output_path") for row in tasks]
    progress_files = sorted(path.name for path in result_dir.glob("progress_shard_*.json"))
    duplicate_task_ids = sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1})
    duplicate_output_paths = sorted({path for path in outputs if outputs.count(path) > 1})
    return {
        "tasks_total": len(tasks),
        "unique_task_ids": len(set(task_ids)),
        "duplicate_task_ids": duplicate_task_ids,
        "unique_output_paths": len(set(outputs)),
        "duplicate_output_paths": duplicate_output_paths,
        "progress_files": progress_files,
        "dry_run_instances": plan.get("instances", []),
        "dry_run_instance_candidate_ids": plan.get("instance_candidate_ids", {}),
        "dry_run_instance_strata": plan.get("instance_strata", {}),
    }


def _plan_integrity_from_tasks(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    task_ids = [row.get("task_id") for row in tasks]
    outputs = [row.get("output_path") for row in tasks]
    return {
        "tasks_total": len(tasks),
        "unique_task_ids": len(set(task_ids)),
        "duplicate_task_ids": sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1}),
        "unique_output_paths": len(set(outputs)),
        "duplicate_output_paths": sorted({path for path in outputs if outputs.count(path) > 1}),
    }


def _duplicate_detection_negative_control(root: Path) -> dict[str, Any]:
    plan = _load_json(root / "results" / DRY_RUN_NAME / "plan.json")
    tasks = [dict(row) for row in plan.get("tasks", [])]
    if len(tasks) < 2:
        return {"passed": False, "reason": "dry-run plan has fewer than 2 tasks"}
    tasks[1]["task_id"] = tasks[0]["task_id"]
    tasks[1]["output_path"] = tasks[0]["output_path"]
    integrity = _plan_integrity_from_tasks(tasks)
    return {
        "passed": bool(integrity["duplicate_task_ids"] and integrity["duplicate_output_paths"]),
        "synthetic_integrity": integrity,
        "note": "Synthetic in-memory duplicate control; no dry-run plan file was modified.",
    }


def _first_task_paths(root: Path) -> tuple[Path, Path, str]:
    plan = _load_json(root / "results" / DRY_RUN_NAME / "plan.json")
    first = plan["tasks"][0]
    return root / first["output_path"], root / first["done_path"], str(first["task_id"])


def _runner_summary(root: Path) -> dict[str, Any]:
    text = _run_text([
        str(root / ".venv" / "Scripts" / "python.exe"),
        str(root / "scripts" / "33_phase_d_kernel_runner.py"),
        "--summary",
        "dry_run",
    ], root)
    return json.loads(text) if text.startswith("{") else {"raw": text}


def _runner_block_summary(root: Path, block_id: str) -> dict[str, Any]:
    text = _run_text([
        str(root / ".venv" / "Scripts" / "python.exe"),
        str(root / "scripts" / "33_phase_d_kernel_runner.py"),
        "--summary",
        "dry_run",
        "--block-id",
        block_id,
    ], root)
    return json.loads(text) if text.startswith("{") else {"raw": text}


def _block_filter_execution_check(root: Path) -> dict[str, Any]:
    block_id = "D1"
    run = _run_shards_parallel(root, shard_count=2, stop_after_tasks=None, label="block_filter_execution_check", mode="dry_run", block_id=block_id)
    summary = _runner_block_summary(root, block_id)
    result_dir = root / "results" / DRY_RUN_NAME
    progress_rows = [
        _load_json(path)
        for path in sorted(result_dir.glob(f"progress_{block_id}_shard_*.json"))
    ]
    tasks_total = int(summary.get("tasks_total", -1))
    completed = int(summary.get("tasks_completed", -1))
    checks = {
        "block_subprocesses_returned_zero": run["all_returncodes_zero"],
        "block_summary_complete": summary.get("status") == "complete",
        "block_tasks_nonzero": tasks_total > 0,
        "block_tasks_completed": completed == tasks_total,
        "progress_files_block_scoped": bool(progress_rows) and all(row.get("block_id") == block_id and row.get("block_scoped") is True for row in progress_rows),
    }
    return {
        "passed": all(checks.values()),
        "block_id": block_id,
        "run": run,
        "summary": summary,
        "progress_files": [path.name for path in sorted(result_dir.glob(f"progress_{block_id}_shard_*.json"))],
        "checks": checks,
        "note": "Validates that the official runner path can be scoped to one Phase D block before the official token is accepted.",
    }


def _missing_task_detection_test(root: Path) -> dict[str, Any]:
    array_path, _done_path, task_id = _first_task_paths(root)
    moved = array_path.with_name(array_path.name + ".missing_task_test")
    if moved.exists():
        moved.unlink()
    array_path.rename(moved)
    try:
        summary = _runner_summary(root)
        passed = summary.get("status") == "incomplete" and task_id in summary.get("corrupt_tasks", [])
    finally:
        moved.rename(array_path)
        _runner_summary(root)
    return {
        "passed": bool(passed),
        "task_id": task_id,
        "observed_summary": summary,
        "note": "Temporarily renamed one array, verified runner summary flags the task, then restored it.",
    }


def _corrupted_hash_detection_test(root: Path) -> dict[str, Any]:
    _array_path, done_path, task_id = _first_task_paths(root)
    original_text = done_path.read_text(encoding="utf-8")
    payload = json.loads(original_text)
    payload["array_sha256"] = "0" * 64
    write_json_atomic(done_path, payload, overwrite=True)
    try:
        summary = _runner_summary(root)
        passed = summary.get("status") == "incomplete" and task_id in summary.get("corrupt_tasks", [])
    finally:
        done_path.write_text(original_text, encoding="utf-8")
        _runner_summary(root)
    return {
        "passed": bool(passed),
        "task_id": task_id,
        "observed_summary": summary,
        "note": "Temporarily corrupted one .done.json hash, verified runner summary flags the task, then restored it.",
    }


def _restore_read_check(root: Path, backup_dir: Path) -> dict[str, Any]:
    arrays_read = 0
    bad_arrays: list[str] = []
    hash_mismatches: list[str] = []
    for backup_file in _stable_files(backup_dir):
        rel = backup_file.relative_to(backup_dir)
        source = root / rel
        if source.is_file() and sha256_file(source) != sha256_file(backup_file):
            hash_mismatches.append(rel.as_posix())
        if backup_file.suffix == ".npy":
            try:
                arr = np.load(backup_file, allow_pickle=False)
                if not bool(np.isfinite(arr).all()):
                    bad_arrays.append(rel.as_posix())
                arrays_read += 1
            except Exception:
                bad_arrays.append(rel.as_posix())
    return {
        "passed": bool(arrays_read > 0 and not bad_arrays and not hash_mismatches),
        "arrays_read": arrays_read,
        "bad_arrays": bad_arrays,
        "hash_mismatches": hash_mismatches,
    }


def _post_merge_completion_check(final_summary: dict[str, Any], integrity: dict[str, Any], backup: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "runner_summary_complete": final_summary.get("status") == "complete",
        "no_pending_tasks": int(final_summary.get("tasks_pending", -1)) == 0,
        "no_corrupt_tasks": final_summary.get("corrupt_tasks") == [],
        "no_duplicate_task_ids": integrity.get("duplicate_task_ids") == [],
        "no_duplicate_output_paths": integrity.get("duplicate_output_paths") == [],
        "backup_verified": backup.get("verified") is True,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def _write_completion_report(result_dir: Path, report: dict[str, Any]) -> dict[str, str]:
    completion_json = result_dir / "phase_d_runner_dry_run_completion_report.json"
    completion_md = result_dir / "phase_d_runner_dry_run_completion_report.md"
    payload = {
        "created_at_utc": _now_utc(),
        "status": report["status"],
        "decision": report["decision"],
        "checks": report["checks"],
        "final_summary": report["final_summary"],
        "integrity": report["integrity"],
        "backup": report["backup"],
    }
    write_json_atomic(completion_json, payload, overwrite=True)
    completion_md.write_text(
        "\n".join([
            "# Phase D Runner Dry-Run Completion Report",
            "",
            f"Status: `{report['status']}`",
            f"Decision: `{report['decision']}`",
            f"Final tasks completed: `{report['final_summary'].get('tasks_completed')}` / `{report['final_summary'].get('tasks_total')}`",
            f"Backup verified: `{report['backup'].get('verified')}`",
            "",
        ]),
        encoding="utf-8",
    )
    return {
        "completion_report_json": sha256_file(completion_json),
        "completion_report_md": sha256_file(completion_md),
    }


def run_phase_d_runner_dry_run(
    shard_count: int = DEFAULT_SHARD_COUNT,
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    result_dir = root / "results" / DRY_RUN_NAME
    raw_dir = root / "data" / "raw" / DRY_RUN_NAME
    summary_path = result_dir / "phase_d_runner_dry_run_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Phase D runner dry-run output: {summary_path}")
    if overwrite:
        _safe_reset(raw_dir, root)
        _safe_reset(result_dir, root)
    ensure_dir(result_dir)

    self_test = _run_self_test(root)
    official_plan = write_phase_d_official_kernel_plan(root)
    thermal_before = _thermal_snapshot(root)
    partial = _run_shards_parallel(root, shard_count, stop_after_tasks=1, label="controlled_interrupt")
    partial_summary_text = _run_text([
        str(root / ".venv" / "Scripts" / "python.exe"),
        str(root / "scripts" / "33_phase_d_kernel_runner.py"),
        "--summary",
        "dry_run",
    ], root)
    partial_summary = json.loads(partial_summary_text) if partial_summary_text.startswith("{") else {"raw": partial_summary_text}
    resume = _run_shards_parallel(root, shard_count, stop_after_tasks=None, label="resume_to_completion")
    final_summary_text = _run_text([
        str(root / ".venv" / "Scripts" / "python.exe"),
        str(root / "scripts" / "33_phase_d_kernel_runner.py"),
        "--summary",
        "dry_run",
    ], root)
    final_summary = json.loads(final_summary_text) if final_summary_text.startswith("{") else {"raw": final_summary_text}
    thermal_after = _thermal_snapshot(root)
    integrity = _summarize_plan_integrity(root)
    manifest_hashes = _collect_file_hashes([raw_dir, result_dir], root)
    manifest_path = result_dir / "dry_run_manifest.json"
    write_json_atomic(manifest_path, {
        "created_at_utc": _now_utc(),
        "files": manifest_hashes,
    }, overwrite=True)
    backup = _copy_and_verify_backup(root, DEFAULT_BACKUP_DIR)
    restore_read = _restore_read_check(root, DEFAULT_BACKUP_DIR)
    duplicate_negative = _duplicate_detection_negative_control(root)
    missing_task = _missing_task_detection_test(root)
    corrupted_hash = _corrupted_hash_detection_test(root)
    block_filter = _block_filter_execution_check(root)
    final_summary = _runner_summary(root)

    all_memory_samples = partial["memory_samples"] + resume["memory_samples"]
    peak_working_set = max((int(row.get("PeakWorkingSet64") or 0) for row in all_memory_samples), default=0)
    ram_bytes = 15.8 * 1024**3
    memory_pass = peak_working_set == 0 or peak_working_set < 0.80 * ram_bytes
    partial_incomplete = int(partial_summary.get("tasks_pending", 0)) > 0 and partial_summary.get("status") == "incomplete"
    final_complete = final_summary.get("status") == "complete" and int(final_summary.get("tasks_pending", -1)) == 0
    duplicate_pass = not integrity["duplicate_task_ids"] and not integrity["duplicate_output_paths"]
    progress_pass = len(integrity["progress_files"]) == shard_count
    post_merge = _post_merge_completion_check(final_summary, integrity, backup)
    status = "phase_d_runner_dry_run_passed" if (
        self_test["passed"]
        and partial["all_returncodes_zero"]
        and resume["all_returncodes_zero"]
        and partial_incomplete
        and final_complete
        and duplicate_pass
        and progress_pass
        and backup["verified"]
        and restore_read["passed"]
        and duplicate_negative["passed"]
        and missing_task["passed"]
        and corrupted_hash["passed"]
        and block_filter["passed"]
        and post_merge["passed"]
        and memory_pass
    ) else "phase_d_runner_dry_run_failed"
    report = {
        "created_at_utc": _now_utc(),
        "phase_d_runner_dry_run_version": PHASE_D_RUNNER_DRY_RUN_VERSION,
        "runner_version": PHASE_D_KERNEL_RUNNER_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "scope": "phase_d_runner_operational_dry_run_not_official_phase_d_data",
        "run_name": DRY_RUN_NAME,
        "official_run_name": OFFICIAL_RUN_NAME,
        "shard_count": shard_count,
        "self_test": self_test,
        "official_plan_written_not_executed": {
            "plan_path": f"results/{OFFICIAL_RUN_NAME}/plan.json",
            "runtime_estimate_path": f"results/{OFFICIAL_RUN_NAME}/runtime_estimate.json",
            "tasks_total": official_plan["plan"]["tasks_total"],
            "groups_total": official_plan["plan"]["groups_total"],
            "expected_total_hours_H1_H12": official_plan["runtime_estimate"]["expected_total_hours_H1_H12"],
        },
        "controlled_interrupt": partial,
        "partial_summary": partial_summary,
        "resume_to_completion": resume,
        "final_summary": final_summary,
        "integrity": integrity,
        "backup": backup,
        "restore_read": restore_read,
        "runner_negative_controls": {
            "duplicate_detection": duplicate_negative,
            "missing_task_detection": missing_task,
            "corrupted_hash_detection": corrupted_hash,
        },
        "block_filter_execution_check": block_filter,
        "post_merge_completion": post_merge,
        "memory": {
            "peak_working_set_bytes": peak_working_set,
            "ram_reference_bytes": int(ram_bytes),
            "peak_working_set_fraction_of_ram": peak_working_set / ram_bytes if peak_working_set else None,
            "passes_80pct_ram_gate": memory_pass,
        },
        "thermal": {
            "before": thermal_before,
            "after": thermal_after,
            "status": "sensor_available" if thermal_before.get("available") or thermal_after.get("available") else "temperature_sensor_unavailable_runtime_gate_proxy_used",
        },
        "checks": {
            "official_plan_written_without_execution": official_plan["runtime_estimate"]["decision"] == "phase_d_official_kernel_runner_plan_ready_not_executed",
            "self_test_tiny_grid_passed": self_test["passed"],
            "controlled_interrupt_created_pending_tasks": partial_incomplete,
            "resume_completed_all_tasks": final_complete,
            "all_subprocesses_returned_zero": partial["all_returncodes_zero"] and resume["all_returncodes_zero"],
            "duplicate_detection_passed": duplicate_pass,
            "duplicate_detection_negative_control_passed": duplicate_negative["passed"],
            "missing_task_detection_passed": missing_task["passed"],
            "corrupted_hash_detection_passed": corrupted_hash["passed"],
            "block_filter_execution_check_passed": block_filter["passed"],
            "restore_read_verified": restore_read["passed"],
            "post_merge_completion_passed": post_merge["passed"],
            "progress_files_match_shard_count": progress_pass,
            "backup_sha256_verified": backup["verified"],
            "memory_gate_passed": memory_pass,
        },
        "decision": "proceed_to_final_scope_snapshot" if status.endswith("passed") else "do_not_freeze_phase_d_v1_0_until_runner_dry_run_is_resolved",
    }
    write_json_atomic(summary_path, report, overwrite=True)
    completion_hashes = _write_completion_report(result_dir, report)
    output_hashes = {
        "phase_d_runner_dry_run_summary_json": sha256_file(summary_path),
        "dry_run_manifest_json": sha256_file(manifest_path),
        "runner_summary_json": sha256_file(result_dir / "runner_summary.json"),
        "dry_run_plan_json": sha256_file(result_dir / "plan.json"),
        "self_test_runner_summary_json": sha256_file(root / "results" / SELF_TEST_RUN_NAME / "runner_summary.json"),
        "official_plan_json": sha256_file(root / "results" / OFFICIAL_RUN_NAME / "plan.json"),
        "official_runtime_estimate_json": sha256_file(root / "results" / OFFICIAL_RUN_NAME / "runtime_estimate.json"),
        **completion_hashes,
    }
    write_json_atomic(result_dir / "phase_d_runner_dry_run_output_hashes.json", {
        "created_at_utc": _now_utc(),
        "phase_d_runner_dry_run_version": PHASE_D_RUNNER_DRY_RUN_VERSION,
        "hashes": output_hashes,
    }, overwrite=True)
    report["output_hashes"] = output_hashes
    return report
