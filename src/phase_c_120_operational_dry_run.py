from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .phase_c_120_kernel_runner import DRY_RUN_NAME
from .phase_c_120_preflight import PROTOCOL_VERSION


OPERATIONAL_DRY_RUN_VERSION = "phase_c_120_operational_dry_run_v1"
DEFAULT_SHARD_COUNT = min(8, max(1, os.cpu_count() or 1))
POLL_INTERVAL_SECONDS = 1.0


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_reset(path: Path, root: Path) -> None:
    resolved = path.resolve()
    allowed = {
        (root / "data" / "raw" / DRY_RUN_NAME).resolve(),
        (root / "results" / DRY_RUN_NAME).resolve(),
        Path("C:/phase_c_120_backups/phase_c_120_operational_dry_run").resolve(),
    }
    if resolved not in allowed:
        raise RuntimeError(f"Refusing to remove non-dry-run path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _run_text(args: list[str], root: Path, timeout: int = 30) -> str:
    try:
        proc = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:
        return f"ERROR: {exc!r}"
    return (proc.stdout or proc.stderr or "").strip()


def _thermal_snapshot(root: Path) -> dict[str, Any]:
    text = _run_text([
        "powershell",
        "-NoProfile",
        "-Command",
        "Get-CimInstance MSAcpi_ThermalZoneTemperature -Namespace root/wmi | Select-Object CurrentTemperature | ConvertTo-Json",
    ], root, timeout=10)
    if not text or text.startswith("ERROR"):
        return {"available": False, "raw": text}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"available": False, "raw": text}
    rows = data if isinstance(data, list) else [data]
    celsius = []
    for row in rows:
        if isinstance(row, dict) and row.get("CurrentTemperature") is not None:
            celsius.append(float(row["CurrentTemperature"]) / 10.0 - 273.15)
    return {"available": bool(celsius), "celsius": celsius, "raw": data}


def _process_memory_sample(pids: list[int], root: Path) -> list[dict[str, Any]]:
    if not pids:
        return []
    id_list = ",".join(str(pid) for pid in pids)
    command = (
        f"$ids=@({id_list}); "
        "Get-Process -Id $ids -ErrorAction SilentlyContinue | "
        "Select-Object Id,WorkingSet64,PeakWorkingSet64,CPU | ConvertTo-Json"
    )
    text = _run_text(["powershell", "-NoProfile", "-Command", command], root, timeout=10)
    if not text or text.startswith("ERROR"):
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    rows = data if isinstance(data, list) else [data]
    return [row for row in rows if isinstance(row, dict)]


def _run_shards_parallel(
    root: Path,
    shard_count: int,
    stop_after_tasks: int | None,
    label: str,
) -> dict[str, Any]:
    script = root / "scripts" / "17_phase_c_120_kernel_runner.py"
    python = root / ".venv" / "Scripts" / "python.exe"
    procs: list[tuple[int, subprocess.Popen[str]]] = []
    started = time.perf_counter()
    for shard_index in range(shard_count):
        args = [
            str(python),
            str(script),
            "--mode",
            "dry_run",
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(shard_count),
        ]
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
        "stop_after_tasks": stop_after_tasks,
        "elapsed_seconds": elapsed,
        "all_returncodes_zero": all(row["returncode"] == 0 for row in outputs),
        "outputs": outputs,
        "memory_samples": memory_samples,
        "peak_working_set_bytes": peak_working_set,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _collect_file_hashes(paths: list[Path], root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for base in paths:
        if base.exists():
            for path in sorted(base.rglob("*")):
                if path.is_file():
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
        "verified": not mismatches and len(copied) == len(source_hashes),
    }


def _summarize_plan_integrity(root: Path) -> dict[str, Any]:
    result_dir = root / "results" / DRY_RUN_NAME
    plan = _load_json(result_dir / "plan.json")
    tasks = plan.get("tasks", [])
    task_ids = [row.get("task_id") for row in tasks]
    outputs = [row.get("output_path") for row in tasks]
    progress_files = sorted(path.name for path in result_dir.glob("progress_shard_*.json"))
    return {
        "tasks_total": len(tasks),
        "unique_task_ids": len(set(task_ids)),
        "duplicate_task_ids": sorted({task_id for task_id in task_ids if task_ids.count(task_id) > 1}),
        "unique_output_paths": len(set(outputs)),
        "duplicate_output_paths": sorted({path for path in outputs if outputs.count(path) > 1}),
        "progress_files": progress_files,
    }


def run_phase_c_120_operational_dry_run(
    shard_count: int = DEFAULT_SHARD_COUNT,
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    result_dir = root / "results" / DRY_RUN_NAME
    raw_dir = root / "data" / "raw" / DRY_RUN_NAME
    summary_path = result_dir / "operational_dry_run_summary.json"
    if summary_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing operational dry-run output: {summary_path}")
    if overwrite:
        _safe_reset(raw_dir, root)
        _safe_reset(result_dir, root)
    ensure_dir(result_dir)

    thermal_before = _thermal_snapshot(root)
    partial = _run_shards_parallel(root, shard_count, stop_after_tasks=1, label="controlled_interrupt")
    partial_summary_text = _run_text([
        str(root / ".venv" / "Scripts" / "python.exe"),
        str(root / "scripts" / "17_phase_c_120_kernel_runner.py"),
        "--summary",
        "dry_run",
    ], root, timeout=120)
    partial_summary = json.loads(partial_summary_text) if partial_summary_text.startswith("{") else {"raw": partial_summary_text}
    resume = _run_shards_parallel(root, shard_count, stop_after_tasks=None, label="resume_to_completion")
    final_summary_text = _run_text([
        str(root / ".venv" / "Scripts" / "python.exe"),
        str(root / "scripts" / "17_phase_c_120_kernel_runner.py"),
        "--summary",
        "dry_run",
    ], root, timeout=120)
    final_summary = json.loads(final_summary_text) if final_summary_text.startswith("{") else {"raw": final_summary_text}
    thermal_after = _thermal_snapshot(root)
    integrity = _summarize_plan_integrity(root)
    manifest_hashes = _collect_file_hashes([raw_dir, result_dir], root)
    write_json_atomic(result_dir / "dry_run_manifest.json", {
        "created_at_utc": _now_utc(),
        "files": manifest_hashes,
    }, overwrite=True)
    backup = _copy_and_verify_backup(root, Path("C:/phase_c_120_backups/phase_c_120_operational_dry_run"))

    all_memory_samples = partial["memory_samples"] + resume["memory_samples"]
    peak_working_set = max((int(row.get("PeakWorkingSet64") or 0) for row in all_memory_samples), default=0)
    ram_bytes = 15.8 * 1024**3
    memory_pass = peak_working_set == 0 or peak_working_set < 0.80 * ram_bytes
    partial_incomplete = int(partial_summary.get("tasks_pending", 0)) > 0 and partial_summary.get("status") == "incomplete"
    final_complete = final_summary.get("status") == "complete" and int(final_summary.get("tasks_pending", -1)) == 0
    duplicate_pass = not integrity["duplicate_task_ids"] and not integrity["duplicate_output_paths"]
    progress_pass = len(integrity["progress_files"]) == shard_count
    status = "phase_c_120_operational_dry_run_passed" if (
        partial["all_returncodes_zero"]
        and resume["all_returncodes_zero"]
        and partial_incomplete
        and final_complete
        and duplicate_pass
        and progress_pass
        and backup["verified"]
        and memory_pass
    ) else "phase_c_120_operational_dry_run_failed"
    report = {
        "created_at_utc": _now_utc(),
        "operational_dry_run_version": OPERATIONAL_DRY_RUN_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "scope": "parallel_operational_dry_run_not_official_phase_c_data",
        "run_name": DRY_RUN_NAME,
        "shard_count": shard_count,
        "controlled_interrupt": partial,
        "partial_summary": partial_summary,
        "resume_to_completion": resume,
        "final_summary": final_summary,
        "integrity": integrity,
        "backup": backup,
        "memory": {
            "peak_working_set_bytes": peak_working_set,
            "ram_reference_bytes": int(ram_bytes),
            "peak_working_set_fraction_of_ram": peak_working_set / ram_bytes if peak_working_set else None,
            "passes_80pct_ram_gate": memory_pass,
        },
        "thermal": {
            "before": thermal_before,
            "after": thermal_after,
            "status": "sensor_available" if thermal_before.get("available") or thermal_after.get("available") else "temperature_sensor_unavailable_throughput_proxy_only",
        },
        "checks": {
            "controlled_interrupt_created_pending_tasks": partial_incomplete,
            "resume_completed_all_tasks": final_complete,
            "all_subprocesses_returned_zero": partial["all_returncodes_zero"] and resume["all_returncodes_zero"],
            "duplicate_detection_passed": duplicate_pass,
            "progress_files_match_shard_count": progress_pass,
            "backup_sha256_verified": backup["verified"],
            "memory_gate_passed": memory_pass,
        },
        "decision": "phase_8_operational_gate_passed" if status.endswith("passed") else "do_not_start_phase_9_until_operational_gate_is_resolved",
    }
    write_json_atomic(summary_path, report, overwrite=True)
    return report
