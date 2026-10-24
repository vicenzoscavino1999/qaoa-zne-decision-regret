from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import tracemalloc
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import transpile
from qiskit_aer import AerSimulator

from .circuits import build_qaoa_circuit, graph_from_config
from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, sha256_file, write_json_atomic
from .landscapes import generate_ideal_landscape, generate_noisy_landscape, generate_zne_landscape
from .mitiq_wrapper import add_measurements, fold_global_odd_integer
from .noise_models import build_depolarizing_noise_model
from .phase_c import _selected_seeds, _simulator_options_for_profile


TARGET_GRID_SIZE = 120
PROTOCOL_VERSION = "phase_c_120_v0.6_draft"
PLANNING_DIR = PROJECT_ROOT / "results" / "phase_c_120_planning"
PACKAGE_NAMES = (
    "PyYAML",
    "matplotlib",
    "mitiq",
    "numpy",
    "pandas",
    "ply",
    "qiskit",
    "qiskit-aer",
    "scipy",
)
ENV_VAR_NAMES = (
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _run_text_command(args: list[str], timeout: int = 20) -> str:
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception as exc:  # pragma: no cover - depends on local OS tools
        return f"ERROR: {exc!r}"
    text = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0 and err:
        return f"ERROR rc={proc.returncode}: {err}"
    return text


def _powershell_json(command: str) -> Any:
    text = _run_text_command([
        "powershell",
        "-NoProfile",
        "-Command",
        f"{command} | ConvertTo-Json -Depth 6",
    ])
    if not text or text.startswith("ERROR"):
        return {"error": text or "empty powershell output"}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _disk_usage(path: str) -> dict[str, int | str]:
    usage = shutil.disk_usage(path)
    return {
        "path": path,
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
    }


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def _numpy_config_text() -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        np.show_config()
    return buffer.getvalue()


def _git_report(root: Path) -> dict[str, Any]:
    git_dir = root / ".git"
    if not git_dir.exists():
        return {"is_git_repo": False}
    return {
        "is_git_repo": True,
        "branch": _run_text_command(["git", "branch", "--show-current"]),
        "commit": _run_text_command(["git", "rev-parse", "HEAD"]),
        "status_short": _run_text_command(["git", "status", "--short"]),
    }


def generate_execution_environment(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    freeze_text = _run_text_command([str(root / ".venv" / "Scripts" / "python.exe"), "-m", "pip", "freeze"], timeout=60)
    if freeze_text.startswith("ERROR"):
        freeze_text = _run_text_command([sys.executable, "-m", "pip", "freeze"], timeout=60)

    report: dict[str, Any] = {
        "created_at_utc": _utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "platform": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor(),
            "machine": platform.machine(),
            "physical_cpu_note": "Use powershell_cim.processor for authoritative Windows core count.",
            "os_cpu_count": os.cpu_count(),
        },
        "powershell_cim": {
            "processor": _powershell_json("Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed"),
            "computer": _powershell_json("Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory"),
            "os": _powershell_json("Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,OSArchitecture"),
            "logical_disks": _powershell_json("Get-CimInstance Win32_LogicalDisk | Select-Object DeviceID,VolumeName,DriveType,FileSystem,Size,FreeSpace"),
            "disk_drives": _powershell_json("Get-CimInstance Win32_DiskDrive | Select-Object Index,Model,MediaType,Size,Partitions"),
        },
        "disk_usage": {
            "C": _disk_usage("C:\\"),
            "D": _disk_usage("D:\\"),
            "project_root": _disk_usage(str(root.anchor or root.drive + "\\")),
        },
        "package_versions": _package_versions(),
        "environment_variables": {name: os.environ.get(name, "") for name in ENV_VAR_NAMES},
        "numpy_config": _numpy_config_text(),
        "pip_freeze": freeze_text.splitlines() if freeze_text and not freeze_text.startswith("ERROR") else [],
        "git": _git_report(root),
    }
    report["environment_hash"] = _json_hash({k: v for k, v in report.items() if k != "environment_hash"})
    return report


def _source_tree_files(root: Path) -> list[Path]:
    include_roots = [
        root / "src",
        root / "scripts",
        root / "config",
    ]
    files: list[Path] = []
    for include_root in include_roots:
        if include_root.exists():
            files.extend(path for path in include_root.rglob("*") if path.is_file())
    root_patterns = ("*.md", "*.txt", "*.json", ".gitignore", ".gitattributes")
    for pattern in root_patterns:
        files.extend(path for path in root.glob(pattern) if path.is_file())
    excluded_names = {"manifest_v1.6.json"}
    filtered: list[Path] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        if rel.split("/", 1)[0] in {".git", ".venv", "data", "results", "logs"}:
            continue
        if path.name in excluded_names:
            continue
        if "__pycache__" in rel or path.suffix == ".pyc":
            continue
        filtered.append(path)
    return sorted(set(filtered), key=lambda p: p.relative_to(root).as_posix())


def generate_source_snapshot(root: Path = PROJECT_ROOT, output_dir: Path = PLANNING_DIR) -> dict[str, Any]:
    ensure_dir(output_dir)
    files = _source_tree_files(root)
    file_hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in files
    }
    manifest = {
        "created_at_utc": _utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "method": "sorted_path_sha256_canonical_json",
        "excluded_roots": [".git", ".venv", "data", "results", "logs"],
        "files": file_hashes,
    }
    manifest["source_tree_hash"] = _json_hash({"files": file_hashes, "method": manifest["method"]})
    manifest_path = output_dir / "source_tree_manifest.json"
    write_json_atomic(manifest_path, manifest)

    zip_path = output_dir / "source_snapshot_phase_c_120.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(root).as_posix())
    manifest["source_snapshot_path"] = zip_path.relative_to(root).as_posix()
    manifest["source_snapshot_sha256"] = sha256_file(zip_path)
    write_json_atomic(manifest_path, manifest)
    return manifest


def generate_task_inventory(cfg: dict[str, Any], seeds_cfg: dict[str, Any]) -> dict[str, Any]:
    instances = [str(x) for x in cfg["phase_c"]["instances"]]
    noise_values = [float(x) for x in cfg["noise"]["p_values"]]
    shot_budgets = [int(x) for x in cfg["shots"]["total_budget_primary"]]
    seed_count = int(cfg["phase_c"]["seed_count"])
    selected_seeds = [int(x) for x in seeds_cfg["official_candidates"][:seed_count]]
    zne_methods = {str(k): [int(s) for s in v] for k, v in cfg["zne"]["methods"].items()}
    n_conditions = len(instances) * len(noise_values) * len(shot_budgets) * len(selected_seeds)
    ideal_tasks = len(instances)
    noisy_tasks = n_conditions
    zne_tasks = n_conditions * len(zne_methods)
    total_output_tasks = ideal_tasks + noisy_tasks + zne_tasks
    grid_points = TARGET_GRID_SIZE * TARGET_GRID_SIZE
    scale_count_no_reuse = 1 + sum(len(v) for v in zne_methods.values())
    scale_count_reuse = len({scale for scales in zne_methods.values() for scale in scales})
    raw_scale_no_reuse = ideal_tasks + n_conditions * scale_count_no_reuse
    raw_scale_with_reuse = ideal_tasks + n_conditions * scale_count_reuse
    array_bytes = grid_points * 8

    return {
        "created_at_utc": _utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "target_grid_size": TARGET_GRID_SIZE,
        "grid_points_per_landscape": grid_points,
        "instances": instances,
        "noise_values": noise_values,
        "shot_budgets": shot_budgets,
        "seed_count": seed_count,
        "selected_seeds": selected_seeds,
        "zne_methods": zne_methods,
        "official_output_tasks": {
            "ideal": ideal_tasks,
            "noisy": noisy_tasks,
            "zne": zne_tasks,
            "total": total_output_tasks,
        },
        "raw_scale_landscapes": {
            "no_scale_reuse": raw_scale_no_reuse,
            "validated_scale_reuse": raw_scale_with_reuse,
            "scale_count_per_condition_no_reuse": scale_count_no_reuse,
            "scale_count_per_condition_with_reuse": scale_count_reuse,
            "scale_reuse_default": "disabled",
        },
        "gridpoint_evaluations": {
            "official_output_values": total_output_tasks * grid_points,
            "raw_no_scale_reuse": raw_scale_no_reuse * grid_points,
            "raw_validated_scale_reuse": raw_scale_with_reuse * grid_points,
        },
        "storage_estimate_bytes": {
            "one_float64_landscape": array_bytes,
            "official_output_raw": total_output_tasks * array_bytes,
            "raw_no_scale_reuse": raw_scale_no_reuse * array_bytes,
            "raw_validated_scale_reuse": raw_scale_with_reuse * array_bytes,
        },
    }


def _weight_sum(graph) -> float:
    return float(sum(weight for _, _, weight in graph.edges))


def generate_circuit_size_report(cfg: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    p_noise = float(cfg["phase_c"].get("benchmark_noise", 0.005))
    simulator = AerSimulator(noise_model=build_depolarizing_noise_model(p_noise), max_parallel_threads=1)
    scale_factors = sorted({int(scale) for scales in cfg["zne"]["methods"].values() for scale in scales})
    for instance in cfg["phase_c"]["instances"]:
        graph = graph_from_config(str(instance), cfg)
        base = build_qaoa_circuit(graph, 0.123, 0.456, measure=False)
        rows.append({
            "instance": graph.name,
            "scale_factor": 1,
            "circuit_kind": "base_unfolded_unmeasured",
            "n_qubits": graph.n_qubits,
            "n_edges": len(graph.edges),
            "sum_edge_weights": _weight_sum(graph),
            "gate_count": int(sum(base.count_ops().values())),
            "depth": int(base.depth()),
            "ops": dict(base.count_ops()),
        })
        for scale in scale_factors:
            folded = fold_global_odd_integer(base, scale)
            measured = add_measurements(folded)
            tqc = transpile(
                measured,
                simulator,
                basis_gates=["rx", "rz", "h", "cx"],
                optimization_level=0,
                seed_transpiler=0,
            )
            rows.append({
                "instance": graph.name,
                "scale_factor": scale,
                "circuit_kind": "folded_measured_transpiled",
                "n_qubits": graph.n_qubits,
                "n_edges": len(graph.edges),
                "sum_edge_weights": _weight_sum(graph),
                "gate_count": int(sum(tqc.count_ops().values())),
                "depth": int(tqc.depth()),
                "ops": dict(tqc.count_ops()),
            })
    return {
        "created_at_utc": _utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "basis_gates": ["rx", "rz", "h", "cx"],
        "seed_transpiler": 0,
        "rows": rows,
    }


def _time_call(label: str, fn) -> dict[str, Any]:
    tracemalloc.start()
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    array = result[0] if isinstance(result, tuple) else result
    return {
        "label": label,
        "seconds": elapsed,
        "tracemalloc_current_bytes": int(current),
        "tracemalloc_peak_bytes": int(peak),
        "shape": list(array.shape) if hasattr(array, "shape") else None,
        "finite": bool(np.isfinite(array).all()) if hasattr(array, "shape") else None,
        "min": float(np.nanmin(array)) if hasattr(array, "shape") else None,
        "max": float(np.nanmax(array)) if hasattr(array, "shape") else None,
    }


def generate_one_task_timing_report(
    cfg: dict[str, Any],
    timing_grid_size: int,
    simulator_profile: str,
) -> dict[str, Any]:
    options = _simulator_options_for_profile(simulator_profile)
    p_noise = float(cfg["phase_c"].get("benchmark_noise", 0.005))
    shots_total = int(cfg["phase_c"].get("benchmark_shots_total", 1024))
    seed = 42
    rows: list[dict[str, Any]] = []
    for instance in ("G1_C4", "G3_weighted6"):
        graph = graph_from_config(instance, cfg)
        rows.append({
            "instance": instance,
            "method": "ideal",
            **_time_call(
                f"{instance}_ideal",
                lambda graph=graph: generate_ideal_landscape(graph, timing_grid_size),
            ),
        })
        rows.append({
            "instance": instance,
            "method": "noisy",
            "p_noise": p_noise,
            "shots_total": shots_total,
            **_time_call(
                f"{instance}_noisy",
                lambda graph=graph: generate_noisy_landscape(
                    graph,
                    timing_grid_size,
                    p_noise,
                    shots_total,
                    seed,
                    "phase_c_120_preflight_noisy",
                    simulator_options=options,
                ),
            ),
        })
        rows.append({
            "instance": instance,
            "method": "zne_cubic",
            "p_noise": p_noise,
            "shots_total": shots_total,
            "scale_factors": [1, 3, 5, 7],
            **_time_call(
                f"{instance}_zne_cubic",
                lambda graph=graph: generate_zne_landscape(
                    graph,
                    timing_grid_size,
                    p_noise,
                    shots_total,
                    seed,
                    [1, 3, 5, 7],
                    "cubic",
                    simulator_options=options,
                ),
            ),
        })
    for row in rows:
        row["timing_grid_size"] = timing_grid_size
        row["grid_points"] = timing_grid_size * timing_grid_size
        row["seconds_per_gridpoint"] = float(row["seconds"]) / float(row["grid_points"])
        row["simulator_profile"] = simulator_profile
    return {
        "created_at_utc": _utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "timing_grid_size": timing_grid_size,
        "simulator_profile": simulator_profile,
        "simulator_options": options,
        "rows": rows,
    }


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_back_of_envelope_runtime(
    task_inventory: dict[str, Any],
    timing_report: dict[str, Any] | None,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    grid_points = int(task_inventory["grid_points_per_landscape"])
    official = task_inventory["official_output_tasks"]
    estimates: dict[str, Any] = {
        "created_at_utc": _utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "target_grid_size": TARGET_GRID_SIZE,
        "notes": [
            "This is a pre-profiling estimate; Phase 1 profiling supersedes it.",
            "Small-grid one-task timing is a sanity check, not a final runtime model.",
        ],
    }
    if timing_report is not None:
        rows = timing_report["rows"]
        ideal_spg = max(float(r["seconds_per_gridpoint"]) for r in rows if r["method"] == "ideal")
        noisy_spg = max(float(r["seconds_per_gridpoint"]) for r in rows if r["method"] == "noisy")
        zne_spg = max(float(r["seconds_per_gridpoint"]) for r in rows if r["method"] == "zne_cubic")
        serial_seconds = (
            int(official["ideal"]) * grid_points * ideal_spg
            + int(official["noisy"]) * grid_points * noisy_spg
            + int(official["zne"]) * grid_points * zne_spg
        )
        local_workers = max(1, int(os.cpu_count() or 1))
        estimates["small_grid_timing_extrapolation"] = {
            "ideal_seconds_per_gridpoint": ideal_spg,
            "noisy_seconds_per_gridpoint": noisy_spg,
            "zne_cubic_seconds_per_gridpoint": zne_spg,
            "serial_hours": serial_seconds / 3600.0,
            "local_workers": local_workers,
            "local_ideal_parallel_hours": serial_seconds / 3600.0 / local_workers,
            "local_75pct_parallel_hours": serial_seconds / 3600.0 / (local_workers * 0.75),
            "warning": "Likely pessimistic/noisy because it extrapolates tiny-grid timings.",
        }

    feasibility = _load_json_if_exists(root / "results" / "phase_c" / "feasibility_gate_default_threads_1.json")
    if feasibility:
        estimate = feasibility.get("estimate", {})
        source_grid_points = int(estimate.get("grid_points", 0) or 0)
        scale = (TARGET_GRID_SIZE * TARGET_GRID_SIZE / source_grid_points) if source_grid_points else math.nan
        estimates["existing_feasibility_default_threads_1_scaled_to_120"] = {
            "source_grid_points": source_grid_points,
            "scale_factor_grid_points": scale,
            "source_selected_total_hours_shot_scaled": estimate.get("selected_total_hours_shot_scaled"),
            "scaled_serial_hours": float(estimate.get("selected_total_hours_shot_scaled", math.nan)) * scale,
            "scaled_local_ideal_parallel_hours": float(estimate.get("local_ideal_parallel_hours_shot_scaled", math.nan)) * scale,
            "scaled_local_75pct_parallel_hours": float(estimate.get("local_75pct_parallel_hours_shot_scaled", math.nan)) * scale,
        }
    optimization = _load_json_if_exists(root / "results" / "phase_c" / "optimization_benchmark.json")
    if optimization:
        best = optimization.get("best_numerically_equivalent_candidate")
        if isinstance(best, dict):
            scale = (TARGET_GRID_SIZE * TARGET_GRID_SIZE) / (80 * 80)
            estimates["existing_optimization_best_numerically_equivalent_scaled_to_120"] = {
                "candidate": best.get("candidate"),
                "source_assumed_grid": 80,
                "scale_factor_grid_points": scale,
                "source_selected_total_hours_shot_scaled": best.get("selected_total_hours_shot_scaled"),
                "scaled_serial_hours": float(best.get("selected_total_hours_shot_scaled", math.nan)) * scale,
                "scaled_local_ideal_parallel_hours": float(best.get("local_ideal_parallel_hours_shot_scaled", math.nan)) * scale,
            }

    for key, value in list(estimates.items()):
        if isinstance(value, dict) and "scaled_local_75pct_parallel_hours" in value:
            hours = float(value["scaled_local_75pct_parallel_hours"])
            value["execution_days_75pct_parallel"] = hours / 24.0
            value["phase_c_120_execution_bucket"] = (
                "<=14_days" if hours <= 14 * 24 else
                "14_to_30_days" if hours <= 30 * 24 else
                "30_to_45_days" if hours <= 45 * 24 else
                ">45_days"
            )
    return estimates


def generate_storage_report(task_inventory: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    storage = task_inventory["storage_estimate_bytes"]
    d_free = int(environment["disk_usage"]["D"]["free_bytes"])
    c_free = int(environment["disk_usage"]["C"]["free_bytes"])
    conservative = int(storage["raw_no_scale_reuse"])
    with_metadata = int(conservative * 1.25)
    return {
        "created_at_utc": _utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "storage_estimate_bytes": storage,
        "conservative_with_25pct_metadata_bytes": with_metadata,
        "D_free_bytes": d_free,
        "C_free_bytes": c_free,
        "D_percent_of_free_for_conservative": with_metadata / d_free if d_free else math.inf,
        "C_percent_of_free_for_conservative": with_metadata / c_free if c_free else math.inf,
        "recommendation": "Use D: for execution; use C: only for small critical backups unless more space is freed.",
    }


def run_phase_c_120_preflight(
    timing_grid_size: int = 2,
    simulator_profile: str = "default_threads_1",
    skip_timing: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_c_120_planning")
    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    seeds_cfg = read_yaml(root / "config" / "seeds.yaml")

    environment = generate_execution_environment(root)
    write_json_atomic(output_dir / "execution_environment.json", environment)
    requirements_text = "\n".join(environment.get("pip_freeze", [])) + "\n"
    (output_dir / "requirements_phase_c.txt").write_text(requirements_text, encoding="utf-8")

    source_snapshot = generate_source_snapshot(root, output_dir)
    task_inventory = generate_task_inventory(cfg, seeds_cfg)
    write_json_atomic(output_dir / "task_inventory.json", task_inventory)
    circuit_report = generate_circuit_size_report(cfg)
    write_json_atomic(output_dir / "circuit_size_report.json", circuit_report)
    _write_csv(output_dir / "circuit_size_report.csv", circuit_report["rows"])

    timing_report = None
    if not skip_timing:
        timing_report = generate_one_task_timing_report(cfg, timing_grid_size, simulator_profile)
        write_json_atomic(output_dir / "one_task_timing_report.json", timing_report)
        _write_csv(output_dir / "one_task_timing_report.csv", timing_report["rows"])

    runtime = generate_back_of_envelope_runtime(task_inventory, timing_report, root)
    write_json_atomic(output_dir / "back_of_envelope_runtime.json", runtime)
    storage = generate_storage_report(task_inventory, environment)
    write_json_atomic(output_dir / "storage_report.json", storage)

    summary = {
        "created_at_utc": _utc_now(),
        "protocol_version": PROTOCOL_VERSION,
        "status": "phase_c_120_preflight_complete",
        "output_dir": output_dir.relative_to(root).as_posix(),
        "timing_grid_size": None if skip_timing else timing_grid_size,
        "simulator_profile": simulator_profile,
        "environment_hash": environment["environment_hash"],
        "source_tree_hash": source_snapshot["source_tree_hash"],
        "task_total": task_inventory["official_output_tasks"]["total"],
        "raw_scale_landscapes_no_reuse": task_inventory["raw_scale_landscapes"]["no_scale_reuse"],
        "next_recommended_action": "Review preflight reports, then run Phase 1 profiling; do not start official Phase C yet.",
        "outputs": {
            "execution_environment": "execution_environment.json",
            "source_tree_manifest": "source_tree_manifest.json",
            "source_snapshot": "source_snapshot_phase_c_120.zip",
            "task_inventory": "task_inventory.json",
            "circuit_size_report": "circuit_size_report.json",
            "one_task_timing_report": None if skip_timing else "one_task_timing_report.json",
            "back_of_envelope_runtime": "back_of_envelope_runtime.json",
            "storage_report": "storage_report.json",
        },
    }
    write_json_atomic(output_dir / "preflight_summary.json", summary)
    return summary
