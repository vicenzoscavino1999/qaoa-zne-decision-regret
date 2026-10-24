from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .phase_c_120_custom_kernel import CUSTOM_KERNEL_VERSION
from .phase_c_120_preflight import PROTOCOL_VERSION


AMENDMENT_VERSION = "phase_c_120_amendment_v1"
AMENDMENT_ID = "phase_c_120_kernel_v2_pre_official"


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _run_text(args: list[str], root: Path) -> str:
    try:
        proc = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=30, check=False)
    except Exception as exc:
        return f"ERROR: {exc!r}"
    return (proc.stdout or proc.stderr or "").strip()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _official_data_state(root: Path) -> dict[str, Any]:
    raw_dir = root / "data" / "raw" / "phase_c_120_kernel_official"
    arrays = sorted(raw_dir.rglob("*.npy")) if raw_dir.exists() else []
    done = sorted(raw_dir.rglob("*.done.json")) if raw_dir.exists() else []
    return {
        "official_raw_dir": raw_dir.relative_to(root).as_posix(),
        "official_raw_dir_exists": raw_dir.exists(),
        "official_npy_count": len(arrays),
        "official_done_count": len(done),
        "official_data_generated_before_amendment": bool(arrays or done),
    }


def run_phase_c_120_kernel_v2_amendment(overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    result_dir = ensure_dir(root / "results" / "phase_c_120_amendments")
    json_path = result_dir / f"{AMENDMENT_ID}.json"
    md_path = result_dir / f"{AMENDMENT_ID}.md"
    if json_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing amendment: {json_path}")

    validation_path = root / "results" / "phase_c_120_custom_kernel" / "custom_kernel_validation_summary.json"
    timed_probe_path = root / "results" / "phase_c_120_timed_probe" / "timed_probe_summary.json"
    official_estimate_path = root / "results" / "phase_c_120_kernel_official" / "runtime_estimate.json"
    validation = _load_json(validation_path)
    timed_probe = _load_json(timed_probe_path)
    official_estimate = _load_json(official_estimate_path)
    official_state = _official_data_state(root)
    validation_passed = (
        validation.get("status") == "custom_kernel_validation_passed"
        and validation.get("custom_kernel_version") == CUSTOM_KERNEL_VERSION
        and int(validation.get("probability_failures", -1)) == 0
        and int(validation.get("observable_failures", -1)) == 0
    )
    timed_probe_passed = (
        timed_probe.get("status") == "phase_c_120_timed_probe_passed"
        and timed_probe.get("custom_kernel_version") == CUSTOM_KERNEL_VERSION
        and timed_probe.get("passes_72h_gate") is True
    )
    status = "phase_c_120_kernel_v2_amendment_accepted" if validation_passed and timed_probe_passed and not official_state["official_data_generated_before_amendment"] else "phase_c_120_kernel_v2_amendment_blocked"
    report = {
        "created_at_utc": _now_utc(),
        "amendment_version": AMENDMENT_VERSION,
        "amendment_id": AMENDMENT_ID,
        "protocol_version_before": PROTOCOL_VERSION,
        "protocol_version_after": f"{PROTOCOL_VERSION}+amendment_kernel_v2",
        "status": status,
        "classification": "pre_official_operational_amendment",
        "restart_required": False,
        "reason": "Timed 120x120 throughput probe showed the validated v1 kernel exceeded the 72h gate; v2 keeps the same numerical semantics while removing avoidable Python/transpile overhead.",
        "scientific_parameters_changed": [],
        "numerical_protocol_changed": "implementation_only_no_change_to_noise_folding_seed_shots_grid_metrics_or_go_no_go_rules",
        "custom_kernel_version_after": CUSTOM_KERNEL_VERSION,
        "official_data_state": official_state,
        "validation": {
            "summary_path": validation_path.relative_to(root).as_posix(),
            "summary_sha256": sha256_file(validation_path) if validation_path.is_file() else None,
            "status": validation.get("status"),
            "rows": validation.get("rows"),
            "probability_failures": validation.get("probability_failures"),
            "observable_failures": validation.get("observable_failures"),
            "max_abs_probability_diff": validation.get("max_abs_probability_diff"),
            "max_abs_expected_cut_diff": validation.get("max_abs_expected_cut_diff"),
        },
        "timed_probe": {
            "summary_path": timed_probe_path.relative_to(root).as_posix(),
            "summary_sha256": sha256_file(timed_probe_path) if timed_probe_path.is_file() else None,
            "status": timed_probe.get("status"),
            "projected_local_75pct_parallel_hours": timed_probe.get("projected_local_75pct_parallel_hours"),
            "passes_72h_gate": timed_probe.get("passes_72h_gate"),
        },
        "official_runtime_estimate": {
            "summary_path": official_estimate_path.relative_to(root).as_posix(),
            "summary_sha256": sha256_file(official_estimate_path) if official_estimate_path.is_file() else None,
            "basis": official_estimate.get("basis"),
            "local_75pct_parallel_hours": official_estimate.get("local_75pct_parallel_hours"),
            "decision": official_estimate.get("decision"),
        },
        "git": {
            "branch": _run_text(["git", "branch", "--show-current"], root),
            "commit": _run_text(["git", "rev-parse", "HEAD"], root),
            "status_short_at_amendment": _run_text(["git", "status", "--short"], root),
        },
        "decision": "amendment_valid_before_phase_9" if status.endswith("accepted") else "do_not_start_phase_9_until_amendment_is_resolved",
    }
    write_json_atomic(json_path, report, overwrite=True)
    md = (
        f"# {AMENDMENT_ID}\n\n"
        f"Status: `{status}`\n\n"
        "This is a pre-official operational amendment. No official Phase C 120x120 arrays existed before the change.\n\n"
        "Scientific parameters changed: none.\n\n"
        f"Validation: `{validation.get('rows')}` Aer-equivalence cases, "
        f"`{validation.get('probability_failures')}` probability failures, "
        f"`{validation.get('observable_failures')}` observable failures.\n\n"
        f"Timed probe projection after amendment: `{timed_probe.get('projected_local_75pct_parallel_hours')}` local 75% parallel hours.\n"
    )
    md_path.write_text(md, encoding="utf-8")
    return report
