"""Progressive reproducibility verifier for the through-Phase-D package.

The verifier supports fast local and CI checks, and reports unavailable large
artifacts as explicit warnings in environments that intentionally omit them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reproducibility" / "reports"
BASELINE = ROOT / "reproducibility" / "config" / "baseline_commit.yaml"
RELEASE_MANIFEST = ROOT / "RELEASE_MANIFEST_THROUGH_PHASE_D.json"
REFERENCE_MANIFEST = ROOT / "reproducibility" / "reference" / "reference_manifest.json"
PAPER_MAPPING = ROOT / "reproducibility" / "config" / "paper_mapping.yaml"
SUBSET_SPEC = ROOT / "reproducibility" / "config" / "subset_specification.yaml"
SCHEMAS = ROOT / "reproducibility" / "config" / "schemas.yaml"
TOLERANCES = ROOT / "reproducibility" / "config" / "tolerances.yaml"
NUMERICAL_SUBSET_SCRIPT = ROOT / "scripts" / "verify_numerical_reproducibility_subset.py"
PHASE_EF_RE = re.compile(r"phase_[ef]|PHASE_[EF]|Phase [EF]")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def baseline_manifest_stale() -> bool:
    return "manifest_status: stale_until_regenerated" in read_text(BASELINE)


def in_docker() -> bool:
    return Path("/.dockerenv").exists()


def allow_missing_large_artifacts() -> bool:
    return os.environ.get("REPRO_CI_ALLOW_MISSING_ARTIFACTS", "").strip().lower() in {"1", "true", "yes"}


def is_large_artifact_path(rel: str) -> bool:
    return rel.startswith(("data/", "results/"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_scope_guard() -> dict[str, Any]:
    matches: list[str] = []
    if RELEASE_MANIFEST.exists():
        try:
            payload = json.loads(RELEASE_MANIFEST.read_text(encoding="utf-8"))
            entries = payload.get("files", payload) if isinstance(payload, dict) else payload
            for entry in entries:
                if isinstance(entry, dict):
                    path = str(entry.get("path", ""))
                else:
                    path = str(entry)
                if PHASE_EF_RE.search(path):
                    matches.append(path)
        except Exception as exc:
            return {
                "status": "WARN",
                "severity": "WARN",
                "message": f"could not parse release manifest for scope guard: {type(exc).__name__}: {exc}",
            }
    return {
        "status": "PASS" if not matches else "FAIL",
        "severity": "BLOCK",
        "phase_ef_absent": not matches,
        "matches": matches[:20],
        "match_count": len(matches),
    }


def check_release_manifest() -> dict[str, Any]:
    if not RELEASE_MANIFEST.exists():
        return {"status": "WARN", "severity": "WARN", "message": "release manifest not present"}
    if baseline_manifest_stale():
        return {
            "status": "WARN",
            "severity": "WARN",
            "message": "release manifest intentionally stale until scope freeze",
            "manifest": RELEASE_MANIFEST.name,
        }
    result = subprocess.run(
        [sys.executable, "scripts/verify_release_manifest.py", str(RELEASE_MANIFEST)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "status": "PASS" if result.returncode == 0 else "FAIL",
        "severity": "BLOCK",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def check_reference_manifest() -> dict[str, Any]:
    if not REFERENCE_MANIFEST.exists():
        return {
            "status": "WARN",
            "severity": "WARN",
            "message": "reference_manifest.json is not available in this environment",
        }
    try:
        payload = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "FAIL", "severity": "BLOCK", "message": f"invalid reference manifest: {exc}"}
    outputs = payload.get("reference_outputs", {})
    missing: list[str] = []
    hash_failures: list[dict[str, str]] = []
    byte_failures: list[dict[str, Any]] = []
    checked = 0
    for rel, expected in outputs.items():
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
            continue
        checked += 1
        if isinstance(expected, str):
            expected_hash = expected.removeprefix("sha256:")
            expected_bytes = None
        elif isinstance(expected, dict):
            expected_hash = str(expected.get("sha256", "")).removeprefix("sha256:")
            expected_bytes = expected.get("bytes")
        else:
            hash_failures.append({"path": rel, "reason": "invalid manifest entry"})
            continue
        actual_hash = sha256_file(path)
        if expected_hash and actual_hash != expected_hash:
            hash_failures.append({"path": rel, "expected": expected_hash, "actual": actual_hash})
        if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
            byte_failures.append({"path": rel, "expected": expected_bytes, "actual": path.stat().st_size})
    if missing and (in_docker() or allow_missing_large_artifacts()) and all(is_large_artifact_path(path) for path in missing):
        return {
            "status": "WARN",
            "severity": "WARN",
            "message": "large reference outputs are not bundled in this environment; use local checkout, mounted data/results, or future verification bundle",
            "declared": len(outputs),
            "missing_count": len(missing),
            "missing_sample": missing[:10],
        }
    failed = bool(missing or hash_failures or byte_failures)
    return {
        "status": "FAIL" if failed else "PASS",
        "severity": "BLOCK",
        "checked": checked,
        "declared": len(outputs),
        "missing": missing,
        "hash_failures": hash_failures,
        "byte_failures": byte_failures,
    }


def check_schema_and_tolerance_files() -> dict[str, Any]:
    missing = [path.relative_to(ROOT).as_posix() for path in (SCHEMAS, TOLERANCES) if not path.exists()]
    if missing:
        return {
            "status": "WARN",
            "severity": "WARN",
            "message": "schema/tolerance files are not complete yet",
            "missing": missing,
        }
    text = TOLERANCES.read_text(encoding="utf-8", errors="replace")
    has_cross_platform = "cross_platform_note" in text and "Windows" in text and "Linux" in text
    return {
        "status": "PASS" if has_cross_platform else "WARN",
        "severity": "WARN" if has_cross_platform else "BLOCK",
        "schemas_present": True,
        "tolerances_present": True,
        "cross_platform_note_present": has_cross_platform,
    }


def check_paper_mapping() -> dict[str, Any]:
    if not PAPER_MAPPING.exists():
        return {
            "status": "WARN",
            "severity": "WARN",
            "message": "paper_mapping.yaml is not available in this environment",
        }
    text = PAPER_MAPPING.read_text(encoding="utf-8")
    try:
        payload = yaml.safe_load(text) or {}
    except Exception as exc:
        return {"status": "FAIL", "severity": "BLOCK", "message": f"invalid paper_mapping.yaml: {exc}"}
    missing: list[str] = []
    checked_outputs = 0
    checked_generators = 0
    checked_dependencies = 0

    def check_path(value: str | None, label: str) -> None:
        nonlocal checked_outputs
        if not value or value in {"manual_static_asset", "not_applicable"}:
            return
        checked_outputs += 1
        if not (ROOT / value).exists():
            missing.append(f"{label}:{value}")

    def check_generator(value: Any, label: str) -> None:
        nonlocal checked_generators
        if value in (None, "manual_static_asset", "not_applicable"):
            return
        values = value if isinstance(value, list) else [value]
        for item in values:
            checked_generators += 1
            if not (ROOT / str(item)).exists():
                missing.append(f"{label}:{item}")

    def check_dependency(value: Any, label: str) -> None:
        nonlocal checked_dependencies
        if value in (None, "not_applicable"):
            return
        values = value if isinstance(value, list) else [value]
        for item in values:
            checked_dependencies += 1
            if not (ROOT / str(item)).exists():
                missing.append(f"{label}:{item}")

    for section in ("figures", "tables", "supplements"):
        for entry in payload.get(section, []) or []:
            entry_id = str(entry.get("id", section))
            check_path(entry.get("output") or entry.get("source"), f"{entry_id}.output")
            for output in entry.get("outputs", []) or []:
                check_path(str(output), f"{entry_id}.output")
            check_generator(entry.get("generator") or entry.get("generators"), f"{entry_id}.generator")
            check_dependency(entry.get("data_dependency") or entry.get("data_dependencies"), f"{entry_id}.dependency")

    phase_ef_absent = not bool(PHASE_EF_RE.search(text))
    if missing and (in_docker() or allow_missing_large_artifacts()) and all(":results/" in item or ":data/" in item for item in missing):
        return {
            "status": "WARN",
            "severity": "WARN",
            "message": "paper-mapping data dependencies are not bundled in this environment",
            "phase_ef_absent": phase_ef_absent,
            "checked_outputs": checked_outputs,
            "checked_generators": checked_generators,
            "checked_dependencies": checked_dependencies,
            "missing_count": len(missing),
            "missing_sample": missing[:10],
        }
    failed = bool(missing) or not phase_ef_absent
    return {
        "status": "FAIL" if failed else "PASS",
        "severity": "BLOCK",
        "phase_ef_absent": phase_ef_absent,
        "checked_outputs": checked_outputs,
        "checked_generators": checked_generators,
        "checked_dependencies": checked_dependencies,
        "missing": missing,
    }


def check_numerical_subset() -> dict[str, Any]:
    if not SUBSET_SPEC.exists():
        return {
            "status": "WARN",
            "severity": "WARN",
            "message": "subset_specification.yaml is not available in this environment",
        }
    if not NUMERICAL_SUBSET_SCRIPT.exists():
        return {
            "status": "WARN",
            "severity": "WARN",
            "message": "numerical subset script is not present",
        }
    required_data = [
        ROOT / "results" / "phase_a_smoke_summary.json",
        ROOT / "results" / "phase_c_120_analysis" / "classification_matrix.csv",
        ROOT / "results" / "phase_d" / "global_D1_D2_D3_D4" / "classification_matrix.csv",
        ROOT / "data" / "raw" / "G1_C4" / "C_ideal.npy",
    ]
    missing = [path.relative_to(ROOT).as_posix() for path in required_data if not path.exists()]
    if missing:
        return {
            "status": "WARN",
            "severity": "WARN",
            "message": "numerical subset requires data/results not present in this environment",
            "missing": missing,
        }
    result = subprocess.run(
        [sys.executable, str(NUMERICAL_SUBSET_SCRIPT)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    passed = result.returncode == 0 and "overall_pass=true" in result.stdout
    return {
        "status": "PASS" if passed else "FAIL",
        "severity": "BLOCK",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def report(mode: str, results: dict[str, Any]) -> dict[str, Any]:
    statuses = [item.get("status") for item in results.values() if isinstance(item, dict)]
    if "FAIL" in statuses:
        overall = "FAIL"
    elif "WARN" in statuses:
        overall = "PASS_WITH_WARNINGS"
    else:
        overall = "PASS"
    return {
        "verify_version": "0.5.0-fdr-audit",
        "timestamp_utc": utc_now(),
        "mode": mode,
        "results": results,
        "overall": overall,
    }


def write_report(payload: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    latest = REPORT_DIR / "last_verify_report.json"
    stamp = payload["timestamp_utc"].replace("-", "").replace(":", "").replace("Z", "Z")
    stamped = REPORT_DIR / f"verify_report_{stamp}.json"
    text = json.dumps(payload, indent=2, sort_keys=True)
    latest.write_text(text + "\n", encoding="utf-8")
    stamped.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Progressive reproducibility verifier.")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--release-manifest", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--paper-mapping", action="store_true")
    parser.add_argument("--numerical-subset", action="store_true")
    parser.add_argument("--clean-cache", action="store_true")
    args = parser.parse_args()

    if args.clean_cache:
        for path in REPORT_DIR.glob("verify_report_*.json"):
            path.unlink()
        print("clean_cache_status=passed")
        return 0

    results: dict[str, Any] = {"scope_guard": check_scope_guard()}
    mode = "fast"
    if args.release_manifest:
        mode = "release-manifest"
        results["release_manifest"] = check_release_manifest()
    elif args.reference_only:
        mode = "reference-only"
        results["reference_manifest"] = check_reference_manifest()
    elif args.paper_mapping:
        mode = "paper-mapping"
        results["paper_mapping"] = check_paper_mapping()
    elif args.numerical_subset:
        mode = "numerical-subset"
        results["numerical_subset"] = check_numerical_subset()
    else:
        results["release_manifest"] = check_release_manifest()
        results["reference_manifest"] = check_reference_manifest()
        results["schema_tolerance_files"] = check_schema_and_tolerance_files()
        results["paper_mapping"] = check_paper_mapping()
        results["numerical_subset"] = check_numerical_subset()

    payload = report(mode, results)
    write_report(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if payload["overall"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
