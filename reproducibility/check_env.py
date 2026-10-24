"""Environment fingerprint and drift checker for the reproducibility package."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT_PATH = ROOT / "reproducibility" / "config" / "environment_fingerprint.yaml"
FROZEN_ENV_PATH = ROOT / "reproducibility" / "reference" / "frozen_env.txt"

PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("PyYAML", "yaml"),
    ("qiskit", "qiskit"),
    ("qiskit-aer", "qiskit_aer"),
    ("mitiq", "mitiq"),
    ("tqdm", "tqdm"),
    ("ply", "ply"),
]

THREAD_VARS = [
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "PYTHONDONTWRITEBYTECODE",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_pip_freeze() -> str:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.replace("\r\n", "\n")


def package_status() -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for distribution, module in PACKAGES:
        row: dict[str, Any] = {"distribution": distribution, "module": module}
        try:
            importlib.import_module(module)
            row["import_ok"] = True
        except Exception as exc:  # pragma: no cover - diagnostic path
            row["import_ok"] = False
            row["error"] = f"{type(exc).__name__}: {exc}"
            missing.append(distribution)
        try:
            row["version"] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            row["version"] = None
            if distribution not in missing:
                missing.append(distribution)
        rows.append(row)
    return rows, missing


def blas_info() -> str:
    try:
        import numpy as np

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            np.__config__.show()
        text = buffer.getvalue().strip()
        return text or "unavailable"
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"unavailable: {type(exc).__name__}: {exc}"


def in_docker() -> bool:
    return Path("/.dockerenv").exists() or os.environ.get("container") == "docker"


def compare_frozen_env(current_freeze: str) -> dict[str, Any]:
    if not FROZEN_ENV_PATH.exists():
        return {
            "status": "WARN",
            "severity": "WARN",
            "message": "frozen_env.txt is not present yet",
        }
    expected = FROZEN_ENV_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    if expected == current_freeze:
        return {"status": "PASS", "severity": "BLOCK", "message": "pip freeze matches frozen_env.txt"}
    return {
        "status": "WARN",
        "severity": "WARN",
        "message": "pip freeze differs from frozen_env.txt during development",
        "current_sha256": hashlib.sha256(current_freeze.encode("utf-8")).hexdigest(),
        "expected_sha256": hashlib.sha256(expected.encode("utf-8")).hexdigest(),
    }


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def write_fingerprint(payload: dict[str, Any], current_freeze: str) -> None:
    FINGERPRINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_ENV_PATH.write_text(current_freeze, encoding="utf-8")

    lines = [
        "version: 1",
        f"generated_utc: {yaml_scalar(payload['timestamp_utc'])}",
        f"python_executable: {yaml_scalar(payload['python']['executable'])}",
        f"python_version: {yaml_scalar(payload['python']['version'])}",
        f"platform: {yaml_scalar(payload['platform']['platform'])}",
        f"machine: {yaml_scalar(payload['platform']['machine'])}",
        f"in_docker: {yaml_scalar(payload['platform']['in_docker'])}",
        f"pip_freeze_sha256: {yaml_scalar(payload['pip_freeze_sha256'])}",
        "thread_environment:",
    ]
    for name, value in payload["thread_environment"].items():
        lines.append(f"  {name}: {yaml_scalar(value)}")
    lines.append("packages:")
    for row in payload["packages"]:
        lines.append(f"  - distribution: {yaml_scalar(row['distribution'])}")
        lines.append(f"    module: {yaml_scalar(row['module'])}")
        lines.append(f"    import_ok: {yaml_scalar(row['import_ok'])}")
        lines.append(f"    version: {yaml_scalar(row.get('version'))}")
    lines.append("blas_backend_note: |")
    blas_text = str(payload["blas_backend"]).splitlines() or ["unavailable"]
    for line in blas_text:
        lines.append(f"  {line}")
    FINGERPRINT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check and optionally fingerprint the Python environment.")
    parser.add_argument("--write-fingerprint", action="store_true", help="Write environment_fingerprint.yaml and frozen_env.txt")
    args = parser.parse_args()

    current_freeze = run_pip_freeze()
    freeze_hash = hashlib.sha256(current_freeze.encode("utf-8")).hexdigest()
    packages, missing = package_status()

    payload: dict[str, Any] = {
        "timestamp_utc": utc_now(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "platform": {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "in_docker": in_docker(),
        },
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARS},
        "packages": packages,
        "blas_backend": blas_info(),
        "pip_freeze_sha256": f"sha256:{freeze_hash}",
        "checks": {},
    }

    if args.write_fingerprint:
        write_fingerprint(payload, current_freeze)
        payload["checks"]["frozen_env"] = {
            "status": "PASS",
            "severity": "WARN",
            "message": "fingerprint files written",
        }
    else:
        payload["checks"]["frozen_env"] = compare_frozen_env(current_freeze)

    if missing:
        payload["checks"]["required_imports"] = {
            "status": "FAIL",
            "severity": "BLOCK",
            "missing": missing,
        }
        payload["overall"] = "FAIL"
    else:
        payload["checks"]["required_imports"] = {
            "status": "PASS",
            "severity": "BLOCK",
            "missing": [],
        }
        has_warning = any(check.get("status") == "WARN" for check in payload["checks"].values())
        payload["overall"] = "PASS_WITH_WARNINGS" if has_warning else "PASS"

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if payload["overall"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
