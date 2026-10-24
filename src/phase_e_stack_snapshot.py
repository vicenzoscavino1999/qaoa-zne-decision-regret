from __future__ import annotations

import io
import os
import platform
import sys
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, write_json_atomic


PHASE_E_STACK_SNAPSHOT_VERSION = "phase_e_stack_snapshot_v0_1"
PACKAGE_NAMES = ("numpy", "scipy", "pandas", "matplotlib", "PyYAML", "qiskit", "qiskit-aer", "mitiq", "tqdm", "ply")
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MKL_CBWR", "NUMEXPR_NUM_THREADS")


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _numpy_config_text() -> str:
    stream = io.StringIO()
    try:
        np.show_config(mode="dicts")  # type: ignore[call-arg]
    except TypeError:
        try:
            old_stdout = sys.stdout
            sys.stdout = stream
            np.show_config()
        finally:
            sys.stdout = old_stdout
        return stream.getvalue()
    except Exception as exc:  # pragma: no cover - diagnostic path
        return repr(exc)

    try:
        old_stdout = sys.stdout
        sys.stdout = stream
        np.show_config()
    finally:
        sys.stdout = old_stdout
    return stream.getvalue()


def build_phase_e_stack_snapshot() -> dict[str, Any]:
    return {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "snapshot_version": PHASE_E_STACK_SNAPSHOT_VERSION,
        "status": "draft_not_frozen",
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": _package_versions(),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "prng_policy": {
            "numpy_generator": "numpy.random.Generator",
            "bit_generator": "PCG64",
            "seed_sequence_policy": "root_seed_to_child_streams_before_parallel_execution",
        },
        "numpy_config": _numpy_config_text(),
    }


def write_phase_e_stack_snapshot(root: Path = PROJECT_ROOT, overwrite: bool = True) -> dict[str, Any]:
    snapshot = build_phase_e_stack_snapshot()
    output_dir = ensure_dir(root / "results" / "phase_e" / "planning")
    write_json_atomic(output_dir / "phase_e_stack_snapshot.json", snapshot, overwrite=overwrite)
    requirements = "\n".join(
        f"{name}=={version}"
        for name, version in sorted(snapshot["packages"].items())
        if version is not None
    )
    (output_dir / "requirements_phase_e.txt").write_text(requirements + "\n", encoding="utf-8")
    return snapshot
