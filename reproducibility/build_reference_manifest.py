"""Build the reference manifest for selected frozen outputs."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reproducibility" / "reference" / "reference_manifest.json"

REFERENCE_FILES = [
    ("results/phase_a_smoke_summary.json", "A", "Phase A smoke-test summary."),
    ("results/phase_b_debug/phase_b_summary.json", "B", "Phase B debug-pilot summary."),
    ("results/phase_b_debug/phase_b_analysis.json", "B", "Phase B analysis and calibration summary."),
    ("results/phase_b_debug/seed_selection_table.csv", "B", "Phase B seed-selection table."),
    ("results/phase_c_120_analysis/classification_matrix.csv", "C", "Phase C 120-grid discovery classification matrix."),
    ("results/phase_c_120_analysis/statistics_summary.json", "C", "Phase C discovery statistics summary."),
    ("results/phase_c_120_analysis/sensitivity_summary.json", "C", "Phase C sensitivity summary."),
    ("results/phase_d/global_D1_D2_D3_D4/fdr_global_540_summary.json", "D", "Phase D global 540-cell FDR summary."),
    ("results/phase_d/global_D1_D2_D3_D4/classification_matrix.csv", "D", "Phase D independent-replication classification matrix."),
    ("results/phase_d/global_D1_D2_D3_D4/instance_primary_endpoint.json", "D", "Phase D instance-level primary endpoint."),
    ("results/phase_d/global_D1_D2_D3_D4/instance_primary_endpoint.csv", "D", "Phase D instance-level endpoint table."),
    ("results/release_checks/phase9_5_numerical_reproducibility.json", "release", "Fast numerical reproducibility verification report."),
    ("NUMERICAL_REPRODUCIBILITY_PHASE9_5.md", "release", "Numerical reproducibility subset documentation."),
    ("config/config_official_pilot_v1.6.yaml", "release", "Official through-Phase-D configuration."),
    ("config/seeds.yaml", "release", "Official seed registry input."),
    ("config/phase_d_instances_v1.0.json", "D", "Frozen generated Phase D instance definitions."),
    ("scripts/verify_numerical_reproducibility_subset.py", "release", "Fast numerical reproducibility subset verifier."),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> dict[str, object]:
    reference_outputs: dict[str, dict[str, object]] = {}
    total_size = 0
    for rel, phase, description in REFERENCE_FILES:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(rel)
        size = path.stat().st_size
        total_size += size
        reference_outputs[rel] = {
            "sha256": sha256_file(path),
            "bytes": size,
            "phase": phase,
            "description": description,
        }
    return {
        "manifest_version": "1.0.0",
        "manifest_schema_version": "1",
        "scope": "through_phase_d_with_auxiliary_diagnostics",
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "baseline_commit": "PENDING_UNTIL_CLEAN_SCOPE_COMMIT",
        "generation_platform": "Windows 11 / Python 3.12.4",
        "total_files": len(reference_outputs),
        "total_size_bytes": total_size,
        "excluded_from_scope": [
            "phase_e",
            "phase_f",
            "real_hardware_pending",
            "extra_experiment_pending",
        ],
        "reference_outputs": reference_outputs,
    }


def main() -> int:
    manifest = build_manifest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"reference_manifest={OUTPUT.relative_to(ROOT).as_posix()}")
    print(f"total_files={manifest['total_files']}")
    print(f"total_size_bytes={manifest['total_size_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
