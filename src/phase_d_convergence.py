from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .phase_d_analysis import GLOBAL_SUMMARY
from .phase_d_completion import GLOBAL_RESULT_DIR


PHASE_D_CONVERGENCE_VERSION = "phase_d_convergence_v1"
CONVERGENCE_DIR = "convergence"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _result_dir(root: Path) -> Path:
    return ensure_dir(root / "results" / "phase_d" / CONVERGENCE_DIR)


def _hashes(root: Path, rels: list[str]) -> dict[str, str | None]:
    return {rel: sha256_file(root / rel) if (root / rel).is_file() else None for rel in rels}


def run_phase_d_convergence(overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    phase_c_path = root / "results" / "phase_c_120_analysis" / "paper_go_no_go_summary.json"
    phase_d_path = root / "results" / "phase_d" / GLOBAL_RESULT_DIR / GLOBAL_SUMMARY
    if not phase_c_path.is_file() or not phase_d_path.is_file():
        raise FileNotFoundError("Run Phase C summary and Phase D global analysis before convergence.")
    result_dir = _result_dir(root)
    json_path = result_dir / "phase_d_convergence_assessment.json"
    md_path = result_dir / "phase_d_convergence_assessment.md"
    if not overwrite:
        for path in (json_path, md_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing convergence output: {path}")
    phase_c = _load_json(phase_c_path)
    phase_d = _load_json(phase_d_path)
    phase_c_positive = phase_c.get("paper_go_strong") is True
    phase_d_label = str(phase_d.get("assigned_label", "unknown"))
    if phase_c_positive and phase_d_label in {"strong_independent_replication", "moderate_independent_replication"}:
        outcome = "convergent_support"
    elif phase_c_positive and phase_d_label == "weak_independent_replication":
        outcome = "partial_convergence"
    elif phase_c_positive and phase_d_label == "not_supported_as_independent_replication":
        outcome = "discordant_evidence"
    else:
        outcome = "inconclusive_replication"
    report = {
        "convergence_version": PHASE_D_CONVERGENCE_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_d_convergence_assessment_complete",
        "outcome": outcome,
        "method": "narrative_comparison_no_combined_fdr",
        "phase_c": {
            "source": phase_c_path.relative_to(root).as_posix(),
            "paper_go_strong": phase_c.get("paper_go_strong"),
            "fdr_family_size": phase_c.get("fdr_family_size"),
            "fdr_q": phase_c.get("fdr_q"),
            "discordant_fdr_count": phase_c.get("discordant_fdr_count"),
        },
        "phase_d": {
            "source": phase_d_path.relative_to(root).as_posix(),
            "assigned_label": phase_d_label,
            "fdr_family_size": phase_d.get("fdr_family_size"),
            "primary_q": phase_d.get("primary_q"),
            "cell_level_primary_endpoint_pass": phase_d.get("cell_level_primary_endpoint_pass"),
            "instance_level_primary_endpoint_pass": phase_d.get("instance_level_primary_endpoint_pass"),
            "discordant_fdr_count": phase_d.get("discordant_fdr_count"),
        },
        "no_combined_fdr_computed": True,
    }
    write_json_atomic(json_path, report, overwrite=True)
    md_path.write_text(
        "\n".join([
            "# Phase C vs Phase D Convergence Assessment",
            "",
            f"Outcome: `{outcome}`",
            "",
            "This assessment compares the frozen Phase C conclusion with the independent Phase D conclusion narratively.",
            "It does not compute a combined Phase C + Phase D FDR family.",
            "",
        ]),
        encoding="utf-8",
    )
    report["output_hashes"] = _hashes(root, [json_path.relative_to(root).as_posix(), md_path.relative_to(root).as_posix()])
    write_json_atomic(json_path, report, overwrite=True)
    return report


def write_phase_d_convergence_preflight(overwrite: bool = True, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    planning_dir = ensure_dir(root / "results" / "phase_d" / "planning")
    required_sources = [
        "src/phase_d_convergence.py",
        "scripts/38_phase_d_convergence.py",
        "src/phase_d_analysis.py",
        "scripts/36_phase_d_analysis.py",
    ]
    checks = {
        "convergence_sources_exist": all((root / rel).is_file() for rel in required_sources),
        "phase_c_summary_available": (root / "results" / "phase_c_120_analysis" / "paper_go_no_go_summary.json").is_file(),
        "method_is_narrative_not_combined_fdr": True,
        "phase_d_global_input_contract": (root / "results" / "phase_d" / GLOBAL_RESULT_DIR / GLOBAL_SUMMARY).relative_to(root).as_posix() == f"results/phase_d/{GLOBAL_RESULT_DIR}/{GLOBAL_SUMMARY}",
    }
    report = {
        "convergence_version": PHASE_D_CONVERGENCE_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_d_convergence_preflight_passed" if all(checks.values()) else "phase_d_convergence_preflight_failed",
        "decision": "convergence_script_ready_for_phase_d_v1_0_freeze" if all(checks.values()) else "block_phase_d_v1_0_freeze_until_convergence_preflight_passes",
        "scope": "pre_official_execution_convergence_script_readiness",
        "checks": checks,
        "post_execution_output_contract": [
            f"results/phase_d/{CONVERGENCE_DIR}/phase_d_convergence_assessment.json",
            f"results/phase_d/{CONVERGENCE_DIR}/phase_d_convergence_assessment.md",
        ],
        "source_hashes": _hashes(root, required_sources),
    }
    output = planning_dir / "phase_d_convergence_preflight.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Phase D convergence preflight: {output}")
    write_json_atomic(output, report, overwrite=True)
    return report
