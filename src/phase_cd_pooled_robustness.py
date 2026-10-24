from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .phase_c_120_analysis import _as_float, _as_int, _benjamini_hochberg, _read_csv, _write_csv
from .phase_d_analysis import CONTINUITY_Q, PRIMARY_Q, _by_adjust, _holm_adjust


PHASE_CD_POOLED_ROBUSTNESS_VERSION = "phase_cd_pooled_15_robustness_v1"
POOLED_DIR = "phase_cd_pooled_15"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _result_dir(root: Path) -> Path:
    return ensure_dir(root / "results" / "phase_d" / "robustness" / POOLED_DIR)


def _hashes(root: Path, rels: list[str]) -> dict[str, str | None]:
    return {rel: sha256_file(root / rel) if (root / rel).is_file() else None for rel in rels}


def _source_paths(root: Path) -> dict[str, Path]:
    return {
        "phase_c_classification": root / "results" / "phase_c_120_analysis" / "classification_matrix.csv",
        "phase_c_summary": root / "results" / "phase_c_120_analysis" / "paper_go_no_go_summary.json",
        "phase_d_classification": root / "results" / "phase_d" / "global_D1_D2_D3_D4" / "classification_matrix.csv",
        "phase_d_summary": root / "results" / "phase_d" / "global_D1_D2_D3_D4" / "fdr_global_540_summary.json",
        "phase_d_convergence": root / "results" / "phase_d" / "convergence" / "phase_d_convergence_assessment.json",
    }


def _normalize_row(row: dict[str, Any], source_phase: str) -> dict[str, Any]:
    source_family = "phase_c_135" if source_phase == "PhaseC" else "phase_d_540"
    cell_id = str(row["cell_id"])
    block_id = "PhaseC" if source_phase == "PhaseC" else str(row.get("block_id", ""))
    excluded = (
        _bool(row.get("excluded", False))
        or _bool(row.get("excluded_below_detection_floor", False))
        or _bool(row.get("excluded_extrapolation_breakdown", False))
    )
    out = {
        "source_phase": source_phase,
        "source_family": source_family,
        "source_cell_id": cell_id,
        "pooled_cell_id": f"{source_phase}:{cell_id}",
        "block_id": block_id,
        "instance": str(row["instance"]),
        "noise": _as_float(row["noise"]),
        "shots_total": _as_int(row["shots_total"]),
        "zne_order": str(row["zne_order"]),
        "allocation": str(row.get("allocation", "proportional_to_abs_weights")),
        "n_seeds": _as_int(row["n_seeds"]),
        "median_r_mse": _as_float(row["median_r_mse"]),
        "median_d_R": _as_float(row["median_d_R"]),
        "ci_low_r_mse": _as_float(row["ci_low_r_mse"]),
        "ci_low_d_R": _as_float(row["ci_low_d_R"]),
        "prob_d_R_positive_seed": _as_float(row["prob_d_R_positive_seed"]),
        "p_conjunction": _as_float(row["p_conjunction"]),
        "excluded": excluded,
        "excluded_below_detection_floor": _bool(row.get("excluded_below_detection_floor", False)),
        "excluded_extrapolation_breakdown": _bool(row.get("excluded_extrapolation_breakdown", False)),
        "source_discordant_pre_fdr": _bool(row.get("discordant_pre_fdr", False)),
        "source_discordant_fdr": _bool(row.get("discordant_fdr", False)),
        "source_classification": str(row.get("classification", "")),
    }
    out["cell_positive_direction"] = bool(out["median_r_mse"] > 0.0 and out["median_d_R"] > 0.0)
    return out


def _criterion_style(rows: list[dict[str, Any]], fdr_key: str) -> bool:
    pre = [row for row in rows if _bool(row.get("pooled_discordant_pre_fdr"))]
    fdr = [row for row in rows if _bool(row.get(fdr_key))]
    spread = bool(
        len({str(row["instance"]) for row in pre}) > 1
        and (len({_as_float(row["noise"]) for row in pre}) > 1 or len({_as_int(row["shots_total"]) for row in pre}) > 1)
    )
    return len(pre) >= 5 and len(fdr) >= 3 and spread


def _classify_pooled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        pre = bool(row["source_discordant_pre_fdr"])
        row["pooled_discordant_pre_fdr"] = pre
        row["pooled_discordant_fdr_q05"] = bool(pre and row["fdr_pooled_bh_q05"])
        row["pooled_discordant_fdr_q10"] = bool(pre and row["fdr_pooled_bh_q10"])
        row["pooled_discordant_fdr_holm"] = bool(pre and row["fdr_pooled_holm_q05"])
        row["pooled_discordant_fdr_by"] = bool(pre and row["fdr_pooled_by_q05"])
        if row["excluded_below_detection_floor"]:
            label = "excluded_below_detection_floor"
        elif row["excluded_extrapolation_breakdown"]:
            label = "excluded_extrapolation_breakdown"
        elif row["pooled_discordant_fdr_q05"]:
            label = "pooled_discordant_fdr_q05"
        elif row["pooled_discordant_fdr_q10"]:
            label = "pooled_discordant_fdr_q10_only"
        elif pre:
            label = "pooled_discordant_not_fdr"
        else:
            label = "not_discordant"
        row["pooled_classification"] = label
    return rows


def _count(rows: list[dict[str, Any]], key: str) -> int:
    return sum(_bool(row.get(key)) for row in rows)


def _by_instance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for instance in sorted({str(row["instance"]) for row in rows}, key=lambda value: (value[0], int(value[1:]) if value[1:].isdigit() else value)):
        subset = [row for row in rows if str(row["instance"]) == instance]
        out.append({
            "instance": instance,
            "source_phase": subset[0]["source_phase"],
            "block_id": subset[0]["block_id"],
            "cells": len(subset),
            "pooled_discordant_pre_fdr_count": _count(subset, "pooled_discordant_pre_fdr"),
            "pooled_discordant_fdr_q05_count": _count(subset, "pooled_discordant_fdr_q05"),
            "pooled_discordant_fdr_q10_count": _count(subset, "pooled_discordant_fdr_q10"),
            "source_discordant_fdr_count": _count(subset, "source_discordant_fdr"),
            "cell_positive_direction_count": _count(subset, "cell_positive_direction"),
        })
    return out


def _by_source(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for source in sorted({str(row["source_phase"]) for row in rows}):
        subset = [row for row in rows if str(row["source_phase"]) == source]
        out[source] = {
            "cells": len(subset),
            "instances": len({str(row["instance"]) for row in subset}),
            "pooled_discordant_pre_fdr_count": _count(subset, "pooled_discordant_pre_fdr"),
            "pooled_discordant_fdr_q05_count": _count(subset, "pooled_discordant_fdr_q05"),
            "pooled_discordant_fdr_q10_count": _count(subset, "pooled_discordant_fdr_q10"),
            "pooled_discordant_fdr_holm_count": _count(subset, "pooled_discordant_fdr_holm"),
            "pooled_discordant_fdr_by_count": _count(subset, "pooled_discordant_fdr_by"),
            "source_discordant_fdr_count": _count(subset, "source_discordant_fdr"),
        }
    return out


def run_phase_cd_pooled_15_robustness(overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    paths = _source_paths(root)
    missing = [name for name, path in paths.items() if name != "phase_d_convergence" and not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required pooled robustness inputs: {missing}")

    result_dir = _result_dir(root)
    summary_path = result_dir / "phase_cd_pooled_15_summary.json"
    matrix_path = result_dir / "phase_cd_pooled_15_classification_matrix.csv"
    by_instance_path = result_dir / "phase_cd_pooled_15_by_instance.csv"
    if not overwrite:
        for path in (summary_path, matrix_path, by_instance_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing pooled robustness output: {path}")

    phase_c_rows = [_normalize_row(row, "PhaseC") for row in _read_csv(paths["phase_c_classification"])]
    phase_d_rows = [_normalize_row(row, "PhaseD") for row in _read_csv(paths["phase_d_classification"])]
    pooled_rows = sorted(phase_c_rows + phase_d_rows, key=lambda row: (row["source_phase"], row["instance"], row["noise"], row["shots_total"], row["zne_order"]))
    for index, row in enumerate(pooled_rows, start=1):
        row["pooled_cell_index"] = index

    _benjamini_hochberg(pooled_rows, "p_conjunction", "q_pooled_bh_q05", "fdr_pooled_bh_q05", PRIMARY_Q)
    _benjamini_hochberg(pooled_rows, "p_conjunction", "q_pooled_bh_q10", "fdr_pooled_bh_q10", CONTINUITY_Q)
    _holm_adjust(pooled_rows, "p_conjunction", "q_pooled_holm", "fdr_pooled_holm_q05", PRIMARY_Q)
    _by_adjust(pooled_rows, "p_conjunction", "q_pooled_by", "fdr_pooled_by_q05", PRIMARY_Q)
    _classify_pooled(pooled_rows)

    by_instance = _by_instance(pooled_rows)
    top3_share = 0.0
    fdr_rows = [row for row in pooled_rows if row["pooled_discordant_fdr_q05"]]
    if fdr_rows:
        by_instance_fdr = Counter(str(row["instance"]) for row in fdr_rows)
        top3_share = sum(count for _, count in by_instance_fdr.most_common(3)) / len(fdr_rows)

    phase_c_summary = _load_json(paths["phase_c_summary"])
    phase_d_summary = _load_json(paths["phase_d_summary"])
    convergence = _load_json(paths["phase_d_convergence"]) if paths["phase_d_convergence"].is_file() else {}
    summary = {
        "pooled_robustness_version": PHASE_CD_POOLED_ROBUSTNESS_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_cd_pooled_15_robustness_complete",
        "scope": "exploratory_pooled_phase_c_g1_g3_plus_phase_d_h1_h12",
        "exploratory_only": True,
        "not_primary": True,
        "primary_result_replaced": False,
        "phase_d_result_replaced": False,
        "convergence_result_replaced": False,
        "no_raw_data_recomputed": True,
        "warning": "This pooled 15-instance family is a post-Phase-D exploratory robustness diagnostic, not a confirmatory endpoint.",
        "pooled_family_size": len(pooled_rows),
        "instances_total": len({str(row["instance"]) for row in pooled_rows}),
        "source_counts": _by_source(pooled_rows),
        "classification_counts": dict(sorted(Counter(str(row["pooled_classification"]) for row in pooled_rows).items())),
        "pooled_discordant_pre_fdr_count": _count(pooled_rows, "pooled_discordant_pre_fdr"),
        "pooled_discordant_fdr_q05_count": _count(pooled_rows, "pooled_discordant_fdr_q05"),
        "pooled_discordant_fdr_q10_count": _count(pooled_rows, "pooled_discordant_fdr_q10"),
        "pooled_discordant_fdr_holm_count": _count(pooled_rows, "pooled_discordant_fdr_holm"),
        "pooled_discordant_fdr_by_count": _count(pooled_rows, "pooled_discordant_fdr_by"),
        "criterion_2_style_q05_pass": _criterion_style(pooled_rows, "pooled_discordant_fdr_q05"),
        "criterion_2_style_q10_pass": _criterion_style(pooled_rows, "pooled_discordant_fdr_q10"),
        "instances_with_pooled_q05_fdr": len({str(row["instance"]) for row in pooled_rows if row["pooled_discordant_fdr_q05"]}),
        "instances_with_pooled_q10_fdr": len({str(row["instance"]) for row in pooled_rows if row["pooled_discordant_fdr_q10"]}),
        "blocks_with_pooled_q05_fdr": len({str(row["block_id"]) for row in pooled_rows if row["pooled_discordant_fdr_q05"]}),
        "top_3_instance_share_q05": top3_share,
        "phase_c_reference": {
            "paper_go_strong": phase_c_summary.get("paper_go_strong"),
            "fdr_family_size": phase_c_summary.get("fdr_family_size"),
            "fdr_q": phase_c_summary.get("fdr_q"),
            "discordant_fdr_count": phase_c_summary.get("discordant_fdr_count"),
        },
        "phase_d_reference": {
            "assigned_label": phase_d_summary.get("assigned_label"),
            "fdr_family_size": phase_d_summary.get("fdr_family_size"),
            "primary_q": phase_d_summary.get("primary_q"),
            "discordant_fdr_count": phase_d_summary.get("discordant_fdr_count"),
            "cell_level_primary_endpoint_pass": phase_d_summary.get("cell_level_primary_endpoint_pass"),
            "instance_level_primary_endpoint_pass": phase_d_summary.get("instance_level_primary_endpoint_pass"),
        },
        "convergence_reference": {
            "outcome": convergence.get("outcome"),
            "status": convergence.get("status"),
        },
        "input_hashes": _hashes(root, [path.relative_to(root).as_posix() for path in paths.values() if path.is_file()]),
        "outputs": {
            "classification_matrix": matrix_path.relative_to(root).as_posix(),
            "by_instance": by_instance_path.relative_to(root).as_posix(),
        },
    }

    _write_csv(matrix_path, pooled_rows, overwrite=True)
    _write_csv(by_instance_path, by_instance, overwrite=True)
    summary["output_hashes"] = _hashes(root, [matrix_path.relative_to(root).as_posix(), by_instance_path.relative_to(root).as_posix()])
    write_json_atomic(summary_path, summary, overwrite=True)
    return summary
