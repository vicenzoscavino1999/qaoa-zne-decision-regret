from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest, norm

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic
from .phase_c_120_analysis import _as_float, _as_int, _benjamini_hochberg, _read_csv, _write_csv
from .phase_d_analysis import (
    BOOTSTRAP_BY_CELL_CSV,
    CLASSIFICATION_MATRIX_CSV,
    COMPARISONS_CSV,
    FDR_SENSITIVITY_CSV,
    GLOBAL_SUMMARY,
    METRICS_CSV,
    PRIMARY_Q,
    TAU_SENSITIVITY_CSV,
)
from .phase_d_completion import GLOBAL_RESULT_DIR
from .phase_d_kernel_runner import PHASE_D_BLOCK_IDS


PHASE_D_ROBUSTNESS_VERSION = "phase_d_robustness_v1"
PHASE_D_ROBUSTNESS_TIER2_VERSION = "phase_d_robustness_tier2_v2"
ROBUSTNESS_DIR = "robustness"
BOOTSTRAP_RESAMPLES = 10000
PERMUTATION_RESAMPLES = 10000
PERMUTATION_SEED = 20260513


class CellKey(tuple):
    __slots__ = ()

    def __new__(cls, instance: str, noise: float, shots_total: int, zne_order: str):
        return tuple.__new__(cls, (instance, float(noise), int(shots_total), zne_order))

    @property
    def instance(self) -> str:
        return self[0]

    @property
    def noise(self) -> float:
        return self[1]

    @property
    def shots_total(self) -> int:
        return self[2]

    @property
    def zne_order(self) -> str:
        return self[3]

    def id(self) -> str:
        return f"{self.instance}|p={self.noise:g}|B={self.shots_total}|zne={self.zne_order}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _global_dir(root: Path) -> Path:
    return root / "results" / "phase_d" / GLOBAL_RESULT_DIR


def _result_dir(root: Path) -> Path:
    return ensure_dir(root / "results" / "phase_d" / ROBUSTNESS_DIR)


def _hashes(root: Path, rels: list[str]) -> dict[str, str | None]:
    return {rel: sha256_file(root / rel) if (root / rel).is_file() else None for rel in rels}


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _cell_key(row: dict[str, Any]) -> CellKey:
    return CellKey(str(row["instance"]), _as_float(row["noise"]), _as_int(row["shots_total"]), str(row["zne_order"]))


def _cell_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (str(row["instance"]), _as_float(row["noise"]), _as_int(row["shots_total"]), str(row["zne_order"]))


def _criterion(rows: list[dict[str, Any]], fdr_key: str = "discordant_fdr_recomputed") -> bool:
    pre = [row for row in rows if str(row.get("discordant_pre_fdr")).lower() == "true" or row.get("discordant_pre_fdr") is True]
    fdr = [row for row in rows if row.get(fdr_key) is True or str(row.get(fdr_key)).lower() == "true"]
    spread = bool(
        len({row["instance"] for row in pre}) > 1
        and (len({row["noise"] for row in pre}) > 1 or len({row["shots_total"] for row in pre}) > 1)
    )
    return len(pre) >= 5 and len(fdr) >= 3 and spread


def _classification_summary(rows: list[dict[str, Any]], fdr_key: str = "discordant_fdr") -> dict[str, Any]:
    pre = [row for row in rows if _bool(row.get("discordant_pre_fdr"))]
    fdr = [row for row in rows if _bool(row.get(fdr_key))]
    return {
        "criterion_2_style_q05_pass": _criterion(rows, fdr_key),
        "discordant_pre_fdr_count": len(pre),
        "discordant_fdr_q05_count": len(fdr),
        "discordant_instances": len({str(row["instance"]) for row in fdr}),
        "discordant_blocks": len({str(row.get("block_id", "")) for row in fdr if str(row.get("block_id", ""))}),
        "classification_counts": dict(sorted(Counter(str(row.get("classification", "")) for row in rows).items())),
    }


def _bootstrap_pair_stats(r_values: np.ndarray, d_values: np.ndarray, seed: int, include_bca: bool = False) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, r_values.size, size=(BOOTSTRAP_RESAMPLES, r_values.size))
    boot_r = np.median(r_values[indices], axis=1)
    boot_d = np.median(d_values[indices], axis=1)
    p_r_positive = float((np.count_nonzero(boot_r <= 0.0) + 1) / (BOOTSTRAP_RESAMPLES + 1))
    p_d_positive = float((np.count_nonzero(boot_d <= 0.0) + 1) / (BOOTSTRAP_RESAMPLES + 1))
    out = {
        "n_seeds": int(r_values.size),
        "resamples": BOOTSTRAP_RESAMPLES,
        "median_r_mse": float(np.median(r_values)),
        "mean_r_mse": float(np.mean(r_values)),
        "ci_low_r_mse": float(np.quantile(boot_r, 0.025)),
        "ci_high_r_mse": float(np.quantile(boot_r, 0.975)),
        "p_r_mse_positive": p_r_positive,
        "prob_r_mse_positive_seed": float(np.mean(r_values > 0.0)),
        "median_d_R": float(np.median(d_values)),
        "mean_d_R": float(np.mean(d_values)),
        "ci_low_d_R": float(np.quantile(boot_d, 0.025)),
        "ci_high_d_R": float(np.quantile(boot_d, 0.975)),
        "p_d_R_positive": p_d_positive,
        "prob_d_R_positive_seed": float(np.mean(d_values > 0.0)),
        "p_conjunction": max(p_r_positive, p_d_positive),
    }
    if include_bca:
        bca_r = _bca_ci(r_values, boot_r, float(out["median_r_mse"]))
        bca_d = _bca_ci(d_values, boot_d, float(out["median_d_R"]))
        out.update({
            "bca_ci_low_r_mse": bca_r[0],
            "bca_ci_high_r_mse": bca_r[1],
            "bca_ci_low_d_R": bca_d[0],
            "bca_ci_high_d_R": bca_d[1],
        })
    return out


def _bca_ci(values: np.ndarray, boot_stats: np.ndarray, observed: float) -> tuple[float, float]:
    prop_less = float(np.mean(boot_stats < observed))
    eps = 1.0 / (2.0 * len(boot_stats))
    prop_less = min(max(prop_less, eps), 1.0 - eps)
    z0 = float(norm.ppf(prop_less))
    if values.size <= 2:
        accel = 0.0
    else:
        jack = np.asarray([np.median(np.delete(values, i)) for i in range(values.size)], dtype=float)
        jack_mean = float(np.mean(jack))
        diffs = jack_mean - jack
        denom = float(6.0 * (np.sum(diffs**2) ** 1.5))
        accel = 0.0 if denom == 0.0 or not math.isfinite(denom) else float(np.sum(diffs**3) / denom)

    def adjusted(prob: float) -> float:
        z = float(norm.ppf(prob))
        denom = 1.0 - accel * (z0 + z)
        if denom == 0.0:
            return prob
        return float(norm.cdf(z0 + (z0 + z) / denom))

    q_low = min(max(adjusted(0.025), 0.0), 1.0)
    q_high = min(max(adjusted(0.975), 0.0), 1.0)
    if q_low > q_high:
        q_low, q_high = q_high, q_low
    return float(np.quantile(boot_stats, q_low)), float(np.quantile(boot_stats, q_high))


def _classify_bootstrap_rows(rows: list[dict[str, Any]], tau_r: float, q_key: str = "q_conjunction_bh_q05", require_ci: bool = True) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        excluded = (
            _bool(row.get("excluded_below_detection_floor"))
            or _bool(row.get("excluded_extrapolation_breakdown"))
            or _bool(row.get("excluded_oob_threshold"))
        )
        pre = bool(
            not excluded
            and _as_float(row["median_r_mse"]) >= 0.10
            and (not require_ci or _as_float(row["ci_low_r_mse"]) > 0.0)
            and _as_float(row["median_d_R"]) > 0.0
            and (not require_ci or _as_float(row["ci_low_d_R"]) >= 0.0)
            and _as_float(row["prob_d_R_positive_seed"]) >= 0.60
        )
        fdr = bool(pre and _as_float(row[q_key]) <= PRIMARY_Q)
        strong = bool(
            fdr
            and require_ci
            and _as_float(row["median_r_mse"]) >= 0.20
            and _as_float(row["median_d_R"]) >= tau_r
            and _as_float(row["ci_low_r_mse"]) > 0.0
            and _as_float(row["ci_low_d_R"]) > 0.0
            and _as_float(row["prob_d_R_positive_seed"]) >= 0.70
        )
        if _bool(row.get("excluded_below_detection_floor")):
            label = "excluded_below_detection_floor"
        elif _bool(row.get("excluded_oob_threshold")):
            label = "excluded_oob_threshold"
        elif _bool(row.get("excluded_extrapolation_breakdown")):
            label = "excluded_extrapolation_breakdown"
        elif strong:
            label = "strongly_discordant_fdr"
        elif fdr:
            label = "discordant_fdr"
        elif pre:
            label = "discordant_not_fdr"
        else:
            label = "not_discordant"
        row["discordant_pre_fdr"] = pre
        row["discordant_fdr"] = fdr
        row["strongly_discordant_fdr"] = strong
        row["classification"] = label
        out.append(row)
    return out


def _expected_seed_set(global_dir: Path, fallback_seeds: set[int]) -> list[int]:
    root = global_dir.parents[2]
    plan_path = root / "results" / "phase_d_120_kernel_official" / "plan.json"
    if plan_path.is_file():
        plan = _load_json(plan_path)
        return sorted(_as_int(seed) for seed in plan.get("seeds", []))
    return sorted(fallback_seeds)


def _validate_seed_sets(
    cell_data: dict[CellKey, dict[str, Any]],
    noisy_regret_by_condition: dict[tuple[str, float, int], list[tuple[int, float]]],
    expected_seeds: list[int],
) -> dict[str, Any]:
    expected = list(expected_seeds)
    expected_set = set(expected)
    cell_issues: list[dict[str, Any]] = []
    for key, data in sorted(cell_data.items()):
        actual = [int(seed) for seed in data["seeds"]]
        counts = Counter(actual)
        actual_set = set(actual)
        issue = {
            "cell_id": key.id(),
            "n_seed_rows": len(actual),
            "missing_seeds": [seed for seed in expected if seed not in actual_set],
            "extra_seeds": sorted(seed for seed in actual_set if seed not in expected_set),
            "duplicate_seeds": sorted(seed for seed, count in counts.items() if count > 1),
        }
        if issue["missing_seeds"] or issue["extra_seeds"] or issue["duplicate_seeds"] or len(actual) != len(expected):
            cell_issues.append(issue)

    condition_issues: list[dict[str, Any]] = []
    for (instance, noise, shots), pairs in sorted(noisy_regret_by_condition.items()):
        actual = [int(seed) for seed, _ in pairs]
        counts = Counter(actual)
        actual_set = set(actual)
        issue = {
            "instance": instance,
            "noise": noise,
            "shots_total": shots,
            "n_seed_rows": len(actual),
            "missing_seeds": [seed for seed in expected if seed not in actual_set],
            "extra_seeds": sorted(seed for seed in actual_set if seed not in expected_set),
            "duplicate_seeds": sorted(seed for seed, count in counts.items() if count > 1),
        }
        if issue["missing_seeds"] or issue["extra_seeds"] or issue["duplicate_seeds"] or len(actual) != len(expected):
            condition_issues.append(issue)

    validation = {
        "expected_seed_count": len(expected),
        "expected_seeds": expected,
        "cell_count": len(cell_data),
        "noisy_condition_count": len(noisy_regret_by_condition),
        "cell_seed_sets_complete": not cell_issues,
        "noisy_condition_seed_sets_complete": not condition_issues,
        "cell_seed_issue_count": len(cell_issues),
        "noisy_condition_seed_issue_count": len(condition_issues),
        "cell_seed_issues": cell_issues[:20],
        "noisy_condition_seed_issues": condition_issues[:20],
    }
    if cell_issues or condition_issues:
        raise RuntimeError(f"Phase D Tier 2 seed completeness validation failed: {validation}")
    return validation


def _build_cell_data(global_dir: Path) -> tuple[dict[CellKey, dict[str, Any]], dict[tuple[str, float, int], list[tuple[int, float]]], dict[str, Any]]:
    comparison_rows = _read_csv(global_dir / COMPARISONS_CSV)
    metric_rows = _read_csv(global_dir / METRICS_CSV)
    grouped: dict[CellKey, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        grouped[_cell_key(row)].append(row)
    metrics_by_key: dict[tuple[str, float, int, int, str], dict[str, Any]] = {}
    noisy_regret_by_condition: dict[tuple[str, float, int], list[tuple[int, float]]] = defaultdict(list)
    for row in metric_rows:
        instance = str(row["instance"])
        noise = _as_float(row["noise"])
        shots = _as_int(row["shots_total"])
        seed = _as_int(row["seed"])
        method = str(row["method"])
        if method == "noisy":
            noisy_regret_by_condition[(instance, noise, shots)].append((seed, _as_float(row["norm_regret"])))
        elif method.startswith("zne_"):
            metrics_by_key[(instance, noise, shots, seed, method)] = row
    cell_data: dict[CellKey, dict[str, Any]] = {}
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda row: _as_int(row["seed"]))
        seeds = np.asarray([_as_int(row["seed"]) for row in rows], dtype=int)
        r_values = np.asarray([_as_float(row["r_mse"]) for row in rows], dtype=float)
        d_values = np.asarray([_as_float(row["d_R"]) for row in rows], dtype=float)
        below: list[bool] = []
        breakdown: list[bool] = []
        f_oob: list[float] = []
        method = f"zne_{key.zne_order}"
        for row in rows:
            metric = metrics_by_key[(key.instance, key.noise, key.shots_total, _as_int(row["seed"]), method)]
            below.append(_bool(metric["below_detection_floor"]))
            breakdown.append(_bool(metric["extrapolation_breakdown"]))
            f_oob.append(_as_float(metric["f_oob"]))
        cell_data[key] = {
            "block_id": str(rows[0].get("block_id", "")),
            "seeds": seeds,
            "r_values": r_values,
            "d_values": d_values,
            "below": np.asarray(below, dtype=bool),
            "breakdown": np.asarray(breakdown, dtype=bool),
            "f_oob": np.asarray(f_oob, dtype=float),
        }
    observed_seeds = {int(seed) for data in cell_data.values() for seed in data["seeds"]}
    seed_validation = _validate_seed_sets(cell_data, noisy_regret_by_condition, _expected_seed_set(global_dir, observed_seeds))
    return cell_data, noisy_regret_by_condition, seed_validation


def _compute_bootstrap_classification(
    cell_data: dict[CellKey, dict[str, Any]],
    tau_r: float,
    seed_offset: int,
    key_filter: set[CellKey] | None = None,
    seed_filter: set[int] | None = None,
    include_bca: bool = False,
    bca_classification: bool = False,
    oob_threshold: float | None = None,
    ignore_detection_floor: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    keys = [key for key in sorted(cell_data) if key_filter is None or key in key_filter]
    for cell_index, key in enumerate(keys, start=1):
        data = cell_data[key]
        seeds = data["seeds"]
        mask = np.ones(seeds.shape, dtype=bool)
        if seed_filter is not None:
            mask &= np.asarray([int(seed) in seed_filter for seed in seeds], dtype=bool)
        r_values = data["r_values"][mask]
        d_values = data["d_values"][mask]
        below = data["below"][mask]
        breakdown = data["breakdown"][mask]
        f_oob = data["f_oob"][mask]
        stats = _bootstrap_pair_stats(r_values, d_values, seed=seed_offset + cell_index, include_bca=include_bca)
        below_excluded = False if ignore_detection_floor else bool(np.any(below))
        breakdown_excluded = bool(np.any(breakdown))
        oob_excluded = bool(np.any(f_oob > oob_threshold)) if oob_threshold is not None else False
        row = {
            "cell_index": cell_index,
            "cell_id": key.id(),
            "instance": key.instance,
            "noise": key.noise,
            "shots_total": key.shots_total,
            "zne_order": key.zne_order,
            "block_id": data["block_id"],
            "allocation": "proportional_to_abs_weights",
            "tau_R": tau_r,
            "below_detection_floor_count": int(np.count_nonzero(below)),
            "below_detection_floor_rate": float(np.mean(below)) if below.size else 0.0,
            "excluded_below_detection_floor": below_excluded,
            "extrapolation_breakdown_count": int(np.count_nonzero(breakdown)),
            "extrapolation_breakdown_rate": float(np.mean(breakdown)) if breakdown.size else 0.0,
            "excluded_extrapolation_breakdown": breakdown_excluded,
            "excluded_oob_threshold": oob_excluded,
            "oob_threshold": "" if oob_threshold is None else oob_threshold,
            "oob_exclusion_count": int(np.count_nonzero(f_oob > oob_threshold)) if oob_threshold is not None else "",
            **stats,
        }
        if bca_classification:
            row["ci_low_r_mse"] = row["bca_ci_low_r_mse"]
            row["ci_high_r_mse"] = row["bca_ci_high_r_mse"]
            row["ci_low_d_R"] = row["bca_ci_low_d_R"]
            row["ci_high_d_R"] = row["bca_ci_high_d_R"]
        rows.append(row)
    _benjamini_hochberg(rows, "p_conjunction", "q_conjunction_bh_q05", "fdr_conjunction_pass_q05", PRIMARY_Q)
    classified = _classify_bootstrap_rows(rows, tau_r=tau_r, q_key="q_conjunction_bh_q05")
    return rows, classified


def _phase_d_tau_null(
    noisy_regret_by_condition: dict[tuple[str, float, int], list[tuple[int, float]]],
    primary_rows: list[dict[str, Any]],
    inherited_tau: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    values: list[float] = []
    condition_rows: list[dict[str, Any]] = []
    for (instance, noise, shots), rows in sorted(noisy_regret_by_condition.items()):
        rows = sorted(rows)
        condition_values: list[float] = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                delta = float(rows[j][1] - rows[i][1])
                values.append(delta)
                condition_values.append(delta)
        sigma = float(np.std(np.asarray(condition_values, dtype=float), ddof=1)) if len(condition_values) > 1 else 0.0
        condition_rows.append({
            "instance": instance,
            "noise": noise,
            "shots_total": shots,
            "seed_count": len(rows),
            "pair_count": len(condition_values),
            "sigma_noisy_vs_noisy_d_R": sigma,
            "tau_R_condition_null": max(0.005, 2.0 * sigma),
            "mean_abs_noisy_vs_noisy_d_R": float(np.mean(np.abs(condition_values))) if condition_values else 0.0,
            "p95_abs_noisy_vs_noisy_d_R": float(np.quantile(np.abs(condition_values), 0.95)) if condition_values else 0.0,
        })
    sigma_phase_d = float(np.std(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0
    tau_phase_d = max(0.005, 2.0 * sigma_phase_d)
    by_cell: list[dict[str, Any]] = []
    for row in sorted(primary_rows, key=_cell_sort_key):
        pre = _bool(row["discordant_pre_fdr"])
        fdr = _bool(row["discordant_fdr"])
        pass_tau = bool(pre and _as_float(row["median_d_R"]) >= tau_phase_d)
        pass_tau_fdr = bool(fdr and _as_float(row["median_d_R"]) >= tau_phase_d)
        by_cell.append({
            "cell_id": row["cell_id"],
            "instance": row["instance"],
            "block_id": row.get("block_id", ""),
            "noise": row["noise"],
            "shots_total": row["shots_total"],
            "zne_order": row["zne_order"],
            "classification": row["classification"],
            "discordant_pre_fdr": pre,
            "discordant_fdr": fdr,
            "median_d_R": row["median_d_R"],
            "tau_R_phaseD_null": tau_phase_d,
            "discordant_pre_fdr_tau_phaseD_null": pass_tau,
            "discordant_fdr_tau_phaseD_null_q05": pass_tau_fdr,
        })
    tau_pre = [row for row in by_cell if row["discordant_pre_fdr_tau_phaseD_null"]]
    tau_fdr = [row for row in by_cell if row["discordant_fdr_tau_phaseD_null_q05"]]
    summary = {
        "status": "phase_d_tau_r_null_diagnostic_complete",
        "inherited_tau_R": inherited_tau,
        "sigma_phaseD_null": sigma_phase_d,
        "tau_R_phaseD_null": tau_phase_d,
        "tau_R_phaseD_null_over_inherited_tau_R": float(tau_phase_d / inherited_tau) if inherited_tau else float("nan"),
        "noisy_vs_noisy_pair_count": len(values),
        "condition_count": len(condition_rows),
        "mean_abs_noisy_vs_noisy_d_R": float(np.mean(np.abs(values))) if values else 0.0,
        "p95_abs_noisy_vs_noisy_d_R": float(np.quantile(np.abs(values), 0.95)) if values else 0.0,
        "discordant_pre_fdr_tau_phaseD_null_count": len(tau_pre),
        "discordant_fdr_tau_phaseD_null_q05_count": len(tau_fdr),
        "instances_with_tau_phaseD_null_fdr": len({str(row["instance"]) for row in tau_fdr}),
        "blocks_with_tau_phaseD_null_fdr": len({str(row["block_id"]) for row in tau_fdr}),
        "pairwise_differences_are_correlated": True,
        "tau_R_phaseD_null_is_diagnostic_only": True,
        "tau_R_phaseD_null_does_not_replace_primary_tau_R": True,
        "interpretation": (
            "inherited_tau_R_conservative_relative_to_phase_d_null"
            if tau_phase_d < inherited_tau
            else "inherited_tau_R_liberal_relative_to_phase_d_null"
            if tau_phase_d > inherited_tau
            else "inherited_tau_R_matches_phase_d_null"
        ),
        "diagnostic_only_not_primary": True,
    }
    return summary, condition_rows, by_cell


def _run_leave_one_seed(cell_data: dict[CellKey, dict[str, Any]], primary_rows: list[dict[str, Any]], tau_r: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    all_seeds = sorted({int(seed) for data in cell_data.values() for seed in data["seeds"]})
    primary_by_id = {str(row["cell_id"]): row for row in primary_rows}
    primary_pass = _criterion(primary_rows, "discordant_fdr")
    summary_rows: list[dict[str, Any]] = []
    by_cell_rows: list[dict[str, Any]] = []
    for omitted_seed in all_seeds:
        keep = {seed for seed in all_seeds if seed != omitted_seed}
        _, classified = _compute_bootstrap_classification(cell_data, tau_r, seed_offset=300000 + omitted_seed, seed_filter=keep)
        summary = _classification_summary(classified)
        for row in classified:
            original = primary_by_id[str(row["cell_id"])]
            by_cell_rows.append({
                "omitted_seed": omitted_seed,
                "cell_id": row["cell_id"],
                "instance": row["instance"],
                "block_id": row.get("block_id", ""),
                "noise": row["noise"],
                "shots_total": row["shots_total"],
                "zne_order": row["zne_order"],
                "primary_classification": original["classification"],
                "loo_classification": row["classification"],
                "primary_discordant_fdr": _bool(original["discordant_fdr"]),
                "loo_discordant_fdr": _bool(row["discordant_fdr"]),
                "same_as_primary": str(original["classification"]) == str(row["classification"]),
            })
        summary_rows.append({
            "omitted_seed": omitted_seed,
            **summary,
            "seed_loo_flip": bool(summary["criterion_2_style_q05_pass"] != primary_pass),
            "complete_dissolution": bool(summary["discordant_fdr_q05_count"] == 0),
        })
    flips = sum(_bool(row["seed_loo_flip"]) for row in summary_rows)
    if flips == 0:
        label = "strong_seed_robustness"
    elif flips <= 2:
        label = "moderate_seed_robustness"
    elif flips <= 5:
        label = "weak_seed_robustness"
    else:
        label = "seed_fragile"
    aggregate = {
        "seed_loo_runs": len(summary_rows),
        "seed_loo_flips": flips,
        "seed_robustness_label": label,
        "complete_dissolution_runs": sum(_bool(row["complete_dissolution"]) for row in summary_rows),
        "discordant_fdr_q05_min": min(_as_int(row["discordant_fdr_q05_count"]) for row in summary_rows),
        "discordant_fdr_q05_max": max(_as_int(row["discordant_fdr_q05_count"]) for row in summary_rows),
        "discordant_pre_fdr_min": min(_as_int(row["discordant_pre_fdr_count"]) for row in summary_rows),
        "discordant_pre_fdr_max": max(_as_int(row["discordant_pre_fdr_count"]) for row in summary_rows),
    }
    return summary_rows, by_cell_rows, aggregate


def _permutation_p(values: np.ndarray, statistic: str, seed: int, observed: float) -> float:
    if observed <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(PERMUTATION_RESAMPLES, values.size))
    sampled = values[None, :] * signs
    null_stats = np.median(sampled, axis=1) if statistic == "median" else np.mean(sampled, axis=1)
    return float((np.count_nonzero(null_stats >= observed) + 1) / (PERMUTATION_RESAMPLES + 1))


def _run_nonparametric(cell_data: dict[CellKey, dict[str, Any]], primary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    primary_disc = {str(row["cell_id"]) for row in primary_rows if _bool(row["discordant_fdr"])}
    test_rows: dict[str, list[dict[str, Any]]] = {"sign": [], "permutation_median": [], "permutation_mean": []}
    for cell_index, key in enumerate(sorted(cell_data), start=1):
        data = cell_data[key]
        r = data["r_values"]
        d = data["d_values"]
        base = {
            "cell_index": cell_index,
            "cell_id": key.id(),
            "instance": key.instance,
            "block_id": data["block_id"],
            "noise": key.noise,
            "shots_total": key.shots_total,
            "zne_order": key.zne_order,
            "median_r_mse": float(np.median(r)),
            "mean_r_mse": float(np.mean(r)),
            "median_d_R": float(np.median(d)),
            "mean_d_R": float(np.mean(d)),
            "prob_r_mse_positive_seed": float(np.mean(r > 0.0)),
            "prob_d_R_positive_seed": float(np.mean(d > 0.0)),
            "excluded_below_detection_floor": bool(np.any(data["below"])),
            "excluded_extrapolation_breakdown": bool(np.any(data["breakdown"])),
            "n_seeds": int(r.size),
        }
        count_r = int(np.count_nonzero(r > 0.0))
        count_d = int(np.count_nonzero(d > 0.0))
        p_r = float(binomtest(count_r, r.size, 0.5, alternative="greater").pvalue)
        p_d = float(binomtest(count_d, d.size, 0.5, alternative="greater").pvalue)
        test_rows["sign"].append({**base, "test": "sign", "p_r_mse_positive": p_r, "p_d_R_positive": p_d, "p_conjunction": max(p_r, p_d)})
        for statistic, name in (("median", "permutation_median"), ("mean", "permutation_mean")):
            observed_r = float(np.median(r) if statistic == "median" else np.mean(r))
            observed_d = float(np.median(d) if statistic == "median" else np.mean(d))
            p_r_perm = _permutation_p(r, statistic, PERMUTATION_SEED + cell_index * 10 + (0 if statistic == "median" else 1), observed_r)
            p_d_perm = _permutation_p(d, statistic, PERMUTATION_SEED + cell_index * 10 + (2 if statistic == "median" else 3), observed_d)
            test_rows[name].append({**base, "test": name, "p_r_mse_positive": p_r_perm, "p_d_R_positive": p_d_perm, "p_conjunction": max(p_r_perm, p_d_perm)})
    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for name, rows in test_rows.items():
        _benjamini_hochberg(rows, "p_conjunction", "q_conjunction_bh_q05", "fdr_conjunction_pass_q05", PRIMARY_Q)
        classified = _classify_bootstrap_rows(rows, tau_r=0.0, q_key="q_conjunction_bh_q05", require_ci=False)
        summary = _classification_summary(classified)
        disc = {str(row["cell_id"]) for row in classified if _bool(row["discordant_fdr"])}
        intersection = primary_disc & disc
        union = primary_disc | disc
        jaccard = float(len(intersection) / len(union)) if union else 1.0
        distributed = bool(summary["discordant_instances"] >= 6 and summary["discordant_blocks"] >= 3)
        if len(disc) >= 0.70 * len(primary_disc) and jaccard >= 0.60 and distributed:
            label = "supportive"
        elif len(disc) >= 3 and distributed:
            label = "partially_supportive"
        else:
            label = "not_supportive"
        summary_rows.append({
            "test": name,
            **summary,
            "primary_discordant_fdr_count": len(primary_disc),
            "jaccard_vs_primary_fdr": jaccard,
            "support_label": label,
        })
        for row in classified:
            detail_rows.append({**row, "nonparametric_test": name})
    aggregate = {row["test"]: row for row in summary_rows}
    return summary_rows, detail_rows, aggregate


def _run_fragility(primary_rows: list[dict[str, Any]], leave_one_seed_by_cell: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stability: dict[str, list[bool]] = defaultdict(list)
    same: dict[str, list[bool]] = defaultdict(list)
    for row in leave_one_seed_by_cell:
        stability[str(row["cell_id"])].append(_bool(row["loo_discordant_fdr"]))
        same[str(row["cell_id"])].append(_bool(row["same_as_primary"]))
    rows: list[dict[str, Any]] = []
    for row in sorted(primary_rows, key=_cell_sort_key):
        cell_id = str(row["cell_id"])
        primary_disc = _bool(row["discordant_fdr"])
        cell_stability = float(np.mean(stability[cell_id])) if primary_disc else float(np.mean(same[cell_id]))
        if cell_stability >= 0.90:
            label = "stable"
        elif cell_stability >= 0.70:
            label = "moderate"
        else:
            label = "fragile"
        rows.append({
            "cell_id": cell_id,
            "instance": row["instance"],
            "block_id": row.get("block_id", ""),
            "noise": row["noise"],
            "shots_total": row["shots_total"],
            "zne_order": row["zne_order"],
            "primary_classification": row["classification"],
            "primary_discordant_fdr": primary_disc,
            "cell_seed_stability": cell_stability,
            "stability_label": label,
            "fragile_primary_discordant": bool(primary_disc and cell_stability < 0.70),
            "q_margin_vs_0p05": PRIMARY_Q - _as_float(row["q_conjunction_bh_q05"]),
            "d_margin_vs_tau_R": _as_float(row["median_d_R"]) - _as_float(row["tau_R"]),
            "d_margin_vs_zero": _as_float(row["median_d_R"]),
        })
    bins: list[dict[str, Any]] = []
    for low in np.arange(0.0, 1.0, 0.05):
        high = round(float(low + 0.05), 2)
        count = sum(float(low) <= _as_float(row["cell_seed_stability"]) < high for row in rows)
        if math.isclose(high, 1.0):
            count = sum(float(low) <= _as_float(row["cell_seed_stability"]) <= high for row in rows)
        bins.append({"stability_bin_low": round(float(low), 2), "stability_bin_high": high, "cell_count": int(count)})
    fragile_count = sum(_bool(row["fragile_primary_discordant"]) for row in rows)
    primary_count = sum(_bool(row["primary_discordant_fdr"]) for row in rows)
    if fragile_count == 0:
        assessment = "strong"
    elif fragile_count <= 2:
        assessment = "acceptable_with_disclosure"
    elif fragile_count <= 4:
        assessment = "moderate_requires_explicit_qualification"
    else:
        assessment = "weak_claim_must_be_downgraded"
    aggregate = {
        "primary_discordant_fdr_count": primary_count,
        "fragile_cell_count_among_primary_discordant_fdr": fragile_count,
        "stable_primary_discordant_fdr_count": sum(_bool(row["primary_discordant_fdr"]) and str(row["stability_label"]) == "stable" for row in rows),
        "cell_fragility_assessment": assessment,
    }
    return rows, bins, aggregate


def _run_bca_and_oob_sensitivity(cell_data: dict[CellKey, dict[str, Any]], tau_r: float, primary_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _, bca_class = _compute_bootstrap_classification(cell_data, tau_r, seed_offset=600000, include_bca=True, bca_classification=True)
    rows = [{"diagnostic": "BCa CI", **_classification_summary(bca_class)}]
    for threshold in (0.05, 0.10, 0.15):
        _, class_rows = _compute_bootstrap_classification(cell_data, tau_r, seed_offset=700000 + int(threshold * 1000), oob_threshold=threshold)
        rows.append({"diagnostic": f"OOB threshold {threshold:g}", **_classification_summary(class_rows)})
    _, relaxed = _compute_bootstrap_classification(cell_data, tau_r, seed_offset=701000, ignore_detection_floor=True)
    rows.append({"diagnostic": "ignore below-detection-floor exclusion", **_classification_summary(relaxed)})
    primary_summary = _classification_summary(primary_rows)
    for row in rows:
        row["primary_discordant_fdr_q05_count"] = primary_summary["discordant_fdr_q05_count"]
        row["primary_criterion_pass"] = primary_summary["criterion_2_style_q05_pass"]
        row["criterion_changed_vs_primary"] = bool(row["criterion_2_style_q05_pass"] != primary_summary["criterion_2_style_q05_pass"])
    return rows, bca_class


def _recompute_subset(rows: list[dict[str, Any]], label: str, dropped_field: str, dropped_value: str) -> dict[str, Any]:
    subset = [dict(row) for row in rows if str(row.get(dropped_field)) != dropped_value]
    for row in subset:
        row["p_conjunction"] = _as_float(row["p_conjunction"])
        row["discordant_pre_fdr"] = str(row.get("discordant_pre_fdr")).lower() == "true"
    _benjamini_hochberg(subset, "p_conjunction", "q_recomputed_bh_q05", "fdr_recomputed_bh_q05", PRIMARY_Q)
    for row in subset:
        row["discordant_fdr_recomputed"] = bool(row["discordant_pre_fdr"] and row["fdr_recomputed_bh_q05"])
    return {
        "diagnostic": label,
        "dropped_field": dropped_field,
        "dropped_value": dropped_value,
        "remaining_cells": len(subset),
        "discordant_pre_fdr_count": sum(bool(row["discordant_pre_fdr"]) for row in subset),
        "discordant_fdr_q05_recomputed_count": sum(bool(row["discordant_fdr_recomputed"]) for row in subset),
        "criterion_2_style_q05_pass": _criterion(subset),
    }


def run_phase_d_robustness(overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    global_dir = _global_dir(root)
    class_path = global_dir / CLASSIFICATION_MATRIX_CSV
    summary_path = global_dir / GLOBAL_SUMMARY
    if not class_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Run Phase D global analysis before robustness.")
    result_dir = _result_dir(root)
    output_paths = [
        result_dir / "phase_d_540_robustness_summary.json",
        result_dir / "phase_d_540_leave_one_instance.csv",
        result_dir / "phase_d_540_leave_one_noise.csv",
        result_dir / "phase_d_540_leave_one_shot_budget.csv",
        result_dir / "phase_d_540_leave_one_zne_order.csv",
        result_dir / "phase_d_540_fdr_sensitivity.csv",
        result_dir / "phase_d_540_tau_r_sensitivity.csv",
        result_dir / "phase_d_540_robustness_limitations.md",
    ]
    if not overwrite:
        for path in output_paths:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing Phase D robustness output: {path}")

    rows = _read_csv(class_path)
    instance_rows = [_recompute_subset(rows, "leave_one_instance", "instance", value) for value in sorted({row["instance"] for row in rows})]
    noise_rows = [_recompute_subset(rows, "leave_one_noise", "noise", value) for value in sorted({row["noise"] for row in rows}, key=float)]
    shot_rows = [_recompute_subset(rows, "leave_one_shot_budget", "shots_total", value) for value in sorted({row["shots_total"] for row in rows}, key=int)]
    zne_rows = [_recompute_subset(rows, "leave_one_zne_order", "zne_order", value) for value in sorted({row["zne_order"] for row in rows})]

    _write_csv(result_dir / "phase_d_540_leave_one_instance.csv", instance_rows, overwrite=True)
    _write_csv(result_dir / "phase_d_540_leave_one_noise.csv", noise_rows, overwrite=True)
    _write_csv(result_dir / "phase_d_540_leave_one_shot_budget.csv", shot_rows, overwrite=True)
    _write_csv(result_dir / "phase_d_540_leave_one_zne_order.csv", zne_rows, overwrite=True)
    fdr_source = global_dir / FDR_SENSITIVITY_CSV
    tau_source = global_dir / TAU_SENSITIVITY_CSV
    if fdr_source.is_file():
        _write_csv(result_dir / "phase_d_540_fdr_sensitivity.csv", _read_csv(fdr_source), overwrite=True)
    if tau_source.is_file():
        _write_csv(result_dir / "phase_d_540_tau_r_sensitivity.csv", _read_csv(tau_source), overwrite=True)

    loo_flips = sum(not row["criterion_2_style_q05_pass"] for row in instance_rows)
    if loo_flips == 0:
        loo_label = "strong_phase_d_robustness"
    elif loo_flips <= 2:
        loo_label = "moderate_phase_d_robustness"
    elif loo_flips <= 4:
        loo_label = "weak_phase_d_robustness"
    else:
        loo_label = "phase_d_fragile"
    summary_output = result_dir / "phase_d_540_robustness_summary.json"
    hashable_outputs = [path for path in output_paths if path != summary_output]
    summary = {
        "robustness_version": PHASE_D_ROBUSTNESS_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_d_540_robustness_complete",
        "scope": "phase_d_540_cells_H1_H12",
        "primary_global_summary": _load_json(summary_path),
        "leave_one_instance_flip_count": loo_flips,
        "leave_one_instance_label": loo_label,
        "diagnostics_completed": [
            "leave_one_instance",
            "leave_one_noise",
            "leave_one_shot_budget",
            "leave_one_zne_order",
            "FDR sensitivity copied from global analysis",
            "tau_R sensitivity copied from global analysis",
        ],
        "deferred_diagnostics": [
            "leave_one_seed bootstrap recomputation",
            "BCa CI sensitivity",
            "OOB threshold sensitivity",
            "permutation test by cell",
            "sign test by cell",
            "fragility index by cell",
        ],
        "output_hash_policy": "The summary JSON omits its own hash to avoid a self-referential checksum.",
        "output_hashes": _hashes(root, [path.relative_to(root).as_posix() for path in hashable_outputs if path.is_file()]),
    }
    write_json_atomic(summary_output, summary, overwrite=True)
    (result_dir / "phase_d_540_robustness_limitations.md").write_text(
        "\n".join([
            "# Phase D 540 Robustness Limitations",
            "",
            "This robustness script recomputes the primary BH q=0.05 family after dropping instances, noise values, shot budgets, and ZNE orders.",
            "Seed-level and nonparametric diagnostics remain separate post-analysis extensions and must be reported if used for stronger robustness claims.",
            "",
        ]),
        encoding="utf-8",
    )
    summary["output_hashes"] = _hashes(root, [path.relative_to(root).as_posix() for path in hashable_outputs if path.is_file()])
    write_json_atomic(summary_output, summary, overwrite=True)
    return summary


def run_phase_d_robustness_tier2(overwrite: bool = False, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    global_dir = _global_dir(root)
    class_path = global_dir / CLASSIFICATION_MATRIX_CSV
    summary_path = global_dir / GLOBAL_SUMMARY
    if not class_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("Run Phase D global analysis before Tier 2 robustness.")
    result_dir = _result_dir(root)
    output_paths = [
        result_dir / "phase_d_540_robustness_tier2_summary.json",
        result_dir / "phase_d_tau_r_phase_d_null_diagnostic.json",
        result_dir / "phase_d_tau_r_phase_d_null_by_condition.csv",
        result_dir / "phase_d_tau_r_phase_d_null_by_cell.csv",
        result_dir / "phase_d_540_leave_one_seed_summary.csv",
        result_dir / "phase_d_540_leave_one_seed_by_cell.csv",
        result_dir / "phase_d_540_nonparametric_summary.csv",
        result_dir / "phase_d_540_nonparametric_by_cell.csv",
        result_dir / "phase_d_540_fragility_by_cell.csv",
        result_dir / "phase_d_540_fragility_distribution.csv",
        result_dir / "phase_d_540_bca_oob_sensitivity.csv",
        result_dir / "phase_d_540_bca_classification.csv",
        result_dir / "phase_d_540_robustness_tier2_limitations.md",
    ]
    if not overwrite:
        for path in output_paths:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing Phase D Tier 2 robustness output: {path}")

    primary_rows = _read_csv(class_path)
    inherited_tau = _as_float(primary_rows[0]["tau_R"])
    cell_data, noisy_regret_by_condition, seed_validation = _build_cell_data(global_dir)
    tau_summary, tau_condition_rows, tau_cell_rows = _phase_d_tau_null(noisy_regret_by_condition, primary_rows, inherited_tau)
    seed_summary_rows, seed_by_cell_rows, seed_agg = _run_leave_one_seed(cell_data, primary_rows, inherited_tau)
    nonparam_summary_rows, nonparam_detail_rows, nonparam_agg = _run_nonparametric(cell_data, primary_rows)
    fragility_rows, fragility_distribution_rows, fragility_agg = _run_fragility(primary_rows, seed_by_cell_rows)
    bca_oob_rows, bca_classification_rows = _run_bca_and_oob_sensitivity(cell_data, inherited_tau, primary_rows)

    _write_csv(result_dir / "phase_d_tau_r_phase_d_null_by_condition.csv", tau_condition_rows, overwrite=True)
    _write_csv(result_dir / "phase_d_tau_r_phase_d_null_by_cell.csv", tau_cell_rows, overwrite=True)
    _write_csv(result_dir / "phase_d_540_leave_one_seed_summary.csv", seed_summary_rows, overwrite=True)
    _write_csv(result_dir / "phase_d_540_leave_one_seed_by_cell.csv", seed_by_cell_rows, overwrite=True)
    _write_csv(result_dir / "phase_d_540_nonparametric_summary.csv", nonparam_summary_rows, overwrite=True)
    _write_csv(result_dir / "phase_d_540_nonparametric_by_cell.csv", nonparam_detail_rows, overwrite=True)
    _write_csv(result_dir / "phase_d_540_fragility_by_cell.csv", fragility_rows, overwrite=True)
    _write_csv(result_dir / "phase_d_540_fragility_distribution.csv", fragility_distribution_rows, overwrite=True)
    _write_csv(result_dir / "phase_d_540_bca_oob_sensitivity.csv", bca_oob_rows, overwrite=True)
    _write_csv(result_dir / "phase_d_540_bca_classification.csv", bca_classification_rows, overwrite=True)
    write_json_atomic(result_dir / "phase_d_tau_r_phase_d_null_diagnostic.json", tau_summary, overwrite=True)
    (result_dir / "phase_d_540_robustness_tier2_limitations.md").write_text(
        "\n".join([
            "# Phase D 540 Robustness Tier 2 Limitations",
            "",
            "Tier 2 uses completed Phase D outputs only. It does not add instances, seeds, noisy executions, or ZNE executions.",
            "The Phase-D-specific tau_R is diagnostic-only and does not replace the inherited primary tau_R used for the official Phase D claim.",
            "The Phase-D-specific tau_R null uses pairwise noisy-vs-noisy differences across seeds; these pairwise differences are correlated and are interpreted only as a diagnostic scale check.",
            "Nonparametric tests are robustness diagnostics over per-seed paired comparisons and are not a new primary FDR family.",
            "",
        ]),
        encoding="utf-8",
    )

    summary_output = result_dir / "phase_d_540_robustness_tier2_summary.json"
    hashable_outputs = [path for path in output_paths if path != summary_output]
    summary = {
        "robustness_version": PHASE_D_ROBUSTNESS_TIER2_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_d_540_robustness_tier2_complete",
        "scope": "phase_d_540_cells_H1_H12_existing_outputs_only",
        "primary_global_summary": _load_json(summary_path),
        "seed_completeness_validation": seed_validation,
        "tau_R_phaseD_null": tau_summary,
        "leave_one_seed": seed_agg,
        "nonparametric": nonparam_agg,
        "fragility": fragility_agg,
        "bca_oob_sensitivity": bca_oob_rows,
        "diagnostics_completed": [
            "Phase-D-specific tau_R noisy-vs-noisy null diagnostic",
            "classification counts under tau_R_phaseD_null",
            "leave-one-seed bootstrap recomputation",
            "sign test by cell",
            "permutation median test by cell",
            "permutation mean test by cell",
            "fragility index by cell from leave-one-seed stability",
            "BCa CI sensitivity",
            "OOB threshold sensitivity at 0.05, 0.10, 0.15",
            "below-detection-floor exclusion sensitivity",
        ],
        "primary_result_replaced": False,
        "output_hash_policy": "The summary JSON omits its own hash to avoid a self-referential checksum.",
        "output_hashes": _hashes(root, [path.relative_to(root).as_posix() for path in hashable_outputs if path.is_file()]),
    }
    write_json_atomic(summary_output, summary, overwrite=True)
    return summary


def write_phase_d_robustness_preflight(overwrite: bool = True, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    planning_dir = ensure_dir(root / "results" / "phase_d" / "planning")
    required_sources = [
        "src/phase_d_robustness.py",
        "scripts/37_phase_d_robustness.py",
        "src/phase_d_analysis.py",
        "scripts/36_phase_d_analysis.py",
    ]
    required_outputs = [
        "phase_c_frozen_robustness_reference.json",
        "phase_d_540_robustness_summary.json",
        "phase_d_540_leave_one_instance.csv",
        "phase_d_540_leave_one_seed.csv",
        "phase_d_540_leave_one_noise.csv",
        "phase_d_540_leave_one_shot_budget.csv",
        "phase_d_540_leave_one_zne_order.csv",
        "phase_d_540_nonparametric_tests.csv",
        "phase_d_540_fragility_by_cell.csv",
        "phase_d_540_fdr_sensitivity.csv",
        "phase_d_540_tau_r_sensitivity.csv",
        "phase_d_540_robustness_limitations.md",
    ]
    checks = {
        "robustness_sources_exist": all((root / rel).is_file() for rel in required_sources),
        "depends_on_global_analysis": (root / "src" / "phase_d_analysis.py").is_file(),
        "block_ids_declared": list(PHASE_D_BLOCK_IDS) == ["D1", "D2", "D3", "D4"],
        "output_contract_declared": len(required_outputs) == 12,
    }
    report = {
        "robustness_version": PHASE_D_ROBUSTNESS_VERSION,
        "created_at_utc": _utc_now(),
        "status": "phase_d_robustness_preflight_passed" if all(checks.values()) else "phase_d_robustness_preflight_failed",
        "decision": "robustness_scripts_ready_for_phase_d_v1_0_freeze" if all(checks.values()) else "block_phase_d_v1_0_freeze_until_robustness_preflight_passes",
        "scope": "pre_official_execution_robustness_script_readiness",
        "checks": checks,
        "post_execution_output_contract": [f"results/phase_d/{ROBUSTNESS_DIR}/{name}" for name in required_outputs],
        "source_hashes": _hashes(root, required_sources),
    }
    output = planning_dir / "phase_d_robustness_preflight.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing Phase D robustness preflight: {output}")
    write_json_atomic(output, report, overwrite=True)
    return report
