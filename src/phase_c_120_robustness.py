from __future__ import annotations

import csv
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.stats import binom, binomtest, norm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .io_utils import PROJECT_ROOT, ensure_dir, read_yaml, sha256_file, write_json_atomic
from .phase_c_120_analysis import (
    BOOTSTRAP_BY_CELL_CSV,
    CLASSIFICATION_MATRIX_CSV,
    COMPARISONS_CSV,
    METRICS_CSV,
    PAPER_GO_NO_GO_JSON,
    _benjamini_hochberg,
    _current_analysis_hashes,
    _read_csv,
    _result_dir as _analysis_result_dir,
)


ROBUSTNESS_VERSION = "phase_c_120_robustness_v0.5"
ROBUSTNESS_RESULT_DIR = "phase_c_120_robustness"
PLAN_PATH = "PHASE_C_120_ROBUSTNESS_EXTENSION_PLAN.md"
RESAMPLES = 10000
PERMUTATION_RESAMPLES = 10000
PERMUTATION_SEED = 20260513
NUMERIC_TOL = 1e-12


@dataclass(frozen=True)
class CellKey:
    instance: str
    noise: float
    shots_total: int
    zne_order: str

    def id(self) -> str:
        return f"{self.instance}|p={self.noise:g}|B={self.shots_total}|zne={self.zne_order}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _result_dir(root: Path) -> Path:
    return ensure_dir(root / "results" / ROBUSTNESS_RESULT_DIR)


def _as_float(value: Any) -> float:
    return float(str(value).strip())


def _as_int(value: Any) -> int:
    return int(str(value).strip())


def _csv_bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(f"Object is not JSON serializable: {type(value)!r}")


def _write_csv_atomic(path: Path, rows: list[dict[str, Any]], overwrite: bool = False) -> None:
    ensure_dir(path.parent)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _cell_key(row: dict[str, Any]) -> CellKey:
    return CellKey(
        instance=str(row["instance"]),
        noise=_as_float(row["noise"]),
        shots_total=_as_int(row["shots_total"]),
        zne_order=str(row["zne_order"]),
    )


def _cell_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["instance"]),
        _as_float(row["noise"]),
        _as_int(row["shots_total"]),
        str(row["zne_order"]),
    )


def _summary_from_classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    discordant = [row for row in rows if _csv_bool(row.get("discordant_pre_fdr", False))]
    discordant_fdr = [row for row in rows if _csv_bool(row.get("discordant_fdr", False))]
    strongly = [row for row in rows if _csv_bool(row.get("strongly_discordant_fdr", False))]
    spread = bool(
        len({str(row["instance"]) for row in discordant}) > 1
        or len({_as_float(row["noise"]) for row in discordant}) > 1
        or len({_as_int(row["shots_total"]) for row in discordant}) > 1
    )
    criterion_1 = len(strongly) >= 2
    criterion_2 = len(discordant) >= 5 and len(discordant_fdr) >= 3 and spread
    return {
        "paper_go_strong": bool(criterion_1 or criterion_2),
        "criterion_1": bool(criterion_1),
        "criterion_2": bool(criterion_2),
        "discordant_pre_fdr_count": len(discordant),
        "discordant_fdr_count": len(discordant_fdr),
        "strongly_discordant_fdr_count": len(strongly),
        "classification_counts": dict(sorted(Counter(str(row["classification"]) for row in rows).items())),
        "discordant_distinct_instances": sorted({str(row["instance"]) for row in discordant}),
        "discordant_distinct_noise_values": sorted({_as_float(row["noise"]) for row in discordant}),
        "discordant_distinct_shot_budgets": sorted({_as_int(row["shots_total"]) for row in discordant}),
        "spread": bool(spread),
    }


def _classify_rows(rows: list[dict[str, Any]], cfg: dict[str, Any], q_threshold: float) -> list[dict[str, Any]]:
    tau_r = float(cfg["phase_c"]["tau_R"])
    discordant_cfg = cfg["go_no_go"]["discordant"]
    strong_cfg = cfg["go_no_go"]["strongly_discordant"]
    out: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        excluded_below = _csv_bool(row.get("excluded_below_detection_floor", False))
        excluded_breakdown = _csv_bool(row.get("excluded_extrapolation_breakdown", False))
        excluded = excluded_below or excluded_breakdown
        row["fdr_conjunction_pass"] = bool(_as_float(row["q_conjunction_bh"]) <= q_threshold)
        discordant_pre_fdr = bool(
            not excluded
            and _as_float(row["median_r_mse"]) >= float(discordant_cfg["r_mse_min"])
            and _as_float(row["ci_low_r_mse"]) > 0.0
            and _as_float(row["median_d_R"]) > 0.0
            and _as_float(row["ci_low_d_R"]) >= 0.0
            and _as_float(row["prob_d_R_positive_seed"]) >= float(discordant_cfg["prob_d_R_positive_min"])
        )
        discordant_fdr = bool(discordant_pre_fdr and row["fdr_conjunction_pass"])
        strong = bool(
            not excluded
            and _as_float(row["median_r_mse"]) >= float(strong_cfg["r_mse_min"])
            and _as_float(row["median_d_R"]) >= tau_r
            and _as_float(row["ci_low_r_mse"]) > 0.0
            and _as_float(row["ci_low_d_R"]) > 0.0
            and _as_float(row["prob_d_R_positive_seed"]) >= float(strong_cfg["prob_d_R_positive_min"])
            and _as_float(row["q_conjunction_bh"]) <= q_threshold
        )
        if excluded_below:
            label = "excluded_below_detection_floor"
        elif excluded_breakdown:
            label = "excluded_extrapolation_breakdown"
        elif strong:
            label = "strongly_discordant_fdr"
        elif discordant_fdr:
            label = "discordant_fdr"
        elif discordant_pre_fdr:
            label = "discordant_not_fdr"
        else:
            label = "not_discordant"
        row["discordant_pre_fdr"] = bool(discordant_pre_fdr)
        row["discordant_fdr"] = bool(discordant_fdr)
        row["strongly_discordant_fdr"] = bool(strong)
        row["classification"] = label
        out.append(row)
    return out


def _bca_ci(values: np.ndarray, boot_stats: np.ndarray, observed: float, confidence: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - confidence
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
        numerator = float(np.sum(diffs**3))
        denominator = float(6.0 * (np.sum(diffs**2) ** 1.5))
        accel = 0.0 if denominator == 0.0 or not math.isfinite(denominator) else numerator / denominator

    def adjusted(prob: float) -> float:
        z = float(norm.ppf(prob))
        denom = 1.0 - accel * (z0 + z)
        if denom == 0.0:
            return prob
        return float(norm.cdf(z0 + (z0 + z) / denom))

    q_low = min(max(adjusted(alpha / 2.0), 0.0), 1.0)
    q_high = min(max(adjusted(1.0 - alpha / 2.0), 0.0), 1.0)
    if q_low > q_high:
        q_low, q_high = q_high, q_low
    return float(np.quantile(boot_stats, q_low)), float(np.quantile(boot_stats, q_high))


def _bootstrap_pair_stats(
    r_values: np.ndarray,
    d_values: np.ndarray,
    resamples: int,
    seed: int,
    include_bca: bool = False,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, r_values.size, size=(int(resamples), int(r_values.size)))
    boot_r = np.median(r_values[indices], axis=1)
    boot_d = np.median(d_values[indices], axis=1)
    p_r_positive = float((np.count_nonzero(boot_r <= 0.0) + 1) / (boot_r.size + 1))
    p_d_positive = float((np.count_nonzero(boot_d <= 0.0) + 1) / (boot_d.size + 1))
    out = {
        "n_seeds": int(r_values.size),
        "resamples": int(resamples),
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
        "nan_inf_exclusion_count": 0,
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


def _load_official_inputs(root: Path) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    analysis_dir = _analysis_result_dir(root)
    cfg = read_yaml(root / "config" / "config_official_pilot_v1.6.yaml")
    comparison_rows = _read_csv(analysis_dir / COMPARISONS_CSV)
    metric_rows = _read_csv(analysis_dir / METRICS_CSV)
    bootstrap_rows = _read_csv(analysis_dir / BOOTSTRAP_BY_CELL_CSV)
    classification_rows = _read_csv(analysis_dir / CLASSIFICATION_MATRIX_CSV)
    return cfg, comparison_rows, metric_rows, bootstrap_rows, classification_rows


def _build_cell_data(
    comparison_rows: list[dict[str, str]],
    metric_rows: list[dict[str, str]],
) -> dict[CellKey, dict[str, Any]]:
    grouped: dict[CellKey, list[dict[str, str]]] = defaultdict(list)
    for row in comparison_rows:
        grouped[_cell_key(row)].append(row)
    metrics_by_key: dict[tuple[str, float, int, int, str], dict[str, str]] = {}
    for row in metric_rows:
        if str(row["method"]).startswith("zne_"):
            metrics_by_key[
                (
                    str(row["instance"]),
                    _as_float(row["noise"]),
                    _as_int(row["shots_total"]),
                    _as_int(row["seed"]),
                    str(row["zne_order"]),
                )
            ] = row
    cell_data: dict[CellKey, dict[str, Any]] = {}
    for key in sorted(grouped, key=lambda k: (k.instance, k.noise, k.shots_total, k.zne_order)):
        rows = sorted(grouped[key], key=lambda row: _as_int(row["seed"]))
        seeds = np.asarray([_as_int(row["seed"]) for row in rows], dtype=int)
        r_values = np.asarray([_as_float(row["r_mse"]) for row in rows], dtype=float)
        d_values = np.asarray([_as_float(row["d_R"]) for row in rows], dtype=float)
        below = []
        breakdown = []
        for row in rows:
            metric = metrics_by_key[(key.instance, key.noise, key.shots_total, _as_int(row["seed"]), key.zne_order)]
            below.append(_csv_bool(metric["below_detection_floor"]))
            breakdown.append(_csv_bool(metric["extrapolation_breakdown"]))
        cell_data[key] = {
            "seeds": seeds,
            "r_values": r_values,
            "d_values": d_values,
            "below": np.asarray(below, dtype=bool),
            "breakdown": np.asarray(breakdown, dtype=bool),
        }
    return cell_data


def _compute_bootstrap_classification(
    cell_data: dict[CellKey, dict[str, Any]],
    cfg: dict[str, Any],
    seed_offset: int = 0,
    include_bca: bool = False,
    key_filter: set[CellKey] | None = None,
    seed_filter: set[int] | None = None,
    q_threshold: float = 0.10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bootstrap_seed = int(cfg["rng"]["bootstrap_seed"]) + int(seed_offset)
    tau_r = float(cfg["phase_c"]["tau_R"])
    rows: list[dict[str, Any]] = []
    ordered_keys = [key for key in sorted(cell_data, key=lambda k: (k.instance, k.noise, k.shots_total, k.zne_order)) if key_filter is None or key in key_filter]
    for cell_index, key in enumerate(ordered_keys, start=1):
        data = cell_data[key]
        seeds = data["seeds"]
        mask = np.ones(seeds.shape, dtype=bool)
        if seed_filter is not None:
            mask &= np.asarray([int(seed) in seed_filter for seed in seeds], dtype=bool)
        r_values = data["r_values"][mask]
        d_values = data["d_values"][mask]
        below = data["below"][mask]
        breakdown = data["breakdown"][mask]
        stats = _bootstrap_pair_stats(
            r_values,
            d_values,
            resamples=RESAMPLES,
            seed=bootstrap_seed + cell_index,
            include_bca=include_bca,
        )
        rows.append({
            "cell_index": cell_index,
            "cell_id": key.id(),
            "instance": key.instance,
            "noise": key.noise,
            "shots_total": key.shots_total,
            "zne_order": key.zne_order,
            "allocation": "proportional_to_abs_weights",
            "tau_R": tau_r,
            "below_detection_floor_count": int(np.count_nonzero(below)),
            "below_detection_floor_rate": float(np.mean(below)) if below.size else 0.0,
            "excluded_below_detection_floor": bool(np.any(below)),
            "extrapolation_breakdown_count": int(np.count_nonzero(breakdown)),
            "extrapolation_breakdown_rate": float(np.mean(breakdown)) if breakdown.size else 0.0,
            "excluded_extrapolation_breakdown": bool(np.any(breakdown)),
            **stats,
        })
    _benjamini_hochberg(rows, "p_conjunction", "q_conjunction_bh", "fdr_conjunction_pass", q_threshold)
    return rows, _classify_rows(rows, cfg, q_threshold=q_threshold)


def _classify_bca_copy(rows: list[dict[str, Any]], cfg: dict[str, Any], q_threshold: float = 0.10) -> list[dict[str, Any]]:
    bca_rows = []
    for source in rows:
        row = dict(source)
        row["ci_low_r_mse"] = row["bca_ci_low_r_mse"]
        row["ci_high_r_mse"] = row["bca_ci_high_r_mse"]
        row["ci_low_d_R"] = row["bca_ci_low_d_R"]
        row["ci_high_d_R"] = row["bca_ci_high_d_R"]
        bca_rows.append(row)
    _benjamini_hochberg(bca_rows, "p_conjunction", "q_conjunction_bh", "fdr_conjunction_pass", q_threshold)
    return _classify_rows(bca_rows, cfg, q_threshold=q_threshold)


def _make_scope_snapshot(root: Path) -> dict[str, Any]:
    analysis_dir = _analysis_result_dir(root)
    files = {
        "plan": root / PLAN_PATH,
        "metrics_per_seed": analysis_dir / METRICS_CSV,
        "comparisons_per_seed": analysis_dir / COMPARISONS_CSV,
        "bootstrap_by_cell": analysis_dir / BOOTSTRAP_BY_CELL_CSV,
        "classification_matrix": analysis_dir / CLASSIFICATION_MATRIX_CSV,
        "paper_go_no_go_summary": analysis_dir / PAPER_GO_NO_GO_JSON,
    }
    script_paths = [
        root / "scripts" / "26_phase_c_120_robustness.py",
        root / "src" / "phase_c_120_robustness.py",
        root / "src" / "phase_c_120_analysis.py",
        root / "src" / "phase_c_120_paper_outputs.py",
    ]
    return {
        "status": "robustness_scope_frozen_post_primary",
        "version": ROBUSTNESS_VERSION,
        "plan_version": "v0.5",
        "created_at_utc": _utc_now(),
        "primary_complete": True,
        "primary_outputs_cannot_be_changed_by_extension": True,
        "primary_output_hashes": {name: sha256_file(path) for name, path in files.items()},
        "code_hashes": {path.relative_to(root).as_posix(): sha256_file(path) for path in script_paths if path.exists()},
        "analysis_hashes_current": _current_analysis_hashes(root),
        "dependency_versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
    }


def _run_synthetic_validation(cfg: dict[str, Any]) -> dict[str, Any]:
    synthetic: dict[CellKey, dict[str, Any]] = {}
    seeds = np.arange(5, dtype=int)
    for idx in range(12):
        key = CellKey(f"S{idx // 4}", 0.001 + 0.001 * (idx % 2), 512 + 512 * ((idx // 2) % 2), ["linear", "quadratic"][idx % 2])
        if idx < 3:
            r = np.full(5, 0.55 + 0.01 * idx)
            d = np.full(5, 0.08 + 0.01 * idx)
            planted = "large_positive"
        elif idx < 6:
            r = np.zeros(5)
            d = np.zeros(5)
            planted = "zero_effect"
        else:
            r = np.asarray([0.16, 0.13, -0.02, 0.12, -0.01]) + 0.002 * idx
            d = np.asarray([0.03, -0.01, 0.02, 0.01, -0.02]) + 0.001 * idx
            planted = "ambiguous"
        synthetic[key] = {
            "seeds": seeds.copy(),
            "r_values": r,
            "d_values": d,
            "below": np.zeros(5, dtype=bool),
            "breakdown": np.zeros(5, dtype=bool),
            "planted": planted,
        }
    _, classification = _compute_bootstrap_classification(synthetic, cfg, seed_offset=500000, include_bca=False)
    planted_by_cell = {key.id(): data["planted"] for key, data in synthetic.items()}
    rows = []
    warnings: list[str] = []
    failures: list[str] = []
    for row in classification:
        planted = planted_by_cell[str(row["cell_id"])]
        is_disc = _csv_bool(row["discordant_fdr"])
        if planted == "large_positive" and not is_disc:
            failures.append(f"large_positive_not_discordant:{row['cell_id']}")
        if planted == "zero_effect" and is_disc:
            failures.append(f"zero_effect_discordant:{row['cell_id']}")
        if planted == "ambiguous":
            # Leave-one-seed instability is expected, but exact 0/1 means the fixture is not probing the middle range.
            key = _cell_key(row)
            stabilities = []
            for omitted_seed in range(5):
                keep = set(int(x) for x in seeds if int(x) != omitted_seed)
                _, loo_class = _compute_bootstrap_classification({key: synthetic[key]}, cfg, seed_offset=510000 + omitted_seed, seed_filter=keep)
                stabilities.append(1.0 if _csv_bool(loo_class[0]["discordant_fdr"]) else 0.0)
            stability = float(np.mean(stabilities))
            if stability in {0.0, 1.0}:
                warnings.append(f"ambiguous_fixture_extreme_stability:{row['cell_id']}:{stability}")
        else:
            stability = 1.0 if is_disc else 0.0
        rows.append({
            "cell_id": row["cell_id"],
            "planted": planted,
            "classification": row["classification"],
            "discordant_fdr": bool(is_disc),
            "cell_seed_stability": stability,
            "median_r_mse": row["median_r_mse"],
            "median_d_R": row["median_d_R"],
        })
    large_sign_p = float(binomtest(5, 5, 0.5, alternative="greater").pvalue)
    zero_sign_p = float(binomtest(0, 5, 0.5, alternative="greater").pvalue)
    if large_sign_p >= 0.05:
        failures.append("large_positive_sign_test_not_below_0.05")
    if zero_sign_p <= 0.50:
        failures.append("zero_effect_sign_test_not_above_0.50")
    return {
        "status": "PASS" if not failures else "FAIL",
        "required_failures": failures,
        "soft_warnings": warnings,
        "large_positive_sign_p": large_sign_p,
        "zero_effect_sign_p": zero_sign_p,
        "rows": rows,
    }


def _run_seed_loo(
    cell_data: dict[CellKey, dict[str, Any]],
    cfg: dict[str, Any],
    primary_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    all_seeds = sorted({int(seed) for data in cell_data.values() for seed in data["seeds"]})
    primary_by_id = {str(row["cell_id"]): row for row in primary_rows}
    primary_go = _summary_from_classification([dict(row) for row in primary_rows])["paper_go_strong"]
    summary_rows: list[dict[str, Any]] = []
    by_cell_rows: list[dict[str, Any]] = []
    discrepancy_total = 0
    total_evaluations = 0
    for omitted_seed in all_seeds:
        keep = set(seed for seed in all_seeds if seed != omitted_seed)
        boot_rows, pct_class = _compute_bootstrap_classification(
            cell_data,
            cfg,
            seed_offset=100000 + int(omitted_seed),
            include_bca=True,
            seed_filter=keep,
        )
        bca_class = _classify_bca_copy(boot_rows, cfg)
        bca_by_id = {str(row["cell_id"]): row for row in bca_class}
        summary = _summary_from_classification(pct_class)
        discrepancies = 0
        for row in pct_class:
            cell_id = str(row["cell_id"])
            primary_class = str(primary_by_id[cell_id]["classification"])
            pct_classification = str(row["classification"])
            bca_classification = str(bca_by_id[cell_id]["classification"])
            if pct_classification != bca_classification:
                discrepancies += 1
            by_cell_rows.append({
                "omitted_seed": omitted_seed,
                "cell_id": cell_id,
                "instance": row["instance"],
                "noise": row["noise"],
                "shots_total": row["shots_total"],
                "zne_order": row["zne_order"],
                "primary_classification": primary_class,
                "loo_classification": pct_classification,
                "loo_bca_classification": bca_classification,
                "primary_discordant_fdr": _csv_bool(primary_by_id[cell_id]["discordant_fdr"]),
                "loo_discordant_fdr": _csv_bool(row["discordant_fdr"]),
                "same_as_primary": primary_class == pct_classification,
            })
        discrepancy_total += discrepancies
        total_evaluations += len(pct_class)
        summary_rows.append({
            "omitted_seed": omitted_seed,
            **summary,
            "seed_loo_flip": bool(bool(summary["paper_go_strong"]) != bool(primary_go)),
            "complete_dissolution": bool(int(summary["discordant_fdr_count"]) == 0),
            "percentile_bca_discrepancy_count": discrepancies,
            "percentile_bca_discrepancy_rate": float(discrepancies / len(pct_class)),
        })
    flips = sum(_csv_bool(row["seed_loo_flip"]) for row in summary_rows)
    complete_dissolutions = sum(_csv_bool(row["complete_dissolution"]) for row in summary_rows)
    if flips == 0:
        seed_label = "strong"
    elif flips <= 2:
        seed_label = "moderate"
    elif flips <= 5:
        seed_label = "weak"
    else:
        seed_label = "fragile"
    discrepancy_rate = float(discrepancy_total / total_evaluations) if total_evaluations else 0.0
    if discrepancy_rate > 0.40:
        bca_label = "severe_report_parallel_labels"
    elif discrepancy_rate > 0.25:
        bca_label = "materially_sensitive_to_ci_method"
    elif discrepancy_rate > 0.10:
        bca_label = "limitation_note_required"
    else:
        bca_label = "minor_or_none"
    aggregate = {
        "seed_loo_runs": len(summary_rows),
        "seed_loo_flips": flips,
        "seed_robust": seed_label,
        "complete_dissolution_runs": complete_dissolutions,
        "loo_percentile_bca_discrepancy_count": discrepancy_total,
        "loo_percentile_bca_total_cell_evaluations": total_evaluations,
        "loo_percentile_bca_discrepancy_rate": discrepancy_rate,
        "loo_percentile_bca_discrepancy_label": bca_label,
    }
    return summary_rows, by_cell_rows, aggregate


def _run_structural_loo(
    cell_data: dict[CellKey, dict[str, Any]],
    cfg: dict[str, Any],
    primary_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    dimensions = {
        "instance": sorted({key.instance for key in cell_data}),
        "noise": sorted({key.noise for key in cell_data}),
        "shot_budget": sorted({key.shots_total for key in cell_data}),
        "zne_order": sorted({key.zne_order for key in cell_data}),
    }
    primary_go = _summary_from_classification([dict(row) for row in primary_rows])["paper_go_strong"]
    rows: list[dict[str, Any]] = []
    for dimension, values in dimensions.items():
        for value in values:
            if dimension == "instance":
                key_filter = {key for key in cell_data if key.instance != value}
            elif dimension == "noise":
                key_filter = {key for key in cell_data if key.noise != value}
            elif dimension == "shot_budget":
                key_filter = {key for key in cell_data if key.shots_total != value}
            else:
                key_filter = {key for key in cell_data if key.zne_order != value}
            _, classified = _compute_bootstrap_classification(
                cell_data,
                cfg,
                seed_offset=200000 + len(rows),
                key_filter=key_filter,
            )
            summary = _summary_from_classification(classified)
            rows.append({
                "dimension": dimension,
                "removed_value": value,
                "remaining_cells": len(classified),
                **summary,
                "paper_go_preserved": bool(summary["paper_go_strong"]) == bool(primary_go),
                "minimally_distributed": bool(
                    len(summary["discordant_distinct_noise_values"]) == 2
                    or len(summary["discordant_distinct_shot_budgets"]) == 2
                ),
            })
    family_rows: list[dict[str, Any]] = []
    for dimension in dimensions:
        subset = [row for row in rows if row["dimension"] == dimension]
        family_pass = all(_csv_bool(row["paper_go_preserved"]) for row in subset)
        family_rows.append({
            "dimension": dimension,
            "tests": len(subset),
            "family_pass": bool(family_pass),
            "failures": sum(not _csv_bool(row["paper_go_preserved"]) for row in subset),
            "complete_dissolution_runs": sum(_as_int(row["discordant_fdr_count"]) == 0 for row in subset),
            "noise_family_has_privileged_interpretive_weight": bool(dimension == "noise"),
        })
    pass_count = sum(_csv_bool(row["family_pass"]) for row in family_rows)
    if pass_count == 4:
        label = "strong"
    elif pass_count == 3:
        label = "moderate"
    elif pass_count == 2:
        label = "weak"
    else:
        label = "structurally_fragile"
    aggregate = {
        "structural_pass_count": pass_count,
        "structural_label": label,
        "structural_pattern": {str(row["dimension"]): bool(row["family_pass"]) for row in family_rows},
    }
    return rows, family_rows, aggregate


def _classify_nonparametric(
    rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    q_threshold: float = 0.10,
) -> list[dict[str, Any]]:
    discordant_cfg = cfg["go_no_go"]["discordant"]
    out = []
    for source in rows:
        row = dict(source)
        row["fdr_conjunction_pass"] = bool(_as_float(row["q_conjunction_bh"]) <= q_threshold)
        excluded = _csv_bool(row["excluded_below_detection_floor"]) or _csv_bool(row["excluded_extrapolation_breakdown"])
        pre = bool(
            not excluded
            and _as_float(row["median_r_mse"]) >= float(discordant_cfg["r_mse_min"])
            and _as_float(row["median_d_R"]) > 0.0
            and _as_float(row["prob_d_R_positive_seed"]) >= float(discordant_cfg["prob_d_R_positive_min"])
        )
        fdr = bool(pre and row["fdr_conjunction_pass"])
        if _csv_bool(row["excluded_below_detection_floor"]):
            label = "excluded_below_detection_floor"
        elif _csv_bool(row["excluded_extrapolation_breakdown"]):
            label = "excluded_extrapolation_breakdown"
        elif fdr:
            label = "discordant_fdr"
        elif pre:
            label = "discordant_not_fdr"
        else:
            label = "not_discordant"
        row["discordant_pre_fdr"] = pre
        row["discordant_fdr"] = fdr
        row["strongly_discordant_fdr"] = False
        row["classification"] = label
        out.append(row)
    return out


def _permutation_p(value: np.ndarray, statistic: str, seed: int, observed: float) -> float:
    if observed <= 0.0:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(PERMUTATION_RESAMPLES, value.size))
    sampled = value[None, :] * signs
    if statistic == "median":
        null_stats = np.median(sampled, axis=1)
    else:
        null_stats = np.mean(sampled, axis=1)
    return float((np.count_nonzero(null_stats >= observed) + 1) / (PERMUTATION_RESAMPLES + 1))


def _run_nonparametric(
    cell_data: dict[CellKey, dict[str, Any]],
    cfg: dict[str, Any],
    primary_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    primary_disc = {str(row["cell_id"]) for row in primary_rows if _csv_bool(row["discordant_fdr"])}
    median_rows: list[dict[str, Any]] = []
    mean_rows: list[dict[str, Any]] = []
    sign_rows: list[dict[str, Any]] = []
    for cell_index, key in enumerate(sorted(cell_data, key=lambda k: (k.instance, k.noise, k.shots_total, k.zne_order)), start=1):
        data = cell_data[key]
        r = data["r_values"]
        d = data["d_values"]
        below = data["below"]
        breakdown = data["breakdown"]
        base = {
            "cell_index": cell_index,
            "cell_id": key.id(),
            "instance": key.instance,
            "noise": key.noise,
            "shots_total": key.shots_total,
            "zne_order": key.zne_order,
            "median_r_mse": float(np.median(r)),
            "mean_r_mse": float(np.mean(r)),
            "median_d_R": float(np.median(d)),
            "mean_d_R": float(np.mean(d)),
            "prob_r_mse_positive_seed": float(np.mean(r > 0)),
            "prob_d_R_positive_seed": float(np.mean(d > 0)),
            "excluded_below_detection_floor": bool(np.any(below)),
            "excluded_extrapolation_breakdown": bool(np.any(breakdown)),
            "n_seeds": int(r.size),
        }
        count_r = int(np.count_nonzero(r > 0))
        count_d = int(np.count_nonzero(d > 0))
        p_r = float(binomtest(count_r, r.size, 0.5, alternative="greater").pvalue)
        p_d = float(binomtest(count_d, d.size, 0.5, alternative="greater").pvalue)
        sign_rows.append({
            **base,
            "test": "sign",
            "positive_r_count": count_r,
            "positive_d_count": count_d,
            "p_r_mse_positive": p_r,
            "p_d_R_positive": p_d,
            "p_conjunction": max(p_r, p_d),
        })
        for statistic, target_rows in [("median", median_rows), ("mean", mean_rows)]:
            observed_r = float(np.median(r) if statistic == "median" else np.mean(r))
            observed_d = float(np.median(d) if statistic == "median" else np.mean(d))
            p_r_perm = _permutation_p(r, statistic, PERMUTATION_SEED + cell_index * 10 + (0 if statistic == "median" else 1), observed_r)
            p_d_perm = _permutation_p(d, statistic, PERMUTATION_SEED + cell_index * 10 + (2 if statistic == "median" else 3), observed_d)
            target_rows.append({
                **base,
                "test": f"permutation_{statistic}",
                "p_r_mse_positive": p_r_perm,
                "p_d_R_positive": p_d_perm,
                "p_conjunction": max(p_r_perm, p_d_perm),
            })

    out_sets = []
    detail_rows: list[dict[str, Any]] = []
    for name, rows in [("sign", sign_rows), ("permutation_median", median_rows), ("permutation_mean", mean_rows)]:
        _benjamini_hochberg(rows, "p_conjunction", "q_conjunction_bh", "fdr_conjunction_pass", 0.10)
        classified = _classify_nonparametric(rows, cfg, 0.10)
        summary = _summary_from_classification(classified)
        disc = {str(row["cell_id"]) for row in classified if _csv_bool(row["discordant_fdr"])}
        intersection = primary_disc & disc
        union = primary_disc | disc
        jaccard = float(len(intersection) / len(union)) if union else 1.0
        weights = {str(row["cell_id"]): abs(_as_float(row["median_d_R"])) for row in primary_rows}
        weighted_intersection = sum(weights.get(cell_id, 0.0) for cell_id in intersection)
        weighted_union = sum(weights.get(cell_id, 0.0) for cell_id in union)
        weighted_jaccard = float(weighted_intersection / weighted_union) if weighted_union else 1.0
        distributed = bool(len(summary["discordant_distinct_noise_values"]) >= 2 and len(summary["discordant_distinct_shot_budgets"]) >= 2)
        supportive = bool(len(disc) >= 0.70 * len(primary_disc) and jaccard >= 0.60 and distributed)
        partial = bool(len(disc) >= 3 and distributed and not supportive)
        if supportive:
            support_label = "supportive"
        elif partial:
            support_label = "partially_supportive"
        else:
            support_label = "not_supportive"
        if weighted_jaccard >= 0.70 and jaccard >= 0.60:
            weighted_label = "strong_identity_and_effect_concordance"
        elif weighted_jaccard < 0.50 and jaccard >= 0.60:
            weighted_label = "set_concordant_but_effect_discordant"
        elif weighted_jaccard < 0.70 or jaccard < 0.60:
            weighted_label = "weak_or_partial_concordance"
        else:
            weighted_label = "intermediate"
        changed = sorted(union - intersection)
        top_by_effect = [
            str(row["cell_id"]) for row in sorted(primary_rows, key=lambda row: abs(_as_float(row["median_d_R"])), reverse=True)[:5]
        ]
        out_sets.append({
            "test": name,
            **summary,
            "primary_discordant_fdr_count": len(primary_disc),
            "jaccard": jaccard,
            "weighted_jaccard": weighted_jaccard,
            "distributed_across_conditions": distributed,
            "support_label": support_label,
            "weighted_jaccard_label": weighted_label,
            "status_changed_cell_count": len(changed),
            "top5_effect_cell_changed": bool(any(cell_id in changed for cell_id in top_by_effect)),
        })
        for row in classified:
            detail_rows.append({**row, "nonparametric_test": name})
    return out_sets, detail_rows, median_rows, {"primary_discordant_fdr_count": len(primary_disc)}


def _run_fragility(
    primary_rows: list[dict[str, str]],
    seed_by_cell_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stability_by_cell: dict[str, list[bool]] = defaultdict(list)
    same_by_cell: dict[str, list[bool]] = defaultdict(list)
    for row in seed_by_cell_rows:
        cell_id = str(row["cell_id"])
        stability_by_cell[cell_id].append(_csv_bool(row["loo_discordant_fdr"]))
        same_by_cell[cell_id].append(_csv_bool(row["same_as_primary"]))
    rows: list[dict[str, Any]] = []
    for row in sorted(primary_rows, key=_cell_sort_key):
        cell_id = str(row["cell_id"])
        primary_disc = _csv_bool(row["discordant_fdr"])
        if primary_disc:
            stability = float(np.mean(stability_by_cell[cell_id])) if stability_by_cell[cell_id] else 0.0
        else:
            stability = float(np.mean(same_by_cell[cell_id])) if same_by_cell[cell_id] else 0.0
        if stability >= 0.90:
            label = "stable"
        elif stability >= 0.70:
            label = "moderate"
        else:
            label = "fragile"
        q_margin = abs(0.10 - _as_float(row["q_conjunction_bh"]))
        rows.append({
            "cell_id": cell_id,
            "instance": row["instance"],
            "noise": row["noise"],
            "shots_total": row["shots_total"],
            "zne_order": row["zne_order"],
            "primary_classification": row["classification"],
            "primary_discordant_fdr": primary_disc,
            "cell_seed_stability": stability,
            "stability_label": label,
            "fragile_primary_discordant": bool(primary_disc and stability < 0.70),
            "r_margin_vs_0p10": _as_float(row["median_r_mse"]) - 0.10,
            "d_margin_vs_zero": _as_float(row["median_d_R"]),
            "ci_r_margin_vs_zero": _as_float(row["ci_low_r_mse"]),
            "ci_d_margin_vs_zero": _as_float(row["ci_low_d_R"]),
            "q_margin": q_margin,
            "borderline_q_margin_0p02": bool(q_margin <= 0.02),
        })
    bin_rows = []
    for low in np.arange(0.0, 1.0, 0.05):
        high = round(float(low + 0.05), 2)
        count = sum(float(low) <= _as_float(row["cell_seed_stability"]) < high for row in rows)
        if math.isclose(high, 1.0):
            count = sum(float(low) <= _as_float(row["cell_seed_stability"]) <= high for row in rows)
        bin_rows.append({"stability_bin_low": round(float(low), 2), "stability_bin_high": high, "cell_count": int(count)})
    fragile_cell_count = sum(_csv_bool(row["fragile_primary_discordant"]) for row in rows)
    if fragile_cell_count == 0:
        assessment = "strong"
    elif fragile_cell_count <= 2:
        assessment = "acceptable_with_disclosure"
    elif fragile_cell_count <= 4:
        assessment = "moderate_requires_explicit_qualification"
    else:
        assessment = "weak_claim_must_be_downgraded"
    aggregate = {
        "fragile_cell_count_among_primary_discordant_fdr": fragile_cell_count,
        "cell_fragility_assessment": assessment,
        "borderline_q_margin_0p02_count": sum(_csv_bool(row["borderline_q_margin_0p02"]) for row in rows),
        "primary_discordant_fdr_count": sum(_csv_bool(row["primary_discordant_fdr"]) for row in rows),
    }
    return rows, bin_rows, aggregate


def _holm_adjust(rows: list[dict[str, Any]], p_key: str, adjusted_key: str, pass_key: str, alpha: float) -> None:
    ranked = sorted(enumerate(rows), key=lambda item: (float(item[1][p_key]), int(item[0])))
    m = len(ranked)
    previous = 0.0
    adjusted: dict[int, float] = {}
    for rank, (index, row) in enumerate(ranked, start=1):
        value = min(1.0, max(previous, float(row[p_key]) * (m - rank + 1)))
        previous = value
        adjusted[index] = value
    for index, row in enumerate(rows):
        row[adjusted_key] = adjusted[index]
        row[pass_key] = bool(adjusted[index] <= alpha)


def _run_fdr_family_sensitivity(primary_rows: list[dict[str, str]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    families: list[tuple[str, str, float]] = [
        ("global_135_q0p10", "global", 0.10),
        ("global_135_q0p05", "global", 0.05),
        ("by_instance_q0p10", "instance", 0.10),
        ("by_noise_q0p10", "noise", 0.10),
        ("by_shot_budget_q0p10", "shots_total", 0.10),
        ("by_zne_order_q0p10", "zne_order", 0.10),
        ("holm_global_135_alpha0p10", "holm_global", 0.10),
    ]
    summary_rows: list[dict[str, Any]] = []
    for family_name, group_key, threshold in families:
        rows = [dict(row) for row in primary_rows]
        if group_key == "global":
            _benjamini_hochberg(rows, "p_conjunction", "q_conjunction_bh", "fdr_conjunction_pass", threshold)
        elif group_key == "holm_global":
            _holm_adjust(rows, "p_conjunction", "q_conjunction_bh", "fdr_conjunction_pass", threshold)
        else:
            grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                grouped[row[group_key]].append(row)
            for subset in grouped.values():
                _benjamini_hochberg(subset, "p_conjunction", "q_conjunction_bh", "fdr_conjunction_pass", threshold)
        classified = _classify_rows(rows, cfg, q_threshold=threshold)
        summary = _summary_from_classification(classified)
        summary_rows.append({
            "family": family_name,
            "grouping": group_key,
            "threshold": threshold,
            **summary,
            "mechanical_smaller_family_note": bool(group_key not in {"global", "holm_global"}),
        })
    aggregate = {
        "global_q0p10_preserved": bool(summary_rows[0]["paper_go_strong"]),
        "global_q0p05_preserved": bool(summary_rows[1]["paper_go_strong"]),
        "holm_global_preserved": bool(summary_rows[-1]["paper_go_strong"]),
        "subfamily_results_are_descriptive_only": True,
    }
    return summary_rows, aggregate


def _run_holdout_power(cell_data: dict[CellKey, dict[str, Any]], primary_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_disc = [row for row in primary_rows if _csv_bool(row["discordant_fdr"])]
    rows: list[dict[str, Any]] = []
    min_power: dict[str, float] = {"H10_optimistic": 1.0, "H10_central": 1.0, "H10_conservative": 1.0, "H20_optimistic": 1.0, "H20_central": 1.0, "H20_conservative": 1.0}
    for row in primary_disc:
        key = _cell_key(row)
        data = cell_data[key]
        positive_fraction = float(np.mean((data["r_values"] > 0.0) & (data["d_values"] > 0.0)))
        out = {
            "cell_id": key.id(),
            "observed_positive_fraction": positive_fraction,
        }
        for label, discount in [("optimistic", 0.0), ("central", 0.10), ("conservative", 0.20)]:
            p = min(max(positive_fraction - discount, 0.0), 1.0)
            for h in [10, 20]:
                threshold = int(math.ceil(h / 2.0))
                power = float(binom.sf(threshold - 1, h, p))
                out[f"power_H{h}_{label}"] = power
                min_power[f"H{h}_{label}"] = min(min_power[f"H{h}_{label}"], power)
        rows.append(out)
    return rows, {"min_power": min_power, "targeted_holdout_is_not_full_grid_replication": True}


def _claim_assessment(
    seed_agg: dict[str, Any],
    structural_agg: dict[str, Any],
    fragility_agg: dict[str, Any],
    nonparametric_summary: list[dict[str, Any]],
    fdr_agg: dict[str, Any],
) -> dict[str, Any]:
    r2_labels = {row["test"]: row["support_label"] for row in nonparametric_summary}
    r2_supportive = any(label == "supportive" for label in r2_labels.values())
    r2_partial = any(label == "partially_supportive" for label in r2_labels.values())
    seed_label = str(seed_agg["seed_robust"])
    structural_label = str(structural_agg["structural_label"])
    fragility_assessment = str(fragility_agg["cell_fragility_assessment"])
    if seed_label == "strong" and r2_supportive and fragility_assessment == "strong" and fdr_agg["global_q0p05_preserved"]:
        claim = "strong evidence of distributed optimization-relevant discordance"
    elif seed_label in {"strong", "moderate"} and (r2_supportive or r2_partial) and fdr_agg["global_q0p10_preserved"]:
        claim = "moderate evidence; condition-dependent qualifications may apply"
    elif seed_label in {"weak", "fragile"} or structural_label in {"weak", "structurally_fragile"}:
        claim = "supportive but fragile evidence; avoid strongly robust language"
    elif not r2_supportive and not r2_partial:
        claim = "primary result remains, but no strong robustness claim is allowed"
    else:
        claim = "mixed robustness; qualify specific weaknesses"
    return {
        "claim_strength": claim,
        "seed_robust": seed_label,
        "structural_label": structural_label,
        "cell_fragility_assessment": fragility_assessment,
        "nonparametric_labels": r2_labels,
        "fdr_global_q05_preserved": bool(fdr_agg["global_q0p05_preserved"]),
    }


def _write_limitations(path: Path, summary: dict[str, Any]) -> None:
    text = f"""# Phase C 120 Robustness Limitations And Interpretation

Status: {summary['status']}

The robustness extension does not redefine the frozen primary GO/NO-GO
decision. It can downgrade, qualify, or limit the scientific interpretation.

Claim assessment:

```text
{summary['claim_assessment']['claim_strength']}
```

Key diagnostics:

```text
seed_robust = {summary['seed_loo']['seed_robust']}
seed_loo_flips = {summary['seed_loo']['seed_loo_flips']}
structural_label = {summary['structural_loo']['structural_label']}
fragile_cell_count = {summary['fragility']['fragile_cell_count_among_primary_discordant_fdr']}
cell_fragility_assessment = {summary['fragility']['cell_fragility_assessment']}
```

Targeted holdout note:

```text
Targeted holdout, if activated later, is not a full-grid replication. It only
tests whether originally discovered cells preserve directional evidence under
new seeds.
```
"""
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def _strip_for_determinism(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"created_at_utc", "wall_clock_seconds", "determinism_check", "output_hashes"}:
                continue
            if key == "timing":
                out[key] = {"timing_primary_summary": _strip_for_determinism(item.get("timing_primary_summary", {}))}
                continue
            out[key] = _strip_for_determinism(item)
        return out
    if isinstance(value, list):
        return [_strip_for_determinism(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _determinism_check(first: dict[str, Any], root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    second = _run_pipeline(root)
    elapsed = time.perf_counter() - started
    first_sanitized = _strip_for_determinism(first)
    second_sanitized = _strip_for_determinism(second)
    first_blob = json.dumps(first_sanitized, sort_keys=True, default=_json_default)
    second_blob = json.dumps(second_sanitized, sort_keys=True, default=_json_default)
    if first_blob == second_blob:
        return {
            "status": "PASS",
            "mode": "full_tier1_second_run_sanitized",
            "numeric_tolerance": NUMERIC_TOL,
            "second_run_wall_clock_seconds": elapsed,
            "allowed_differences": ["created_at_utc", "wall_clock_seconds", "timing_runtime_seconds", "output_hashes"],
        }
    return {
        "status": "FAIL",
        "mode": "full_tier1_second_run_sanitized",
        "numeric_tolerance": NUMERIC_TOL,
        "second_run_wall_clock_seconds": elapsed,
        "allowed_differences": ["created_at_utc", "wall_clock_seconds", "timing_runtime_seconds", "output_hashes"],
        "first_sanitized_sha256": _sha256_text(first_blob),
        "second_sanitized_sha256": _sha256_text(second_blob),
    }


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run_pipeline(root: Path) -> dict[str, Any]:
    cfg, comparison_rows, metric_rows, bootstrap_rows, primary_rows = _load_official_inputs(root)
    cell_data = _build_cell_data(comparison_rows, metric_rows)
    t0 = time.perf_counter()
    _, timing_class = _compute_bootstrap_classification(cell_data, cfg, seed_offset=900000)
    primary_pass_seconds = time.perf_counter() - t0
    timing_summary = {
        "machine_id": platform.node(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version.split()[0],
        "primary_pass_time_seconds": primary_pass_seconds,
        "estimated_R1_seed_loo_seconds": primary_pass_seconds * 30,
        "estimated_R1_structural_seconds": primary_pass_seconds * 13,
        "estimated_R2_nonparametric_seconds": "measured_during_full_run",
        "timing_primary_summary": _summary_from_classification(timing_class),
    }
    synthetic_summary = _run_synthetic_validation(cfg)
    seed_summary_rows, seed_by_cell_rows, seed_agg = _run_seed_loo(cell_data, cfg, primary_rows)
    structural_rows, structural_family_rows, structural_agg = _run_structural_loo(cell_data, cfg, primary_rows)
    nonparametric_summary, nonparametric_detail, _, _ = _run_nonparametric(cell_data, cfg, primary_rows)
    fragility_rows, fragility_bins, fragility_agg = _run_fragility(primary_rows, seed_by_cell_rows)
    fdr_rows, fdr_agg = _run_fdr_family_sensitivity(primary_rows, cfg)
    holdout_power_rows, holdout_power_agg = _run_holdout_power(cell_data, primary_rows)
    claim_assessment = _claim_assessment(seed_agg, structural_agg, fragility_agg, nonparametric_summary, fdr_agg)
    checklist_rows = [
        {"diagnostic": "Seed LOO", "paper_go_strong_preserved": seed_agg["seed_loo_flips"] == 0, "discordant_fdr_count": "see leave_one_seed_summary.csv", "key_change": seed_agg["seed_robust"]},
        {"diagnostic": "Structural LOO instance", "paper_go_strong_preserved": structural_agg["structural_pattern"].get("instance"), "discordant_fdr_count": "see structural_loo_summary.csv", "key_change": "family pass/fail"},
        {"diagnostic": "Structural LOO noise", "paper_go_strong_preserved": structural_agg["structural_pattern"].get("noise"), "discordant_fdr_count": "see structural_loo_summary.csv", "key_change": "privileged interpretive weight"},
        {"diagnostic": "Structural LOO shot budget", "paper_go_strong_preserved": structural_agg["structural_pattern"].get("shot_budget"), "discordant_fdr_count": "see structural_loo_summary.csv", "key_change": "family pass/fail"},
        {"diagnostic": "Structural LOO ZNE order", "paper_go_strong_preserved": structural_agg["structural_pattern"].get("zne_order"), "discordant_fdr_count": "see structural_loo_summary.csv", "key_change": "family pass/fail"},
        {"diagnostic": "Nonparametric sign", "paper_go_strong_preserved": next(row for row in nonparametric_summary if row["test"] == "sign")["paper_go_strong"], "discordant_fdr_count": next(row for row in nonparametric_summary if row["test"] == "sign")["discordant_fdr_count"], "key_change": next(row for row in nonparametric_summary if row["test"] == "sign")["support_label"]},
        {"diagnostic": "Nonparametric permutation", "paper_go_strong_preserved": next(row for row in nonparametric_summary if row["test"] == "permutation_median")["paper_go_strong"], "discordant_fdr_count": next(row for row in nonparametric_summary if row["test"] == "permutation_median")["discordant_fdr_count"], "key_change": next(row for row in nonparametric_summary if row["test"] == "permutation_median")["support_label"]},
        {"diagnostic": "Fragility", "paper_go_strong_preserved": fragility_agg["cell_fragility_assessment"] not in {"weak_claim_must_be_downgraded"}, "discordant_fdr_count": fragility_agg["primary_discordant_fdr_count"], "key_change": fragility_agg["cell_fragility_assessment"]},
        {"diagnostic": "FDR q=0.05", "paper_go_strong_preserved": fdr_agg["global_q0p05_preserved"], "discordant_fdr_count": next(row for row in fdr_rows if row["family"] == "global_135_q0p05")["discordant_fdr_count"], "key_change": "sensitivity"},
        {"diagnostic": "FDR Holm", "paper_go_strong_preserved": fdr_agg["holm_global_preserved"], "discordant_fdr_count": next(row for row in fdr_rows if row["family"] == "holm_global_135_alpha0p10")["discordant_fdr_count"], "key_change": "sensitivity"},
    ]
    summary = {
        "version": ROBUSTNESS_VERSION,
        "plan_version": "v0.5",
        "created_at_utc": _utc_now(),
        "status": "phase_c_120_robustness_tier1_complete",
        "primary_go_no_go_unchanged": True,
        "synthetic_validation_status": synthetic_summary["status"],
        "timing": timing_summary,
        "seed_loo": seed_agg,
        "structural_loo": structural_agg,
        "nonparametric": {row["test"]: row for row in nonparametric_summary},
        "fragility": fragility_agg,
        "fdr_family": fdr_agg,
        "holdout_power": holdout_power_agg,
        "claim_assessment": claim_assessment,
        "outputs": {
            "robustness_scope_snapshot": "results/phase_c_120_robustness/robustness_scope_snapshot.json",
            "synthetic_validation_summary": "results/phase_c_120_robustness/synthetic_validation_summary.json",
            "robustness_timing_calibration": "results/phase_c_120_robustness/robustness_timing_calibration.json",
            "leave_one_seed_summary": "results/phase_c_120_robustness/leave_one_seed_summary.csv",
            "leave_one_seed_by_cell": "results/phase_c_120_robustness/leave_one_seed_by_cell.csv",
            "structural_loo_summary": "results/phase_c_120_robustness/structural_loo_summary.csv",
            "structural_loo_family_summary": "results/phase_c_120_robustness/structural_loo_family_summary.csv",
            "nonparametric_summary": "results/phase_c_120_robustness/nonparametric_summary.csv",
            "nonparametric_by_cell": "results/phase_c_120_robustness/nonparametric_by_cell.csv",
            "fragility_by_cell": "results/phase_c_120_robustness/fragility_by_cell.csv",
            "fragility_distribution": "results/phase_c_120_robustness/fragility_distribution.csv",
            "fdr_family_sensitivity": "results/phase_c_120_robustness/fdr_family_sensitivity.csv",
            "holdout_power_analysis": "results/phase_c_120_robustness/holdout_power_analysis.csv",
            "robustness_reviewer_checklist": "results/phase_c_120_robustness/robustness_reviewer_checklist.csv",
            "robustness_limitations": "results/phase_c_120_robustness/robustness_limitations.md",
        },
        "_rows": {
            "seed_summary_rows": seed_summary_rows,
            "seed_by_cell_rows": seed_by_cell_rows,
            "structural_rows": structural_rows,
            "structural_family_rows": structural_family_rows,
            "nonparametric_summary": nonparametric_summary,
            "nonparametric_detail": nonparametric_detail,
            "fragility_rows": fragility_rows,
            "fragility_bins": fragility_bins,
            "fdr_rows": fdr_rows,
            "holdout_power_rows": holdout_power_rows,
            "checklist_rows": checklist_rows,
            "synthetic_rows": synthetic_summary["rows"],
        },
        "_synthetic": synthetic_summary,
    }
    return summary


def run_phase_c_120_robustness(
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    result_dir = _result_dir(root)
    output_paths = [
        result_dir / "robustness_summary.json",
        result_dir / "robustness_scope_snapshot.json",
        result_dir / "synthetic_validation_summary.json",
        result_dir / "robustness_timing_calibration.json",
    ]
    if not overwrite:
        existing = [path for path in output_paths if path.exists()]
        if existing:
            raise FileExistsError(f"Refusing to overwrite robustness outputs: {existing[0]}")
    started = time.perf_counter()
    scope_snapshot = _make_scope_snapshot(root)
    pipeline = _run_pipeline(root)
    determinism = _determinism_check(pipeline, root)
    elapsed = time.perf_counter() - started
    rows = pipeline.pop("_rows")
    synthetic = pipeline.pop("_synthetic")
    pipeline["wall_clock_seconds"] = elapsed
    pipeline["determinism_check"] = determinism

    write_json_atomic(result_dir / "robustness_scope_snapshot.json", scope_snapshot, overwrite=True)
    write_json_atomic(result_dir / "synthetic_validation_summary.json", {k: v for k, v in synthetic.items() if k != "rows"}, overwrite=True)
    _write_csv_atomic(result_dir / "synthetic_validation_rows.csv", rows["synthetic_rows"], overwrite=True)
    write_json_atomic(result_dir / "robustness_timing_calibration.json", pipeline["timing"], overwrite=True)
    _write_csv_atomic(result_dir / "leave_one_seed_summary.csv", rows["seed_summary_rows"], overwrite=True)
    _write_csv_atomic(result_dir / "leave_one_seed_by_cell.csv", rows["seed_by_cell_rows"], overwrite=True)
    _write_csv_atomic(result_dir / "structural_loo_summary.csv", rows["structural_rows"], overwrite=True)
    _write_csv_atomic(result_dir / "structural_loo_family_summary.csv", rows["structural_family_rows"], overwrite=True)
    _write_csv_atomic(result_dir / "nonparametric_summary.csv", rows["nonparametric_summary"], overwrite=True)
    _write_csv_atomic(result_dir / "nonparametric_by_cell.csv", rows["nonparametric_detail"], overwrite=True)
    _write_csv_atomic(result_dir / "fragility_by_cell.csv", rows["fragility_rows"], overwrite=True)
    _write_csv_atomic(result_dir / "fragility_distribution.csv", rows["fragility_bins"], overwrite=True)
    _write_csv_atomic(result_dir / "fdr_family_sensitivity.csv", rows["fdr_rows"], overwrite=True)
    _write_csv_atomic(result_dir / "holdout_power_analysis.csv", rows["holdout_power_rows"], overwrite=True)
    _write_csv_atomic(result_dir / "robustness_reviewer_checklist.csv", rows["checklist_rows"], overwrite=True)
    _write_limitations(result_dir / "robustness_limitations.md", pipeline)

    output_hashes = {}
    for path in sorted(result_dir.iterdir()):
        if path.is_file() and path.name != "robustness_summary.json":
            output_hashes[path.name] = sha256_file(path)
    pipeline["output_hashes"] = output_hashes
    write_json_atomic(result_dir / "robustness_summary.json", pipeline, overwrite=True)
    return pipeline


def make_phase_c_120_robustness_figures(
    overwrite: bool = False,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    result_dir = _result_dir(root)
    figure_dir = ensure_dir(root / "results" / "figures")
    fragility_path = result_dir / "fragility_by_cell.csv"
    fragility_distribution_path = result_dir / "fragility_distribution.csv"
    nonparametric_path = result_dir / "nonparametric_summary.csv"
    for path in [fragility_path, fragility_distribution_path, nonparametric_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing robustness input: {path}")

    fragility_rows = _read_csv(fragility_path)
    distribution_rows = _read_csv(fragility_distribution_path)
    nonparam_rows = _read_csv(nonparametric_path)

    seed_fig = figure_dir / "phase_c_120_robustness_seed_stability.png"
    dist_fig = figure_dir / "phase_c_120_fragility_distribution.png"
    nonparam_fig = figure_dir / "phase_c_120_nonparametric_comparison.png"
    for path in [seed_fig, dist_fig, nonparam_fig]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite figure: {path}")

    ordered = sorted(fragility_rows, key=lambda row: (_as_float(row["cell_seed_stability"]), str(row["cell_id"])))
    colors = ["#c43c39" if _csv_bool(row["primary_discordant_fdr"]) else "#4f81bd" for row in ordered]
    plt.figure(figsize=(11, 4.5))
    plt.bar(range(len(ordered)), [_as_float(row["cell_seed_stability"]) for row in ordered], color=colors, linewidth=0)
    plt.axhline(0.90, color="#333333", linestyle="--", linewidth=1, label="stable threshold 0.90")
    plt.axhline(0.70, color="#777777", linestyle=":", linewidth=1, label="fragility threshold 0.70")
    plt.ylim(0, 1.04)
    plt.xlabel("Cells sorted by seed stability")
    plt.ylabel("Cell seed stability")
    plt.title("Phase C 120 robustness: seed stability by cell")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(seed_fig, dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    mids = [(_as_float(row["stability_bin_low"]) + _as_float(row["stability_bin_high"])) / 2.0 for row in distribution_rows]
    counts = [_as_int(row["cell_count"]) for row in distribution_rows]
    plt.bar(mids, counts, width=0.045, color="#5b8cc0", edgecolor="white")
    plt.axvline(0.70, color="#777777", linestyle=":", linewidth=1)
    plt.axvline(0.90, color="#333333", linestyle="--", linewidth=1)
    plt.xlabel("Cell seed stability")
    plt.ylabel("Cell count")
    plt.title("Phase C 120 robustness: fragility distribution")
    plt.tight_layout()
    plt.savefig(dist_fig, dpi=180)
    plt.close()

    labels = [str(row["test"]) for row in nonparam_rows]
    disc_counts = [_as_int(row["discordant_fdr_count"]) for row in nonparam_rows]
    jaccards = [_as_float(row["jaccard"]) for row in nonparam_rows]
    x = np.arange(len(labels))
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(x - 0.18, disc_counts, width=0.36, color="#6aa84f", label="discordant FDR count")
    ax1.axhline(13, color="#333333", linestyle="--", linewidth=1, label="primary count 13")
    ax1.set_ylabel("Discordant FDR count")
    ax1.set_xticks(x, labels, rotation=20, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, jaccards, color="#c27c0e", marker="o", linewidth=2, label="Jaccard")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Jaccard vs primary set")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=False)
    plt.title("Phase C 120 robustness: nonparametric concordance")
    plt.tight_layout()
    plt.savefig(nonparam_fig, dpi=180)
    plt.close()

    outputs = {
        "status": "phase_c_120_robustness_figures_complete",
        "created_at_utc": _utc_now(),
        "figures": {
            "fig_robustness_seed_stability": seed_fig.relative_to(root).as_posix(),
            "fig_fragility_distribution": dist_fig.relative_to(root).as_posix(),
            "fig_nonparametric_comparison": nonparam_fig.relative_to(root).as_posix(),
        },
        "figure_hashes": {
            seed_fig.name: sha256_file(seed_fig),
            dist_fig.name: sha256_file(dist_fig),
            nonparam_fig.name: sha256_file(nonparam_fig),
        },
    }
    write_json_atomic(result_dir / "robustness_figures_summary.json", outputs, overwrite=True)
    return outputs
