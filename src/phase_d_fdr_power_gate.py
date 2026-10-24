from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic


FDR_POWER_VERSION = "phase_d_fdr_power_gate_v1"
DEFAULT_SIMULATIONS = 10_000
DEFAULT_SEED = 20260514
FAMILY_SIZE = 540
INSTANCE_COUNT = 12
CELLS_PER_INSTANCE = 45
SEEDS_PER_CELL = 30
Q_PRIMARY = 0.05
Q_CONTINUITY = 0.10
BY_C_M = float(np.sum(1.0 / np.arange(1, FAMILY_SIZE + 1)))
INSTANCE_BINOMIAL_ALPHA = 0.05
INSTANCE_STRESS_NOISE_MIN = 0.005
INSTANCE_STRESS_SHOT_MIN = 1024


@dataclass(frozen=True)
class Scenario:
    name: str
    effect_multiplier: float


SCENARIOS = (
    Scenario("optimistic", 1.00),
    Scenario("central", 0.75),
    Scenario("conservative", 0.50),
)


def _binomial_pass_count(n: int, alpha: float, p0: float = 0.5) -> int:
    for k in range(n + 1):
        tail = sum(math.comb(n, j) * (p0 ** j) * ((1.0 - p0) ** (n - j)) for j in range(k, n + 1))
        if tail <= alpha:
            return k
    return n + 1


INSTANCE_BINOMIAL_PASS_COUNT = _binomial_pass_count(INSTANCE_COUNT, INSTANCE_BINOMIAL_ALPHA)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _as_float(value: Any) -> float:
    return float(value)


def _as_int(value: Any) -> int:
    return int(float(value))


def _csv_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _normal_sf(z: np.ndarray) -> np.ndarray:
    # Vectorized standard-normal survival function without requiring scipy.
    erf = np.vectorize(math.erf, otypes=[float])
    return 0.5 * (1.0 - erf(z / math.sqrt(2.0)))


def _bh_pass(p_values: np.ndarray, q: float) -> np.ndarray:
    order = np.argsort(p_values, axis=1, kind="mergesort")
    sorted_p = np.take_along_axis(p_values, order, axis=1)
    thresholds = (np.arange(1, p_values.shape[1] + 1, dtype=float) / p_values.shape[1]) * q
    below = sorted_p <= thresholds
    any_pass = np.any(below, axis=1)
    max_rank = np.where(any_pass, below.shape[1] - 1 - np.argmax(below[:, ::-1], axis=1), -1)
    sorted_pass = np.arange(p_values.shape[1])[None, :] <= max_rank[:, None]
    sorted_pass[~any_pass, :] = False
    out = np.zeros_like(sorted_pass, dtype=bool)
    np.put_along_axis(out, order, sorted_pass, axis=1)
    return out


def _holm_pass(p_values: np.ndarray, alpha: float) -> np.ndarray:
    order = np.argsort(p_values, axis=1, kind="mergesort")
    sorted_p = np.take_along_axis(p_values, order, axis=1)
    thresholds = alpha / (p_values.shape[1] - np.arange(p_values.shape[1], dtype=float))
    passes_step = sorted_p <= thresholds
    first_fail = np.argmax(~passes_step, axis=1)
    all_pass = np.all(passes_step, axis=1)
    cutoff = np.where(all_pass, p_values.shape[1], first_fail)
    sorted_pass = np.arange(p_values.shape[1])[None, :] < cutoff[:, None]
    out = np.zeros_like(sorted_pass, dtype=bool)
    np.put_along_axis(out, order, sorted_pass, axis=1)
    return out


def _template_table(root: Path) -> dict[str, Any]:
    comparison_path = root / "results" / "phase_c_120_analysis" / "comparisons_per_seed.csv"
    bootstrap_path = root / "results" / "phase_c_120_analysis" / "bootstrap_by_cell.csv"
    class_path = root / "results" / "phase_c_120_analysis" / "classification_matrix.csv"
    comparison_rows = _read_csv(comparison_path)
    bootstrap_rows = _read_csv(bootstrap_path)
    class_rows = _read_csv(class_path)
    grouped: dict[tuple[str, float, int, str], list[dict[str, str]]] = {}
    for row in comparison_rows:
        key = (row["instance"], _as_float(row["noise"]), _as_int(row["shots_total"]), row["zne_order"])
        grouped.setdefault(key, []).append(row)
    bootstrap_by_key = {
        (row["instance"], _as_float(row["noise"]), _as_int(row["shots_total"]), row["zne_order"]): row
        for row in bootstrap_rows
    }
    class_by_key = {
        (row["instance"], _as_float(row["noise"]), _as_int(row["shots_total"]), row["zne_order"]): row
        for row in class_rows
    }

    records: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: _as_int(row["seed"]))
        r = np.asarray([_as_float(row["r_mse"]) for row in rows], dtype=float)
        d = np.asarray([_as_float(row["d_R"]) for row in rows], dtype=float)
        sd_r = float(np.std(r, ddof=1))
        sd_d = float(np.std(d, ddof=1))
        corr = float(np.corrcoef(r, d)[0, 1]) if sd_r > 1e-12 and sd_d > 1e-12 else 0.0
        corr = min(0.95, max(-0.95, corr)) if math.isfinite(corr) else 0.0
        b = bootstrap_by_key[key]
        c = class_by_key[key]
        records.append({
            "cell_id": b["cell_id"],
            "instance": key[0],
            "noise": key[1],
            "shots_total": key[2],
            "zne_order": key[3],
            "mean_r": float(np.mean(r)),
            "mean_d": float(np.mean(d)),
            "sd_r": max(sd_r, 1e-9),
            "sd_d": max(sd_d, 1e-9),
            "corr": corr,
            "excluded": bool(_csv_bool(b["excluded_below_detection_floor"]) or _csv_bool(b["excluded_extrapolation_breakdown"])),
            "phase_c_discordant_pre_fdr": _csv_bool(c["discordant_pre_fdr"]),
            "phase_c_discordant_fdr": _csv_bool(c["discordant_fdr"]),
        })

    return {
        "records": records,
        "source_hashes": {
            "comparisons_per_seed": sha256_file(comparison_path),
            "bootstrap_by_cell": sha256_file(bootstrap_path),
            "classification_matrix": sha256_file(class_path),
        },
    }


def _phase_d_slot_candidates(records: list[dict[str, Any]]) -> tuple[list[np.ndarray], np.ndarray, list[dict[str, Any]]]:
    by_condition: dict[tuple[float, int, str], list[int]] = {}
    for index, row in enumerate(records):
        key = (float(row["noise"]), int(row["shots_total"]), str(row["zne_order"]))
        by_condition.setdefault(key, []).append(index)

    conditions = sorted(by_condition, key=lambda key: (key[0], key[1], key[2]))
    slots_one_instance = [
        {
            "noise": noise,
            "shots_total": shots_total,
            "zne_order": zne_order,
            "stress_endpoint_cell": bool(noise >= INSTANCE_STRESS_NOISE_MIN and shots_total >= INSTANCE_STRESS_SHOT_MIN),
        }
        for noise, shots_total, zne_order in conditions
    ]
    if len(slots_one_instance) != CELLS_PER_INSTANCE:
        raise RuntimeError(f"Expected {CELLS_PER_INSTANCE} condition slots, found {len(slots_one_instance)}")

    slot_candidates: list[np.ndarray] = []
    stress_mask: list[bool] = []
    for _instance in range(INSTANCE_COUNT):
        for slot in slots_one_instance:
            key = (float(slot["noise"]), int(slot["shots_total"]), str(slot["zne_order"]))
            slot_candidates.append(np.asarray(by_condition[key], dtype=int))
            stress_mask.append(bool(slot["stress_endpoint_cell"]))
    return slot_candidates, np.asarray(stress_mask, dtype=bool), slots_one_instance


def _simulate_chunk(
    rng: np.random.Generator,
    template: dict[str, np.ndarray],
    scenario: Scenario,
    n_families: int,
) -> dict[str, np.ndarray]:
    slot_candidates = template["slot_candidates"]
    template_idx = np.empty((n_families, FAMILY_SIZE), dtype=int)
    for slot_index, candidates in enumerate(slot_candidates):
        template_idx[:, slot_index] = candidates[rng.integers(0, len(candidates), size=n_families)]
    mean_r = template["mean_r"][template_idx] * scenario.effect_multiplier
    mean_d = template["mean_d"][template_idx] * scenario.effect_multiplier
    sd_r = template["sd_r"][template_idx]
    sd_d = template["sd_d"][template_idx]
    corr = template["corr"][template_idx]
    excluded = template["excluded"][template_idx]

    z1 = rng.standard_normal(size=(n_families, FAMILY_SIZE, SEEDS_PER_CELL))
    z2 = rng.standard_normal(size=(n_families, FAMILY_SIZE, SEEDS_PER_CELL))
    r = mean_r[..., None] + sd_r[..., None] * z1
    d = mean_d[..., None] + sd_d[..., None] * (corr[..., None] * z1 + np.sqrt(1.0 - corr[..., None] ** 2) * z2)
    median_r = np.median(r, axis=2)
    median_d = np.median(d, axis=2)
    sample_sd_r = np.std(r, axis=2, ddof=1)
    sample_sd_d = np.std(d, axis=2, ddof=1)
    se_r = 1.2533141373155001 * sample_sd_r / math.sqrt(SEEDS_PER_CELL)
    se_d = 1.2533141373155001 * sample_sd_d / math.sqrt(SEEDS_PER_CELL)
    se_r = np.maximum(se_r, 1e-12)
    se_d = np.maximum(se_d, 1e-12)
    p_r = np.clip(_normal_sf(median_r / se_r), 0.0, 1.0)
    p_d = np.clip(_normal_sf(median_d / se_d), 0.0, 1.0)
    p_conj = np.maximum(p_r, p_d)
    ci_low_r = median_r - 1.959963984540054 * se_r
    ci_low_d = median_d - 1.959963984540054 * se_d
    prob_d_positive_seed = np.mean(d > 0.0, axis=2)
    pre = (
        ~excluded
        & (median_r >= 0.10)
        & (ci_low_r > 0.0)
        & (median_d > 0.0)
        & (ci_low_d >= 0.0)
        & (prob_d_positive_seed >= 0.60)
    )
    cell_positive_direction = ~excluded & (median_r > 0.0) & (median_d > 0.0)
    return {
        "p_conj": p_conj,
        "pre": pre,
        "cell_positive_direction": cell_positive_direction,
        "excluded": excluded,
        "median_r": median_r,
        "median_d": median_d,
    }


def _endpoint_metrics(
    simulated: dict[str, np.ndarray],
    fdr_pass_q05: np.ndarray,
    fdr_pass_q10: np.ndarray,
    holm_pass: np.ndarray,
    by_pass: np.ndarray,
    stress_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    pre = simulated["pre"]
    cell_pos = simulated["cell_positive_direction"]
    nonexcluded = ~simulated["excluded"]
    fdr_q05 = pre & fdr_pass_q05
    fdr_q10 = pre & fdr_pass_q10
    fdr_holm = pre & holm_pass
    fdr_by = pre & by_pass

    def criterion_2(fdr: np.ndarray) -> np.ndarray:
        pre_count = np.sum(pre, axis=1)
        fdr_count = np.sum(fdr, axis=1)
        by_instance = np.reshape(pre, (pre.shape[0], INSTANCE_COUNT, CELLS_PER_INSTANCE))
        by_noise = np.reshape(pre, (pre.shape[0], INSTANCE_COUNT, 5, 3, 3))
        by_shot = np.reshape(pre, (pre.shape[0], INSTANCE_COUNT, 5, 3, 3))
        spread = (
            (np.sum(np.any(by_instance, axis=2), axis=1) > 1)
            | (np.sum(np.any(by_noise, axis=(1, 3, 4)), axis=1) > 1)
            | (np.sum(np.any(by_shot, axis=(1, 2, 4)), axis=1) > 1)
        )
        return (pre_count >= 5) & (fdr_count >= 3) & spread

    def legacy_instance_fdr_diagnostic(fdr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        shaped_pos = np.reshape(cell_pos, (cell_pos.shape[0], INSTANCE_COUNT, CELLS_PER_INSTANCE))
        shaped_non = np.reshape(nonexcluded, (nonexcluded.shape[0], INSTANCE_COUNT, CELLS_PER_INSTANCE))
        shaped_pre = np.reshape(pre, (pre.shape[0], INSTANCE_COUNT, CELLS_PER_INSTANCE))
        shaped_fdr = np.reshape(fdr, (fdr.shape[0], INSTANCE_COUNT, CELLS_PER_INSTANCE))
        shaped_r = np.reshape(simulated["median_r"], (fdr.shape[0], INSTANCE_COUNT, CELLS_PER_INSTANCE))
        shaped_d = np.reshape(simulated["median_d"], (fdr.shape[0], INSTANCE_COUNT, CELLS_PER_INSTANCE))
        non_count = np.maximum(1, np.sum(shaped_non, axis=2))
        pos_fraction = np.sum(shaped_pos, axis=2) / non_count
        masked_r = np.where(shaped_non, shaped_r, np.nan)
        masked_d = np.where(shaped_non, shaped_d, np.nan)
        instance_positive = (
            (pos_fraction >= 0.50)
            & (np.nanmedian(masked_r, axis=2) > 0.0)
            & (np.nanmedian(masked_d, axis=2) > 0.0)
        )
        instance_supportive = (
            instance_positive
            & (np.sum(shaped_fdr, axis=2) >= 1)
            & (np.sum(shaped_pre, axis=2) >= 3)
        )
        return np.sum(instance_supportive, axis=1), np.sum(instance_positive, axis=1)

    def stress_instance_endpoint() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        stress_one_instance = np.reshape(stress_mask, (INSTANCE_COUNT, CELLS_PER_INSTANCE))[0]
        shaped_pos = np.reshape(cell_pos, (cell_pos.shape[0], INSTANCE_COUNT, CELLS_PER_INSTANCE))
        shaped_non = np.reshape(nonexcluded, (nonexcluded.shape[0], INSTANCE_COUNT, CELLS_PER_INSTANCE))
        shaped_pre = np.reshape(pre, (pre.shape[0], INSTANCE_COUNT, CELLS_PER_INSTANCE))
        stress_pos = shaped_pos[:, :, stress_one_instance]
        stress_non = shaped_non[:, :, stress_one_instance]
        stress_pre = shaped_pre[:, :, stress_one_instance]
        stress_non_count = np.maximum(1, np.sum(stress_non, axis=2))
        stress_positive_fraction = np.sum(stress_pos, axis=2) / stress_non_count
        instance_positive_stress = np.sum(stress_pre, axis=2) >= 3
        instance_positive_stress_count = np.sum(instance_positive_stress, axis=1)
        endpoint_pass = instance_positive_stress_count >= INSTANCE_BINOMIAL_PASS_COUNT
        return endpoint_pass, instance_positive_stress_count, np.mean(stress_positive_fraction, axis=1)

    supportive_q05, legacy_positive_count = legacy_instance_fdr_diagnostic(fdr_q05)
    supportive_q10, _ = legacy_instance_fdr_diagnostic(fdr_q10)
    endpoint_q05, positive_stress_count, mean_stress_positive_fraction = stress_instance_endpoint()
    criterion_q05 = criterion_2(fdr_q05)
    criterion_q10 = criterion_2(fdr_q10)
    return {
        "pre_count": np.sum(pre, axis=1),
        "fdr_q05_count": np.sum(fdr_q05, axis=1),
        "fdr_q10_count": np.sum(fdr_q10, axis=1),
        "holm_count": np.sum(fdr_holm, axis=1),
        "by_count": np.sum(fdr_by, axis=1),
        "criterion2_q05": criterion_q05,
        "criterion2_q10": criterion_q10,
        "zero_q05": np.sum(fdr_q05, axis=1) == 0,
        "zero_q10": np.sum(fdr_q10, axis=1) == 0,
        "instance_endpoint_q05": endpoint_q05,
        "instance_endpoint_q10": endpoint_q05,
        "co_primary_q05": criterion_q05 & endpoint_q05,
        "instance_supportive_q05_count": supportive_q05,
        "instance_supportive_q10_count": supportive_q10,
        "legacy_instance_positive_direction_count": legacy_positive_count,
        "instance_positive_stress_count": positive_stress_count,
        "mean_instance_stress_positive_fraction": mean_stress_positive_fraction,
    }


def _summarize_values(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _summarize_scenario(name: str, multiplier: float, accum: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    merged = {key: np.concatenate(value) for key, value in accum.items()}
    return {
        "scenario": name,
        "effect_multiplier": multiplier,
        "simulated_families": int(len(merged["pre_count"])),
        "expected_discordant_pre_fdr_count": float(np.mean(merged["pre_count"])),
        "expected_discordant_fdr_q05_count": float(np.mean(merged["fdr_q05_count"])),
        "expected_discordant_fdr_q10_count": float(np.mean(merged["fdr_q10_count"])),
        "expected_holm_count": float(np.mean(merged["holm_count"])),
        "expected_by_count": float(np.mean(merged["by_count"])),
        "probability_of_criterion_2_pass_q05": float(np.mean(merged["criterion2_q05"])),
        "probability_of_criterion_2_pass_q10": float(np.mean(merged["criterion2_q10"])),
        "probability_of_zero_discoveries_q05": float(np.mean(merged["zero_q05"])),
        "probability_of_zero_discoveries_q10": float(np.mean(merged["zero_q10"])),
        "probability_instance_endpoint_pass_q05": float(np.mean(merged["instance_endpoint_q05"])),
        "probability_instance_endpoint_pass_q10": float(np.mean(merged["instance_endpoint_q10"])),
        "probability_co_primary_pass_q05": float(np.mean(merged["co_primary_q05"])),
        "expected_instance_supportive_q05_count": float(np.mean(merged["instance_supportive_q05_count"])),
        "expected_instance_supportive_q10_count": float(np.mean(merged["instance_supportive_q10_count"])),
        "expected_legacy_instance_positive_direction_count": float(np.mean(merged["legacy_instance_positive_direction_count"])),
        "expected_instance_positive_stress_count": float(np.mean(merged["instance_positive_stress_count"])),
        "expected_mean_instance_stress_positive_fraction": float(np.mean(merged["mean_instance_stress_positive_fraction"])),
        "pre_count_distribution": _summarize_values(merged["pre_count"]),
        "fdr_q05_count_distribution": _summarize_values(merged["fdr_q05_count"]),
        "fdr_q10_count_distribution": _summarize_values(merged["fdr_q10_count"]),
        "instance_supportive_q05_count_distribution": _summarize_values(merged["instance_supportive_q05_count"]),
        "instance_positive_stress_count_distribution": _summarize_values(merged["instance_positive_stress_count"]),
    }


def _decision(scenario_rows: list[dict[str, Any]]) -> dict[str, Any]:
    central = next(row for row in scenario_rows if row["scenario"] == "central")
    conservative = next(row for row in scenario_rows if row["scenario"] == "conservative")
    confirmatory_pass = bool(
        central["probability_of_criterion_2_pass_q05"] >= 0.50
        and central["probability_of_zero_discoveries_q05"] <= 0.25
    )
    amend_required = not confirmatory_pass
    conservative_warning = bool(
        conservative["probability_of_criterion_2_pass_q05"] < 0.50
        and confirmatory_pass
    )
    co_primary_warning = bool(
        central["probability_co_primary_pass_q05"] < 0.50
        and confirmatory_pass
    )
    if confirmatory_pass:
        recommendation = "proceed_to_runtime_gate"
    else:
        recommendation = "do_not_execute_confirmatory_phase_d_without_amendment"
    return {
        "created_at_utc": _utc_now(),
        "fdr_power_version": FDR_POWER_VERSION,
        "power_gate_confirmatory_pass": confirmatory_pass,
        "power_gate_amend_required": amend_required,
        "power_gate_conservative_warning": conservative_warning,
        "co_primary_endpoint_power_warning": co_primary_warning,
        "recommendation": recommendation,
        "gate_rule": {
            "central_probability_of_criterion_2_pass_q05_min": 0.50,
            "central_probability_of_zero_discoveries_q05_max": 0.25,
            "instance_endpoint": "one-sided exact binomial/sign test over 12 instances",
            "instance_endpoint_alpha": INSTANCE_BINOMIAL_ALPHA,
            "instance_endpoint_success_threshold": INSTANCE_BINOMIAL_PASS_COUNT,
        },
        "central_metrics": {
            "probability_of_criterion_2_pass_q05": central["probability_of_criterion_2_pass_q05"],
            "probability_of_zero_discoveries_q05": central["probability_of_zero_discoveries_q05"],
            "probability_instance_endpoint_pass_q05": central["probability_instance_endpoint_pass_q05"],
            "probability_co_primary_pass_q05": central["probability_co_primary_pass_q05"],
            "expected_instance_positive_stress_count": central["expected_instance_positive_stress_count"],
        },
        "conservative_metrics": {
            "probability_of_criterion_2_pass_q05": conservative["probability_of_criterion_2_pass_q05"],
            "probability_of_zero_discoveries_q05": conservative["probability_of_zero_discoveries_q05"],
        },
    }


def _interpretation_md(summary: dict[str, Any], decision: dict[str, Any]) -> str:
    lines = [
        "# Phase D FDR Power Gate Interpretation",
        "",
        f"Created UTC: {summary['created_at_utc']}",
        "",
        "This pre-execution simulation uses Phase C seed-level effect summaries as the calibration source. "
        "For each Phase D scenario, the mean r_MSE and d_R effects are scaled toward zero while Phase C residual "
        "variance and r/d correlation are retained. No Phase D noisy or ZNE outcomes are used.",
        "",
        "## Gate Decision",
        "",
        f"- Recommendation: `{decision['recommendation']}`",
        f"- Confirmatory FDR gate pass: `{decision['power_gate_confirmatory_pass']}`",
        f"- Amendment required: `{decision['power_gate_amend_required']}`",
        f"- Conservative scenario warning: `{decision['power_gate_conservative_warning']}`",
        f"- Co-primary endpoint power warning: `{decision['co_primary_endpoint_power_warning']}`",
        "",
        "## Scenario Summary",
        "",
        "| Scenario | Multiplier | P(criterion-2 q05) | P(zero q05) | E[fdr q05] | E[positive instances] | P(instance endpoint q05) | P(co-primary q05) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["scenario_results"]:
        lines.append(
            f"| {row['scenario']} | {row['effect_multiplier']:.2f} | "
            f"{row['probability_of_criterion_2_pass_q05']:.3f} | "
            f"{row['probability_of_zero_discoveries_q05']:.3f} | "
            f"{row['expected_discordant_fdr_q05_count']:.2f} | "
            f"{row['expected_instance_positive_stress_count']:.2f} | "
            f"{row['probability_instance_endpoint_pass_q05']:.3f} | "
            f"{row['probability_co_primary_pass_q05']:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])
    if decision["power_gate_confirmatory_pass"]:
        lines.append(
            "The central q=0.05 FDR gate passes under the pre-specified criterion. "
            "Phase D may proceed to the runtime gate, but execution remains blocked until runtime, dry-run, "
            "and final v1.0 freeze are complete."
        )
    else:
        lines.append(
            "The central q=0.05 FDR gate does not pass. Phase D should not be executed as a confirmatory "
            "replication without a pre-execution amendment, stop decision, or explicit exploratory downgrade."
        )
    if decision["co_primary_endpoint_power_warning"]:
        lines.append(
            "The co-primary strong-label endpoint has lower simulated power than the FDR-only gate. "
            "This should be addressed before Phase D freeze if the strong label is intended as a co-primary claim."
        )
    lines.append("")
    return "\n".join(lines)


def run_phase_d_fdr_power_gate(
    root: Path = PROJECT_ROOT,
    simulations: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
    chunk_size: int = 250,
    force: bool = False,
) -> dict[str, Any]:
    planning_dir = ensure_dir(root / "results" / "phase_d" / "planning")
    summary_path = planning_dir / "phase_d_fdr_power_simulation.json"
    csv_path = planning_dir / "phase_d_fdr_power_simulation.csv"
    decision_path = planning_dir / "phase_d_fdr_power_gate_decision.json"
    md_path = planning_dir / "phase_d_fdr_power_interpretation.md"
    hashes_path = planning_dir / "phase_d_fdr_power_output_hashes.json"
    if any(path.exists() for path in (summary_path, csv_path, decision_path, md_path, hashes_path)) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase D FDR power outputs.")

    template_payload = _template_table(root)
    records = template_payload["records"]
    slot_candidates, stress_mask, slots_one_instance = _phase_d_slot_candidates(records)
    template = {
        key: np.asarray([row[key] for row in records], dtype=float if key != "excluded" else bool)
        for key in ("mean_r", "mean_d", "sd_r", "sd_d", "corr", "excluded")
    }
    template["slot_candidates"] = slot_candidates
    rng = np.random.default_rng(seed)
    scenario_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for scenario in SCENARIOS:
        remaining = simulations
        accum: dict[str, list[np.ndarray]] = {}
        while remaining > 0:
            n = min(chunk_size, remaining)
            remaining -= n
            simulated = _simulate_chunk(rng, template, scenario, n)
            p = simulated["p_conj"]
            metrics = _endpoint_metrics(
                simulated=simulated,
                fdr_pass_q05=_bh_pass(p, Q_PRIMARY),
                fdr_pass_q10=_bh_pass(p, Q_CONTINUITY),
                holm_pass=_holm_pass(p, Q_PRIMARY),
                by_pass=_bh_pass(p, Q_PRIMARY / BY_C_M),
                stress_mask=stress_mask,
            )
            for key, value in metrics.items():
                accum.setdefault(key, []).append(value)
        scenario_results.append(_summarize_scenario(scenario.name, scenario.effect_multiplier, accum))

    decision = _decision(scenario_results)
    summary = {
        "created_at_utc": _utc_now(),
        "fdr_power_version": FDR_POWER_VERSION,
        "status": "complete",
        "simulations_per_scenario": simulations,
        "rng_seed": seed,
        "family_size": FAMILY_SIZE,
        "phase_d_instances": INSTANCE_COUNT,
        "cells_per_instance": CELLS_PER_INSTANCE,
        "seeds_per_cell": SEEDS_PER_CELL,
        "primary_q": Q_PRIMARY,
        "continuity_q": Q_CONTINUITY,
        "by_c_m": BY_C_M,
        "instance_endpoint": {
            "definition": "one-sided binomial/sign test over H1-H12 using instance_positive_stress indicators",
            "stress_subfamily": {
                "noise_min": INSTANCE_STRESS_NOISE_MIN,
                "shots_min": INSTANCE_STRESS_SHOT_MIN,
                "cells_per_instance": int(np.sum(stress_mask[:CELLS_PER_INSTANCE])),
            },
            "instance_positive_stress_rule": ">=3 discordant_pre_fdr cells inside the stress subfamily",
            "alpha": INSTANCE_BINOMIAL_ALPHA,
            "success_threshold": INSTANCE_BINOMIAL_PASS_COUNT,
            "null": "P(instance_positive_stress) <= 0.50",
        },
        "phase_d_condition_slots_one_instance": slots_one_instance,
        "scenario_results": scenario_results,
        "decision": decision,
        "calibration_source": {
            "phase_c_family_size": 135,
            "source_hashes": template_payload["source_hashes"],
            "model": "Phase C seed-level normal residual model by matched noise/shot/ZNE condition; scenario scales mean r_MSE and d_R effects, keeps residual SD and r/d correlation.",
            "no_phase_d_noisy_or_zne_used": True,
        },
        "output_hash_manifest": "results/phase_d/planning/phase_d_fdr_power_output_hashes.json",
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_csv(csv_path, scenario_results)
    write_json_atomic(summary_path, summary, overwrite=True)
    write_json_atomic(decision_path, decision, overwrite=True)
    md_path.write_text(_interpretation_md(summary, decision), encoding="utf-8")
    output_hashes = {
        "phase_d_fdr_power_simulation_json": sha256_file(summary_path),
        "phase_d_fdr_power_simulation_csv": sha256_file(csv_path),
        "phase_d_fdr_power_gate_decision_json": sha256_file(decision_path),
        "phase_d_fdr_power_interpretation_md": sha256_file(md_path),
    }
    write_json_atomic(hashes_path, {
        "created_at_utc": _utc_now(),
        "fdr_power_version": FDR_POWER_VERSION,
        "hashes": output_hashes,
    }, overwrite=True)
    summary["output_hashes"] = output_hashes
    summary["output_hashes_file"] = hashes_path.relative_to(root).as_posix()
    return summary
