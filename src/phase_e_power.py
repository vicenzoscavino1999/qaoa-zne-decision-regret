from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

from .io_utils import PROJECT_ROOT, ensure_dir, sha256_file, write_json_atomic


PHASE_E_POWER_VERSION = "phase_e_power_sizing_v0_1_draft"
DEFAULT_SEED = 20260517
DEFAULT_SIMULATIONS = 5000
DEFAULT_CHUNK_SIZE = 250
DEFAULT_N_SEED_CANDIDATES = (20, 30, 50)
DEFAULT_EFFECT_MULTIPLIERS = (1.0, 0.75, 0.5, 0.25)
PHASE_E_DEPTH2_FAMILY_SIZE = 270
DEFAULT_POWER = 0.80


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


def _as_float(value: Any) -> float:
    return float(value)


def _as_int(value: Any) -> int:
    return int(float(value))


def _source_files(root: Path) -> dict[str, Path]:
    return {
        "phase_c_120": root / "results" / "phase_c_120_analysis" / "comparisons_per_seed.csv",
        "phase_d_global": root / "results" / "phase_d" / "global_D1_D2_D3_D4" / "comparisons_per_seed.csv",
    }


def _load_seed_comparisons(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for source, path in _source_files(root).items():
        if not path.exists():
            continue
        source_hashes[source] = sha256_file(path)
        for row in _read_csv(path):
            try:
                d_r = _as_float(row["d_R"])
                r_mse = _as_float(row["r_mse"])
            except (KeyError, TypeError, ValueError):
                continue
            if not (math.isfinite(d_r) and math.isfinite(r_mse)):
                continue
            records.append(
                {
                    "source": source,
                    "phase": row.get("phase", source),
                    "instance": row["instance"],
                    "noise": _as_float(row["noise"]),
                    "shots_total": _as_int(row["shots_total"]),
                    "zne_order": row["zne_order"],
                    "seed": _as_int(row["seed"]),
                    "d_R": d_r,
                    "r_mse": r_mse,
                }
            )
    if not records:
        raise RuntimeError("No Phase C/D seed-level comparison rows were found for Phase E power sizing.")
    return records, source_hashes


def _group_cell_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in records:
        key = (row["source"], row["instance"], row["noise"], row["shots_total"], row["zne_order"])
        grouped.setdefault(key, []).append(row)

    cells: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        d = np.asarray([row["d_R"] for row in rows], dtype=float)
        r = np.asarray([row["r_mse"] for row in rows], dtype=float)
        if len(d) < 2:
            continue
        sd_d = float(np.std(d, ddof=1))
        sd_r = float(np.std(r, ddof=1))
        corr = float(np.corrcoef(d, r)[0, 1]) if sd_d > 1e-12 and sd_r > 1e-12 else 0.0
        if not math.isfinite(corr):
            corr = 0.0
        cells.append(
            {
                "source": key[0],
                "instance": key[1],
                "noise": key[2],
                "shots_total": key[3],
                "zne_order": key[4],
                "n_seed_source": len(rows),
                "mean_d_R": float(np.mean(d)),
                "median_d_R": float(np.median(d)),
                "sd_d_R": max(sd_d, 1e-12),
                "mean_r_mse": float(np.mean(r)),
                "median_r_mse": float(np.median(r)),
                "sd_r_mse": max(sd_r, 1e-12),
                "corr_d_R_r_mse": min(0.95, max(-0.95, corr)),
                "sign_discordant_depth1": bool(np.median(d) > 0.0 and np.median(r) > 0.0),
            }
        )
    if not cells:
        raise RuntimeError("Phase C/D comparison rows did not produce any usable seed-level cells.")
    return cells


def _summarize_vector(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def _mde_rows(cells: list[dict[str, Any]], n_seed_candidates: tuple[int, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pools = {
        "phase_c_120": [cell for cell in cells if cell["source"] == "phase_c_120"],
        "phase_d_global": [cell for cell in cells if cell["source"] == "phase_d_global"],
        "phase_c_plus_d": cells,
    }
    z_power = stats.norm.ppf(DEFAULT_POWER)
    alpha_uncorrected = 0.05
    alpha_bonferroni = 0.05 / PHASE_E_DEPTH2_FAMILY_SIZE
    for pool_name, pool_cells in pools.items():
        if not pool_cells:
            continue
        for metric, sd_key in (("d_R", "sd_d_R"), ("r_mse", "sd_r_mse")):
            sd_values = np.asarray([cell[sd_key] for cell in pool_cells], dtype=float)
            sd_summary = _summarize_vector(sd_values)
            for n_seed in n_seed_candidates:
                for sd_label, sd_value in (
                    ("median_sd", sd_summary["median"]),
                    ("p75_sd", sd_summary["p75"]),
                    ("p90_sd", sd_summary["p90"]),
                ):
                    rows.append(
                        {
                            "row_type": "minimum_detectable_effect",
                            "source_pool": pool_name,
                            "metric": metric,
                            "sd_reference": sd_label,
                            "sd_value": sd_value,
                            "n_seed": n_seed,
                            "target_power": DEFAULT_POWER,
                            "alpha_uncorrected_two_sided": alpha_uncorrected,
                            "mde_uncorrected_two_sided": (stats.norm.ppf(1.0 - alpha_uncorrected / 2.0) + z_power)
                            * sd_value
                            / math.sqrt(n_seed),
                            "alpha_bonferroni_two_sided_family_270": alpha_bonferroni,
                            "mde_bonferroni_two_sided_family_270": (stats.norm.ppf(1.0 - alpha_bonferroni / 2.0) + z_power)
                            * sd_value
                            / math.sqrt(n_seed),
                            "alpha_uncorrected_one_sided": alpha_uncorrected,
                            "mde_uncorrected_one_sided": (stats.norm.ppf(1.0 - alpha_uncorrected) + z_power)
                            * sd_value
                            / math.sqrt(n_seed),
                        }
                    )
    return rows


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


def _simulate_power(
    cells: list[dict[str, Any]],
    simulations: int,
    seed: int,
    n_seed_candidates: tuple[int, ...],
    effect_multipliers: tuple[float, ...],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    templates = [cell for cell in cells if cell["sign_discordant_depth1"]]
    if len(templates) < 5:
        templates = cells
    mean_d = np.asarray([cell["mean_d_R"] for cell in templates], dtype=float)
    mean_r = np.asarray([cell["mean_r_mse"] for cell in templates], dtype=float)
    sd_d = np.asarray([cell["sd_d_R"] for cell in templates], dtype=float)
    sd_r = np.asarray([cell["sd_r_mse"] for cell in templates], dtype=float)
    corr = np.asarray([cell["corr_d_R_r_mse"] for cell in templates], dtype=float)

    rows: list[dict[str, Any]] = []
    for n_seed in n_seed_candidates:
        for multiplier in effect_multipliers:
            fdr_q05_counts = []
            fdr_q10_counts = []
            sign_counts = []
            remaining = simulations
            while remaining > 0:
                current = min(chunk_size, remaining)
                remaining -= current
                selected = rng.integers(0, len(templates), size=(current, PHASE_E_DEPTH2_FAMILY_SIZE))
                mu_d = mean_d[selected] * multiplier
                mu_r = mean_r[selected] * multiplier
                sigma_d = sd_d[selected]
                sigma_r = sd_r[selected]
                rho = corr[selected]
                z1 = rng.standard_normal(size=(current, PHASE_E_DEPTH2_FAMILY_SIZE, n_seed))
                z2 = rng.standard_normal(size=(current, PHASE_E_DEPTH2_FAMILY_SIZE, n_seed))
                d = mu_d[..., None] + sigma_d[..., None] * z1
                r = mu_r[..., None] + sigma_r[..., None] * (rho[..., None] * z1 + np.sqrt(1.0 - rho[..., None] ** 2) * z2)
                mean_d_hat = np.mean(d, axis=2)
                mean_r_hat = np.mean(r, axis=2)
                se_d = np.maximum(np.std(d, axis=2, ddof=1) / math.sqrt(n_seed), 1e-12)
                se_r = np.maximum(np.std(r, axis=2, ddof=1) / math.sqrt(n_seed), 1e-12)
                p_d = stats.norm.sf(mean_d_hat / se_d)
                p_r = stats.norm.sf(mean_r_hat / se_r)
                p_conj = np.maximum(p_d, p_r)
                sign_positive = (mean_d_hat > 0.0) & (mean_r_hat > 0.0)
                fdr_q05 = _bh_pass(p_conj, 0.05) & sign_positive
                fdr_q10 = _bh_pass(p_conj, 0.10) & sign_positive
                fdr_q05_counts.append(np.sum(fdr_q05, axis=1))
                fdr_q10_counts.append(np.sum(fdr_q10, axis=1))
                sign_counts.append(np.sum(sign_positive, axis=1))
            fdr_q05_count = np.concatenate(fdr_q05_counts)
            fdr_q10_count = np.concatenate(fdr_q10_counts)
            sign_count = np.concatenate(sign_counts)
            rows.append(
                {
                    "row_type": "monte_carlo_depth2_family_power",
                    "source_pool": "depth1_sign_discordant_templates" if len(templates) != len(cells) else "all_depth1_templates",
                    "family_size": PHASE_E_DEPTH2_FAMILY_SIZE,
                    "n_seed": n_seed,
                    "effect_multiplier": multiplier,
                    "simulations": simulations,
                    "expected_sign_positive_cells": float(np.mean(sign_count)),
                    "expected_fdr_q05_cells": float(np.mean(fdr_q05_count)),
                    "expected_fdr_q10_cells": float(np.mean(fdr_q10_count)),
                    "prob_at_least_one_fdr_q05": float(np.mean(fdr_q05_count >= 1)),
                    "prob_at_least_five_fdr_q05": float(np.mean(fdr_q05_count >= 5)),
                    "prob_zero_fdr_q05": float(np.mean(fdr_q05_count == 0)),
                    "prob_at_least_one_fdr_q10": float(np.mean(fdr_q10_count >= 1)),
                    "prob_at_least_five_fdr_q10": float(np.mean(fdr_q10_count >= 5)),
                    "prob_zero_fdr_q10": float(np.mean(fdr_q10_count == 0)),
                }
            )
    return rows


def _interpretation_md(summary: dict[str, Any]) -> str:
    central = [
        row
        for row in summary["simulation_rows"]
        if row["row_type"] == "monte_carlo_depth2_family_power"
        and row["n_seed"] == 30
        and abs(row["effect_multiplier"] - 0.5) < 1e-12
    ][0]
    mde_rows = [
        row
        for row in summary["mde_rows"]
        if row["source_pool"] == "phase_c_plus_d"
        and row["metric"] == "d_R"
        and row["sd_reference"] == "median_sd"
        and row["n_seed"] == 30
    ][0]
    lines = [
        "# Phase E Preliminary Power/Sizing Note",
        "",
        f"Created UTC: {summary['created_at_utc']}",
        f"Status: `{summary['status']}`",
        "",
        "This is a pre-pilot sizing artifact. It uses frozen depth-1 Phase C/Phase D seed-level comparison rows to estimate seed-level variability for Phase E planning. It does not freeze `tau_R_opt`, `K`, `T`, `N_LHS`, selected instances, or official execution.",
        "",
        "## Inputs",
        "",
        f"- Phase C rows: {summary['input_summary']['phase_c_120_rows']}",
        f"- Phase D rows: {summary['input_summary']['phase_d_global_rows']}",
        f"- Usable seed-level cells: {summary['input_summary']['usable_seed_level_cells']}",
        "",
        "## Planning Readout",
        "",
        f"- For `n_seed=30`, pooled median seed-level `d_R` SD gives uncorrected two-sided 80% MDE `{mde_rows['mde_uncorrected_two_sided']:.6g}`.",
        f"- The same row's conservative Bonferroni-family-270 planning MDE is `{mde_rows['mde_bonferroni_two_sided_family_270']:.6g}`.",
        f"- In the illustrative depth-2 family simulation with 50% of depth-1 sign-discordant template effects and `n_seed=30`, `P(at least one q<=0.05 cell)` is `{central['prob_at_least_one_fdr_q05']:.3f}` and `P(zero q<=0.05 cells)` is `{central['prob_zero_fdr_q05']:.3f}`.",
        "",
        "## Interpretation",
        "",
        "The current default `n_seed=30` remains plausible for effects comparable to depth-1 seed-level variability, but smaller effects can be underpowered after a 270-cell multiplicity burden. The unified depth-2 pilot still must estimate depth-2-specific variance before freeze.",
        "",
        "## Next Required Gates",
        "",
        "- finite-size/lightcone diagnostics and depth-2 instance selection;",
        "- unified depth-2 pilot;",
        "- calibration of `T`, `K`, `K_ref`, and `N_LHS`; and",
        "- runtime/storage gate before final freeze.",
        "",
    ]
    return "\n".join(lines)


def run_phase_e_power_sizing(
    root: Path = PROJECT_ROOT,
    simulations: int = DEFAULT_SIMULATIONS,
    seed: int = DEFAULT_SEED,
    n_seed_candidates: tuple[int, ...] = DEFAULT_N_SEED_CANDIDATES,
    effect_multipliers: tuple[float, ...] = DEFAULT_EFFECT_MULTIPLIERS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = ensure_dir(root / "results" / "phase_e" / "power_sizing")
    json_path = output_dir / "phase_e_power_simulation.json"
    csv_path = output_dir / "phase_e_power_simulation.csv"
    md_path = output_dir / "phase_e_power_interpretation.md"
    cell_csv_path = output_dir / "phase_e_depth1_power_cell_summary.csv"
    hash_path = output_dir / "phase_e_power_output_hashes.json"
    if any(path.exists() for path in (json_path, csv_path, md_path, cell_csv_path, hash_path)) and not force:
        raise FileExistsError("Refusing to overwrite existing Phase E power sizing outputs.")

    records, source_hashes = _load_seed_comparisons(root)
    cells = _group_cell_records(records)
    mde_rows = _mde_rows(cells, n_seed_candidates)
    simulation_rows = _simulate_power(cells, simulations, seed, n_seed_candidates, effect_multipliers, chunk_size)
    input_summary = {
        "phase_c_120_rows": sum(1 for row in records if row["source"] == "phase_c_120"),
        "phase_d_global_rows": sum(1 for row in records if row["source"] == "phase_d_global"),
        "usable_seed_level_cells": len(cells),
        "sign_discordant_depth1_cells": sum(1 for cell in cells if cell["sign_discordant_depth1"]),
    }
    summary = {
        "created_at_utc": _utc_now(),
        "status": "phase_e_preliminary_power_sizing_complete",
        "power_version": PHASE_E_POWER_VERSION,
        "scope": "pre_pilot_depth1_informed_planning_only",
        "official_execution_authorized": False,
        "simulations": simulations,
        "chunk_size": chunk_size,
        "rng_seed": seed,
        "n_seed_candidates": list(n_seed_candidates),
        "effect_multipliers": list(effect_multipliers),
        "phase_e_depth2_family_size": PHASE_E_DEPTH2_FAMILY_SIZE,
        "input_summary": input_summary,
        "source_hashes": source_hashes,
        "mde_rows": mde_rows,
        "simulation_rows": simulation_rows,
        "limitations": [
            "Uses depth-1 grid-based Phase C/D seed-level variance as a planning proxy for optimizer-based depth-2 variance.",
            "Does not select Phase E instances and does not use any Phase E depth-2 outcomes.",
            "Monte Carlo rows are illustrative pre-pilot sizing diagnostics, not a frozen confirmatory power model.",
            "Final seed count, tau_R_opt, and FDR interpretation must be revisited after the unified depth-2 pilot.",
        ],
        "outputs": {
            "json": "results/phase_e/power_sizing/phase_e_power_simulation.json",
            "csv": "results/phase_e/power_sizing/phase_e_power_simulation.csv",
            "interpretation_md": "results/phase_e/power_sizing/phase_e_power_interpretation.md",
            "cell_summary_csv": "results/phase_e/power_sizing/phase_e_depth1_power_cell_summary.csv",
            "hashes": "results/phase_e/power_sizing/phase_e_power_output_hashes.json",
        },
    }
    _write_csv(csv_path, mde_rows + simulation_rows)
    _write_csv(cell_csv_path, cells)
    write_json_atomic(json_path, summary, overwrite=True)
    md_path.write_text(_interpretation_md(summary), encoding="utf-8")
    hashes = {
        "created_at_utc": _utc_now(),
        "power_version": PHASE_E_POWER_VERSION,
        "hashes": {
            "phase_e_power_simulation_json": sha256_file(json_path),
            "phase_e_power_simulation_csv": sha256_file(csv_path),
            "phase_e_power_interpretation_md": sha256_file(md_path),
            "phase_e_depth1_power_cell_summary_csv": sha256_file(cell_csv_path),
        },
    }
    write_json_atomic(hash_path, hashes, overwrite=True)
    summary["output_hashes"] = hashes["hashes"]
    return summary
