"""Reproduce a transparent post-analysis revision from unchanged seed CSVs.

Run from any directory: python revision_20261015/reanalyze_statistics.py
Requires the repository's pinned NumPy, pandas, SciPy and matplotlib.
Never writes to original data/results/manuscript paths.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import binom, binomtest, false_discovery_control

NUMERICAL_REGRET_TOLERANCE = 1e-12
SENSITIVITY_TOLERANCES = (0., 1e-14, 1e-12, 1e-10, 1e-6)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


@lru_cache(None)
def multinomial_coefficients(n):
    h = n // 2
    lo, hi = np.meshgrid(np.arange(h), np.arange(h), indexing="ij")
    lo, hi = lo.ravel(), hi.ravel()
    mid = n - lo - hi
    coeff = np.array([math.factorial(n) / (math.factorial(int(a)) * math.factorial(int(b)) * math.factorial(int(c))) for a, b, c in zip(lo, hi, mid)])
    return lo, hi, mid, coeff


def empirical_bootstrap_median_distribution(x):
    """Exact conditional distribution for an even-size empirical bootstrap.

    This removes Monte Carlo error only; its positivity tail need not be a
    calibrated frequentist p-value. Formula is checked by exhaustive enumeration.
    """
    n = len(x)
    if n % 2 or n < 2:
        raise ValueError("An even sample size >= 2 is required")
    h = n // 2
    values, counts = np.unique(x, return_counts=True)
    prob = counts / n
    cum = np.cumsum(prob)
    cum[-1] = 1.
    prev = np.r_[0., cum[:-1]]
    upper = np.maximum(0., 1. - cum)
    lo, hi, mid, coeff = multinomial_coefficients(n)
    diag = np.sum(coeff[None, :] * prev[:, None] ** lo * upper[:, None] ** hi * prob[:, None] ** mid, axis=1)
    ii, jj = np.triu_indices(len(values), k=1)
    off = math.comb(n, h) * (cum[ii] ** h - prev[ii] ** h) * ((1. - prev[jj]) ** h - upper[jj] ** h)
    support = np.r_[values, (values[ii] + values[jj]) / 2]
    mass = np.r_[diag, off]
    if abs(mass.sum() - 1.) > 1e-11:
        raise AssertionError("Bootstrap distribution does not normalize")
    return support, mass


def method_checks():
    checks = {}
    errors = []
    for x in ([-2., -.1, .3, 2.], [-1., -1., 0., 2.], [-3., 1.], [-1., 1.]):
        direct = np.array([np.median(a) for a in itertools.product(x, repeat=len(x))])
        support, mass = empirical_bootstrap_median_distribution(np.array(x))
        errors.extend(abs(np.mean(direct <= t) - mass[support <= t].sum()) for t in np.unique(direct))
    checks["bootstrap_max_cdf_error_vs_enumeration"] = float(max(errors))
    # Exact enumeration of all possible binomial counts under boundary null.
    k = np.arange(31)
    p = binom.sf(k - 1, 30, .5)
    probs = binom.pmf(k, 30, .5)
    max_excess = max(float(probs[p <= a].sum() - a) for a in np.r_[p, .05, .05/540])
    checks["binomial_null_size_max_excess"] = max_excess
    for count, value in zip(k, p):
        assert abs(value - binomtest(int(count), 30, .5, alternative="greater").pvalue) < 1e-14
    np.testing.assert_allclose(holm([.01, .04, .03, .2]), [.04, .09, .09, .2])
    # Shared helper from the frozen analysis: independent implementation check.
    checks["holm_known_vector_pass"] = True
    assert checks["bootstrap_max_cdf_error_vs_enumeration"] < 1e-12
    assert max_excess < 1e-12
    return checks


def summarize_phase(root, phase, relative, out):
    paths = {name: root / relative / name for name in ("comparisons_per_seed.csv", "classification_matrix.csv")}
    seed = pd.read_csv(paths["comparisons_per_seed.csv"])
    legacy = pd.read_csv(paths["classification_matrix.csv"])
    keys = ["instance", "noise", "shots_total", "zne_order"]
    frozen = legacy.set_index(keys)
    rows = []
    seed_set = set(seed.seed.unique())
    assert len(seed_set) == 30
    for key, group in seed.groupby(keys, sort=True):
        group = group.sort_values("seed")
        assert len(group) == 30 and set(group.seed) == seed_set
        old = frozen.loc[key]
        x, y = group.r_mse.to_numpy(), group.d_R.to_numpy()
        assert np.isfinite(x).all() and np.isfinite(y).all()
        excluded = bool(old.excluded_below_detection_floor or old.excluded_extrapolation_breakdown)
        px = float(binom.sf(np.sum(x > 0) - 1, 30, .5))
        py = float(binom.sf(np.sum(y > NUMERICAL_REGRET_TOLERANCE) - 1, 30, .5))
        tails = []
        for values in (x, y):
            support, mass = empirical_bootstrap_median_distribution(values)
            tails.append(float(mass[support <= 0].sum()))
        row = dict(zip(keys, key))
        row.update(cell_id=old.cell_id, n=30, excluded=excluded,
            positive_mse=int(np.sum(x > 0)), positive_regret=int(np.sum(y > NUMERICAL_REGRET_TOLERANCE)),
            positive_regret_strict=int(np.sum(y > 0)),
            joint_positive=int(np.sum((x > 0) & (y > NUMERICAL_REGRET_TOLERANCE))), zero_regret=int(np.sum(y == 0)),
            numerical_ties_regret=int(np.sum(np.abs(y) <= NUMERICAL_REGRET_TOLERANCE)),
            p_mse=px, p_regret=py, p_conjunction=1. if excluded else max(px, py),
            conditional_bootstrap_tail=1. if excluded else max(tails),
            median_r_mse=float(np.median(x)), median_d_R=float(np.median(y)),
            q25_d_R=float(np.quantile(y, .25)), q75_d_R=float(np.quantile(y, .75)),
            legacy_nominal=bool(old.discordant_pre_fdr), legacy_bh=bool(old.discordant_fdr),
            legacy_p=float(old.p_conjunction), legacy_strong=bool(old.strongly_discordant_fdr),
            tau_R=float(old.tau_R))
        for eps in SENSITIVITY_TOLERANCES:
            row[f"p_conjunction_eps_{eps:g}"] = 1. if excluded else max(px, float(binom.sf(np.sum(y > eps) - 1, 30, .5)))
        rows.append(row)
    df = pd.DataFrame(rows)
    assert len(df) == (135 if phase == "C" else 540)
    q = .10 if phase == "C" else .05
    for prefix, p in [("sign", df.p_conjunction.to_numpy()), ("conditional_bootstrap", df.conditional_bootstrap_tail.to_numpy())]:
        df[prefix + "_holm"] = holm(p)
        df[prefix + "_bh"] = false_discovery_control(p, method="bh")
        df[prefix + "_by"] = false_discovery_control(p, method="by")
    df["revised_holm_reject"] = df.sign_holm <= .05
    df["revised_by_reject"] = df.sign_by <= q
    df["conditional_bh_reject"] = df.sign_bh <= q
    if phase == "D":
        graph_path = root / "config/phase_d_instances_v1.0.json"
        paths["graph_config"] = graph_path
        graphs = json.loads(graph_path.read_text(encoding="utf-8"))["selected_instances_H1_to_H12"]
        scales = {}
        for p in sorted((root / "data/raw/phase_d").glob("D*/H*/grid_120/C_ideal.npy")):
            name = p.parts[-3]
            a = np.load(p, allow_pickle=False)
            scales[name] = (float(np.ptp(a)), float(sum(abs(e[2]) for e in graphs[name]["edges"])))
            paths["ideal_" + name] = p
        assert len(scales) == 12
        df["ideal_range"] = df.instance.map(lambda x: scales[x][0])
        df["total_edge_weight"] = df.instance.map(lambda x: scales[x][1])
        df["median_d_objective"] = df.median_d_R * df.ideal_range
        df["median_d_percent_edge_weight"] = 100 * df.median_d_objective / df.total_edge_weight
    # Legacy and resolution-only counts retain the original descriptive filters;
    # revised sign rejection sets above deliberately do not filter after testing.
    summary = dict(phase=phase, cells=len(df), seeds_per_cell=30, excluded=int(df.excluded.sum()),
        legacy_nominal=int(df.legacy_nominal.sum()), legacy_bh=int(df.legacy_bh.sum()),
        legacy_by=int((legacy.discordant_fdr_by).sum()) if "discordant_fdr_by" in legacy else None,
        legacy_holm=int((legacy.discordant_fdr_holm).sum()) if "discordant_fdr_holm" in legacy else None,
        revised_sign_holm=int(df.revised_holm_reject.sum()), revised_sign_by=int(df.revised_by_reject.sum()),
        conditional_sign_bh=int(df.conditional_bh_reject.sum()),
        resolution_only_bh_legacy_filter=int((df.legacy_nominal & (df.conditional_bootstrap_bh <= q)).sum()),
        resolution_only_by_legacy_filter=int((df.legacy_nominal & (df.conditional_bootstrap_by <= q)).sum()),
        resolution_only_holm_legacy_filter=int((df.legacy_nominal & (df.conditional_bootstrap_holm <= .05)).sum()),
        revised_holm_instances=sorted(df.loc[df.revised_holm_reject, "instance"].unique().tolist()),
        revised_holm_min_median_r=float(df.loc[df.revised_holm_reject, "median_r_mse"].min()) if df.revised_holm_reject.any() else None,
        revised_holm_max_median_d=float(df.loc[df.revised_holm_reject, "median_d_R"].max()) if df.revised_holm_reject.any() else None,
        revised_holm_above_tau=int((df.revised_holm_reject & (df.median_d_R >= df.tau_R)).sum()),
        revised_holm_above_original_mse_threshold=int((df.revised_holm_reject & (df.median_r_mse >= .10)).sum()),
        numerical_regret_tolerance=NUMERICAL_REGRET_TOLERANCE,
        numerical_tolerance_sensitivity=[],
        input_sha256={str(p.relative_to(root)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()})
    for eps in SENSITIVITY_TOLERANCES:
        p_eps = df[f"p_conjunction_eps_{eps:g}"].to_numpy()
        summary["numerical_tolerance_sensitivity"].append(dict(epsilon=eps,
            holm=int((holm(p_eps) <= .05).sum()),
            by=int((false_discovery_control(p_eps, method="by") <= q).sum()),
            bh=int((false_discovery_control(p_eps, method="bh") <= q).sum())))
    if phase == "D":
        stress = df[(df.noise >= .005) & (df.shots_total >= 1024) & ~df.excluded]
        counts = stress.groupby("instance").legacy_nominal.sum()
        summary["legacy_stress_positive_instances_descriptive"] = int((counts >= 3).sum())
        selected = df.loc[df.revised_holm_reject]
        summary["revised_holm_median_native_shift"] = float(selected.median_d_objective.median())
        summary["revised_holm_native_shift_range"] = [float(selected.median_d_objective.min()), float(selected.median_d_objective.max())]
        summary["revised_holm_percent_weight_range"] = [float(selected.median_d_percent_edge_weight.min()), float(selected.median_d_percent_edge_weight.max())]
    df.to_csv(out / f"cells_{phase}.csv", index=False)
    df.loc[df.revised_holm_reject].to_csv(out / f"holm_cells_{phase}.csv", index=False)
    if phase == "D":
        order_number = {"linear": 1, "quadratic": 2, "cubic": 3}
        tex_rows = []
        for r in df.loc[df.revised_holm_reject].itertuples():
            tex_rows.append(f"{r.instance} & $({r.noise:g},{r.shots_total},{order_number[r.zne_order]})$ & {r.positive_regret} & {r.median_r_mse:.4f} & {r.median_d_R:.5f} & {r.median_d_objective:.5f} & {r.sign_holm:.6f}" + r"\\")
        (out / "revised_cells_rows.tex").write_text("\n".join(tex_rows) + "\n", encoding="utf-8")
    return summary, df


def plots(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(4.8, 5.1), layout="constrained")
    retained = df.revised_holm_reject
    excluded = df.excluded
    other = ~retained & ~excluded
    axes[0].scatter(df.loc[other, "median_r_mse"], df.loc[other, "median_d_R"], s=10, c="#a9b5bf", alpha=.7, label="Other eligible cells")
    axes[0].scatter(df.loc[retained, "median_r_mse"], df.loc[retained, "median_d_R"], s=36, c="#b24a25", label="Sign + Holm retained", zorder=3)
    axes[0].axhline(0, c="black", lw=.6); axes[0].axvline(0, c="black", lw=.6)
    axes[0].set_xscale("symlog", linthresh=1.)
    axes[0].set(xlabel="Median relative MSE improvement (symlog)", ylabel="Median normalized regret change", title="A. Effects across 507 eligible cells")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    subset = df[retained].sort_values(["instance", "shots_total"])
    labels = [f"{r.instance}, p={r.noise:g}, B={r.shots_total}\n{r.zne_order}" for r in subset.itertuples()]
    y = np.arange(len(subset))
    axes[1].hlines(y, subset.q25_d_R, subset.q75_d_R, color="#b24a25", lw=2)
    axes[1].scatter(subset.median_d_R, y, color="#b24a25", s=24, zorder=3)
    axes[1].axvline(0, color="black", lw=.6)
    axes[1].set_yticks(y, labels, fontsize=8)
    axes[1].set(xlabel="Normalized regret change", title="B. Retained cells: median and seed IQR")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "revised_statistics.pdf")
    fig.savefig(out / "revised_statistics.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "statistics")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"amendment": "post-analysis; not preregistered", "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}, "method_checks": method_checks(), "phases": []}
    for phase, rel in [("C", "results/phase_c_120_analysis"), ("D", "results/phase_d/global_D1_D2_D3_D4")]:
        summary, df = summarize_phase(args.root, phase, rel, args.output)
        report["phases"].append(summary)
        print(json.dumps(summary), flush=True)
        if phase == "D":
            plots(df, args.output)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
