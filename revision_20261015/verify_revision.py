"""Read-only numerical and integrity verification of the compact review bundle.

No quantum campaign is run. Optional report output is the only written file.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import tempfile
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from reanalyze_statistics import summarize_phase, method_checks, NUMERICAL_REGRET_TOLERANCE
from reproducibility.verify_manuscript import verify_manuscript


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independently_adjust(p):
    pairs = sorted(enumerate(p), key=lambda x: (x[1], x[0]))
    n = len(p)
    h = sum(1. / k for k in range(1, n + 1))
    out_holm, out_bh, out_by = [0.] * n, [0.] * n, [0.] * n
    running = 0.
    for rank, (i, value) in enumerate(pairs):
        running = max(running, (n-rank)*value)
        out_holm[i] = min(1., running)
    for factor, output in [(1., out_bh), (h, out_by)]:
        running = 1.
        for rank in range(n-1, -1, -1):
            i, value = pairs[rank]
            running = min(running, n*factor*value/(rank+1))
            output[i] = running
    return out_holm, out_bh, out_by


def verify(root):
    here = root / "revision_20261015"
    result = {"status": "PASS", "scope": "numerical reproduction only; no long C/D rerun", "inference_caveat": "Historical 31-bit seed collisions preclude certifying IID sampling and exact inferential calibration for the archive. See followup_review/math_code/math_code_review.txt. PASS does not certify statistical significance.", "checks": {}}
    manifest_path = root / "REVIEW_BUNDLE_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [rel for rel, sha in manifest["sha256"].items() if not (root / rel).is_file() or digest(root / rel) != sha]
        if mismatches:
            raise AssertionError({"bundle_hash_mismatches": mismatches})
        result["checks"]["bundle_files_verified"] = len(manifest["sha256"])
    result["checks"]["method_checks"] = method_checks()
    archived = json.loads((here / "statistics/summary.json").read_text())
    for phase, relative in [("C", "results/phase_c_120_analysis"), ("D", "results/phase_d/global_D1_D2_D3_D4")]:
        saved = pd.read_csv(here / f"statistics/cells_{phase}.csv")
        with tempfile.TemporaryDirectory(prefix="zne-stat-check-") as tmp:
            summary, fresh = summarize_phase(root, phase, relative, Path(tmp))
        pd.testing.assert_frame_equal(saved, fresh, check_dtype=False, check_exact=False, atol=1e-13, rtol=1e-12)
        expected = next(p for p in archived["phases"] if p["phase"] == phase)
        if summary != expected:
            raise AssertionError("Regenerated phase summary differs: " + phase)
        seed = pd.read_csv(root / relative / "comparisons_per_seed.csv")
        pvalues = []
        grouped = seed.groupby(["instance", "noise", "shots_total", "zne_order"])
        for row in saved.itertuples():
            group = grouped.get_group((row.instance, row.noise, row.shots_total, row.zne_order))
            kr = sum(float(x) > 0 for x in group.r_mse)
            kd = sum(float(x) > NUMERICAL_REGRET_TOLERANCE for x in group.d_R)
            pr = sum(math.comb(30, k) for k in range(kr, 31)) / 2**30
            pd_ = sum(math.comb(30, k) for k in range(kd, 31)) / 2**30
            assert (kr, kd) == (row.positive_mse, row.positive_regret)
            pvalues.append(1. if row.excluded else max(pr, pd_))
        ha, bha, bya = independently_adjust(pvalues)
        for got, column in [(ha, "sign_holm"), (bha, "sign_bh"), (bya, "sign_by")]:
            np.testing.assert_allclose(got, saved[column], atol=1e-14, rtol=1e-12)
        q = .1 if phase == "C" else .05
        result["checks"][phase] = dict(cells=len(saved), holm=int(sum(x <= .05 for x in ha)), by=int(sum(x <= q for x in bya)), bh=int(sum(x <= q for x in bha)), independently_recomputed=True)
    control = here / "bias_variance"
    info = json.loads((control / "results.json").read_text())
    for rel, sha in info["hashes"].items():
        assert digest(root / rel.replace("\\", "/")) == sha, rel
    point = pd.read_csv(control / "pointwise.csv")
    samples = pd.read_csv(control / "replicates.csv")
    aggregate = pd.read_csv(control / "summary.csv")
    assert len(point) == 3456 and len(samples) == 12000 and len(aggregate) == 24
    np.testing.assert_allclose(point["mse"], point.bias**2 + point.variance, atol=1e-13, rtol=1e-12)
    point_groups = point.groupby(["graph", "p", "method"])
    sample_groups = samples.groupby(["graph", "p", "method"])
    for row in aggregate.itertuples():
        key = row.graph, row.p, row.method
        group = point_groups.get_group(key).sort_values("index")
        draws = sample_groups.get_group(key)
        assert len(group) == 144 and len(draws) == 500 and draws.replicate.nunique() == 500
        ideal = group.ideal.to_numpy()
        recomputed_regret = (ideal[draws.selected.to_numpy(dtype=int)] - ideal.min()) / np.ptp(ideal)
        np.testing.assert_allclose(draws.regret, recomputed_regret, atol=1e-14, rtol=1e-12)
        np.testing.assert_allclose([row.bias2, row.variance, row.expected_mse, row.mc_mse, row.mc_regret], [np.mean(group.bias**2), group.variance.mean(), group.mse.mean(), draws.mse.mean(), draws.regret.mean()], atol=1e-13, rtol=1e-12)
    result["checks"]["bounded_control"] = dict(configurations=24, pointwise_rows=3456, repetitions=12000, source_hashes_match=True, identities_and_regret_recomputed=True)
    result["checks"]["manuscript_tables"] = verify_manuscript(root)
    result["checks"]["manuscript_numeric_table_current"] = True
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()
    result = verify(args.root)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
