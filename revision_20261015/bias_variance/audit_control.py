"""Verify saved outputs without rerunning or resampling the control.

Run from any directory: python -B path/to/audit_control.py
The ideal reference uses Qiskit's statevector path, independently of the
production density kernel. A supplemental manifest records current hashes;
the original run manifest and protocol remain untouched.
"""
import csv
import hashlib
import json
import platform
import site
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
site.addsitedir(str(ROOT / '.venv/Lib/site-packages'))
sys.path.insert(0, str(ROOT))
import matplotlib
import numpy as np
import yaml
from src.circuits import graph_from_config, ideal_expected_cut
from src.mitiq_wrapper import deterministic_seed

t0 = time.perf_counter()
data = json.loads((OUT / 'results.json').read_text())
rows = data['rows']
with (OUT / 'pointwise.csv').open(newline='', encoding='utf-8') as f:
    points = list(csv.DictReader(f))
with (OUT / 'replicates.csv').open(newline='', encoding='utf-8') as f:
    reps = list(csv.DictReader(f))
with (OUT / 'summary.csv').open(newline='', encoding='utf-8') as f:
    saved_summary = list(csv.DictReader(f))
assert (len(rows), len(points), len(reps), len(saved_summary)) == (24, 3456, 12000, 24)
hash_check = {n: hashlib.sha256((ROOT / n.replace('\\', '/')).read_bytes()).hexdigest() == h
              for n, h in data['hashes'].items()}
assert all(hash_check.values()), hash_check
cfg = yaml.safe_load((ROOT / 'config/config_official_pilot_v1.6.yaml').read_text())
expected_weights = {'noisy': [1], 'linear': [1.5, -.5],
                    'quadratic': [1.875, -1.25, .375],
                    'cubic': [2.1875, -2.1875, 1.3125, -.3125]}
expected_shots = {'noisy': [1024], 'linear': [768, 256],
                  'quadratic': [548, 366, 110], 'cubic': [373, 373, 224, 54]}
ideal_reference = {}
ideal_checks = []
for name in ('G1_C4', 'G2_weighted4', 'G3_weighted6'):
    pp = sorted([p for p in points if p['graph'] == name
                 and float(p['p']) == .001 and p['method'] == 'noisy'],
                key=lambda p: int(p['index']))
    graph = graph_from_config(name, cfg)
    reference = np.array([-ideal_expected_cut(graph, float(p['gamma']), float(p['beta'])) for p in pp])
    saved = np.array([float(p['ideal']) for p in pp])
    delta = float(np.max(np.abs(reference - saved)))
    assert delta < 1e-11, (name, delta)
    ideal_reference[name] = saved
    ideal_checks.append({'graph': name, 'max_abs_statevector_difference': delta,
                         'equivalent_minima_tolerance_1e-10': int(np.sum(saved <= saved.min() + 1e-10))})

checks = []
seeds = []
for r, sr in zip(rows, saved_summary):
    name, noise, method = r['graph'], r['p'], r['method']
    seed = deterministic_seed('bias_variance_v1', name, noise, method, 20261015)
    seeds.append(seed)
    pp = sorted([p for p in points if p['graph'] == name and float(p['p']) == noise and p['method'] == method], key=lambda p: int(p['index']))
    rr = sorted([s for s in reps if s['graph'] == name and float(s['p']) == noise and s['method'] == method], key=lambda s: int(s['replicate']))
    assert len(pp) == 144 and len(rr) == 500
    assert [int(p['index']) for p in pp] == list(range(144))
    assert [int(s['replicate']) for s in rr] == list(range(500))
    ideal = np.array([float(p['ideal']) for p in pp])
    mean = np.array([float(p['mean']) for p in pp])
    bias = np.array([float(p['bias']) for p in pp])
    variance = np.array([float(p['variance']) for p in pp])
    mse = np.array([float(s['mse']) for s in rr])
    regret = np.array([float(s['regret']) for s in rr])
    contrast = np.array([float(s['contrast']) for s in rr])
    selected = np.array([int(s['selected']) for s in rr])
    assert np.all(np.isfinite(np.concatenate((ideal, mean, bias, variance, mse, regret, contrast))))
    assert np.all(variance >= 0) and np.all((selected >= 0) & (selected < 144))
    np.testing.assert_allclose(bias, mean - ideal, atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose([float(p['mse']) for p in pp], bias**2 + variance, atol=1e-13, rtol=1e-12)
    span = np.ptp(ideal)
    np.testing.assert_allclose(regret, (ideal[selected] - ideal.min()) / span, atol=1e-13, rtol=1e-12)
    expected = {'bias2': np.mean(bias**2), 'variance': np.mean(variance),
                'expected_mse': np.mean(bias**2+variance), 'mc_mse': np.mean(mse),
                'mc_mse_se': np.std(mse, ddof=1)/np.sqrt(500),
                'mc_regret': np.mean(regret), 'mc_regret_se': np.std(regret, ddof=1)/np.sqrt(500),
                'mean_landscape_regret': (ideal[np.argmin(mean)]-ideal.min())/span,
                'contrast_mc_mean': np.mean(contrast), 'contrast_mc_variance': np.var(contrast, ddof=1),
                'contrast_reversal_fraction': np.mean(contrast < 0)}
    for k, v in expected.items():
        np.testing.assert_allclose(r[k], v, atol=1e-13, rtol=1e-12)
    for k, v in r.items():
        if isinstance(v, (int, float)):
            np.testing.assert_allclose(float(sr[k]), v, atol=1e-13, rtol=1e-12)
        else:
            assert sr[k] == v
    np.testing.assert_array_equal(json.loads(r['weights']), expected_weights[method])
    np.testing.assert_array_equal(json.loads(r['shots']), expected_shots[method])
    assert sum(json.loads(r['shots'])) == 1024
    i, j = r['ideal_best'], r['competitor']
    assert i == int(np.argmin(ideal))
    candidates = np.flatnonzero(ideal > ideal.min()+1e-10)
    assert j == int(candidates[np.argmin(ideal[candidates])])
    np.testing.assert_allclose(r['exact_contrast'], mean[j]-mean[i], atol=1e-13, rtol=1e-12)
    np.testing.assert_allclose(r['contrast_variance'], variance[i]+variance[j], atol=1e-13, rtol=1e-12)
    checks.append({'graph': name, 'p': noise, 'method': method, 'seed': seed,
                   'mse_mc_z': r['mc_mse_z'], 'mc_material_error_fraction_tolerance_1e-10': float(np.mean(regret > 1e-10)),
                   'contrast_mc_variance_over_exact': r['contrast_mc_variance']/r['contrast_variance']})
assert len(set(seeds)) == 24
result = {'status': 'PASS', 'audit_utc': datetime.now(timezone.utc).isoformat(),
          'audit_elapsed_seconds': time.perf_counter()-t0,
          'run_elapsed_seconds': data['elapsed_seconds'], 'original_hash_checks': hash_check,
          'record_counts': {'configurations': 24, 'pointwise': 3456, 'replicates': 12000},
          'independent_ideal_reference': ideal_checks, 'configurations': checks,
          'max_abs_mse_mc_z': max(abs(r['mc_mse_z']) for r in rows),
          'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                       'matplotlib': matplotlib.__version__, 'pyyaml': yaml.__version__},
          'notes': ['MC agreement is descriptive; no post hoc acceptance threshold was used.',
                    'Replicate labels match across methods, but samples and seeds are independent.',
                    'Protocol and original run manifest are preserved byte for byte.',
                    'The protocol contains a literal backtick-n before its pre-run clarification; this cosmetic issue is preserved for hash integrity.',
                    'The independent ideal comparison verifies all 432 grid points; noisy second moments have the two original Aer checks only.']}
(OUT / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
sources = sorted(p for p in (ROOT / 'src').glob('*.py') if not p.name.startswith('phase_e'))
inputs = sources + [ROOT / 'config/config_official_pilot_v1.6.yaml'] + sorted(p for p in OUT.iterdir() if p.is_file() and p.name != 'manifest_final.json')
manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'Current non-Phase-E source snapshot and supplemental output hashes, not a replacement for run-time provenance in results.json.',
            'sha256': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
(OUT / 'manifest_final.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps({'status': result['status'], 'audit_seconds': result['audit_elapsed_seconds'],
                  'ideal_reference': ideal_checks, 'max_abs_mse_mc_z': result['max_abs_mse_mc_z']}))
