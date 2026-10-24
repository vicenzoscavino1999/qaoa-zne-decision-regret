"""Post hoc oracle allocation sensitivity from saved exact probabilities.

Requires Python 3.12 and NumPy. No circuit evaluation or stochastic sampling
is performed. The confirmation inputs are opened read-only, and all results
are written exclusively to the path explicitly supplied by --output.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def integer_optimum(q, B):
    """Exact positive-integer optimum by diminishing marginal-benefit greedy.

    Each row minimizes sum(q_j/n_j), n_j>=1, sum n_j=B. Starting at
    all ones, each next shot realizes the largest q_j/(n_j*(n_j+1)).
    These benefits decrease separately in each j, so greedy is globally exact.
    Stable first-index ties do not affect the optimum objective value.
    """
    if B < q.shape[1] or np.any(q < 0) or not np.isfinite(q).all():
        raise ValueError('Integer allocation requires finite nonnegative q and B>=number of scales.')
    n = np.ones(q.shape, dtype=np.int64)
    rows = np.arange(len(q))
    for _ in range(B-q.shape[1]):
        j = np.argmax(q/(n*(n+1)), axis=1)
        n[rows, j] += 1
    return n, np.sum(q/n, axis=1)


def stats(a):
    return dict(zip(('min', 'q01', 'q25', 'median', 'q75', 'q99', 'max'),
                    map(float, np.quantile(a, [0, .01, .25, .5, .75, .99, 1]))))


def analyze(root, output):
    t0 = time.perf_counter()
    root = root.resolve()
    conf = root/'confirmation_20261019'
    output = output.resolve()
    if output.is_relative_to(conf):
        raise ValueError('--output must be outside the confirmation source directory.')
    if output == Path(__file__).resolve():
        raise ValueError('--output must not overwrite the analysis script.')
    protocol = json.loads((conf/'protocol.json').read_text(encoding='utf-8'))
    graphs = json.loads((conf/'inputs/graphs.json').read_text(encoding='utf-8'))
    manifest = json.loads((conf/'probabilities/manifest.json').read_text(encoding='utf-8'))
    assert sha(conf/'protocol.json') == manifest['protocol_sha256']
    hashes = {}

    def record(path):
        relative = path.relative_to(root).as_posix()
        hashes[relative] = sha(path)
        return relative

    for name, digest in manifest['sha256'].items():
        path = conf/'probabilities'/name
        assert record(path) in hashes
        assert hashes[path.relative_to(root).as_posix()] == digest
    for rel in ['protocol.json', 'inputs/graphs.json', 'probabilities/manifest.json']:
        record(conf/rel)
    result = {'analysis': 'Exact saved-probability moments only; no sampling or circuit reruns.',
              'scope': 'Post hoc oracle allocation sensitivity, not new regret or confirmatory outcomes.',
              'source_sha256': hashes, 'numpy': np.__version__,
              'analysis_script_sha256': sha(Path(__file__)), 'cells': []}
    for c in protocol['cells']:
        group = next(g for g in protocol['groups'] if g['graph'] == c['graph'])
        g = graphs[c['graph']]
        costs = np.zeros(1 << g['n_nodes'])
        for state in range(len(costs)):
            costs[state] = -sum(w for a, b, w in g['edges']
                                if ((state >> a) & 1) != ((state >> b) & 1))
        probability_path = conf/'probabilities'/f'{c["graph"]}.npy'
        ideal_path = conf/'inputs'/f'{c["graph"]}_ideal.npy'
        record(ideal_path)
        probs = np.load(probability_path, mmap_mode='r')
        assert np.isfinite(probs).all() and np.min(probs) >= 0
        normalization_error = float(np.max(np.abs(probs.sum(axis=2)-1)))
        assert normalization_error < 1e-9
        ind = [group['scales'].index(s) for s in c['scales']]
        pp = probs[:, ind, :]
        means = pp@costs
        v = np.sum(pp*(costs[None, None, :]-means[:, :, None])**2, axis=2)
        # Check the independent second-moment identity without clipping values.
        assert np.max(np.abs(v-(pp@(costs**2)-means**2))) < 1e-11
        a = np.abs(np.array(c['weights']))
        q = a*a*v
        B = c['budget']
        n = np.array(c['allocation'])
        assert np.sum(n) == B and np.min(n) > 0
        implemented = (q/n).sum(axis=1)
        proportional = (a.sum()/B)*(a*v).sum(axis=1)
        uniform = (len(a)/B)*q.sum(axis=1)
        oracle = (a*np.sqrt(v)).sum(axis=1)**2/B
        ni, oracle_i = integer_optimum(q, B)
        # Certificate: no one-shot exchange improves the integer optimum.
        gain = q/(ni*(ni+1))
        loss = np.where(ni > 1, q/(ni*np.maximum(ni-1, 1)), np.inf)
        assert np.all(np.max(gain, axis=1) <= np.min(loss, axis=1)+1e-13)
        fixed_n, fixed_i = integer_optimum(q.mean(axis=0, keepdims=True), B)
        fixed_variance = (q/fixed_n[0]).sum(axis=1)
        fixed_cont = (a*np.sqrt(v.mean(axis=0))).sum()**2/B
        baseline_variance = v[:, 0]/B
        ideal = np.load(ideal_path).reshape(-1)
        bias_z = (means@np.array(c['weights']))-ideal
        bias_n = means[:, 0]-ideal
        positive = oracle > 0
        row = {'id': c['id'], 'B': B, 'weights': c['weights'],
               'recorded_allocation': c['allocation'],
               'source_inputs': {'protocol': 'confirmation_20261019/protocol.json',
                                 'graphs': 'confirmation_20261019/inputs/graphs.json',
                                 'probabilities': probability_path.relative_to(root).as_posix(),
                                 'ideal': ideal_path.relative_to(root).as_posix()},
               'n_grid_points': len(v),
               'single_shot_variance_by_scale': [stats(v[:, j]) for j in range(len(a))],
               'single_shot_zero_variance_count': int(np.sum(v == 0)),
               'probability_normalization_max_error': normalization_error,
               'mean_variance': {'implemented': float(implemented.mean()),
                                 'baseline': float(baseline_variance.mean()),
                                 'proportional_continuous': float(proportional.mean()),
                                 'uniform_continuous': float(uniform.mean()),
                                 'pointwise_neyman_continuous': float(oracle.mean()),
                                 'pointwise_neyman_integer_exact': float(oracle_i.mean()),
                                 'fixed_neyman_continuous': float(fixed_cont),
                                 'fixed_neyman_integer_exact': float(fixed_i[0])},
               'pointwise_ratio_implemented_over_continuous_oracle': stats(implemented[positive]/oracle[positive]),
               'pointwise_ratio_implemented_over_integer_oracle': stats(implemented[positive]/oracle_i[positive]),
               'aggregate_factor_implemented_over_pointwise_continuous_oracle': float(implemented.mean()/oracle.mean()),
               'aggregate_factor_implemented_over_pointwise_integer_oracle': float(implemented.mean()/oracle_i.mean()),
               'aggregate_factor_implemented_over_fixed_continuous_oracle': float(implemented.mean()/fixed_cont),
               'aggregate_factor_implemented_over_fixed_integer_oracle': float(implemented.mean()/fixed_i[0]),
               'fixed_integer_oracle_allocation': fixed_n[0].tolist(),
               'aggregate_factor_implemented_over_uniform_continuous': float(implemented.mean()/uniform.mean()),
               'fractional_variance_reduction_pointwise_integer_oracle': float(1-oracle_i.mean()/implemented.mean()),
               'fractional_variance_reduction_fixed_integer_oracle': float(1-fixed_i[0]/implemented.mean()),
               'mean_excess_variance_implemented_over_baseline': float(np.mean(implemented-baseline_variance)),
               'fractional_reduction_excess_variance_pointwise_integer_oracle': float(np.mean(implemented-oracle_i)/np.mean(implemented-baseline_variance)),
               'mean_squared_bias': {'zne': float(np.mean(bias_z**2)), 'baseline': float(np.mean(bias_n**2))},
               'exact_expected_grid_MSE': {'implemented': float(np.mean(bias_z**2+implemented)),
                                         'pointwise_integer_oracle': float(np.mean(bias_z**2+oracle_i)),
                                         'baseline': float(np.mean(bias_n**2+baseline_variance))},
               'integer_oracle_max_relative_gap_over_continuous': float(np.max(oracle_i[positive]/oracle[positive]-1)),
               'recorded_rounding_max_relative_abs_variance_change': float(np.max(np.abs(implemented/proportional-1))),
               'max_abs_integer_fixed_mean_crosscheck': float(abs(fixed_variance.mean()-fixed_i[0]))}
        assert np.all(oracle <= oracle_i+1e-13)
        assert np.all(oracle_i <= implemented+1e-13)
        result['cells'].append(row)
    result['elapsed_seconds'] = time.perf_counter()-t0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(f'Wrote {len(result["cells"])} exact-moment sensitivity cells to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT,
                        help='Project root; defaults to the root derived from this script location.')
    parser.add_argument('--output', type=Path, required=True,
                        help='Result JSON path, outside the confirmation source directory.')
    args = parser.parse_args()
    analyze(args.root, args.output)


if __name__ == '__main__':
    main()
