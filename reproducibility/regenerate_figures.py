"""Regenerate and verify the three CURRENT manuscript figures from saved numbers.

Run from any directory with Python 3.12, the project's locked plotting stack,
and Poppler (pdfinfo, pdftotext, pdftoppm) on PATH, or supply --pdf-bin.
--output must be a NEW directory outside the project/package. All generated
PDFs, raster comparisons, plot traces and the report are written only there.

Plotting source: MANUSCRITO_ACTUAL/propuesta_reencuadre/figuras_regeneradas/
regen.py, mode iop, whose PDFs equal the current canonical figure PDFs.
This standalone copy needs neither that historical folder nor circuit modules.
It reads cells_D.csv and bias_variance/results.json; it draws no new samples.
PDF byte equality is reported but is not required (creation dates may differ).
Verification requires plotted values to floating-point precision, page sizes, extracted text and
pixel equality at 150 dpi using the same Poppler renderer for both PDFs.
Different fonts/Matplotlib/renderer versions may prevent visual equivalence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def external_output(root, output):
    output = output.expanduser().resolve()
    if output == root or root in output.parents:
        raise ValueError('--output must be outside the project/package')
    if output.exists():
        raise ValueError('--output must be a NEW directory; existing paths are refused')
    return output


def poppler_tools(directory):
    found = {}
    for name in ('pdfinfo', 'pdftotext', 'pdftoppm'):
        if directory:
            path = directory / (name + '.exe' if platform.system() == 'Windows' else name)
            found[name] = str(path) if path.is_file() else None
        else:
            found[name] = shutil.which(name)
        if not found[name]:
            raise RuntimeError(f'{name} is required; install Poppler or supply --pdf-bin')
    return found


def generate(root, output):
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font = {'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'pdf.fonttype': 42}
    traces = {}
    df = pd.read_csv(root / 'revision_20261015/statistics/cells_D.csv')
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42, **font})
    fig, axes = plt.subplots(2, 1, figsize=(4.8, 5.1), layout='constrained')
    retained = df.revised_holm_reject
    excluded = df.excluded
    other = ~retained & ~excluded
    for group, size, color, label, extra in (
        (other, 10, '#a9b5bf', 'Other eligible conditions', {'alpha': .7}),
        (retained, 36, '#b24a25', 'Sign + Holm retained', {'zorder': 3}),
    ):
        artist = axes[0].scatter(df.loc[group, 'median_r_mse'], df.loc[group, 'median_d_R'],
                                 s=size, c=color, label=label, **extra)
        expected = df.loc[group, ['median_r_mse', 'median_d_R']].to_numpy()
        np.testing.assert_array_equal(artist.get_offsets(), expected)
        traces[label] = expected.tolist()
    axes[0].axhline(0, c='black', lw=.6)
    axes[0].axvline(0, c='black', lw=.6)
    axes[0].set_xscale('symlog', linthresh=1.)
    axes[0].set(xlabel='Median relative MSE improvement (symlog)',
                ylabel='Median normalized regret change')
    axes[0].set_title('(a)', loc='left')
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    subset = df[retained].sort_values(['instance', 'shots_total'])
    labels = [f'{r.instance}, p={r.noise:g}, B={r.shots_total}\n{r.zne_order}'
              for r in subset.itertuples()]
    y = np.arange(len(subset))
    lines = axes[1].hlines(y, subset.q25_d_R, subset.q75_d_R, color='#b24a25', lw=2)
    expected_segments = [[[lo, int(i)], [hi, int(i)]] for i, lo, hi in
                         zip(y, subset.q25_d_R, subset.q75_d_R)]
    np.testing.assert_array_equal(lines.get_segments(), expected_segments)
    dots = axes[1].scatter(subset.median_d_R, y, color='#b24a25', s=24, zorder=3)
    np.testing.assert_array_equal(dots.get_offsets(), np.column_stack([subset.median_d_R, y]))
    axes[1].axvline(0, color='black', lw=.6)
    axes[1].set_yticks(y, labels, fontsize=8)
    axes[1].set(xlabel='Normalized regret change')
    axes[1].set_title('(b)', loc='left')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(output / 'figure1.pdf')
    plt.close(fig)
    traces['figure1_retained_IQR'] = {'labels': labels, 'segments': expected_segments,
                                     'medians': subset.median_d_R.tolist()}

    plt.rcdefaults()
    rows = json.loads((root / 'revision_20261015/bias_variance/results.json').read_text())['rows']
    names = ['G1_C4', 'G2_weighted4', 'G3_weighted6']
    methods = ['noisy', 'linear', 'quadratic', 'cubic']
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'axes.labelsize': 10,
                         'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
                         'legend.fontsize': 9.5, 'axes.linewidth': .7, **font})
    fig, axs = plt.subplots(3, 2, figsize=(5.2, 6), layout='constrained')
    fig2, axs2 = plt.subplots(3, 2, figsize=(5.2, 6), layout='constrained')
    display = ['G1 (C4)', 'G2 (weighted4)', 'G3 (weighted6)']
    traces['control_panels'] = []
    for j, p in enumerate((.001, .02)):
        for i, name in enumerate(names):
            r = [next(r for r in rows if r['graph'] == name and r['p'] == p and r['method'] == m)
                 for m in methods]
            x = np.arange(4)
            b = np.array([z['bias2'] for z in r])
            v = np.array([z['variance'] for z in r])
            mse = np.array([z['mc_mse'] for z in r])
            mse_error = np.array([1.96 * z['mc_mse_se'] for z in r])
            exact = np.array([z['mean_landscape_regret'] for z in r])
            finite = np.array([z['mc_regret'] for z in r])
            regret_error = np.array([1.96 * z['mc_regret_se'] for z in r])
            ax = axs[i, j]
            bars_b = ax.bar(x, b, label='Squared bias', color='#245980')
            bars_v = ax.bar(x, v, bottom=b, label='Sampling variance', color='#d58d38')
            errors = ax.errorbar(x, mse, yerr=mse_error, fmt='k.', label='MC MSE ± 1.96 SE',
                                capsize=2.5, elinewidth=.9, markersize=3.5)
            np.testing.assert_array_equal([z.get_height() for z in bars_b], b)
            # Matplotlib forms stacked rectangles through bottom+height-bottom,
            # which can introduce a few final-bit roundoff differences.
            np.testing.assert_allclose([z.get_height() for z in bars_v], v, rtol=2e-14, atol=1e-18)
            np.testing.assert_array_equal([z.get_y() for z in bars_v], b)
            np.testing.assert_array_equal(errors.lines[0].get_ydata(), mse)
            np.testing.assert_array_equal(errors.lines[2][0].get_segments(),
                                          np.stack([np.column_stack([x, mse-mse_error]),
                                                    np.column_stack([x, mse+mse_error])], axis=1))
            ax.set(title=f'{display[i]}, p={p:g}', xticks=x, xticklabels=methods,
                   ylabel='Mean squared error' if j == 0 else '')
            ax.set_ylim(bottom=0)
            ax.tick_params(axis='x', labelrotation=32, pad=1)
            for label in ax.get_xticklabels():
                label.set_ha('right')
            ax = axs2[i, j]
            bars_e = ax.bar(x-.18, exact, width=.36, label='Exact mean', color='#245980')
            bars_f = ax.bar(x+.18, finite, width=.36, label='Finite shots ± 1.96 SE',
                            color='#d58d38', yerr=regret_error, capsize=2.5, error_kw={'elinewidth': .9})
            np.testing.assert_array_equal([z.get_height() for z in bars_e], exact)
            np.testing.assert_array_equal([z.get_height() for z in bars_f], finite)
            np.testing.assert_array_equal(bars_f.errorbar.lines[2][0].get_segments(),
                                          np.stack([np.column_stack([x+.18, finite-regret_error]),
                                                    np.column_stack([x+.18, finite+regret_error])], axis=1))
            ax.set(title=f'{display[i]}, p={p:g}', xticks=x, xticklabels=methods,
                   ylabel='Normalized ideal regret' if j == 0 else '')
            ax.tick_params(axis='x', labelrotation=32, pad=1)
            for label in ax.get_xticklabels():
                label.set_ha('right')
            traces['control_panels'].append({'graph': name, 'p': p, 'methods': methods,
                'bias2': b.tolist(), 'variance': v.tolist(), 'mc_mse': mse.tolist(),
                'mse_error_1_96_SE': mse_error.tolist(), 'mean_landscape_regret': exact.tolist(),
                'mc_regret': finite.tolist(), 'regret_error_1_96_SE': regret_error.tolist()})
    regret_top = max(r['mc_regret'] + 1.96*r['mc_regret_se'] for r in rows)*1.12
    for ax in axs2.flat:
        ax.set_ylim(0, regret_top)
    for ax in axs2[0, :]:
        ax.text(.5, .60, 'All regrets < 1e-14\n(floating-point scale)',
                ha='center', transform=ax.transAxes, fontsize=9.5)
    for f, a in ((fig, axs), (fig2, axs2)):
        handles, labels = a[0, 0].get_legend_handles_labels()
        f.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False,
                 columnspacing=1, handletextpad=.5, handlelength=1.4, labelspacing=.4)
    fig.savefig(output / 'figure2.pdf')
    fig2.savefig(output / 'figure3.pdf')
    plt.close('all')
    (output / 'plot_traces.json').write_text(json.dumps(traces, indent=2, allow_nan=False)+'\n')
    return {'plotted_values_verified': True, 'other_conditions': int(other.sum()),
            'retained_conditions': int(retained.sum()), 'control_configurations': len(rows),
            'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                         'pandas': pd.__version__, 'matplotlib': matplotlib.__version__},
            'font': Path(font_manager.findfont(font_manager.FontProperties(family=['sans-serif']))).name}


def compare_pdfs(root, output, tools):
    import numpy as np
    from PIL import Image
    comparisons = []
    def call(name, *args):
        return subprocess.run([tools[name], *map(str, args)], check=True, capture_output=True).stdout
    for i in range(1, 4):
        name = f'figure{i}'
        canonical = root / 'MANUSCRITO_ACTUAL/fuentes/figures' / (name+'.pdf')
        generated = output / (name+'.pdf')
        infos, texts, images = [], [], []
        for kind, path in (('canonical', canonical), ('regenerated', generated)):
            info = call('pdfinfo', '-box', path).decode('utf-8', errors='replace')
            # Ignore volatile PDF dates; retain page count, dimensions and boxes.
            infos.append([line.strip() for line in info.splitlines()
                          if re.match(r'^(Pages:|Page size:|Page rot:|MediaBox:|CropBox:|BleedBox:|TrimBox:|ArtBox:)', line)])
            texts.append(call('pdftotext', '-layout', '-enc', 'UTF-8', path, '-'))
            prefix = output / f'{name}_{kind}'
            call('pdftoppm', '-r', '150', '-singlefile', '-png', path, prefix)
            with Image.open(prefix.with_suffix('.png')) as im:
                images.append(np.array(im.convert('RGB')))
        same_shape = images[0].shape == images[1].shape
        diff = np.abs(images[0].astype(np.int16)-images[1].astype(np.int16)) if same_shape else None
        item = {'figure': name, 'canonical_sha256': sha(canonical), 'generated_sha256': sha(generated),
                'pdf_bytes_equal': canonical.read_bytes() == generated.read_bytes(),
                'page_geometry_equal': infos[0] == infos[1], 'page_geometry': infos,
                'extracted_text_equal': texts[0] == texts[1], 'raster_dpi': 150,
                'raster_shape_equal': same_shape, 'raster_pixels_equal': bool(same_shape and not diff.any()),
                'differing_pixels': int(np.any(diff, axis=2).sum()) if same_shape else None,
                'max_channel_difference': int(diff.max()) if same_shape else None}
        item['equivalent'] = all(item[k] for k in ('page_geometry_equal', 'extracted_text_equal',
                                                   'raster_shape_equal', 'raster_pixels_equal'))
        comparisons.append(item)
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True, help='NEW directory outside this package/project')
    parser.add_argument('--pdf-bin', type=Path, help='directory containing Poppler executables, if absent from PATH')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = external_output(root, args.output)
    tools = poppler_tools(args.pdf_bin)
    sources = ['revision_20261015/statistics/cells_D.csv', 'revision_20261015/bias_variance/results.json']
    # Read and hash before authoring: missing inputs must not leave partial outputs.
    hashes = {rel: sha(root / rel) for rel in sources}
    canonical_hashes = {f'figure{i}.pdf': sha(root / 'MANUSCRITO_ACTUAL/fuentes/figures' / f'figure{i}.pdf')
                        for i in range(1, 4)}
    output.mkdir(parents=True, exist_ok=False)
    report = {'scope': 'Saved numerical inputs only; no circuits, new sampling or statistical reruns.',
              'sources_sha256': hashes, 'script_sha256': sha(Path(__file__)),
              'plotting_origin': 'Historical figuras_regeneradas/regen.py, iop mode (standalone plotting copy).',
              'plot_checks': generate(root, output)}
    report['figures'] = compare_pdfs(root, output, tools)
    report['inputs_unchanged'] = all(sha(root / rel) == h for rel, h in hashes.items())
    report['canonical_figures_unchanged'] = all(
        sha(root / 'MANUSCRITO_ACTUAL/fuentes/figures' / name) == h for name, h in canonical_hashes.items())
    report['status'] = 'PASS' if all(r['equivalent'] for r in report['figures']) and report['inputs_unchanged'] and report['canonical_figures_unchanged'] else 'FAIL'
    report['limits'] = ['PDF byte equality is optional because metadata dates may differ.',
                        'Raster equality applies at 150 dpi with the same Poppler renderer.',
                        'This verifies reproduction of saved results and presentation, not original simulation accuracy.',
                        'Fonts and plotting/rendering versions can affect output on another platform.']
    (output / 'figure_verification.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': report['status'], 'output': str(output),
                      'figures': [{k: r[k] for k in ('figure', 'equivalent', 'pdf_bytes_equal', 'differing_pixels')}
                                  for r in report['figures']]}, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
