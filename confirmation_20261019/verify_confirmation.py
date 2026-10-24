"""Read-only confirmation audit. An optional --report is the only written file."""
from pathlib import Path
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import site
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
site.addsitedir(str(ROOT / '.venv/Lib/site-packages'))
sys.path.insert(0, str(ROOT))
from reproducibility.verify_manuscript import verify_manuscript


def verify():
    manifest = ROOT / 'REVIEW_BUNDLE_MANIFEST.json'
    checked = 0
    if manifest.exists():
        for rel, digest in json.loads(manifest.read_text(encoding='utf-8'))['sha256'].items():
            assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == digest, rel
            checked += 1
    path = HERE / 'review_code/audit_completed_confirmation.py'
    spec = importlib.util.spec_from_file_location('confirmation_independent_audit', path)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    captured = io.StringIO()
    with redirect_stdout(captured):
        audit.audit_completed(verify_only=True)
    numerical_audit = json.loads(captured.getvalue())
    tables = verify_manuscript(ROOT)
    saved = json.loads((HERE / 'analysis/results.json').read_text(encoding='utf-8'))
    return {
        'status': 'PASS', 'bundle_hashes': checked, 'confirmed': saved['n_confirmed'],
        'main_fresh_table_current': True, 'manuscript_tables': tables,
        'numerical_audit': numerical_audit,
        'scope': 'Four targeted conditions; numerical audit does not validate historical independence.',
        'limits': [
            'All 400 saved paired endpoints are recomputed; only three draws of each scheduled stream are replayed.',
            'Six fixed Aer points do not certify every cached probability.',
            'PRNG independence remains a statistical modeling assumption.',
            'Fresh results concern four historically selected targets, not prevalence over the 540-condition screen.',
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    rendered = json.dumps(verify(), indent=2) + '\n'
    if args.report:
        args.report.write_text(rendered, encoding='utf-8')
    print(rendered, end='')


if __name__ == '__main__':
    main()
