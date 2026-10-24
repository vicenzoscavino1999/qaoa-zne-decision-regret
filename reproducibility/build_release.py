"""Build a new compact, portable release without changing scientific inputs.

Run with Python 3.12.10. Both destination and archive must be new and outside
the source tree. ZIP timestamps and ordering are fixed for repeatable builds
with the same Python/zlib and identical source bytes.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.0.0'

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def files_to_include(root):
    files = set()
    def add(relative):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.add(path)
    def tree(relative):
        for path in (root / relative).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                files.add(path)
    for relative in ('requirements-lock.txt', 'requirements-reproduction-lock.txt',
                     'LICENSE', 'LICENSE-DATA-DOCS.md', 'LICENSE_BUNDLE.md',
                     'CITATION.cff', '.zenodo.json', 'REPRODUCIBILITY.txt'):
        add(relative)
    tree('reproducibility')
    # The unmodified 90-file lock includes all source modules, including unused
    # Phase E modules. Inclusion records provenance; verification does not run them.
    lock = json.loads((root/'confirmation_20261019/protocol_lock.json').read_text())['sha256']
    for relative, expected in lock.items():
        add(relative)
        if sha(root/relative) != expected:
            raise ValueError('Frozen file changed: '+relative)
    add('confirmation_20261019/protocol_lock.json')
    for name in ('PROTOCOL.txt','README_CONFIRMATION.txt','verify_confirmation.py'):
        add('confirmation_20261019/'+name)
    for folder in ('inputs','probabilities','samples','analysis'):
        tree('confirmation_20261019/'+folder)
    for name in ('audit_completed_confirmation.py','independent_audit_results.json',
                 'independent_per_repetition.csv','audit_preparation_self_test.json'):
        add('confirmation_20261019/review_code/'+name)
    for name in ('check_results_interpretation.py','results_interpretation_check.json','design_sensitivity.json'):
        add('confirmation_20261019/review_protocol/'+name)
    for name in ('reanalyze_statistics.py','verify_revision.py','README_REVISION.txt','statistical_amendment.txt'):
        add('revision_20261015/'+name)
    tree('revision_20261015/statistics')
    tree('revision_20261015/bias_variance')
    tree('revision_20261015/allocation_sensitivity_20261020')
    for name in ('core_audit.json','archive_audit.json'):
        add('revision_20261015/prior_audit/'+name)
    for name in ('math_code_review.txt','seed_collisions.json'):
        add('revision_20261015/followup_review/math_code/'+name)
    for name in ('config_official_pilot_v1.6.yaml','phase_d_instances_v1.0.json','seeds.yaml'):
        add('config/'+name)
    for folder in ('results/phase_c_120_analysis','results/phase_d/global_D1_D2_D3_D4'):
        for name in ('comparisons_per_seed.csv','classification_matrix.csv'):
            add(folder+'/'+name)
    ideals = list((root/'data/raw/phase_d').glob('D*/H*/grid_120/C_ideal.npy'))
    if len(ideals) != 12:
        raise ValueError('Expected twelve historical D ideal arrays')
    files.update(ideals)
    for name in ('table_cobyla_endpoint_consistency.csv','table_depth2_exploratory.csv','table_noise_composition_sensitivity.csv'):
        add('manuscript_qst/tables/'+name)
    add('results/phase_d/robustness/no_oracle_decision_diagnostics/no_oracle_diagnostics_summary.csv')
    tree('MANUSCRITO_ACTUAL/fuentes')
    for name in ('manuscrito.pdf','compilar.py'):
        add('MANUSCRITO_ACTUAL/'+name)
    return sorted(files)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--destination',type=Path,required=True)
    ap.add_argument('--archive',type=Path,required=True)
    args=ap.parse_args()
    root=args.root.resolve();dest=args.destination.resolve();archive=args.archive.resolve()
    for path in (dest,archive):
        if path.exists() or path==root or root in path.parents:
            raise SystemExit('Choose a new output path outside the source tree: '+str(path))
    if dest in archive.parents:
        raise SystemExit('Archive must be outside the staging directory')
    selected=files_to_include(root)
    dest.mkdir(parents=True)
    for path in selected:
        target=dest/path.relative_to(root)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    shutil.copyfile(root/'REPRODUCIBILITY.txt',dest/'README.txt')
    for source, target in [('gitattributes.txt', '.gitattributes'), ('gitignore.txt', '.gitignore')]:
        shutil.copyfile(root/'reproducibility/release_assets'/source, dest/target)
    payload=sorted(p for p in dest.rglob('*') if p.is_file())
    manifest={'version':VERSION,'scope':'Unified manuscript, numerical reanalysis, complete targeted-confirmation data, control and allocation sensitivity. Historical noisy C/D landscapes are excluded; twelve historical ideal arrays are included. Public deposition pending.',
              'sha256':{p.relative_to(dest).as_posix():sha(p) for p in payload}}
    (dest/'REVIEW_BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    archive.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(dest.rglob('*')):
            if path.is_file():
                info=zipfile.ZipInfo(path.relative_to(dest).as_posix(),date_time=(2026,10,23,0,0,0))
                info.compress_type=zipfile.ZIP_DEFLATED;info.create_system=3;info.external_attr=0o100644<<16
                z.writestr(info,path.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for relative,digest in manifest['sha256'].items():
            assert hashlib.sha256(z.read(relative)).hexdigest()==digest,relative
    print(json.dumps({'version':VERSION,'payload_files':len(payload),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'destination':str(dest),'archive':str(archive)},indent=2))

if __name__=='__main__':main()
