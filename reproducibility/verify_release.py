"""Read-only verification of an extracted compact release; no long campaign.

Optional report must be outside the package. Intermediate outputs are temporary.
The confirmation check includes six bounded Aer comparisons and partial replay
of recorded RNG streams; it does not generate a new experimental family.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]

def check_manifest(root):
    manifest=json.loads((root/'REVIEW_BUNDLE_MANIFEST.json').read_text(encoding='utf-8'))
    expected=manifest['sha256']
    actual=set()
    for current, directories, filenames in os.walk(root):
        base=Path(current)
        # A checkout adds Git bookkeeping, not release payload. Prune only the
        # root .git entry; unexpected files elsewhere must still be rejected.
        if base==root:
            directories[:]=[name for name in directories if name!='.git']
        for name in filenames:
            if base==root and name=='.git':
                continue
            actual.add((base/name).relative_to(root).as_posix())
    if actual != set(expected)|{'REVIEW_BUNDLE_MANIFEST.json'}:
        raise AssertionError({'extra_files':sorted(actual-set(expected)-{'REVIEW_BUNDLE_MANIFEST.json'}),'missing_files':sorted(set(expected)-actual)})
    for relative,sha in expected.items():
        path=(root/relative).resolve()
        if root not in path.parents:
            raise AssertionError('Unsafe manifest path: '+relative)
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=sha:
                raise AssertionError('Hash mismatch: '+relative)
    return len(expected)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report',type=Path)
    args=ap.parse_args()
    if args.report and (args.report.resolve()==ROOT or ROOT in args.report.resolve().parents):
        raise SystemExit('Report must be outside the release directory')
    started=time.perf_counter()
    count=check_manifest(ROOT)
    if platform.python_version()!='3.12.10':
        raise SystemExit('The frozen confirmation requires Python 3.12.10 exactly')
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    results={}
    with tempfile.TemporaryDirectory(prefix='qaoa-release-check-') as temp:
        tmp=Path(temp)
        for name,relative in [('historical_and_control','revision_20261015/verify_revision.py'),('confirmation','confirmation_20261019/verify_confirmation.py')]:
            run=subprocess.run([sys.executable,'-B',str(ROOT/relative)],cwd=tmp,env=env,capture_output=True,text=True,encoding='utf-8')
            if run.returncode:
                raise RuntimeError(name+' failed:\n'+run.stdout+'\n'+run.stderr)
            results[name]=json.loads(run.stdout)
        output=tmp/'allocation.json'
        run=subprocess.run([sys.executable,'-B',str(ROOT/'revision_20261015/allocation_sensitivity_20261020/allocation_sensitivity.py'),'--output',str(output)],cwd=tmp,env=env,capture_output=True,text=True,encoding='utf-8')
        if run.returncode:
            raise RuntimeError('Allocation check failed:\n'+run.stdout+'\n'+run.stderr)
        saved=json.loads((ROOT/'revision_20261015/allocation_sensitivity_20261020/allocation_sensitivity.json').read_text())
        fresh=json.loads(output.read_text())
        for key in ('cells','source_sha256','analysis_script_sha256','numpy'):
            if saved[key]!=fresh[key]:raise AssertionError('Allocation mismatch: '+key)
        results['allocation']={'status':'PASS','conditions':len(fresh['cells']),'saved_results_exactly_reproduced':True}
    after=check_manifest(ROOT)
    report={'status':'PASS','python':platform.python_version(),'payload_files':count,'package_unchanged':after==count,'checks':results,'elapsed_seconds':time.perf_counter()-started,
            'scope':'Numerical reproduction from included saved data, current main tables and manifest integrity. No long C/D rerun; historical seed-collision limitation remains. Figures and LaTeX compilation are separate documented commands.'}
    rendered=json.dumps(report,indent=2)+'\n'
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(rendered,encoding='utf-8')
    print(rendered)

if __name__=='__main__':main()
