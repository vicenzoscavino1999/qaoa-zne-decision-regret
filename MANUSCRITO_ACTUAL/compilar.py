"""Compile the editable unified sources to manuscrito.pdf.

Usage: python compilar.py [--tex-bin PATH_TO_TEX_EXECUTABLES]
Requires a working LaTeX distribution; does not run scientific calculations.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile

HERE=Path(__file__).resolve().parent
SRC=HERE/'fuentes'
CHECKS=HERE/'verificacion'
CHECKS.mkdir(exist_ok=True)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--tex-bin',type=Path)
args=parser.parse_args()

def executable(name):
    suffix='.exe' if os.name=='nt' else ''
    result=str(args.tex_bin/(name+suffix)) if args.tex_bin else shutil.which(name)
    if not result or not Path(result).is_file():
        raise SystemExit(f'Cannot find {name}; use --tex-bin to select your LaTeX installation.')
    return result

latex,bibtex=executable('pdflatex'),executable('bibtex')
env=os.environ.copy()
for variable in ('BIBINPUTS','BSTINPUTS'):
    env[variable]=str(SRC)+os.pathsep+env.get(variable,'')
with tempfile.TemporaryDirectory(prefix='qaoa_manuscrito_') as temporary:
    build=Path(temporary)
    pdflatex=[latex,'-interaction=nonstopmode','-halt-on-error','-output-directory='+str(build),'main.tex']
    commands=[(pdflatex,SRC),([bibtex,'main'],build),(pdflatex,SRC),(pdflatex,SRC),(pdflatex,SRC)]
    for index,(command,cwd) in enumerate(commands):
        run=subprocess.run(command,cwd=cwd,env=env,capture_output=True)
        (CHECKS/f'compilacion_{index}.txt').write_bytes(run.stdout+run.stderr)
        if run.returncode:
            raise SystemExit(run.stdout.decode(errors='replace')[-5000:]+run.stderr.decode(errors='replace')[-1000:])
    log=(build/'main.log').read_text(errors='replace')
    warnings=[line for line in log.splitlines() if any(token in line for token in
              ('Overfull','undefined','Rerun to get','Missing character','multiply defined'))]
    shutil.copyfile(build/'main.log',CHECKS/'main.log')
    shutil.copyfile(build/'main.aux',CHECKS/'main.aux')
    shutil.copyfile(build/'main.bbl',SRC/'main.bbl')
    if warnings:
        print(json.dumps({'status':'NEEDS_REVIEW','warnings':warnings},indent=2))
    shutil.copyfile(build/'main.pdf',HERE/'manuscrito.pdf')
    bbl=(build/'main.bbl').read_text(encoding='utf-8')
    status={'status':'PASS' if not warnings else 'NEEDS_REVIEW','warnings':warnings,
            'bibliography_items':bbl.count(r'\bibitem'),
            'pdf_sha256':hashlib.sha256((HERE/'manuscrito.pdf').read_bytes()).hexdigest(),
            'scientific_execution':False}
    (CHECKS/'compilacion.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(status,indent=2))
