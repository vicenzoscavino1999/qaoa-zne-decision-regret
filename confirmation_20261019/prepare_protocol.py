"""Run once before producing confirmatory landscapes; no new endpoints inspected."""
from pathlib import Path
import sys,site,json,hashlib,shutil,secrets,platform
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
site.addsitedir(str(ROOT/'.venv/Lib/site-packages'));sys.path.insert(0,str(ROOT))
import numpy as np
import scipy,qiskit,qiskit_aer
from src.circuits import GraphInstance,ideal_expected_cut
from src.landscapes import parameter_grid

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    if (HERE/'protocol.json').exists() or (HERE/'protocol_lock.json').exists():raise RuntimeError('Protocol already prepared; no overwrite')
    assert not (HERE/'samples').exists() and not (HERE/'analysis').exists()
    inputs=HERE/'inputs';inputs.mkdir(exist_ok=False)
    source=ROOT/'results/phase_d/planning/phase_d_selected_instances_final.json'
    original=json.loads(source.read_text());graphs={k:original[k] for k in ('H1','H3','H8')}
    save(inputs/'graphs.json',graphs)
    shutil.copyfile(ROOT/'revision_20261015/statistics/holm_cells_D.csv',inputs/'historical_selection.csv')
    checks=[];gammas,betas=parameter_grid(120)
    for name,record in graphs.items():
        files=list((ROOT/'data/raw/phase_d').glob(f'D*/{name}/grid_120/C_ideal.npy'));assert len(files)==1
        shutil.copyfile(files[0],inputs/f'{name}_ideal.npy')
        ideal=np.load(files[0]);assert ideal.shape==(120,120) and np.isfinite(ideal).all() and np.ptp(ideal)>0
        graph=GraphInstance(name,tuple(tuple(e) for e in record['edges']));assert graph.n_qubits==record['n_nodes']
        errors=[]
        for idx in (0,167,1249,3317,5611,8017,11119,14399):
            independent=-ideal_expected_cut(graph,gammas[idx//120],betas[idx%120])
            errors.append(abs(float(ideal.ravel()[idx])-independent))
        assert max(errors)<1e-10
        checks.append({'graph':name,'original_ideal':files[0].relative_to(ROOT).as_posix(),'sha256':sha(files[0]),'max_statevector_error_8_fixed_points':max(errors)})
    save(HERE/'input_preflight.json',{'graphs_source_sha256':sha(source),'checks':checks})
    cells=[
      {'id':'H1_q','graph':'H1','noise':.02,'budget':1024,'order':'quadratic','scales':[1,3,5],'weights':[1.875,-1.25,.375],'allocation':[548,366,110]},
      {'id':'H3_q','graph':'H3','noise':.02,'budget':1024,'order':'quadratic','scales':[1,3,5],'weights':[1.875,-1.25,.375],'allocation':[548,366,110]},
      {'id':'H8_c','graph':'H8','noise':.01,'budget':4096,'order':'cubic','scales':[1,3,5,7],'weights':[2.1875,-2.1875,1.3125,-.3125],'allocation':[1493,1493,896,214]},
      {'id':'H8_q','graph':'H8','noise':.01,'budget':4096,'order':'quadratic','scales':[1,3,5],'weights':[1.875,-1.25,.375],'allocation':[2194,1463,439]}]
    stamp=datetime.now(timezone.utc).isoformat()
    p={'campaign':'targeted_confirmation_20261019','frozen_at_utc':stamp,
       'scope':'Prospective targeted confirmation of four historically selected conditions, conditional on that selection; not the original 540-condition family or graph-population inference.',
       'selection':'Four candidates in historical_selection.csv; all old data excluded from confirmatory endpoints.',
       'repetitions':100,'alpha':.05,'regret_tolerance':1e-12,'legacy_strong_regret':.0399621871036988,
       'cells':cells,'groups':[{'graph':'H1','noise':.02,'budget':1024,'scales':[1,3,5]},{'graph':'H3','noise':.02,'budget':1024,'scales':[1,3,5]},{'graph':'H8','noise':.01,'budget':4096,'scales':[1,3,5,7]}],
       'root_entropy_uint32':[secrets.randbits(32) for _ in range(8)],
       'sampling':'NumPy SeedSequence with 256-bit OS entropy; spawn groups in listed order, repetitions 0..99, then baseline and each cell/scale in listed order. PCG64DXSM. One persistent RNG per stream, sequential row-major draws. H8 baseline shared across orders; baseline and ZNE scale1 are separate streams. Pilot streams excluded.',
       'grid':{'shape':[120,120],'gamma':'linspace(0,2*pi,120,endpoint=False)','beta':'linspace(0,pi,120,endpoint=False)','flatten':'C order; gamma outer, beta inner','argmin':'np.argmin first index; no tolerance adjustment of minimizer'},
       'probabilities':'Unchanged density-matrix kernel; depolarizing lambda=p on h/rx/rz and 2p on cx; global odd-integer folding. Require finite nonnegative probabilities, abs(sum-1)<=1e-9; normalize by sum before multinomial. Cache deterministic probabilities with chunk and full SHA256 hashes.',
       'metrics':'float64 grid means MSE_N,MSE_Z; r=(MSE_N-MSE_Z)/MSE_N. R=(ideal[argmin]-min(ideal))/(max(ideal)-min(ideal)); d=R_Z-R_N. Positive MSE_N and D0 required.',
       'primary':'K_r=sum(r>0), K_d=sum(d>1e-12); all 100 repetitions count; ties fail. Upper-tail Binomial(100,.5) tests. Per-cell IUT=max(p_r,p_d), Holm adjustment over exactly four IUT p-values at alpha=.05. No practical-effect filter.',
       'secondary':'Median r,d; d quartiles; median d in objective units; joint-positive count; individual 95% Clopper-Pearson intervals, descriptive without simultaneous coverage. Historical 10% MSE and strong-regret thresholds descriptive only.',
       'failure_policy':'No early stopping, sample-size extension, seed replacement, metric-based exclusions or partial endpoint analysis. Technical I/O interruption permits deterministic recovery with identical scheduled streams; completed groups verified and reused. Numerical/source failure stops inference and requires documented audit/amendment, preserving all prior data.',
       'assumptions':'Usual independent PRNG stream/repetition model; not literal independence proof for deterministic random generators. Dependence across endpoints and H8 orders permitted by IUT/Holm.',
       'design_sensitivity':'At n=100, K>=62 in both components is sufficient to pass Holm4. Component probability of K>=62 is .38219 at pi=.60, .76987 at .65, .96602 at .70; no power claim based on selected historical success rates.',
       'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'qiskit':qiskit.__version__,'qiskit_aer':qiskit_aer.__version__}}
    save(HERE/'protocol.json',p)
    human='TARGETED PROSPECTIVE CONFIRMATION - FROZEN PROTOCOL\n\n'
    human+='This is a local timestamped protocol fixed before new sampling, not an externally registered preregistration.\nFrozen UTC: '+stamp+'\n\n'
    for key in ('scope','selection','sampling','probabilities','metrics','primary','secondary','failure_policy','assumptions','design_sensitivity'):
        human+=key.upper()+'\n'+p[key]+'\n\n'
    human+='FIXED CELLS, GRID, SAMPLE SIZE AND SOFTWARE\n'+json.dumps({k:p[k] for k in ('repetitions','cells','grid','versions','root_entropy_uint32')},indent=2)+'\n'
    (HERE/'PROTOCOL.txt').write_text(human,encoding='utf-8')
    frozen=[HERE/'protocol.json',HERE/'PROTOCOL.txt',HERE/'run_confirmation.py',HERE/'prepare_protocol.py',HERE/'input_preflight.json',HERE/'pilot.py',HERE/'pilot_results.json']
    frozen+=list(inputs.iterdir())+sorted((ROOT/'src').glob('*.py'))
    lock={'frozen_at_utc':stamp,'no_confirmatory_samples_created':True,'sha256':{f.relative_to(ROOT).as_posix():sha(f) for f in frozen}}
    save(HERE/'protocol_lock.json',lock)
    print(json.dumps({'frozen_at_utc':stamp,'protocol_sha256':sha(HERE/'protocol.json'),'locked_files':len(frozen),'preflight':checks},indent=2))

if __name__=='__main__':main()
