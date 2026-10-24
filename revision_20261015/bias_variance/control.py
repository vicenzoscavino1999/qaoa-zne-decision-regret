"""Bounded mechanistic control. Run Python -B control.py --pilot, then without --pilot."""
import sys, site, time, json, csv, hashlib, platform
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
site.addsitedir(str(ROOT/'.venv/Lib/site-packages')); sys.path.insert(0,str(ROOT))
import numpy as np
import yaml, qiskit, qiskit_aer
from src.circuits import graph_from_config
from src.phase_c_120_custom_kernel import kernel_probabilities, aer_density_probabilities, cut_values_vector
from src.mitiq_wrapper import richardson_weights, allocate_shots, deterministic_seed
from src.landscapes import parameter_grid
START=time.perf_counter()
CFG=ROOT/'config/config_official_pilot_v1.6.yaml'
cfg=yaml.safe_load(CFG.read_text())
names=['G1_C4','G2_weighted4','G3_weighted6']
orders={'noisy':[1],'linear':[1,3],'quadratic':[1,3,5],'cubic':[1,3,5,7]}
if '--pilot' in sys.argv:
    g=graph_from_config(names[-1],cfg); timing=[]
    for s in (1,3,5,7):
        t=time.perf_counter(); kernel_probabilities(g,.731,.419,.02,s)
        timing.append({'scale':s,'seconds':time.perf_counter()-t})
    result={'timing':timing,'estimated_full_seconds':sum(x['seconds'] for x in timing)*144*2*3,'elapsed':time.perf_counter()-START}
    (OUT/'pilot.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result)); sys.exit()
N=12; R=500; B=1024
rows=[]; points=[]; reps=[]; validations=[]
gammas,betas=parameter_grid(N)
for name in names:
    graph=graph_from_config(name,cfg); cost=-cut_values_vector(graph)
    ideal=np.array([kernel_probabilities(graph,float(a),float(b),0,1).probabilities@cost for a in gammas for b in betas])
    span=float(np.ptp(ideal)); best=int(np.argmin(ideal))
    # First row-major point strictly worse than the minimum with the smallest ideal gap.
    candidates=np.where(ideal>ideal[best]+1e-10)[0]; competitor=int(candidates[np.argmin(ideal[candidates])])
    for p in (.001,.02):
        probs=np.array([[kernel_probabilities(graph,float(a),float(b),p,s).probabilities for s in (1,3,5,7)] for a in gammas for b in betas])
        mu=probs@cost; mom2=probs@(cost*cost); sigma2=np.maximum(mom2-mu*mu,0)
        if name in (names[0],names[-1]) and p==.02:
            a=aer_density_probabilities(graph,float(gammas[2]),float(betas[3]),p,7).probabilities
            k=probs[2*N+3,3]
            validations.append({'graph':name,'mean_error':float(abs((a-k)@cost)),'second_moment_error':float(abs((a-k)@(cost*cost)))})
        for method,scales in orders.items():
            w=np.array(richardson_weights(scales)); shots=np.array(allocate_shots(B,w)); count=len(scales)
            exact=mu[:,:count]@w; variance=sigma2[:,:count]@(w*w/shots); bias=exact-ideal
            rng=np.random.default_rng(deterministic_seed('bias_variance_v1',name,p,method,20261015))
            sampled=np.zeros((R,N*N))
            for i in range(N*N):
                for j in range(count):
                    sampled[:,i]+=w[j]*(rng.multinomial(int(shots[j]),probs[i,j],size=R)@cost)/shots[j]
            mse=np.mean((sampled-ideal)**2,axis=1)
            regret=(ideal[np.argmin(sampled,axis=1)]-ideal[best])/span
            analytic=float(np.mean(bias*bias+variance)); mse_se=float(np.std(mse,ddof=1)/np.sqrt(R))
            # Decision contrast: C(competitor)-C(best), positive preserves the ideal order.
            contrast=sampled[:,competitor]-sampled[:,best]
            exact_contrast=float(exact[competitor]-exact[best]); contrast_var=float(variance[competitor]+variance[best])
            row=dict(graph=name,p=p,method=method,grid=N,B=B,replicates=R,bias2=float(np.mean(bias*bias)),variance=float(np.mean(variance)),expected_mse=analytic,mc_mse=float(np.mean(mse)),mc_mse_se=mse_se,mc_mse_z=float((np.mean(mse)-analytic)/mse_se),mean_landscape_regret=float((ideal[np.argmin(exact)]-ideal[best])/span),mc_regret=float(np.mean(regret)),mc_regret_se=float(np.std(regret,ddof=1)/np.sqrt(R)),ideal_best=best,competitor=competitor,ideal_gap=float(ideal[competitor]-ideal[best]),exact_contrast=exact_contrast,contrast_variance=contrast_var,contrast_mc_mean=float(np.mean(contrast)),contrast_mc_variance=float(np.var(contrast,ddof=1)),contrast_reversal_fraction=float(np.mean(contrast<0)),weights=json.dumps(w.tolist()),shots=json.dumps(shots.tolist()))
            rows.append(row)
            for i in range(N*N): points.append(dict(graph=name,p=p,method=method,index=i,gamma=float(gammas[i//N]),beta=float(betas[i%N]),ideal=float(ideal[i]),mean=float(exact[i]),bias=float(bias[i]),variance=float(variance[i]),mse=float(bias[i]**2+variance[i])))
            for r in range(R): reps.append(dict(graph=name,p=p,method=method,replicate=r,mse=float(mse[r]),regret=float(regret[r]),selected=int(np.argmin(sampled[r])),contrast=float(contrast[r])))
        print(name,p,'completed',time.perf_counter()-START,flush=True)
def write_csv(filename,data):
    with (OUT/filename).open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(data[0])); wr.writeheader(); wr.writerows(data)
write_csv('summary.csv',rows); write_csv('pointwise.csv',points); write_csv('replicates.csv',reps)
inputs=[CFG,ROOT/'src/phase_c_120_custom_kernel.py',ROOT/'src/circuits.py',ROOT/'src/mitiq_wrapper.py',ROOT/'src/noise_models.py',OUT/'protocol.md',Path(__file__)]
meta={'elapsed_seconds':time.perf_counter()-START,'python':platform.python_version(),'numpy':np.__version__,'qiskit':qiskit.__version__,'qiskit_aer':qiskit_aer.__version__,'hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},'validations':validations,'rows':rows}
(OUT/'results.json').write_text(json.dumps(meta,indent=2)); print(json.dumps({'elapsed':meta['elapsed_seconds'],'validations':validations}))
