"""Frozen targeted confirmation with persistent probabilities and fresh shot streams.

Run pilot.py first, prepare_protocol.py once, then this script --probabilities,
--sample, --analyze. No historical source/data is modified. All stochastic
inference uses the ordinary pseudorandom independent-stream model.
"""
from pathlib import Path
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
import argparse,sys,site,json,hashlib,time,math,platform
from importlib.metadata import version
from concurrent.futures import ProcessPoolExecutor,as_completed
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
site.addsitedir(str(ROOT/'.venv/Lib/site-packages'))
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.stats import binom, beta

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path, data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)

def protocol():
    p=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
    freeze=json.loads((HERE/'protocol_lock.json').read_text(encoding='utf-8'))
    for rel,digest in freeze['sha256'].items():
        if sha(ROOT/rel)!=digest: raise RuntimeError(f'Frozen input changed: {rel}')
    actual={'python':platform.python_version(),'numpy':np.__version__,'scipy':version('scipy'),'qiskit':version('qiskit'),'qiskit_aer':version('qiskit-aer')}
    if actual!=p['versions']: raise RuntimeError(f'Software version mismatch: {actual}')
    assert p['repetitions']==100 and len(p['cells'])==4
    assert [g['graph'] for g in p['groups']]==['H1','H3','H8']
    assert len({c['id'] for c in p['cells']})==4
    for g in p['groups']:
        assert g['scales'][0]==1 and len(set(g['scales']))==len(g['scales'])
        cells=[c for c in p['cells'] if c['graph']==g['graph']]
        assert cells and len({c['order'] for c in cells})==len(cells)
        for c in cells:
            assert c['budget']==g['budget'] and c['noise']==g['noise']
            assert len(c['scales'])==len(c['weights'])==len(c['allocation'])
            assert all(isinstance(b,int) and b>0 for b in c['allocation'])
            assert sum(c['allocation'])==c['budget'] and set(c['scales'])<=set(g['scales'])
    return p

def verify_manifest(path,expected):
    manifest=json.loads(Path(path).read_text())
    if manifest['protocol_sha256']!=sha(HERE/'protocol.json'): raise RuntimeError('Manifest protocol mismatch')
    if set(manifest['sha256'])!=set(expected): raise RuntimeError('Unexpected manifest file set')
    for name,digest in manifest['sha256'].items():
        if sha(Path(path).parent/name)!=digest: raise RuntimeError(f'Manifest integrity failure: {name}')
    return manifest

def probability_chunk(job):
    from src.circuits import GraphInstance
    from src.phase_c_120_custom_kernel import kernel_probabilities
    from src.landscapes import parameter_grid
    group,edges,start,stop=job
    graph=GraphInstance(group['graph'],tuple(tuple(e) for e in edges))
    gamma,beta_grid=parameter_grid(120)
    out=np.empty((stop-start,len(group['scales']),1<<graph.n_qubits))
    for offset,idx in enumerate(range(start,stop)):
        for j,scale in enumerate(group['scales']):
            v=kernel_probabilities(graph,gamma[idx//120],beta_grid[idx%120],group['noise'],scale).probabilities
            if not np.isfinite(v).all() or v.min()<0 or abs(v.sum()-1)>1e-9:
                raise RuntimeError('Probability validation failed')
            out[offset,j]=v/v.sum()
    return group['graph'],start,stop,out

def build_probabilities(workers):
    p=protocol(); graphs=json.loads((HERE/'inputs/graphs.json').read_text())
    cache=HERE/'probabilities';cache.mkdir(exist_ok=True)
    arrays={}; jobs=[]; done={}; started=time.perf_counter()
    manifest_path=cache/'manifest.json'
    if manifest_path.exists():
        verify_manifest(manifest_path,[f'{g["graph"]}.npy' for g in p['groups']])
        print('Probability cache already complete and verified.',flush=True);return
    for group in p['groups']:
        name=group['graph']; n=graphs[name]['n_nodes']; path=cache/f'{name}.npy'
        shape=(14400,len(group['scales']),1<<n)
        checkpoint=cache/f'{name}_completed.json'
        if path.exists()!=checkpoint.exists(): raise RuntimeError('Incomplete cache initialization; inspect before recovery')
        if path.exists():
            arrays[name]=np.lib.format.open_memmap(path,mode='r+')
            if arrays[name].shape!=shape: raise RuntimeError('Cache shape mismatch')
            saved=json.loads(checkpoint.read_text());done[name]=saved['ranges']
            if saved['protocol_sha256']!=sha(HERE/'protocol.json'): raise RuntimeError('Checkpoint protocol mismatch')
            for entry in done[name]:
                a,b,digest=entry
                if hashlib.sha256(arrays[name][a:b].tobytes()).hexdigest()!=digest: raise RuntimeError('Checkpoint chunk hash mismatch')
        else:
            arrays[name]=np.lib.format.open_memmap(path,mode='w+',dtype='float64',shape=shape)
            arrays[name][:]=np.nan;arrays[name].flush();done[name]=[]
            write_json(checkpoint,{'protocol_sha256':sha(HERE/'protocol.json'),'ranges':[]})
        complete={(x[0],x[1]) for x in done[name]}
        for a in range(0,14400,96):
            b=min(a+96,14400)
            if (a,b) not in complete: jobs.append((group,graphs[name]['edges'],a,b))
    total=len(jobs); count=0; last=0
    # Largest circuits first; chunk completion order cannot affect deterministic probabilities.
    jobs.sort(key=lambda j:graphs[j[0]['graph']]['n_nodes'],reverse=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending=[pool.submit(probability_chunk,j) for j in jobs]
        for future in as_completed(pending):
            name,a,b,values=future.result(); arrays[name][a:b]=values; arrays[name].flush()
            done[name].append([a,b,hashlib.sha256(values.tobytes()).hexdigest()])
            write_json(cache/f'{name}_completed.json',{'protocol_sha256':sha(HERE/'protocol.json'),'ranges':done[name]})
            count+=1; elapsed=time.perf_counter()-started
            if elapsed-last>=20 or count==total:
                progress={'stage':'probabilities','completed_chunks':count,'total_pending_chunks':total,'elapsed_seconds':elapsed,'completed_points':{k:sum(b-a for a,b,_ in v) for k,v in done.items()}}
                write_json(HERE/'progress.json',progress);print(json.dumps(progress),flush=True);last=elapsed
    for name,arr in arrays.items():
        if not np.isfinite(arr).all(): raise RuntimeError('Unfinished probabilities')
    write_json(manifest_path,{'protocol_sha256':sha(HERE/'protocol.json'),'scope':'exact deterministic probabilities, no shot samples','seconds_this_execution':time.perf_counter()-started,'sha256':{f'{g["graph"]}.npy':sha(cache/f'{g["graph"]}.npy') for g in p['groups']}})

def sample():
    p=protocol(); graphs=json.loads((HERE/'inputs/graphs.json').read_text())
    cache=HERE/'probabilities'
    verify_manifest(cache/'manifest.json',[f'{g["graph"]}.npy' for g in p['groups']])
    target=HERE/'samples';target.mkdir(exist_ok=True)
    expected=[f'{g["graph"]}{suffix}' for g in p['groups'] for suffix in ('.npz','_streams.json')]
    if (target/'manifest.json').exists():
        verify_manifest(target/'manifest.json',expected)
        print('Samples already complete and verified; no draws performed.',flush=True);return
    root_seed=np.random.SeedSequence(p['root_entropy_uint32'])
    groups=root_seed.spawn(len(p['groups']));t0=time.perf_counter();all_paths=[];initial_states=[]
    for group,group_seed in zip(p['groups'],groups):
        name=group['graph'];output=target/f'{name}.npz';meta_path=target/f'{name}_streams.json'
        pending=target/f'{name}.pending.npz'
        cells=[c for c in p['cells'] if c['graph']==name]
        n=p['repetitions']; count=1+sum(len(c['scales']) for c in cells)
        rep_seeds=group_seed.spawn(n)
        child_seeds=[s.spawn(count) for s in rep_seeds]
        group_paths=[list(s.spawn_key) for seeds in child_seeds for s in seeds]
        state_hashes=[hashlib.sha256(json.dumps(np.random.PCG64DXSM(s).state,sort_keys=True).encode()).hexdigest() for seeds in child_seeds for s in seeds]
        all_paths.extend(group_paths);initial_states.extend(state_hashes)
        if meta_path.exists():
            saved=json.loads(meta_path.read_text())
            assert saved['protocol_sha256']==sha(HERE/'protocol.json')
            assert saved['stream_spawn_keys']==group_paths and saved['initial_state_sha256']==state_hashes
            assert saved['root_entropy_uint32']==p['root_entropy_uint32'] and saved['repetitions']==n
            committed=output if output.exists() else pending
            if sha(committed)!=saved['sample_sha256']: raise RuntimeError('Committed sample hash mismatch')
            if not output.exists(): os.replace(pending,output)
            print(f'{name}: completed group verified and reused without resampling.',flush=True);continue
        if output.exists(): raise RuntimeError('Orphan output requires audit; no replacement permitted')
        from src.circuits import GraphInstance
        from src.phase_c_120_custom_kernel import cut_values_vector
        graph=GraphInstance(name,tuple(tuple(e) for e in graphs[name]['edges']))
        cost=-cut_values_vector(graph)
        probs=np.load(cache/f'{name}.npy',mmap_mode='r')
        B=group['budget'];baseline=np.empty((n,14400))
        estimates={c['order']:np.empty_like(baseline) for c in cells}
        for rep,seeds in enumerate(child_seeds):
            rngs=[np.random.Generator(np.random.PCG64DXSM(s)) for s in seeds]
            for idx in range(14400): baseline[rep,idx]=(rngs[0].multinomial(B,probs[idx,0])@cost)/B
            offset=1
            for c in cells:
                out=np.zeros(14400)
                for j,(weight,shots) in enumerate(zip(c['weights'],c['allocation'])):
                    rng=rngs[offset+j]; scale_index=group['scales'].index(c['scales'][j])
                    for idx in range(14400): out[idx]+=weight*(rng.multinomial(shots,probs[idx,scale_index])@cost)/shots
                estimates[c['order']][rep]=out;offset+=len(c['scales'])
            if (rep+1)%20==0:
                progress={'stage':'sampling','graph':name,'completed_repetitions':rep+1,'planned_repetitions':n,'elapsed_seconds':time.perf_counter()-t0}
                write_json(HERE/'progress.json',progress);print(json.dumps(progress),flush=True)
        # A crashed, uncommitted group is reconstructed from exactly the same seeds.
        # Commit metadata precedes the atomic final rename, permitting recovery.
        np.savez_compressed(pending,baseline=baseline,**estimates)
        write_json(meta_path,{'protocol_sha256':sha(HERE/'protocol.json'),'graph':name,'root_entropy_uint32':p['root_entropy_uint32'],'group_spawn_key':list(group_seed.spawn_key),'stream_spawn_keys':group_paths,'initial_state_sha256':state_hashes,'bit_generator':'PCG64DXSM','numpy':np.__version__,'repetitions':n,'baseline_shared_across_orders':len(cells)>1,'sample_sha256':sha(pending)})
        os.replace(pending,output)
    assert len(all_paths)==len(set(map(tuple,all_paths)))
    assert len(initial_states)==len(set(initial_states))
    write_json(target/'manifest.json',{'protocol_sha256':sha(HERE/'protocol.json'),'elapsed_seconds_this_execution':time.perf_counter()-t0,'unique_stream_paths':len(all_paths),'unique_initial_states':len(set(initial_states)),'sha256':{name:sha(target/name) for name in expected}})

def binomial_interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0., float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]

def analyze():
    import csv
    p=protocol(); target=HERE/'analysis'
    if target.exists(): raise RuntimeError('Analysis output exists')
    verify_manifest(HERE/'samples/manifest.json',[f'{g["graph"]}{suffix}' for g in p['groups'] for suffix in ('.npz','_streams.json')])
    rows=[];observations=[]
    for c in p['cells']:
        ideal=np.load(HERE/'inputs'/f'{c["graph"]}_ideal.npy').reshape(-1)
        data=np.load(HERE/'samples'/f'{c["graph"]}.npz')
        noisy=data['baseline'];zne=data[c['order']];lo=ideal.min();D=float(ideal.max()-lo)
        assert noisy.shape==zne.shape==(p['repetitions'],14400)
        assert np.isfinite(noisy).all() and np.isfinite(zne).all() and np.isfinite(ideal).all()
        mseN=np.mean((noisy-ideal)**2,axis=1);mseZ=np.mean((zne-ideal)**2,axis=1)
        assert np.isfinite(mseN).all() and (mseN>0).all() and D>0
        r=(mseN-mseZ)/mseN
        iN=np.argmin(noisy,axis=1);iZ=np.argmin(zne,axis=1)
        RN=(ideal[iN]-lo)/D;RZ=(ideal[iZ]-lo)/D;d=RZ-RN
        n=len(r);assert n==p['repetitions'];kr=int((r>0).sum());kd=int((d>p['regret_tolerance']).sum())
        pr=float(binom.sf(kr-1,n,.5));pd=float(binom.sf(kd-1,n,.5));joint=int(((r>0)&(d>p['regret_tolerance'])).sum())
        rows.append({'id':c['id'],'graph':c['graph'],'noise':c['noise'],'budget':c['budget'],'order':c['order'],'n':n,'positive_mse':kr,'positive_regret':kd,'joint_positive':joint,'p_mse':pr,'p_regret':pd,'p_iut':max(pr,pd),'median_r_mse':float(np.median(r)),'median_d_R':float(np.median(d)),'q25_d_R':float(np.quantile(d,.25)),'q75_d_R':float(np.quantile(d,.75)),'native_median_d':float(np.median(d)*D),'above_descriptive_mse_10pct':bool(np.median(r)>=.10),'above_original_strong_regret':bool(np.median(d)>=p['legacy_strong_regret']),'pi_mse_95pct_CP_descriptive':binomial_interval(kr,n),'pi_regret_95pct_CP_descriptive':binomial_interval(kd,n)})
        for rep in range(n): observations.append({'id':c['id'],'rep':rep,'mse_noisy':mseN[rep],'mse_zne':mseZ[rep],'r_mse':r[rep],'regret_noisy':RN[rep],'regret_zne':RZ[rep],'d_R':d[rep],'argmin_noisy':iN[rep],'argmin_zne':iZ[rep]})
    order=sorted(range(4),key=lambda i:rows[i]['p_iut']); running=0.
    for rank,i in enumerate(order):
        running=max(running,(4-rank)*rows[i]['p_iut']); rows[i]['p_holm']=min(1.,running);rows[i]['confirmed']=rows[i]['p_holm']<=p['alpha']
    target.mkdir()
    with (target/'per_repetition.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(observations[0]));writer.writeheader();writer.writerows(observations)
    write_json(target/'results.json',{'scope':'targeted prospective family of four selected historical conditions; not reconstruction of 540-condition search','protocol_sha256':sha(HERE/'protocol.json'),'n_confirmed':sum(r['confirmed'] for r in rows),'cells':rows,'all_four_confirmed':all(r['confirmed'] for r in rows),'interval_note':'Individual 95% Clopper-Pearson intervals are descriptive; Holm controls the four conjunction tests under the independent-repetition PRNG model.'})
    print(json.dumps({'confirmed':sum(r['confirmed'] for r in rows),'cells':rows},indent=2),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--probabilities',action='store_true');ap.add_argument('--sample',action='store_true');ap.add_argument('--analyze',action='store_true');ap.add_argument('--workers',type=int,default=6);a=ap.parse_args()
    if a.probabilities:build_probabilities(a.workers)
    if a.sample:sample()
    if a.analyze:analyze()
