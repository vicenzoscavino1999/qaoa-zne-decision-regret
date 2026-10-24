"""Independent, read-only audit after BOTH final completion files exist.

Preparation: python -B audit_completed_confirmation.py --self-test
Only after the coordinator's completion signal:
    python -B audit_completed_confirmation.py --audit
Read-only verification of the completed or relocated package:
    python -B audit_completed_confirmation.py --verify-only

Does not import run_confirmation or call any campaign entry point. The replay
uses scheduled frozen RNG states, never fresh entropy. All output is confined
to review_code/. Original files are read only.
"""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import sys,site,argparse,json,hashlib,math,statistics,csv,time,platform
from datetime import datetime,timezone
from importlib.metadata import version
OUT=Path(__file__).resolve().parent
CONF=OUT.parent
ROOT=CONF.parent
sys.dont_write_bytecode=True
site.addsitedir(str(ROOT/'.venv/Lib/site-packages'))
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import betaincinv

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def require(condition,message):
    if not condition: raise AssertionError(message)

def exact_tail(k,n):
    require(0<=k<=n,'Invalid positive count')
    return sum(math.comb(n,j) for j in range(k,n+1))/2**n

def holm_manual(values):
    m=len(values);ordered=sorted(range(m),key=lambda i:(values[i],i))
    adjusted=[0.]*m
    for rank,i in enumerate(ordered):
        adjusted[i]=min(1.,max((m-j)*values[ordered[j]] for j in range(rank+1)))
    return adjusted

def cp_interval(k,n):
    return [float(betaincinv(k,n-k+1,.025)) if k else 0.,
            float(betaincinv(k+1,n-k,.975)) if k<n else 1.]

def quantile_linear(values,q):
    ordered=sorted(float(v) for v in values);pos=(len(ordered)-1)*q
    lo=math.floor(pos);hi=math.ceil(pos);weight=pos-lo
    return ordered[lo]+weight*(ordered[hi]-ordered[lo])

def scalar_metrics(observed,ideal):
    require(len(observed)==len(ideal)>0,'Metric dimensions')
    # fsum is intentionally independent of the campaign's vectorized np.mean.
    mse=math.fsum((float(a)-float(b))**2 for a,b in zip(observed,ideal))/len(ideal)
    selected=min(range(len(observed)),key=observed.__getitem__)
    span=float(max(ideal)-min(ideal))
    require(span>0,'Constant ideal reference')
    regret=(float(ideal[selected])-float(min(ideal)))/span
    return mse,regret,selected

def close(got,expected,label,*,rtol=1e-11,atol=1e-13):
    require(math.isfinite(float(got)) and math.isfinite(float(expected)),f'Nonfinite {label}')
    require(math.isclose(float(got),float(expected),rel_tol=rtol,abs_tol=atol),
            f'{label}: got {got}, expected {expected}')

def check_manifest(path,expected_names,protocol_sha):
    manifest=load_json(path)
    require(manifest['protocol_sha256']==protocol_sha,f'Wrong protocol: {path}')
    require(set(manifest['sha256'])==set(expected_names),f'Wrong file set: {path}')
    for name,sha in manifest['sha256'].items():
        require(Path(name).name==name,f'Nonlocal manifest path {name}')
        require(digest(Path(path).parent/name)==sha,f'Wrong hash {name}')
    return manifest

def cost_from_edges(record):
    # Construct computational costs independently, without importing cut_values_vector.
    n=record['n_nodes']
    return np.array([-math.fsum(float(w) for a,b,w in record['edges']
                                if ((state>>int(a))&1)!=((state>>int(b))&1))
                     for state in range(1<<n)],dtype=float)

def audit_streams_and_replay(p,graphs,caches,samples,protocol_sha):
    """Replay the first THREE draws of every scheduled stream, 1600 streams.

    Individual scale counts were not saved, so compare weighted combinations
    to ZNE landscapes and direct draws to baseline landscapes at points0,1,2.
    This verifies the stream map and those saved values, not every sample.
    """
    all_paths=[];all_states=[];reports=[];counts_hash=hashlib.sha256()
    first_points=3
    for gi,group in enumerate(p['groups']):
        name=group['graph'];cells=[c for c in p['cells'] if c['graph']==name]
        metadata=load_json(CONF/'samples'/f'{name}_streams.json')
        require(metadata['protocol_sha256']==protocol_sha,'Stream metadata protocol')
        require(metadata['root_entropy_uint32']==p['root_entropy_uint32'],'Stream root entropy')
        require(metadata['group_spawn_key']==[gi],'Group spawn path')
        require(metadata['numpy']==p['versions']['numpy'],'Stream NumPy version')
        require(metadata['bit_generator']=='PCG64DXSM','Wrong bit generator')
        require(metadata['repetitions']==p['repetitions'],'Wrong stream repetition count')
        require(metadata['baseline_shared_across_orders']==(len(cells)>1),'Baseline metadata')
        require(metadata['sample_sha256']==digest(CONF/'samples'/f'{name}.npz'),'NPZ/metadata linkage')
        group_paths=[];group_states=[];cost=cost_from_edges(graphs[name]);prob=caches[name]
        expected_baseline=np.empty((p['repetitions'],first_points))
        expected_zne={c['order']:np.zeros_like(expected_baseline) for c in cells}
        count=1+sum(len(c['scales']) for c in cells)
        for rep in range(p['repetitions']):
            generators=[]
            for stream in range(count):
                path=(gi,rep,stream)
                rng=np.random.Generator(np.random.PCG64DXSM(
                    np.random.SeedSequence(p['root_entropy_uint32'],spawn_key=path)))
                group_paths.append(list(path));all_paths.append(path)
                state=hashlib.sha256(json.dumps(rng.bit_generator.state,sort_keys=True).encode()).hexdigest()
                group_states.append(state);all_states.append(state);generators.append(rng)
            for idx in range(first_points):
                counts=generators[0].multinomial(group['budget'],prob[idx,0])
                counts_hash.update(np.asarray(counts,dtype='<i8').tobytes())
                expected_baseline[rep,idx]=(counts@cost)/group['budget']
            offset=1
            for cell in cells:
                for j,(weight,shots) in enumerate(zip(cell['weights'],cell['allocation'])):
                    scale_index=group['scales'].index(cell['scales'][j])
                    for idx in range(first_points):
                        counts=generators[offset+j].multinomial(shots,prob[idx,scale_index])
                        counts_hash.update(np.asarray(counts,dtype='<i8').tobytes())
                        expected_zne[cell['order']][rep,idx]+=weight*(counts@cost)/shots
                offset+=len(cell['scales'])
            require(offset==count,'Stream offset final value')
        require(metadata['stream_spawn_keys']==group_paths,'Stored paths do not match direct identities')
        require(metadata['initial_state_sha256']==group_states,'Initial states do not match direct identities')
        np.testing.assert_array_equal(samples[name]['baseline'][:,:first_points],expected_baseline)
        for cell in cells:
            np.testing.assert_array_equal(samples[name][cell['order']][:,:first_points],expected_zne[cell['order']])
        reports.append({'graph':name,'streams':len(group_paths),'replayed_grid_indices':list(range(first_points)),
                        'baseline_and_weighted_ZNE_exactly_equal':True})
    require(len(all_paths)==len(set(all_paths))==1600,'Duplicate/wrong number of spawn paths')
    require(len(all_states)==len(set(all_states))==1600,'Duplicate/wrong number of initial states')
    return {'unique_paths':len(all_paths),'unique_initial_states':len(all_states),
            'replayed_multinomial_calls':1600*first_points,'counts_sha256':counts_hash.hexdigest(),'groups':reports}

def aer_cache_checks(p,graphs,caches):
    # Six fixed checks, chosen here before any new endpoint is read.
    from src.circuits import GraphInstance
    from src.phase_c_120_custom_kernel import aer_density_probabilities
    records=[]
    for group in p['groups']:
        name=group['graph'];record=graphs[name]
        graph=GraphInstance(name,tuple(tuple(e) for e in record['edges']))
        cost=cost_from_edges(record)
        for idx,scale in ((167,1),(8017,max(group['scales']))):
            gamma=(2.*math.pi/120)*(idx//120);beta=(math.pi/120)*(idx%120)
            reference=aer_density_probabilities(graph,gamma,beta,group['noise'],scale).probabilities
            cached=caches[name][idx,group['scales'].index(scale)]
            errors=[float(np.max(np.abs(reference-cached))),float(abs((reference-cached)@cost)),
                    float(abs((reference-cached)@(cost*cost)))]
            require(errors[0]<1e-10 and max(errors[1:])<1e-9,f'Aer cache mismatch {name} {idx} {scale}')
            records.append({'graph':name,'theta_index':idx,'scale':scale,'errors_probability_mean_second_moment':errors})
    return records

def audit_completed(*,verify_only=False):
    # Completion gate MUST precede reading any campaign endpoints or caches.
    if not (CONF/'analysis/results.json').is_file() or not (CONF/'samples/manifest.json').is_file():
        raise SystemExit('Not complete: audit refused; no campaign endpoints or caches were read.')
    started=time.perf_counter()
    p=load_json(CONF/'protocol.json');protocol_sha=digest(CONF/'protocol.json')
    lock=load_json(CONF/'protocol_lock.json')
    require(p['repetitions']==100 and len(p['cells'])==4,'Wrong fixed family')
    for rel,sha in lock['sha256'].items(): require(digest(ROOT/rel)==sha,f'Frozen file changed: {rel}')
    runtime={'python':platform.python_version(),'numpy':np.__version__,'scipy':version('scipy'),
             'qiskit':version('qiskit'),'qiskit_aer':version('qiskit-aer')}
    require(runtime==p['versions'],'Frozen runtime mismatch')
    names=[g['graph'] for g in p['groups']]
    require(names==['H1','H3','H8'],'Wrong groups/order')
    cache_manifest=check_manifest(CONF/'probabilities/manifest.json',[name+'.npy' for name in names],protocol_sha)
    sample_manifest=check_manifest(CONF/'samples/manifest.json',
            [name+suffix for name in names for suffix in ('.npz','_streams.json')],protocol_sha)
    require(not list((CONF/'samples').glob('*.pending.npz')),'Uncommitted pending files remain')
    graphs=load_json(CONF/'inputs/graphs.json');caches={};samples={};ideals={}
    for group in p['groups']:
        name=group['graph'];dim=1<<graphs[name]['n_nodes']
        cache=np.load(CONF/'probabilities'/f'{name}.npy',mmap_mode='r',allow_pickle=False)
        require(cache.shape==(14400,len(group['scales']),dim) and cache.dtype==np.float64,'Cache dimensions/type')
        require(np.isfinite(cache).all() and cache.min()>=0,'Invalid cache probabilities')
        require(float(np.max(abs(cache.sum(axis=2)-1)))<1e-12,'Cache normalization')
        caches[name]=cache
        expected_keys={'baseline'}|{c['order'] for c in p['cells'] if c['graph']==name}
        with np.load(CONF/'samples'/f'{name}.npz',allow_pickle=False) as data:
            require(set(data.files)==expected_keys,'Unexpected NPZ keys')
            samples[name]={key:data[key] for key in data.files}
        for data in samples[name].values():
            require(data.shape==(100,14400) and data.dtype==np.float64 and np.isfinite(data).all(),'Invalid NPZ landscape')
        ideal=np.load(CONF/'inputs'/f'{name}_ideal.npy',allow_pickle=False)
        require(ideal.shape==(120,120) and np.isfinite(ideal).all() and np.ptp(ideal)>0,'Invalid ideal grid')
        ideals[name]=ideal.ravel(order='C')
    streams=audit_streams_and_replay(p,graphs,caches,samples,protocol_sha)
    require(sample_manifest['unique_stream_paths']==streams['unique_paths'],'Manifest path count')
    require(sample_manifest['unique_initial_states']==streams['unique_initial_states'],'Manifest state count')
    summary=load_json(CONF/'analysis/results.json')
    require(summary['protocol_sha256']==protocol_sha,'Analysis protocol mismatch')
    with (CONF/'analysis/per_repetition.csv').open(newline='',encoding='utf-8') as f:
        archived_observations=list(csv.DictReader(f))
    require(len(archived_observations)==400,'Expected 400 observations')
    observed_by_key={(r['id'],int(r['rep'])):r for r in archived_observations}
    require(len(observed_by_key)==400,'Duplicated observations')
    archived_cells={r['id']:r for r in summary['cells']}
    require(set(archived_cells)=={c['id'] for c in p['cells']},'Wrong analysis cells')
    rows=[];observations=[];max_metric_abs_error=0.
    baseline_metrics={name:[scalar_metrics(row,ideals[name]) for row in samples[name]['baseline']] for name in names}
    for cell in p['cells']:
        name=cell['graph'];ideal=ideals[name];r_values=[];d_values=[]
        for rep,z in enumerate(samples[name][cell['order']]):
            mseN,RN,iN=baseline_metrics[name][rep];mseZ,RZ,iZ=scalar_metrics(z,ideal)
            require(mseN>0,'MSE_N denominator')
            r=(mseN-mseZ)/mseN;d=RZ-RN;r_values.append(r);d_values.append(d)
            observation=dict(id=cell['id'],rep=rep,mse_noisy=mseN,mse_zne=mseZ,r_mse=r,
                             regret_noisy=RN,regret_zne=RZ,d_R=d,argmin_noisy=iN,argmin_zne=iZ)
            old=observed_by_key[(cell['id'],rep)]
            for key,value in observation.items():
                if key=='id':require(old[key]==value,'Observation cell id')
                elif key in ('rep','argmin_noisy','argmin_zne'):require(int(old[key])==value,f'Index mismatch {key}')
                else:
                    max_metric_abs_error=max(max_metric_abs_error,abs(value-float(old[key])))
                    close(value,float(old[key]),f'{cell["id"]},rep{rep},{key}')
            observations.append(observation)
        kr=sum(v>0 for v in r_values);kd=sum(v>p['regret_tolerance'] for v in d_values)
        row={'id':cell['id'],'graph':name,'noise':cell['noise'],'budget':cell['budget'],'order':cell['order'],
             'n':100,'positive_mse':kr,'positive_regret':kd,
             'joint_positive':sum(r>0 and d>p['regret_tolerance'] for r,d in zip(r_values,d_values)),
             'p_mse':exact_tail(kr,100),'p_regret':exact_tail(kd,100),
             'median_r_mse':statistics.median(r_values),'median_d_R':statistics.median(d_values),
             'q25_d_R':quantile_linear(d_values,.25),'q75_d_R':quantile_linear(d_values,.75),
             'native_median_d':statistics.median(d_values)*float(np.ptp(ideal)),
             'above_descriptive_mse_10pct':statistics.median(r_values)>=.1,
             'above_original_strong_regret':statistics.median(d_values)>=p['legacy_strong_regret'],
             'pi_mse_95pct_CP_descriptive':cp_interval(kr,100),
             'pi_regret_95pct_CP_descriptive':cp_interval(kd,100)}
        row['p_iut']=max(row['p_mse'],row['p_regret']);rows.append(row)
    for row,adjusted in zip(rows,holm_manual([r['p_iut'] for r in rows])):
        row['p_holm']=adjusted;row['confirmed']=adjusted<=p['alpha'];old=archived_cells[row['id']]
        for key,value in row.items():
            if isinstance(value,(str,bool,int)):require(old[key]==value,f'Cell categorical/count mismatch {row["id"]},{key}')
            elif isinstance(value,list):
                for a,b in zip(value,old[key]):close(a,b,f'CP interval {key}',rtol=1e-10,atol=1e-12)
            else:close(value,old[key],f'Cell summary {row["id"]},{key}',rtol=1e-11,atol=0 if key.startswith('p_') else 1e-13)
    require(sum(r['confirmed'] for r in rows)==summary['n_confirmed'],'Confirmation total')
    require(all(r['confirmed'] for r in rows)==summary['all_four_confirmed'],'All-four flag')
    aer=aer_cache_checks(p,graphs,caches)
    result={'status':'PASS','audited_at_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.perf_counter()-started,
            'auditor_sha256':digest(__file__),'protocol_sha256':protocol_sha,'versions':runtime,
            'frozen_files_verified':len(lock['sha256']),'observations_recomputed':len(observations),
            'max_abs_metric_difference_fsum_vs_campaign':max_metric_abs_error,'streams':streams,
            'aer_cache_checks':aer,'n_confirmed':sum(r['confirmed'] for r in rows),'cells':rows,
            'limits':['Arithmetic and partial deterministic replay verified; not every draw is replayed.',
                      'Six fixed Aer points do not certify every cached probability.',
                      'PRNG independence is a statistical modeling assumption, not a mathematical guarantee.',
                      'Results concern four historically selected targets with fresh samples, not 540-cell prevalence.']}
    if verify_only:
        print(json.dumps({'status':result['status'],'mode':'verify-only; no files written',
                'protocol_sha256':protocol_sha,'frozen_files_verified':result['frozen_files_verified'],
                'observations_recomputed':len(observations),'unique_streams':streams['unique_paths'],
                'replayed_draws':streams['replayed_multinomial_calls'],'aer_cases':len(aer),
                'n_confirmed':result['n_confirmed']},indent=2))
    else:
        with (OUT/'independent_per_repetition.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(observations[0]));writer.writeheader();writer.writerows(observations)
        (OUT/'independent_audit_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result,indent=2))

def self_test():
    # Deliberately reads no campaign file, including no completion flags.
    require(exact_tail(0,100)==1. and exact_tail(100,100)==2.**-100,'Exact tail extremes')
    require(exact_tail(1,4)==15/16 and exact_tail(3,4)==5/16,'Exact tail known values')
    require(holm_manual([.001,.03,.02,.9])==[.004,.06,.06,.9],'Holm vector')
    for k in range(101):
        lo,hi=cp_interval(k,100);require(0<=lo<=k/100<=hi<=1,'CP bounds')
    mse,regret,idx=scalar_metrics([.6,.4],[0.,1.])
    close(mse,.36,'Synthetic MSE');require(regret==1. and idx==1,'Synthetic regret')
    require(scalar_metrics([1.,1.,3.],[0.,0.,2.])[2]==0,'Tie first index')
    require(quantile_linear(range(100),.25)==24.75,'Quantile')
    root=[1,2,3,4,5,6,7,8]
    spawned=np.random.SeedSequence(root).spawn(3)[2].spawn(100)[23].spawn(8)[5]
    direct=np.random.SeedSequence(root,spawn_key=(2,23,5))
    a=np.random.Generator(np.random.PCG64DXSM(spawned));b=np.random.Generator(np.random.PCG64DXSM(direct))
    np.testing.assert_array_equal(a.multinomial(54,[.2,.3,.5],size=3),b.multinomial(54,[.2,.3,.5],size=3))
    require(cost_from_edges({'n_nodes':2,'edges':[[0,1,2.5]]}).tolist()==[0.,-2.5,-2.5,0.],'Independent cut costs')
    parser=build_parser()
    for flag,attribute in (('--self-test','self_test'),('--audit','audit'),('--verify-only','verify_only')):
        parsed=parser.parse_args([flag])
        require(getattr(parsed,attribute) is True,'CLI mode mapping')
        require(sum(bool(getattr(parsed,a)) for a in ('self_test','audit','verify_only'))==1,'CLI exclusive modes')
    result={'status':'PASS','scope':'Synthetic helpers only; no campaign endpoints/probabilities read',
            'auditor_sha256':digest(__file__),'future_aer_cases':{'graphs':['H1','H3','H8'],
                'pairs_theta_scale':[[167,1],[8017,'maximum scheduled scale']]},
            'parser_modes_verified':['--self-test','--audit','--verify-only']}
    (OUT/'audit_preparation_self_test.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

def build_parser():
    parser=argparse.ArgumentParser()
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--self-test',action='store_true');modes.add_argument('--audit',action='store_true')
    modes.add_argument('--verify-only',action='store_true',help='Run full completed-package audit without writing any file')
    return parser

if __name__=='__main__':
    args=build_parser().parse_args()
    if args.self_test:self_test()
    else:audit_completed(verify_only=args.verify_only)
