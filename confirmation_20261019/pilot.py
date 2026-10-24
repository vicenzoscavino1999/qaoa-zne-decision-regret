"""Timing/implementation pilot only: no landscape endpoints or significance tests."""
from pathlib import Path
import sys, site, json, time, hashlib
ROOT=Path(__file__).resolve().parents[1]
site.addsitedir(str(ROOT/'.venv/Lib/site-packages'))
sys.path.insert(0,str(ROOT))
import numpy as np
from src.circuits import GraphInstance
from src.phase_c_120_custom_kernel import kernel_probabilities,aer_density_probabilities,cut_values_vector
from src.landscapes import parameter_grid

HERE=Path(__file__).resolve().parent
def run():
    target=HERE/'pilot_results.json'
    if target.exists(): raise RuntimeError('Pilot output already exists')
    graphs=json.loads((ROOT/'results/phase_d/planning/phase_d_selected_instances_final.json').read_text())
    gamma,beta=parameter_grid(120)
    indices=[0,167,1249,3317,5611,8017,11119,14399]
    result={'scope':'timing and pointwise validation only; no selected-landscape endpoints', 'points':indices,'groups':[]}
    for name,p,scales,B in [('H1',.02,[1,3,5],1024),('H3',.02,[1,3,5],1024),('H8',.01,[1,3,5,7],4096)]:
        graph=GraphInstance(name,tuple(tuple(e) for e in graphs[name]['edges']))
        elapsed=[]; max_error=0.; probs=None
        for idx in indices:
            for scale in scales:
                t=time.perf_counter()
                probs=kernel_probabilities(graph,gamma[idx//120],beta[idx%120],p,scale).probabilities
                elapsed.append(time.perf_counter()-t)
                assert np.isfinite(probs).all() and probs.min()>=0 and abs(probs.sum()-1)<1e-9
                if idx==167:
                    other=aer_density_probabilities(graph,gamma[idx//120],beta[idx%120],p,scale).probabilities
                    error=float(np.max(abs(probs-other))); max_error=max(max_error,error)
                    assert error<1e-8
        # Pilot streams have a distinct namespace. Sample timing discards outcomes.
        entropy=list(hashlib.sha256(('pilot_20261019_'+name).encode()).digest())
        rng=np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entropy)))
        t=time.perf_counter()
        for _ in range(10000): rng.multinomial(B,probs)
        sample_seconds=time.perf_counter()-t
        row={'graph':name,'n_qubits':graph.n_qubits,'scales':scales,'probability_calls':len(elapsed),'seconds':sum(elapsed),'mean_seconds_per_probability':float(np.mean(elapsed)),'projected_serial_probability_seconds':float(np.mean(elapsed)*14400*len(scales)),'sample_seconds_per_10000_calls':sample_seconds,'aer_max_probability_error':max_error}
        result['groups'].append(row);print(json.dumps(row),flush=True)
    target.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__': run()
