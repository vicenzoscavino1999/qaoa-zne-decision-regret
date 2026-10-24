"""Limited statistical interpretation check; does not read raw landscapes."""
from pathlib import Path
from fractions import Fraction as F
from math import comb,isclose
from statistics import median
from collections import defaultdict
import csv,json

ROOT=Path(__file__).resolve().parents[1]
results=json.loads((ROOT/'analysis/results.json').read_text())
protocol=json.loads((ROOT/'protocol.json').read_text())
data=list(csv.DictReader((ROOT/'analysis/per_repetition.csv').open(newline='')))
grouped=defaultdict(list)
for row in data: grouped[row['id']].append(row)
eps=protocol['regret_tolerance']
out=[]
for c in results['cells']:
    rows=grouped[c['id']]
    assert len(rows)==100 and {int(r['rep']) for r in rows}==set(range(100))
    r=[float(x['r_mse']) for x in rows]
    d=[float(x['d_R']) for x in rows]
    kr=sum(x>0 for x in r); kd=sum(x>eps for x in d)
    kj=sum(x>0 and y>eps for x,y in zip(r,d))
    pr=sum(F(comb(100,k),2**100) for k in range(kr,101))
    pd=sum(F(comb(100,k),2**100) for k in range(kd,101))
    assert (kr,kd,kj)==(c['positive_mse'],c['positive_regret'],c['joint_positive'])
    assert isclose(float(pr),c['p_mse'],rel_tol=1e-13)
    assert isclose(float(pd),c['p_regret'],rel_tol=1e-13)
    assert isclose(median(r),c['median_r_mse'],rel_tol=1e-13)
    assert isclose(median(d),c['median_d_R'],rel_tol=1e-13)
    out.append(dict(id=c['id'],kr=kr,kd=kd,joint=kj,p_exact=max(pr,pd),
                    median_r=median(r),median_d=median(d),
                    native_median=c['native_median_d'],above_10pct=median(r)>=.1,
                    above_strong=median(d)>=protocol['legacy_strong_regret']))
prev=F(0)
for rank,c in enumerate(sorted(out,key=lambda c:c['p_exact'])):
    prev=min(F(1),max(prev,(4-rank)*c['p_exact']))
    declared=next(x for x in results['cells'] if x['id']==c['id'])
    assert isclose(float(prev),declared['p_holm'],rel_tol=1e-13)
    assert (prev<=F(1,20))==declared['confirmed']
    c['p_exact']=str(c['p_exact'])
    c['Holm_exact']=str(prev)
    c['Holm_float']=float(prev)

q={int(x['rep']):x for x in grouped['H8_q']}
c={int(x['rep']):x for x in grouped['H8_c']}
for rep in range(100):
    for name in ('mse_noisy','regret_noisy','argmin_noisy'):
        assert q[rep][name]==c[rep][name]

summary=dict(status='PASS',scope='Endpoint CSV statistics and result interpretation only; raw audit delegated separately',
    cells=out,n_confirmed=sum(x['confirmed'] for x in results['cells']),
    H8_baseline_match_all_repetitions=True,
    median_MSE_improvement_percent_range=[100*min(x['median_r'] for x in out),100*max(x['median_r'] for x in out)],
    median_regret_range=[min(x['median_d'] for x in out),max(x['median_d'] for x in out)],
    median_native_regret_range=[min(x['native_median'] for x in out),max(x['native_median'] for x in out)],
    n_above_10pct=sum(x['above_10pct'] for x in out),n_above_strong=sum(x['above_strong'] for x in out))
(ROOT/'review_protocol/results_interpretation_check.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
