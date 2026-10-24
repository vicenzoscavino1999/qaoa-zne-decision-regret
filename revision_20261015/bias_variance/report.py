"""Create standalone figures and LaTeX note from control.py outputs."""
import sys,site,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parent
site.addsitedir(str(ROOT/'.venv/Lib/site-packages'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
data=json.loads((OUT/'results.json').read_text()); rows=data['rows']
names=['G1_C4','G2_weighted4','G3_weighted6']; methods=['noisy','linear','quadratic','cubic']
plt.rcParams.update({'font.size':10, 'axes.titlesize':10, 'axes.labelsize':10,
                     'xtick.labelsize':9.5, 'ytick.labelsize':9.5,
                     'legend.fontsize':9.5, 'axes.linewidth':.7})
fig,axs=plt.subplots(3,2,figsize=(5.2,6),layout='constrained')
fig2,axs2=plt.subplots(3,2,figsize=(5.2,6),layout='constrained')
display_names=['G1 (C4)','G2 (weighted4)','G3 (weighted6)']
for j,p in enumerate((.001,.02)):
    for i,name in enumerate(names):
        r=[next(r for r in rows if r['graph']==name and r['p']==p and r['method']==m) for m in methods]
        x=np.arange(4); b=np.array([z['bias2'] for z in r]); v=np.array([z['variance'] for z in r]); ax=axs[i,j]
        ax.bar(x,b,label='Squared bias',color='#245980'); ax.bar(x,v,bottom=b,label='Sampling variance',color='#d58d38')
        ax.errorbar(x,[z['mc_mse'] for z in r],yerr=[1.96*z['mc_mse_se'] for z in r],fmt='k.',label='MC MSE ± 1.96 SE',capsize=2.5,elinewidth=.9,markersize=3.5)
        ax.set(title=f'{display_names[i]}, p={p:g}',xticks=x,xticklabels=methods,ylabel='Mean squared error' if j==0 else '')
        ax.set_ylim(bottom=0)
        ax.tick_params(axis='x',labelrotation=32,pad=1)
        for label in ax.get_xticklabels(): label.set_ha('right')
        ax=axs2[i,j]; ax.bar(x-.18,[z['mean_landscape_regret'] for z in r],width=.36,label='Exact mean',color='#245980')
        ax.bar(x+.18,[z['mc_regret'] for z in r],width=.36,label='Finite shots ± 1.96 SE',color='#d58d38',yerr=[1.96*z['mc_regret_se'] for z in r],capsize=2.5,error_kw={'elinewidth':.9})
        ax.set(title=f'{display_names[i]}, p={p:g}',xticks=x,xticklabels=methods,ylabel='Normalized ideal regret' if j==0 else '')
        ax.tick_params(axis='x',labelrotation=32,pad=1)
        for label in ax.get_xticklabels(): label.set_ha('right')
# Common decision scale prevents floating-point differences between equivalent
# G1 minima (about 1e-16) from looking like scientific effects.
regret_top=max(r['mc_regret']+1.96*r['mc_regret_se'] for r in rows)*1.12
for ax in axs2.flat:
    ax.set_ylim(0,regret_top)
for ax in axs2[0,:]:
    ax.text(.5,.60,'All regrets < 1e-14\n(floating-point scale)',ha='center',transform=ax.transAxes,fontsize=9.5)
for f,a,title in ((fig,axs,'Post hoc bias and variance control\n12 × 12 grid; B=1024 per point; 500 replicates'),(fig2,axs2,'Post hoc decision diagnostic\n12 × 12 grid; B=1024 per point; 500 replicates')):
    handles,labels=a[0,0].get_legend_handles_labels()
    f.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,
             columnspacing=1,handletextpad=.5,handlelength=1.4,labelspacing=.4)
    f.suptitle(title,fontsize=10)
for f,name in ((fig,'bias_variance'),(fig2,'decision_regret')):
    f.savefig(OUT/(name+'.pdf')); f.savefig(OUT/(name+'.png'),dpi=180)
lines=[]
for r in rows:
    if r['method']=='noisy': continue
    n=next(z for z in rows if z['graph']==r['graph'] and z['p']==r['p'] and z['method']=='noisy')
    lines.append(f"{r['graph']} p={r['p']:g} {r['method']}: bias2 ratio={r['bias2']/n['bias2']:.4g}; variance ratio={r['variance']/n['variance']:.4g}; MSE ratio={r['expected_mse']/n['expected_mse']:.4g}; mean regret {r['mean_landscape_regret']:.5g}; MC regret {r['mc_regret']:.5g} versus {n['mc_regret']:.5g}")
(OUT/'findings.txt').write_text('\n'.join(lines))
table='\n'.join(f"{r['graph'].replace('_',chr(92)+'_')} & {r['p']:g} & {r['method']} & {r['bias2']:.5g} & {r['variance']:.5g} & {r['expected_mse']:.5g} & {r['mean_landscape_regret']:.4g} & {r['mc_regret']:.4g} \\\\" for r in rows)
tex=r'''\documentclass{article}
\usepackage[margin=1.6cm]{geometry}
\usepackage{amsmath,graphicx,booktabs}
\begin{document}
\section*{Post hoc bounded bias--variance control}
All three known Phase C graphs were retained for structural coverage, with no screening on favorable outcomes. The two endpoint noise rates $p=0.001,0.02$, all three Richardson orders, and the noisy baseline were evaluated at $B=1024$ shots per grid point. This new $12\times12$ finite grid is not the archived $120\times120$ campaign; weighted gamma is treated as a prescribed finite domain, without asserting periodicity. Each configuration has 500 independent multinomial landscape replicates, with independent points, scales and methods. No campaign-level endpoint is retested here.

For objective $C=-\mathrm{cut}$ and scale $j$, the exact density distribution gives $\mu_j=E[C_j]$ and $v_j=E[C_j^2]-\mu_j^2$. Integer shot allocation is the existing largest-remainder allocation proportional to absolute Richardson weights. Thus
\[b(\theta)=\sum_jw_j\mu_j(\theta)-C_0(\theta),\qquad V(\theta)=\sum_j\frac{w_j^2v_j(\theta)}{n_j},\qquad E[(\widehat C-C_0)^2]=b^2+V.\]
We average these quantities over the grid. Regret is measured at the selected grid minimum using the exact ideal landscape, normalized by its range. The exact mean landscape isolates residual deterministic distortion, while finite-shot selection includes sampling. A preregistered-for-this-control contrast compares the row-major ideal minimum with the smallest strictly positive ideal gap (tolerance $10^{-10}$); it is illustrative rather than a worst-case competitor. The protocol file predates execution, but this entire control remains post hoc relative to the original campaign.

\begin{center}\scriptsize
\begin{tabular}{lllrrrrr}\toprule
Graph & $p$ & Method & Bias$^2$ & Variance & Expected MSE & Mean regret & MC regret\\\midrule
TABLE
\bottomrule\end{tabular}\end{center}
\includegraphics[width=\linewidth]{bias_variance.pdf}
\includegraphics[width=\linewidth]{decision_regret.pdf}

At $p=0.02$, linear and quadratic mitigation reduce expected grid-mean MSE on all three graphs: their MSE ratios to the noisy baseline are respectively $0.373$--$0.489$ and $0.758$--$0.839$. Cubic mitigation increases it ($2.116$--$2.265$), despite lowering squared bias. At $p=0.001$, all orders increase MSE, with ratios of about $3.79$--$3.82$, $11.60$--$11.67$, and $34.02$--$34.19$. Actual sampling-variance ratios are approximately $3.90$--$4.00$, $11.63$--$12.27$, and $33.29$--$36.09$; they need not equal the constant-variance amplification factors $4,12.25,36$ because scale-dependent physical variances and integer allocations enter the exact formula.

All exact mean landscapes select an ideal minimum, up to floating-point differences below $10^{-14}$ in normalized regret. This small grid therefore provides no observed deterministic argmin error, although deterministic pointwise bias remains. For the weighted graphs, finite-shot mean regret is higher with mitigation in every examined configuration; at $p=0.02$ this coexists with lower expected MSE for the first two orders. Sampling is necessary for these observed selection errors, but its effect can interact with residual bias through the decision margins; no counterfactual isolates a pure variance effect here. G1 finite-shot regrets are also below $10^{-14}$, so the decision plot uses one common vertical scale rather than magnifying roundoff. An independent statevector calculation agrees with all 432 ideal grid values to within $6.22\times10^{-15}$; there are 32, 2, and 2 ideal minimizers on G1, G2, and G3 within absolute objective tolerance $10^{-10}$. A reversal against one fixed ideal minimizer need not cause selection of a suboptimal grid point: another equivalent ideal minimizer can win. The chosen contrast is not a substitute for the full selection calculation. Zero empirical errors among 500 replicates do not imply zero population error probability.

The plotted MC error bars are descriptive $1.96$ standard errors over independent landscape replicates, not simultaneous confidence intervals or tests of cross-method differences. All 24 MC MSE estimates lie within $1.919$ estimated standard errors of their exact expectations. Exact mean and second moments were checked against Aer at two preselected points (maximum discrepancies $2.73\times10^{-14}$ and $1.33\times10^{-13}$ respectively); source hashes, software versions, timings, all pointwise values, and replicate outcomes are supplied. The production calculation took 69.3 seconds on this machine, below the declared 600-second bound. Small-grid observations do not identify the mechanism of every archived discordant cell, and the single selected contrast cannot characterize the full argmin competition.
\end{document}
'''.replace('TABLE',table)
(OUT/'note.tex').write_text(tex)
print('\n'.join(lines)); print('max MC MSE z',max(abs(r['mc_mse_z']) for r in rows))
