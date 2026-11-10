"""Figure for the sustainability-resilience study (RQ5), from sustainability.csv."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'font.family': 'serif',
                     'axes.spines.top': False, 'axes.spines.right': False})
d = pd.read_csv('sus.csv')
b = d[d.kappa == 0].set_index(['setting', 'seed'])
d = d.join(b[['emissions', 'operating_cost', 'spot_emissions', 'reliability']], on=['setting', 'seed'], rsuffix='0')
d['dE'] = 100 * (d.emissions / d.emissions0 - 1); d['dC'] = 100 * (d.operating_cost / d.operating_cost0 - 1)
d['dS'] = 100 * (d.spot_emissions / d.spot_emissions0 - 1); d['dR'] = 100 * (d.reliability - d.reliability0)
C = {'base': '#2a78d6', 'stressed': '#eb6834'}; L = {'base': 'Base disruption', 'stressed': 'Stressed disruption'}
K = [0, 0.5, 1, 2, 4]
fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.9))
for s in ('base', 'stressed'):
    q = d[d.setting == s]; g = q.groupby('kappa')[['dE', 'dC', 'share_road', 'share_rail']].mean()
    ax[0].scatter(q.dC, -q.dE, s=10, color=C[s], alpha=0.25, lw=0)
    ax[0].plot(g.dC, -g.dE, '-o', color=C[s], ms=4, label=L[s])
    for k in (4,):
        ax[0].annotate(fr'$\kappa={k:g}$', (g.dC[k], -g.dE[k]), textcoords='offset points',
                       xytext=(-6, 6), fontsize=7, color=C[s])
    ax[1].plot(range(5), g.share_road, '-o', color=C[s], ms=4)
    ax[1].plot(range(5), g.share_rail, '--s', color=C[s], ms=4)
    q2 = q[q.kappa > 0]
    ax[2].scatter(q2.dS, q2.dR, s=14, color=C[s], alpha=0.8, lw=0)
ax[0].set_xlabel('Operating-cost increase (%)'); ax[0].set_ylabel(r'CO$_2$e reduction (%)')
ax[0].set_title('(a) Cost of abatement', fontsize=9); pass
ax[1].set_xticks(range(5)); ax[1].set_xticklabels([f'{k:g}' for k in K]); ax[1].set_xlabel(r'Carbon price $\kappa$')
ax[1].set_ylabel('Tonne-km share'); ax[1].set_title('(b) Road (solid), rail (dashed)', fontsize=9)
ax[2].axhline(0, color='#a8a7a0', lw=0.8); ax[2].axvline(0, color='#a8a7a0', lw=0.8)
ax[2].set_xlabel('Change in spot CO$_2$e (%)'); ax[2].set_ylabel('Change in reliability (pts)')
ax[2].set_title(r'(c) Rebound, $\kappa>0$', fontsize=9)
r = d[(d.setting == 'stressed') & (d.seed == 1) & (d.kappa == 4)].iloc[0]
ax[2].annotate('stressed, seed 1,\n' + r'$\kappa=4$', (r.dS, r.dR), textcoords='offset points', xytext=(-48, 2), fontsize=7)
for a in ax: a.grid(color='#e6e5e0', lw=0.7); a.set_axisbelow(True)
h_, l_ = ax[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=2, frameon=False, fontsize=8); fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig("fig_sustainability.pdf"); fig.savefig('fig_sustainability.png', dpi=160)
