"""Mean lower-bound gap to z* by dual iteration, Studies A and B (equal bundle sizes only).
Inputs (per study): history.csv of the main run, history.csv of the balanced control, de.csv."""
import sys, pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
def load(h, hb, de):
    H = pd.read_csv(h); Hb = pd.read_csv(hb); Hb['partition'] += '_bal'
    H = pd.concat([H, Hb]); z = dict(pd.read_csv(de)[['seed', 'de_obj']].values)
    H['gap'] = 100 * (H.seed.map(z) - H.LB) / H.seed.map(z)
    return H
studies = [("Study A: $|\\Omega|=20$, $m=5$", load('hA.csv', 'hAb.csv', 'deA.csv')),
           ("Study B: $|\\Omega|=30$, $m=6$", load('hB.csv', 'hBb.csv', 'deB.csv'))]
P = [('stratified', 'Stratified', '#eb6834', '-'), ('random', 'Random', '#2a78d6', '--'),
     ('costsig_bal', 'Cost-signature (equal sizes)', '#1baf7a', '-.'),
     ('kmeans_bal', 'Demand–disruption $k$-means (equal sizes)', '#8a63d2', ':')]
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
for ax, (title, H) in zip(axes, studies):
    for key, lab, col, ls in P:
        m = H[H.partition == key].groupby('t').gap.mean()
        ax.plot(m.index, m.values, color=col, ls=ls, lw=1.8, label=lab)
        print(title[:7], key, 'L(0) gap', round(m.iloc[0], 2), 'it10', round(m.loc[10], 2), 'it30', round(m.iloc[-1], 2))
    ax.set_title(title, fontsize=9); ax.set_xlabel('Dual iteration')
    ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True); ax.set_xlim(0, 29)
axes[0].set_ylabel('Mean lower-bound gap to $z^\\star$ (%)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig('studies_bound_profile.pdf'); fig.savefig('studies_bound_profile.png', dpi=160)
