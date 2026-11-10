"""Diversity within bundles versus the initial bound, over the 1,096 candidate partitions of the
60 training instances (data/train_small/partitions.csv). Both axes are standardized within each
instance, so the figure compares partitions of the same instance."""
import pandas as pd, numpy as np, matplotlib
from scipy.stats import spearmanr
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
d = pd.read_csv('partitions.csv')
z = lambda v: (v - v.mean()) / (v.std(ddof=0) if v.std(ddof=0) > 1e-12 else 1.0)
d['x'] = d.groupby('seed').sig_within.transform(z)
d['y'] = d.groupby('seed').L0.transform(z)
d['type'] = d.name.str.replace(r'\d+', '', regex=True).replace(
    {'stratified_c': 'stratified', 'stratified_f': 'stratified'})
G = [('kmeans', '$k$-means', '#8a63d2', 'D'), ('costsig', 'Cost-signature', '#1baf7a', '^'),
     ('random', 'Random', '#2a78d6', 's'), ('swap', 'Stratified + swaps', '#a8a7a0', '.'),
     ('stratified', 'Stratified', '#eb6834', 'o')]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0), gridspec_kw={'width_ratios': [1.6, 1]})
for t, lab, c, mk in G:
    q = d[d.type == t]
    a1.scatter(q.x, q.y, s=10 if mk != '.' else 14, marker=mk, color=c, alpha=0.75, lw=0, label=lab, zorder=3)
bins = np.linspace(d.x.quantile(0.02), d.x.quantile(0.98), 9)
mid = 0.5 * (bins[1:] + bins[:-1])
m = [d[(d.x >= lo) & (d.x < hi)].y.mean() for lo, hi in zip(bins[:-1], bins[1:])]
a1.plot(mid, m, color='#1f1f1d', lw=1.6, label='Binned mean', zorder=4)
a1.set_xlabel('Within-bundle diversity (standardized)'); a1.set_ylabel('$L(0)$ (standardized)')
a1.grid(color='#eeede8', lw=0.6); a1.set_axisbelow(True)
a1.legend(frameon=False, fontsize=7, loc='upper left', markerscale=1.4, handletextpad=0.2)
eq = d[~d.type.isin(['kmeans', 'costsig'])]          # equal bundle sizes only (no size confound)
r = {f: [spearmanr(g[f], g.L0).correlation for _, g in eq.groupby('seed') if g[f].std() > 0]
     for f in ('sig_within', 'phi_within')}
a2.hist([r['sig_within'], r['phi_within']], bins=np.linspace(-1, 1, 11), color=['#eb6834', '#2a78d6'],
        label=['Cost signatures', 'Demand and disruption'])
a2.axvline(0, color='#6b6a64', lw=0.9)
a2.set_xlabel('Spearman correlation with $L(0)$\nper instance (equal sizes)'); a2.set_ylabel('Instances')
a2.legend(frameon=False, fontsize=7, loc='upper left')
fig.tight_layout(); fig.savefig('fig_diversity.pdf'); fig.savefig('fig_diversity.png', dpi=160)
for f in r: print(f, round(np.mean(r[f]), 3), round(np.mean(np.array(r[f]) > 0), 3), len(r[f]))
print(spearmanr(d.x, d.y).correlation)
