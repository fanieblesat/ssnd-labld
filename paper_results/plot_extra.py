"""Three additional figures for Section 6, from the result files already in hand.
  fig_bundling_instances.pdf : initial bound gap per instance and partition (Studies A and B)
  fig_scaling_summary.pdf    : LD lower bound relative to the DE lower bound, by scenario count
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'font.family': 'serif',
                     'axes.spines.top': False, 'axes.spines.right': False})
C = {'stratified': '#eb6834', 'random': '#2a78d6', 'costsig_bal': '#1baf7a', 'kmeans_bal': '#8a63d2',
     'zero': '#6b6a64', 'ml': '#eb6834', 'oracle': '#2a78d6'}

# ------------------------------------------------------------------ 1. bundling per instance
def load(s, b, tag):
    x = pd.read_csv(s); y = pd.read_csv(b); y['partition'] += '_bal'
    d = pd.concat([x, y]); d['inst'] = tag + d.seed.astype(str); return d
d = pd.concat([load('A.csv', 'Ab.csv', 'A'), load('B.csv', 'Bb.csv', 'B')])
parts = [('stratified', 'Stratified', 'o'), ('random', 'Random', 's'),
         ('costsig_bal', 'Cost-signature (equal sizes)', '^'), ('kmeans_bal', '$k$-means (equal sizes)', 'D')]
insts = [f'A{i}' for i in range(1, 6)] + [f'B{i}' for i in range(1, 6)]
fig, ax = plt.subplots(figsize=(6.6, 3.9))
for k, (p, lab, mk) in enumerate(parts):
    v = d[d.partition == p].set_index('inst').L0_gap_pct.reindex(insts)
    ax.scatter(v.values, np.arange(len(insts)) + (k - 1.5) * 0.12, marker=mk, s=28, color=C[p],
               label=lab, zorder=3)
for i in range(len(insts)):
    ax.axhline(i, color='#e6e5e0', lw=0.8, zorder=1)
ax.axhline(4.5, color='#a8a7a0', lw=0.8, ls=':')
ax.set_yticks(range(len(insts)))
ax.set_yticklabels([f'Study {s[0]}, seed {s[1]}' for s in insts])
ax.invert_yaxis(); ax.set_xlabel(r'Initial bound gap $(z^\star-L(0))/z^\star$ (%)')
ax.legend(loc='upper center', bbox_to_anchor=(0.42, -0.17), ncol=4, frameon=False, fontsize=8, handletextpad=0.3, columnspacing=1.0)
fig.tight_layout(); fig.savefig('fig_bundling_instances.pdf'); fig.savefig('fig_bundling_instances.png', dpi=160)

# 2. warm-start trajectories: see plot_starts.py

# ------------------------------------------------------------------ 3. scaling summary
S = pd.read_csv('scaling.csv')
S['adv'] = 100 * (S.ld_lb_1800s.fillna(S.ld_lb) - S.de_lb) / S.de_lb
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for proved, mk, lab in [(1, 'o', 'DE proved optimality'), (0, 's', 'DE stopped at the time limit')]:
    q = S[S.de_proved == proved]
    ax.scatter(q.S + np.where(q.seed % 2, -4, 4), q.adv, marker=mk, s=34,
               color='#6b6a64' if proved else '#eb6834', label=lab, zorder=3)
ax.axhline(0, color='#a8a7a0', lw=0.9)
ax.set_xticks([60, 120, 240]); ax.set_xlabel(r'Number of scenarios $|\Omega|$')
ax.set_ylabel('LD lower bound vs. DE lower bound (%)')
ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout(); fig.savefig('fig_scaling_summary.pdf'); fig.savefig('fig_scaling_summary.png', dpi=160)
print(S[['seed', 'S', 'de_proved', 'adv']].round(2).to_string())
