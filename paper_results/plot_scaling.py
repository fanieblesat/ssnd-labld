"""Scaling study figure: LD lower/upper bound over wall-clock time versus the DE bounds at the
time limit, |Omega| = 240. Inputs: scaling.csv, history.csv of run_scaling.py."""
import pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
S = pd.read_csv('scaling.csv'); H = pd.read_csv('hS.csv')
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
rows = S[(S.S == 240) & (S.de_proved == 0)].sort_values('seed')
fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.4), sharey=False); axes = axes.ravel()
for ax, (_, r) in zip(axes, rows.iterrows()):
    h = H[(H.seed == r.seed) & (H.S == 240)].sort_values('t')
    ax.step(h.time, h.LB, where='post', color='#2a78d6', lw=1.8, label='LD lower bound')
    ax.step(h.time, h.UB, where='post', color='#2a78d6', lw=1.8, ls='--', label='LD upper bound')
    ax.axhline(r.de_lb, color='#eb6834', lw=1.6, label='DE lower bound at 1800 s')
    ax.axhline(r.de_ub, color='#eb6834', lw=1.6, ls='--', label='DE upper bound at 1800 s')
    ax.set_title(f'Seed {int(r.seed)}, $|\\Omega|=240$, $m=48$', fontsize=9)
    ax.set_xlabel('Wall-clock time (s)'); ax.set_xlim(0, 1850)
    lo, hi = min(h.LB.min(), r.de_lb), max(h.UB.max(), r.de_ub)
    ax.set_ylim(lo - 0.03 * (hi - lo), hi + 0.03 * (hi - lo))
    ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
[a.set_ylabel('Objective value') for a in (axes[0], axes[2])]
hd, lb = axes[0].get_legend_handles_labels()
fig.legend(hd, lb, loc='lower center', ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig('scaling_bounds.pdf'); fig.savefig('scaling_bounds.png', dpi=160)
