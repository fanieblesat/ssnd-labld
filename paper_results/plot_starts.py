"""Figures for the corrected warm-start policies.
  fig_warmstart.pdf        : Study A, LB gap by iteration, zero / P12 / P60 / oracle start
  fig_scaling_starts.pdf   : |Omega|=240, LB over time, zero / P12 / P60 start, DE lower bound
"""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'font.family': 'serif', 'axes.spines.top': False, 'axes.spines.right': False})
C = {'zero': '#6b6a64', 'p12': '#2a78d6', 'p60': '#eb6834', 'p60tw': '#8a63d2', 'oracle': '#1baf7a'}
z = dict(pd.read_csv('deA.csv')[['seed', 'de_obj']].values)
hb = pd.read_csv('hW_bundle.csv'); h60 = pd.read_csv('hW_s60.csv'); ho = pd.read_csv('hW.csv')
series = [(hb, 'zero', 'zero', 'Zero start', '-'), (hb, 'ml', 'p12', 'Learned start $P_{12}$', '-'),
          (h60, 'ml', 'p60', 'Learned start $P_{60}$', '-'), (ho, 'oracle', 'oracle', 'Oracle start', '--')]
fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.2)); axes = axes.ravel()
for s in range(1, 6):
    ax = axes[s - 1]
    for H, lab, key, name, ls in series:
        h = H[(H.seed == s) & (H.label == lab)].sort_values('t')
        g = 100 * (z[s] - h.LB) / z[s]
        t = list(h.t) + ([29] if h.t.max() < 29 else []); g = list(g) + ([g.iloc[-1]] if h.t.max() < 29 else [])
        ax.step(t, g, where='post', color=C[key], lw=1.5, ls=ls, label=name)
    ax.set_title(f'Seed {s}', fontsize=9); ax.set_xlim(0, 29)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
    if s in (1, 4): ax.set_ylabel('LB gap to $z^\\star$ (%)')
    if s >= 3: ax.set_xlabel('Dual iteration')
axes[5].axis('off'); h_, l_ = axes[0].get_legend_handles_labels(); axes[5].legend(h_, l_, loc='center', frameon=False)
fig.tight_layout(); fig.savefig('fig_warmstart.pdf'); fig.savefig('fig_warmstart.png', dpi=160)

S = pd.read_csv('scaling.csv'); S = S[S.S == 240].set_index('seed')
hz = pd.read_csv('hS.csv'); hz = hz[hz.S == 240]; ha = pd.read_csv('hS_mlb.csv'); hc = pd.read_csv('hS_s60.csv'); hd = pd.read_csv('hS_s60tw.csv')
fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.4)); axes = axes.ravel()
for s in range(1, 6):
    ax = axes[s - 1]; de = S.loc[s, 'de_lb']
    for H, key, name in [(hz, 'zero', 'Zero start'), (ha, 'p12', 'Learned start $P_{12}$'), (hc, 'p60', 'Learned start $P_{60}$'), (hd, 'p60tw', r'$P_{60}$, $\theta_w=0.5$')]:
        h = H[(H.seed == s) & (H.time <= 1800)].sort_values('time')
        ax.step(list(h.time / 60) + [30], list(h.LB) + [h.LB.iloc[-1]], where='post', color=C[key], lw=1.5, label=name)
    ax.axhline(de, color='black', ls='--', lw=1, label='DE lower bound' if s != 4 else None)
    ax.set_title(f'Seed {s}' + (' (DE optimal)' if S.loc[s, 'de_proved'] else ''), fontsize=9); ax.set_xlim(0, 30)
    ax.grid(axis='y', color='#e6e5e0', lw=0.8); ax.set_axisbelow(True)
    if s in (1, 4): ax.set_ylabel('Lower bound')
    if s >= 3: ax.set_xlabel('Time (min)')
axes[5].axis('off'); h_, l_ = axes[0].get_legend_handles_labels(); axes[5].legend(h_, l_, loc='center', frameon=False)
fig.tight_layout(); fig.savefig('fig_scaling_starts.pdf'); fig.savefig('fig_scaling_starts.png', dpi=160)
