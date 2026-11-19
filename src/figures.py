"""Figures of the manuscript (static, print). Palette: reference categorical slots 1-4, with distinct markers
and line styles as secondary encoding."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

import json, sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = sys.argv[1] if len(sys.argv) > 1 else f'{_ROOT}/results'
FIG = f'{_ROOT}/figures'; os.makedirs(FIG, exist_ok=True)
COL = {'raw': '#2a78d6', 'mass_true': '#eb6834', 'margin_true': '#1baf7a', 'fusion_shapley': '#eda100'}
MK = {'raw': 'o', 'mass_true': 's', 'margin_true': '^', 'fusion_shapley': 'D'}
LS = {'raw': '-', 'mass_true': '--', 'margin_true': '-', 'fusion_shapley': ':'}
NAME = {'raw': 'Raw fusion weight', 'mass_true': 'Probability-mass contribution',
        'margin_true': 'Margin contribution (proposed)', 'fusion_shapley': 'Within-fusion Shapley value'}
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e3',
                     'grid.linewidth': 0.6, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'legend.frameon': False})


def fig_manipulation():
    M = pd.read_csv(f'{D}/manipulation.csv', dtype={'split': str}); M = M[M.split != 'paper']
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    panels = [('uniform_mix', 'shape', 'Shape expert mixed with uniform\n(mixing weight $\\lambda$)', 'par', True),
              ('temperature', 'shape', 'Temperature on shape expert\n(sharper $\\leftarrow$ $\\tau$ $\\rightarrow$ flatter)', 'par', False),
              ('temperature', 'texture', 'Temperature on texture expert\n(sharper $\\leftarrow$ $\\tau$ $\\rightarrow$ flatter)', 'par', False)]
    for ax, (kind, ex, title, xcol, rev) in zip(axes, panels):
        g = M[(M.kind == kind) & (M.expert == ex)]
        for a in ('raw', 'mass_true', 'fusion_shapley', 'margin_true'):
            st = g.groupby(xcol)[f'{a}_{ex}'].agg(['mean', 'std']).sort_index()
            x = st.index.values
            ax.plot(x, st['mean'], color=COL[a], marker=MK[a], ms=4.5, lw=1.6, ls=LS[a], label=NAME[a])
            ax.fill_between(x, st['mean'] - st['std'], st['mean'] + st['std'], color=COL[a], alpha=0.12, lw=0)
        ax.set_xscale('log')
        if rev:
            ax.invert_xaxis()
        ax.set_xticks(sorted(g[xcol].unique())); ax.set_xticklabels([f'{v:g}' for v in sorted(g[xcol].unique())])
        ax.minorticks_off()
        ax.set_title(title, fontsize=8.5); ax.set_ylabel(f'Share attributed to {ex}')
        ax.set_ylim(-0.05, 1.0)
    axes[0].set_xlabel('$\\lambda$ (1 = original expert)'); axes[1].set_xlabel('$\\tau$ (1 = original expert)'); axes[2].set_xlabel('$\\tau$ (1 = original expert)')
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(f'{FIG}/fig_manipulation.png', dpi=600); fig.savefig(f'{FIG}/fig_manipulation.pdf'); plt.close(fig)


def fig_plateau():
    PL = pd.read_csv(f'{D}/plateau.csv', dtype={'split': str}); PL = PL[(PL.split != 'paper') & (PL.tol == 0.2)]
    orders = ['CTS', 'CST', 'SCT', 'TCS', 'TSC', 'STC']
    lab = {'CTS': 'C>T>S', 'CST': 'C>S>T', 'SCT': 'S>C>T', 'TCS': 'T>C>S', 'TSC': 'T>S>C', 'STC': 'S>T>C'}
    PL = PL.sort_values('raw_CTS').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    bottom = np.zeros(len(PL)); cols = ['#1baf7a', '#eb6834', '#2a78d6', '#eda100', '#e87ba4', '#4a3aa7']
    for o, c in zip(orders, cols):
        v = PL[f'raw_{o}'].values
        if v.sum() == 0:
            continue
        ax.bar(np.arange(len(PL)), v, bottom=bottom, color=c, width=0.82, edgecolor='white', linewidth=0.6, label=lab[o])
        bottom += v
    ax.plot(np.arange(len(PL)), PL.margin_ok.values, color='#0b0b0b', marker='^', ms=3.5, lw=0, label='Margin contribution in order C>T>S')
    ax.set_xlim(-0.6, len(PL) - 0.4); ax.set_ylim(0, 1.04)
    ax.set_xlabel('Re-split (sorted by the share of raw weightings in the reference order C>T>S)')
    ax.set_ylabel('Share of near-optimal\nweightings'); ax.set_xticks([])
    ax.legend(ncol=5, fontsize=7.5, loc='upper center', bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout(); fig.savefig(f'{FIG}/fig_plateau.png', dpi=600); fig.savefig(f'{FIG}/fig_plateau.pdf'); plt.close(fig)


if __name__ == '__main__':
    fig_manipulation(); fig_plateau(); print('figures written')


def fig_scatter():
    M = pd.read_csv(f'{D}/main.csv', dtype={'split': str}); M = M[(M.split != 'paper') & (M.objective == 'macroF1')]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharex=False)
    for ax, an in zip(axes, ('raw', 'mass_true', 'margin_true')):
        g = M[M.attribution == an]
        ok = g.v_texture > g.v_shape
        ax.scatter(g.v_texture[ok], g.v_shape[ok], s=22, color=COL[an], marker=MK[an], edgecolor='white', linewidth=0.6,
                   label='texture > shape (agrees with references)')
        ax.scatter(g.v_texture[~ok], g.v_shape[~ok], s=26, facecolor='none', edgecolor=COL[an], marker=MK[an], linewidth=1.1,
                   label='shape > texture (contradicts references)')
        lim = max(0.5, float(max(g.v_texture.max(), g.v_shape.max())) * 1.1)
        ax.plot([0, lim], [0, lim], color='#8a8984', lw=0.8, ls='--')
        ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect('equal')
        ax.set_xlabel('Share attributed to texture'); ax.set_ylabel('Share attributed to shape')
        ax.set_title({'raw':'Raw fusion weight','mass_true':'Probability-mass contribution','margin_true':'Margin contribution'}[an] + f'\n{int(ok.sum())} of {len(g)} re-splits agree', fontsize=8.5)
    h1 = plt.Line2D([], [], marker='o', ls='', color='#52514e', label='Filled: texture > shape (agrees with all references)')
    h2 = plt.Line2D([], [], marker='o', ls='', markerfacecolor='none', markeredgecolor='#52514e', label='Hollow: shape > texture (contradicts them)')
    fig.legend(handles=[h1, h2], loc='lower center', ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f'{FIG}/fig_scatter.png', dpi=600); fig.savefig(f'{FIG}/fig_scatter.pdf'); plt.close(fig)


fig_scatter()
