"""Turn the per-partition CSVs into the statistics quoted in the manuscript (results.json)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

import json, sys, os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, binomtest, beta, spearmanr

D = sys.argv[1] if len(sys.argv) > 1 else f'{_ROOT}/results'
G = ('color', 'texture', 'shape')
R = {}


def wilson(k, n, z=1.959964):
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


def boot_ci(x, f=np.mean, B=10000, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (B, len(x)))
    v = np.array([f(x[i]) for i in idx]); return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def mcnemar_exact(a_ok, b_ok):
    """one-sided exact McNemar: is b better than a?"""
    b = int((~a_ok & b_ok).sum()); c = int((a_ok & ~b_ok).sum())
    p = binomtest(b, b + c, 0.5, alternative='greater').pvalue if b + c else 1.0
    return b, c, float(p)


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


main = pd.read_csv(f'{D}/main.csv', dtype={'split': str})
meta = pd.read_csv(f'{D}/meta.csv', dtype={'split': str})
RS = main[main.split != 'paper']; MR = meta[meta.split != 'paper']
n = RS.split.nunique(); R['n_resplits'] = int(n)

# ---- accuracy summary
for col in ('base_f1', 'base_acc', 'fused_f1', 'fused_acc', 'uniform_f1', 'nsv_base',
            'solo_color', 'solo_texture', 'solo_shape', 'sharp_color', 'sharp_texture', 'sharp_shape',
            'ent_color', 'ent_texture', 'ent_shape', 'alpha_color', 'alpha_texture', 'alpha_shape',
            'logo_color', 'logo_texture', 'logo_shape', 'shapley_color', 'shapley_texture', 'shapley_shape'):
    if col in MR:
        R[f'{col}_mean'] = float(MR[col].mean()); R[f'{col}_sd'] = float(MR[col].std())
P = meta[meta.split == 'paper'].iloc[0].to_dict() if (meta.split == 'paper').any() else {}
R['paper'] = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in P.items()}
R['orders'] = {r: MR[f'order_{r}'].value_counts().to_dict() for r in ('solo', 'logo', 'shapley') if f'order_{r}' in MR}
win = (MR.fused_f1 > MR.uniform_f1).values; diff = (MR.fused_f1 - MR.uniform_f1).values
R['fusion_vs_uniform'] = dict(wins=int(win.sum()), n=int(n), diff_mean=float(diff.mean()), diff_sd=float(diff.std(ddof=1)),
                              ci=boot_ci(diff), p_wilcoxon=float(wilcoxon(MR.fused_f1, MR.uniform_f1, alternative='greater').pvalue))
fdiff = (MR.base_f1 - MR.fused_f1).values
R['interpretability_cost'] = dict(mean=float(fdiff.mean()), sd=float(fdiff.std(ddof=1)), ci=boot_ci(fdiff))

# ---- attribution recovery, per objective/attribution/reference
att = {}
for (obj, an), g in RS.groupby(['objective', 'attribution']):
    e = {}
    for ref in ('solo', 'logo', 'shapley'):
        if f'ok_{ref}' not in g or g[f'ok_{ref}'].isna().all():
            continue
        k = int(g[f'ok_{ref}'].sum())
        e[ref] = dict(k=k, n=len(g), ci=wilson(k, len(g)), rho_mean=float(g[f'rho_{ref}'].mean()),
                      rho_sd=float(g[f'rho_{ref}'].std()), rho_ci=boot_ci(g[f'rho_{ref}'].fillna(0)))
    e['shares'] = {x: [float(g[f'v_{x}'].mean()), float(g[f'v_{x}'].std())] for x in G}
    e['fused_f1'] = [float(g.fused_f1.mean()), float(g.fused_f1.std())]
    e['orders'] = g.order.value_counts().to_dict()
    att[f'{obj}|{an}'] = e
R['attribution'] = att
R['paper_attribution'] = {f'{r.objective}|{r.attribution}': dict(order=r.order, **{x: float(r[f'v_{x}']) for x in G},
                                                                  ok_solo=bool(r.ok_solo))
                          for _, r in main[main.split == 'paper'].iterrows()}

# ---- paired tests vs raw (same objective), Holm across the family
tests = []
for obj in ('macroF1', 'calibrated'):
    base = RS[(RS.objective == obj) & (RS.attribution == 'raw')].set_index('split')
    for an in ('mass_true', 'margin_true', 'margin_pseudo', 'margin_fused', 'fusion_shapley'):
        e = RS[(RS.objective == obj) & (RS.attribution == an)].set_index('split')
        if e.empty:
            continue
        e = e.loc[base.index]
        for ref in ('solo', 'logo', 'shapley'):
            b, c, p = mcnemar_exact(base[f'ok_{ref}'].values.astype(bool), e[f'ok_{ref}'].values.astype(bool))
            d = (e[f'rho_{ref}'] - base[f'rho_{ref}']).values
            pw = float(wilcoxon(e[f'rho_{ref}'], base[f'rho_{ref}'], alternative='greater', zero_method='zsplit').pvalue) if np.any(d != 0) else 1.0
            # matched-pairs rank-biserial correlation
            nz = d[d != 0]; ranks = pd.Series(np.abs(nz)).rank().values
            rrb = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()) if len(nz) else 0.0
            dp = (e[f'ok_{ref}'].mean() - base[f'ok_{ref}'].mean())
            tests.append(dict(objective=obj, attribution=an, reference=ref, wins=b, losses=c, p_mcnemar=p,
                              p_wilcoxon=pw, rank_biserial=rrb, diff_prop=float(dp),
                              diff_rho=float(d.mean()), diff_rho_ci=boot_ci(d)))
T = pd.DataFrame(tests)
T['p_mcnemar_holm'] = holm(T.p_mcnemar); T['p_wilcoxon_holm'] = holm(T.p_wilcoxon)
T.to_csv(f'{D}/tests.csv', index=False)
R['tests'] = T.to_dict(orient='records')

# ---- smooth objectives
for obj in ('logistic_stacking', 'convex_nll', 'convex_brier'):
    g = RS[(RS.objective == obj) & (RS.attribution == 'raw')]
    R[f'{obj}_raw'] = dict(k=int(g.ok_solo.sum()), n=len(g), f1=[float(g.fused_f1.mean()), float(g.fused_f1.std())],
                           shares={x: [float(g[f'v_{x}'].mean()), float(g[f'v_{x}'].std())] for x in G})

# ---- manipulation (fixed competence)
if os.path.exists(f'{D}/manipulation.csv'):
    M = pd.read_csv(f'{D}/manipulation.csv', dtype={'split': str}); MRs = M[M.split != 'paper']
    man = {}
    for (kind, ex, par), g in MRs.groupby(['kind', 'expert', 'par']):
        man[f'{kind}|{ex}|{par}'] = dict(n=len(g), solo=float(g.solo_f1.mean()), gap=float(g.gap.mean()),
                                         fused_f1=float(g.fused_f1.mean()),
                                         **{f'sharp_{x}': float(g[f'sharp_{x}'].mean()) for x in G},
                                         **{f'{a}_ok': int(g[f'{a}_ok'].sum()) for a in ('raw', 'mass_true', 'margin_true', 'margin_pseudo', 'fusion_shapley')},
                                         **{f'{a}_{x}': [float(g[f'{a}_{x}'].mean()), float(g[f'{a}_{x}'].std())]
                                            for a in ('raw', 'mass_true', 'margin_true') for x in G})
    R['manipulation'] = man
    # how much does each attribution move when only sharpness changes? (uniform mix on shape)
    um = MRs[MRs.kind == 'uniform_mix']
    mv = {}
    for a in ('raw', 'mass_true', 'margin_true', 'fusion_shapley'):
        piv = um.pivot_table(index='split', columns='par', values=f'{a}_shape')
        mv[a] = dict(shape_at=[float(piv[c].mean()) for c in sorted(piv.columns, reverse=True)],
                     max_abs_change=float((piv.max(1) - piv.min(1)).mean()))
    R['uniform_mix_movement'] = mv

# ---- plateau
if os.path.exists(f'{D}/plateau.csv'):
    PL = pd.read_csv(f'{D}/plateau.csv', dtype={'split': str}); PLr = PL[PL.split != 'paper']
    R['plateau'] = {str(t): dict(n_mean=float(g.n.mean()), raw_ok=float(g.raw_ok.mean()), mass_ok=float(g.mass_ok.mean()),
                                 margin_ok=float(g.margin_ok.mean()),
                                 orders={o: float(g[f'raw_{o}'].mean()) for o in ('CTS', 'CST', 'TCS', 'TSC', 'SCT', 'STC')},
                                 splits_raw_multiple_orders=int((g[[f'raw_{o}' for o in ('CTS', 'CST', 'TCS', 'TSC', 'SCT', 'STC')]].gt(0).sum(1) > 1).sum()))
                    for t, g in PLr.groupby('tol')}
    R['plateau_paper'] = PL[PL.split == 'paper'].to_dict(orient='records')

# ---- calibration
if os.path.exists(f'{D}/calibration.csv'):
    CA = pd.read_csv(f'{D}/calibration.csv', dtype={'split': str}); CAr = CA[CA.split != 'paper']
    R['calibration'] = {ex: {c: [float(g[c].mean()), float(g[c].std())] for c in ('T', 'ece_before', 'ece_after', 'maxp_before', 'maxp_after', 'ent_before', 'ent_after')}
                        for ex, g in CAr.groupby('expert')}
    gap_b = CAr.pivot_table(index='split', columns='expert', values='maxp_before'); gap_a = CAr.pivot_table(index='split', columns='expert', values='maxp_after')
    R['calibration_gap'] = dict(before=float((gap_b['color'] - gap_b['shape']).mean()), after=float((gap_a['color'] - gap_a['shape']).mean()))

# ---- reject option and failure analysis
if os.path.exists(f'{D}/reject.csv'):
    RJ = pd.read_csv(f'{D}/reject.csv', dtype={'split': str}); RJr = RJ[RJ.split != 'paper']
    R['reject'] = {f'{m}|{t}': {c: [float(g[c].mean()), float(g[c].std())] for c in ('coverage', 'acc', 'f1', 'deferred', 'auroc', 'errors')}
                   for (m, t), g in RJr.groupby(['model', 'target'])}
    R['reject_paper'] = RJ[RJ.split == 'paper'].to_dict(orient='records')
if os.path.exists(f'{D}/instance.csv'):
    IN = pd.read_csv(f'{D}/instance.csv', dtype={'split': str}); INr = IN[IN.split != 'paper']
    R['failure'] = {c: [float(INr[c].mean()), float(INr[c].std())] for c in INr.columns if c != 'split'}
    pooled_cross = (INr.color_resp_cross * INr.n_cross).sum() / INr.n_cross.sum()
    pooled_within = (INr.color_resp_within * INr.n_within).sum() / INr.n_within.sum()
    R['failure_pooled'] = dict(cross=float(pooled_cross), within=float(pooled_within),
                               n_cross=int(INr.n_cross.sum()), n_within=int(INr.n_within.sum()))
    R['failure_paper'] = IN[IN.split == 'paper'].to_dict(orient='records')

json.dump(R, open(f'{D}/results.json', 'w'), indent=1, default=float)
print('results.json written;', n, 're-splits')
