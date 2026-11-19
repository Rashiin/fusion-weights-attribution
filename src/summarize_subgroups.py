"""Summary of the five-subgroup control. Usage: python summarize_subgroups.py <out_dir>"""
import sys, json, numpy as np, pandas as pd
from scipy.stats import binom, wilcoxon
OUT = sys.argv[1]
SUB = ('rgb', 'hsv', 'glcm', 'lbp', 'shape')
me = pd.read_csv(f'{OUT}/meta.csv'); m = pd.read_csv(f'{OUT}/main.csv'); mm = pd.read_csv(f'{OUT}/manipulation.csv')
sd = lambda x: float(x.std(ddof=1))
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(100 * (c - h), 1), round(100 * (c + h), 1)]
R = {'n_partitions': int(len(me))}
R['performance'] = {k: [round(me[k].mean(), 2), round(sd(me[k]), 2)] for k in ('base_f1', 'fused_f1', 'uniform_f1')}
R['experts'] = {g: {k: [round(me[f'{k}_{g}'].mean(), 3), round(sd(me[f'{k}_{g}']), 3)]
                    for k in ('solo', 'logo', 'shapley', 'sharp', 'alpha')} for g in SUB}
R['reference_orders'] = {r: me[f'order_{r}'].value_counts().to_dict() for r in ('solo', 'logo', 'shapley')}
R['solo_equals_shapley_order'] = int((me.order_solo == me.order_shapley).sum())
raw = m[m.attribution == 'raw'].set_index('split')
R['attributions'] = {}
for an, x in m.groupby('attribution'):
    x = x.set_index('split'); e = {'mean_share': {g: round(x[f'v_{g}'].mean(), 3) for g in SUB}}
    for r in ('solo', 'logo', 'shapley'):
        k = int(x[f'ok_{r}'].sum())
        e[r] = dict(exact=k, wilson=wilson(k, len(x)), rho=[round(x[f'rho_{r}'].mean(), 2), round(sd(x[f'rho_{r}']), 2)],
                    tau=[round(x[f'tau_{r}'].mean(), 2), round(sd(x[f'tau_{r}']), 2)],
                    pair=round(100 * x[f'pair_{r}'].mean(), 1))
        if an != 'raw':
            d = x[f'rho_{r}'] - raw[f'rho_{r}']
            e[r]['rho_better_worse_tie'] = [int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())]
            e[r]['p_wilcoxon_rho_vs_raw'] = float(wilcoxon(d, alternative='greater').pvalue) if (d != 0).any() else 1.0
            w = int((x[f'ok_{r}'] & ~raw[f'ok_{r}']).sum()); l = int((~x[f'ok_{r}'] & raw[f'ok_{r}']).sum())
            e[r]['mcnemar_wins_losses_p'] = [w, l, float(binom.cdf(l, w + l, 0.5)) if w + l else 1.0]
    R['attributions'][an] = e
R['manipulation'] = {}
for (g, lam), x in mm.groupby(['expert', 'lam']):
    R['manipulation'][f'{g}|{lam}'] = dict(
        sharp=round(x.sharp.mean(), 3), fused_f1=round(x.fused_f1.mean(), 2),
        **{f'{a}_self': round(x[f'{a}_self'].mean(), 3) for a in ('raw', 'mass_true', 'margin_true', 'fusion_shapley')},
        **{f'{a}_rho': round(x[f'{a}_rho'].mean(), 2) for a in ('raw', 'mass_true', 'margin_true', 'margin_fused', 'fusion_shapley')},
        **{f'{a}_pair': round(100 * x[f'{a}_pair'].mean(), 1) for a in ('raw', 'mass_true', 'margin_true', 'margin_fused', 'fusion_shapley')},
        **{f'{a}_exact': int(x[f'{a}_ok'].sum()) for a in ('raw', 'mass_true', 'margin_true', 'margin_fused', 'fusion_shapley')})
json.dump(R, open(f'{OUT}/summary.json', 'w'), indent=1)
print(json.dumps(R, indent=1))
