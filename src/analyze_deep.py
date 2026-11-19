"""Analysis of the deep experts (linear heads on frozen MobileNetV2 embeddings of three
cue-isolating streams). Same protocol as analyze.py: V1 for weights, V2 for temperatures.
usage: python analyze_deep.py <deep_dir> <out_dir>"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

import glob, os, sys, itertools, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, _HERE); sys.path.insert(0, _HERE)
from common import load_all
from lib import *

DD, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
train, val, test, _ = load_all()
y = pd.concat([train, val, test], ignore_index=True)['label_id'].values
rows, man_rows, meta_rows, cal_rows = [], [], [], []
for fn in sorted(glob.glob(f'{DD}/deep_*.npz')):
    s = os.path.basename(fn)[5:-4]
    if s.startswith('sweep'):
        continue
    d = np.load(fn); iva, ite = d['iva'], d['ite']; yv_all, yt = y[iva], y[ite]
    v1, v2 = train_test_split(np.arange(len(iva)), test_size=0.5, stratify=yv_all, random_state=0)
    Pv = [d[f'Pval_{g}'].astype(float) for g in G]; Pt = [d[f'Pte_{g}'].astype(float) for g in G]
    Pv = [P / P.sum(1, keepdims=True) for P in Pv]; Pt = [P / P.sum(1, keepdims=True) for P in Pt]
    Pv1 = [P[v1] for P in Pv]; Pv2 = [P[v2] for P in Pv]; y1, y2 = yv_all[v1], yv_all[v2]
    solo = np.array([mf1(yt, P.argmax(1)) for P in Pt]); refs = {'solo': solo}
    def add(att, obj, f1):
        for an, av in att.items():
            r = dict(split=s, objective=obj, attribution=an, fused_f1=f1, order=order_str(av), **{f'v_{g}': x for g, x in zip(G, av)},
                     ok_solo=order_ok(av, solo), rho_solo=rho(av, solo))
            rows.append(r)
    a = CANDS[int(search_scores(Pv1, y1).argmax())]; yhat = fuse(a, Pt).argmax(1)
    add(all_attributions(a, Pt, yt, None, True), 'macroF1', mf1(yt, yhat))
    T = [fit_temp(P, y2) for P in Pv2]; Pv1c = [sharpen(P, t) for P, t in zip(Pv1, T)]; Ptc = [sharpen(P, t) for P, t in zip(Pt, T)]
    ac = CANDS[int(search_scores(Pv1c, y1).argmax())]
    add(all_attributions(ac, Ptc, yt, None, True), 'calibrated', mf1(yt, fuse(ac, Ptc).argmax(1)))
    for g, t, P, Pc in zip(G, T, Pt, Ptc):
        cal_rows.append(dict(split=s, expert=g, T=t, ece_before=ece(P, yt), ece_after=ece(Pc, yt),
                             maxp_before=P.max(1).mean(), maxp_after=Pc.max(1).mean()))
    meta_rows.append(dict(split=s, fused_f1=mf1(yt, yhat), fused_acc=(yhat == yt).mean() * 100,
                          **{f'solo_{g}': x for g, x in zip(G, solo)}, **{f'sharp_{g}': P.max(1).mean() for g, P in zip(G, Pt)},
                          **{f'ent_{g}': entropy(P) for g, P in zip(G, Pt)}, **{f'alpha_{g}': x for g, x in zip(G, a)}))
    k = int(np.argmin(solo))
    for lam in (1.0, 0.5, 0.25, 0.1):
        Pv1m = list(Pv1); Ptm = list(Pt)
        Pv1m[k] = lam * Pv1[k] + (1 - lam) / C; Ptm[k] = lam * Pt[k] + (1 - lam) / C
        am = CANDS[int(search_scores(Pv1m, y1).argmax())]; att = all_attributions(am, Ptm, yt, None, True)
        r = dict(split=s, kind='uniform_mix', expert=G[k], par=lam, fused_f1=mf1(yt, fuse(am, Ptm).argmax(1)),
                 **{f'sharp_{g}': P.max(1).mean() for g, P in zip(G, Ptm)})
        for an in ('raw', 'mass_true', 'margin_true', 'fusion_shapley'):
            r.update({f'{an}_{g}': x for g, x in zip(G, att[an])}); r[f'{an}_ok'] = order_ok(att[an], solo)
        man_rows.append(r)
    print(s, 'ok', flush=True)

# ---- sweep on the paper partition
sw_rows = []
sp = f'{DD}/deep_sweep_paper.npz'
if os.path.exists(sp):
    z = np.load(sp); budgets = [int(b) for b in z['budgets']]
    d = np.load(f'{DD}/deep_paper.npz'); iva, ite = d['iva'], d['ite']; yv_all, yt = y[iva], y[ite]
    v1, _ = train_test_split(np.arange(len(iva)), test_size=0.5, stratify=yv_all, random_state=0); y1 = yv_all[v1]
    for dc, dt, ds in itertools.product(budgets, budgets, budgets):
        keys = [('color', dc), ('texture', dt), ('shape', ds)]
        Pv1 = [z[f'Pval_{g}_{b}'].astype(float)[v1] for g, b in keys]; Pt = [z[f'Pte_{g}_{b}'].astype(float) for g, b in keys]
        Pv1 = [P / P.sum(1, keepdims=True) for P in Pv1]; Pt = [P / P.sum(1, keepdims=True) for P in Pt]
        solo = np.array([float(z[f'solo_{g}_{b}']) for g, b in keys])
        if np.diff(np.sort(solo)).min() < 1.0:
            continue
        a = CANDS[int(search_scores(Pv1, y1).argmax())]; att = all_attributions(a, Pt, yt, None, True)
        sh = [P.max(1).mean() for P in Pt]
        # sharpness and competence ranks
        r = dict(dims=f'{dc}/{dt}/{ds}', gap=max(sh) - min(sh), spread=solo.max() - solo.min(),
                 sharp_order=order_str(sh), solo_order=order_str(solo),
                 sharp_matches_solo=order_str(sh) == order_str(solo))
        for an in ('raw', 'mass_true', 'margin_true', 'fusion_shapley'):
            r[f'{an}_ok'] = order_ok(att[an], solo); r[f'{an}_rho'] = rho(att[an], solo)
        sw_rows.append(r)
for nm, R in (('deep_main', rows), ('deep_manip', man_rows), ('deep_meta', meta_rows), ('deep_cal', cal_rows), ('deep_sweep', sw_rows)):
    if R:
        pd.DataFrame(R).to_csv(f'{OUT}/{nm}.csv', index=False)
print('written')
