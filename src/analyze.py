"""Analysis of the handcrafted experts (one environment, one protocol).

Per partition the validation split is divided once, stratified 50/50 (seed 0), into
V1 (fusion-weight selection only) and V2 (temperature fitting and reject thresholds only).
Platt scaling inside every SVM uses internal cross-validation on the training split only.
Weights: the best of 4,000 Dirichlet(1,1,1) candidates (seed 42) on V1 Macro-F1.

usage: python analyze.py <cache_dir> <out_dir> [--tuned]
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

import glob, os, sys, itertools, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
sys.path.insert(0, _HERE); sys.path.insert(0, _HERE)
from common import load_all
from lib import *

cache_dir, OUT = sys.argv[1], sys.argv[2]; tuned = '--tuned' in sys.argv
os.makedirs(OUT, exist_ok=True)
train, val, test, _ = load_all()
pool = pd.concat([train, val, test], ignore_index=True)
y = pool['label_id'].values
labels = pool[['label_id', 'label']].drop_duplicates().sort_values('label_id')['label'].tolist()
crop = np.array([l.split('_')[0] for l in labels])
TAUS = [0.25, 0.4, 0.6, 1.0, 1.6, 2.5, 4.0]
LAMS = [1.0, 0.5, 0.25, 0.1]

rows, man_rows, plat_rows, inst_rows, rej_rows, cal_rows, meta_rows = [], [], [], [], [], [], []
files = sorted(glob.glob(f'{cache_dir}/cache_*.npz'))
for fn in files:
    s = os.path.basename(fn)[6:-4]; d = np.load(fn)
    iva, ite = d['iva'], d['ite']; yv_all, yt = y[iva], y[ite]
    v1, v2 = train_test_split(np.arange(len(iva)), test_size=0.5, stratify=yv_all, random_state=0)
    Pv = [d[f'Pval_{g}'] for g in G]; Pt = [d[f'Pte_{g}'] for g in G]
    Pv1 = [P[v1] for P in Pv]; Pv2 = [P[v2] for P in Pv]; y1, y2 = yv_all[v1], yv_all[v2]
    base_pred = d['pred_base']
    if tuned:
        refs = {'solo': np.array([mf1(yt, d[f'pred_{g}']) for g in G])}
    else:
        v = {(): 0.0}
        for r in (1, 2, 3):
            for c in itertools.combinations(G, r):
                v[c] = mf1(yt, d['pred_' + '+'.join(c)])
        refs = references(v)
    solo = refs['solo']
    sharp = [P.max(1).mean() for P in Pt]
    meta = dict(split=s, base_f1=mf1(yt, base_pred), base_acc=(base_pred == yt).mean() * 100,
                **{f'solo_{g}': x for g, x in zip(G, solo)}, **{f'sharp_{g}': x for g, x in zip(G, sharp)},
                **{f'ent_{g}': entropy(P) for g, P in zip(G, Pt)})
    for rn, rv in refs.items():
        meta.update({f'{rn}_{g}': x for g, x in zip(G, rv)}); meta[f'order_{rn}'] = order_str(rv)
    if 'nsv_base' in d.files:
        meta['nsv_base'] = int(d['nsv_base'])
    if tuned:
        meta.update({f'hp_{g}': str(d[f'hp_{g}'].tolist()) for g in G})

    def add(att, obj, fused_f1):
        for an, av in att.items():
            r = dict(split=s, objective=obj, attribution=an, fused_f1=fused_f1, order=order_str(av),
                     **{f'v_{g}': x for g, x in zip(G, av)})
            for rn, rv in refs.items():
                r[f'ok_{rn}'] = order_ok(av, rv); r[f'rho_{rn}'] = rho(av, rv)
            rows.append(r)

    # ---- primary fusion
    sc = search_scores(Pv1, y1); a = CANDS[int(sc.argmax())]
    Fte = fuse(a, Pt); yhat = Fte.argmax(1)
    meta.update(fused_f1=mf1(yt, yhat), fused_acc=(yhat == yt).mean() * 100,
                uniform_f1=mf1(yt, fuse(np.ones(3) / 3, Pt).argmax(1)), v1_best=sc.max(),
                **{f'alpha_{g}': x for g, x in zip(G, a)})
    add(all_attributions(a, Pt, yt, base_pred, with_shapley=True), 'macroF1', meta['fused_f1'])
    if not tuned:
        # ---- smooth objectives
        for obj in ('nll', 'brier'):
            aa = w_convex(Pv1, y1, obj)
            add(all_attributions(aa, Pt, yt, base_pred, with_shapley=False), f'convex_{obj}', mf1(yt, fuse(aa, Pt).argmax(1)))
        wl = w_loglinear(Pv1, y1); Z = sum(wi * np.log(P + 1e-9) for wi, P in zip(wl, Pt))
        add({'raw': norm(wl)}, 'logistic_stacking', mf1(yt, Z.argmax(1)))
        meta.update({f'stack_w_{g}': x for g, x in zip(G, wl)})
        # ---- plateau of the Macro-F1 objective
        for tol in (0.1, 0.2, 0.5):
            near = np.where(sc >= sc.max() - tol)[0]
            cnt = {}; nr = nm = nmar = 0
            for i in near:
                ai = CANDS[i]; o = order_str(ai); cnt[o] = cnt.get(o, 0) + 1
                nr += order_ok(ai, solo); nm += order_ok(mass_contrib(ai, Pt, yt), solo)
                nmar += order_ok(margin_contrib(ai, Pt, yt), solo)
            plat_rows.append(dict(split=s, tol=tol, n=len(near), raw_ok=nr / len(near), mass_ok=nm / len(near),
                                  margin_ok=nmar / len(near),
                                  **{f'raw_{k}': cnt.get(k, 0) / len(near) for k in ('CTS', 'CST', 'TCS', 'TSC', 'SCT', 'STC')}))
        # ---- calibration control: temperatures on V2, weights re-searched on V1
        T = [fit_temp(P, y2) for P in Pv2]
        Pv1c = [sharpen(P, t) for P, t in zip(Pv1, T)]; Ptc = [sharpen(P, t) for P, t in zip(Pt, T)]
        ac = CANDS[int(search_scores(Pv1c, y1).argmax())]
        add(all_attributions(ac, Ptc, yt, base_pred, with_shapley=True), 'calibrated', mf1(yt, fuse(ac, Ptc).argmax(1)))
        for g, t, P, Pc in zip(G, T, Pt, Ptc):
            cal_rows.append(dict(split=s, expert=g, T=t, ece_before=ece(P, yt), ece_after=ece(Pc, yt),
                                 maxp_before=P.max(1).mean(), maxp_after=Pc.max(1).mean(),
                                 ent_before=entropy(P), ent_after=entropy(Pc)))
        # ---- fixed-competence manipulations (argmax of the manipulated expert unchanged)
        def manip(kind, k, par):
            Pv1m = list(Pv1); Ptm = list(Pt)
            if kind == 'temperature':
                Pv1m[k] = sharpen(Pv1[k], par); Ptm[k] = sharpen(Pt[k], par)
            else:
                Pv1m[k] = par * Pv1[k] + (1 - par) / C; Ptm[k] = par * Pt[k] + (1 - par) / C
            am = CANDS[int(search_scores(Pv1m, y1).argmax())]
            att = all_attributions(am, Ptm, yt, base_pred, with_shapley=True)
            sh = [P.max(1).mean() for P in Ptm]
            r = dict(split=s, kind=kind, expert=G[k], par=par, solo_f1=mf1(yt, Ptm[k].argmax(1)),
                     gap=max(sh) - min(sh), fused_f1=mf1(yt, fuse(am, Ptm).argmax(1)),
                     **{f'sharp_{g}': x for g, x in zip(G, sh)})
            for an in ('raw', 'mass_true', 'margin_true', 'margin_pseudo', 'fusion_shapley'):
                r.update({f'{an}_{g}': x for g, x in zip(G, att[an])}); r[f'{an}_ok'] = order_ok(att[an], solo)
            man_rows.append(r)
        for k in (1, 2):
            for tau in TAUS:
                manip('temperature', k, tau)
        for lam in LAMS:
            manip('uniform_mix', 2, lam)
        # ---- reject option: baseline vs fused confidence, thresholds on V2
        Pbt = d['Pte_base']; Pbv2 = d['Pval_base'][v2]
        conf_b_v2 = Pbv2[np.arange(len(v2)), d['predval_base'][v2]]; conf_b_t = Pbt[np.arange(len(yt)), base_pred]
        Fv2 = fuse(a, Pv2); conf_f_v2 = Fv2.max(1); conf_f_t = Fte.max(1)
        for name, cv, ct, pred in (('baseline', conf_b_v2, conf_b_t, base_pred), ('fused', conf_f_v2, conf_f_t, yhat)):
            err = pred != yt
            for cov in (1.0, 0.95, 0.9, 0.85, 0.8):
                thr = -np.inf if cov == 1.0 else np.quantile(cv, 1 - cov)
                keep = ct >= thr
                rej_rows.append(dict(split=s, model=name, target=cov, coverage=keep.mean() * 100,
                                     acc=(pred[keep] == yt[keep]).mean() * 100, f1=mf1(yt[keep], pred[keep]),
                                     deferred=(err & ~keep).sum() / max(err.sum(), 1) * 100, errors=int(err.sum()),
                                     auroc=roc_auc_score(err.astype(int), -ct)))
        # ---- failure analysis: which expert supplies the margin of each wrong decision over the true class
        ii = np.arange(len(yt)); err = yhat != yt; cross = err & (crop[yt] != crop[yhat]); within = err & ~cross
        tw = np.stack([ai * (P[ii, yhat] - P[ii, yt]) for ai, P in zip(a, Pt)], 1)  # >0 summed for errors
        resp = tw.argmax(1)
        cshare_w = tw[:, 0] / np.maximum(tw.sum(1), 1e-12)
        # class-level: colour share of the margin for the true class vs class error rate
        mt = margin_terms(a, Pt, yt)
        cls_share = np.array([mt[yt == c, 0].sum() / mt[yt == c].sum() for c in range(C)])
        cls_err = np.array([(yhat[yt == c] != c).mean() for c in range(C)])
        from scipy.stats import spearmanr as _sp
        inst_rows.append(dict(split=s, n_err=int(err.sum()), n_cross=int(cross.sum()), n_within=int(within.sum()),
                              color_resp_cross=(resp[cross] == 0).mean() if cross.any() else np.nan,
                              color_resp_within=(resp[within] == 0).mean() if within.any() else np.nan,
                              texture_resp_err=(resp[err] == 1).mean(), shape_resp_err=(resp[err] == 2).mean(),
                              color_resp_err=(resp[err] == 0).mean(),
                              cshare_cross_median=np.median(cshare_w[cross]) if cross.any() else np.nan,
                              cshare_within_median=np.median(cshare_w[within]) if within.any() else np.nan,
                              rho_cls_share_err=_sp(cls_share, cls_err)[0]))
        share = cshare_w
        if s == 'paper':
            np.savez(f'{OUT}/paper_instance.npz', share=share, yt=yt, yhat=yhat, a=a, Pt=np.stack(Pt),
                     base_pred=base_pred, conf_b=conf_b_t, conf_f=conf_f_t, crop=crop)
    meta_rows.append(meta)
    print(s, 'ok', flush=True)

for nm, R in (('main', rows), ('manipulation', man_rows), ('plateau', plat_rows), ('instance', inst_rows),
              ('reject', rej_rows), ('calibration', cal_rows), ('meta', meta_rows)):
    if R:
        pd.DataFrame(R).to_csv(f'{OUT}/{nm}.csv', index=False)
print('written', OUT)
