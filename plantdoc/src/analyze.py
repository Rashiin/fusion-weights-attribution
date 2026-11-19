"""Attribution analysis of the PlantDoc deep experts with the paper's own functions (src/lib.py) and the
protocol of src/analyze_deep.py: V1 for the weight search, V2 for temperatures, single-stream test Macro-F1
as reference. Adds uniform flattening (lambda = 0.1) of each stream in turn, decisions held fixed.
usage: python analyze.py <paper src dir> <index.csv> <post_dir> <out_dir>"""
import glob, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

src, index, PD, OUT = sys.argv[1:5]
sys.path.insert(0, src)
import lib
y = pd.read_csv(index).label_id.values
lib.C = int(y.max()) + 1                       # lib's functions read the module-level class count
G, CANDS = lib.G, lib.CANDS
ATT = ('raw', 'mass_true', 'margin_true', 'margin_fused', 'fusion_shapley')
os.makedirs(OUT, exist_ok=True)


def v1v2(yv):
    try:
        return train_test_split(np.arange(len(yv)), test_size=0.5, stratify=yv, random_state=0)
    except ValueError:                          # a class with one validation image cannot be stratified
        return train_test_split(np.arange(len(yv)), test_size=0.5, random_state=0)


rows, meta, man = [], [], []
for fn in sorted(glob.glob(f'{PD}/deep_*.npz'), key=lambda f: int(f.split('_')[-1][:-4])):
    s = int(fn.split('_')[-1][:-4]); d = np.load(fn)
    iva, ite = d['iva'], d['ite']; yv, yt = y[iva], y[ite]
    v1, v2 = v1v2(yv); y1, y2 = yv[v1], yv[v2]
    Pv = [d[f'Pval_{g}'] / d[f'Pval_{g}'].sum(1, keepdims=True) for g in G]
    Pt = [d[f'Pte_{g}'] / d[f'Pte_{g}'].sum(1, keepdims=True) for g in G]
    Pv1 = [P[v1] for P in Pv]; Pv2 = [P[v2] for P in Pv]
    solo = np.array([lib.mf1(yt, P.argmax(1)) for P in Pt])

    def add(obj, a, Pl, extra=None):
        att = lib.all_attributions(a, Pl, yt, None, True); yhat = lib.fuse(a, Pl).argmax(1)
        for an in ATT:
            rows.append(dict(split=s, objective=obj, attribution=an, fused_f1=lib.mf1(yt, yhat),
                             order=lib.order_str(att[an]), ok=lib.order_ok(att[an], solo), rho=lib.rho(att[an], solo),
                             **{f'v_{g}': x for g, x in zip(G, att[an])}, **(extra or {})))
        return att, yhat

    a = CANDS[int(lib.search_scores(Pv1, y1).argmax())]
    att, yhat = add('macroF1', a, Pt)
    T = [lib.fit_temp(P, y2) for P in Pv2]
    Pv1c = [lib.sharpen(P, t) for P, t in zip(Pv1, T)]; Ptc = [lib.sharpen(P, t) for P, t in zip(Pt, T)]
    add('calibrated', CANDS[int(lib.search_scores(Pv1c, y1).argmax())], Ptc)
    # sign of the true-class margin terms (the failure mode seen with handcrafted experts)
    terms = lib.margin_terms(a, Pt, yt).mean(0)
    meta.append(dict(split=s, n_test=len(yt), n_v1=len(v1), fused_f1=lib.mf1(yt, yhat), fused_acc=(yhat == yt).mean() * 100,
                     solo_order=lib.order_str(solo), all_true_margin_negative=bool((terms < 0).all()),
                     **{f'solo_{g}': x for g, x in zip(G, solo)}, **{f'sharp_{g}': P.max(1).mean() for g, P in zip(G, Pt)},
                     **{f'alpha_{g}': x for g, x in zip(G, a)}, **{f'T_{g}': t for g, t in zip(G, T)}))
    # fixed-competence flattening of each stream in turn (lambda = 0.1), weights re-searched
    for k in range(3):
        lam = 0.1
        Pv1m = list(Pv1); Ptm = list(Pt)
        Pv1m[k] = lam * Pv1[k] + (1 - lam) / lib.C; Ptm[k] = lam * Pt[k] + (1 - lam) / lib.C
        am = CANDS[int(lib.search_scores(Pv1m, y1).argmax())]
        add(f'flat_{G[k]}', am, Ptm, dict(flattened=G[k], weakest=bool(k == int(np.argmin(solo)))))
    print(s, f'F1 {lib.mf1(yt, yhat):.1f}', lib.order_str(solo), {an: lib.order_str(att[an]) for an in ATT}, flush=True)

R = pd.DataFrame(rows); M = pd.DataFrame(meta)
R.to_csv(f'{OUT}/main.csv', index=False); M.to_csv(f'{OUT}/meta.csv', index=False)

f = lambda c: f'{M[c].mean():.2f} ± {M[c].std():.2f}'
print('\n=== experts and ensemble (test, mean ± sd over', len(M), 'partitions)')
for g in G:
    print(f'  {g:8s} Macro-F1 {f("solo_" + g)}   sharpness {M["sharp_" + g].mean():.3f}   weight {M["alpha_" + g].mean():.2f}')
print(f'  ensemble Macro-F1 {f("fused_f1")}  accuracy {f("fused_acc")}   |V1| = {M.n_v1.mean():.0f}  |test| = {M.n_test.mean():.0f}')
print('  single-stream orderings:', M.solo_order.value_counts().to_dict())
print('  partitions where all three true-class margin terms are negative:', int(M.all_true_margin_negative.sum()))
for obj in R.objective.unique():
    print(f'\n=== {obj}')
    sub = R[R.objective == obj]
    for an in ATT:
        q = sub[sub.attribution == an]
        print(f'  {an:15s} {int(q.ok.sum()):2d}/{len(q)}   rho {q.rho.mean():.2f} ± {q.rho.std():.2f}')
