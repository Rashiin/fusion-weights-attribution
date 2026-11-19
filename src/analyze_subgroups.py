"""Five-subgroup control (K=5): attributions against retraining references with five experts.
Same protocol as analyze.py: V split once 50/50 (seed 0) into V1 (weights) and V2 (unused here);
weights = best of 4,000 Dirichlet(1,...,1) candidates (seed 42) on V1 Macro-F1.
Usage: python analyze_subgroups.py <src_dir> <cache_dir> <out_dir>"""
import sys, os, glob, math, itertools, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.model_selection import train_test_split
SRC, CACHE, OUT = sys.argv[1:4]
sys.path.insert(0, SRC)
from common import load_all
from lib import mf1, fuse, search_scores, norm, mass_contrib, margin_contrib, margin_pred, fusion_shapley

SUB = ('rgb', 'hsv', 'glcm', 'lbp', 'shape'); K = len(SUB)
CANDS5 = np.random.default_rng(42).dirichlet(np.ones(K), size=4000)
LAMS = [1.0, 0.5, 0.25, 0.1]
os.makedirs(OUT, exist_ok=True)
train, val, test, _ = load_all()
y = pd.concat([train, val, test], ignore_index=True)['label_id'].values

def references(v):
    solo = np.array([v[(g,)] for g in SUB]); full = v[SUB]
    logo = np.array([full - v[tuple(h for h in SUB if h != g)] for g in SUB])
    shap = []
    for g in SUB:
        oth = [h for h in SUB if h != g]; phi = 0.0
        for r in range(K):
            for S in itertools.combinations(oth, r):
                w = math.factorial(r) * math.factorial(K - 1 - r) / math.factorial(K)
                phi += w * (v[tuple(h for h in SUB if h in S or h == g)] - v[tuple(h for h in SUB if h in S)])
        shap.append(phi)
    return {'solo': solo, 'logo': logo, 'shapley': np.array(shap)}

def order(v): return '>'.join(SUB[i] for i in np.argsort(-np.asarray(v)))
def pair_agree(v, ref):
    v, ref = np.asarray(v), np.asarray(ref)
    return float(np.mean([(v[i] > v[j]) == (ref[i] > ref[j]) for i, j in itertools.combinations(range(K), 2)]))
def atts(a, Pl, yt):
    return {'raw': norm(a), 'mass_true': mass_contrib(a, Pl, yt), 'margin_true': margin_contrib(a, Pl, yt),
            'margin_fused': margin_pred(a, Pl), 'fusion_shapley': fusion_shapley(a, Pl, yt)}
def flatten(P, lam): return lam * P + (1 - lam) / P.shape[1]

meta_rows, rows, man_rows = [], [], []
for fn in sorted(glob.glob(f'{CACHE}/cache_*.npz'), key=lambda f: int(os.path.basename(f)[6:-4])):
    s = os.path.basename(fn)[6:-4]; d = np.load(fn)
    iva, ite = d['iva'], d['ite']; yv, yt = y[iva], y[ite]
    v1, _v2 = train_test_split(np.arange(len(iva)), test_size=0.5, stratify=yv, random_state=0)
    Pv1 = [d[f'Pval_{g}'][v1] for g in SUB]; Pt = [d[f'Pte_{g}'] for g in SUB]; y1 = yv[v1]
    v = {(): 0.0}
    for r in range(1, K + 1):
        for c in itertools.combinations(SUB, r):
            v[c] = mf1(yt, d['pred_' + '+'.join(c)])
    refs = references(v)
    sc = search_scores(Pv1, y1, CANDS5); a = CANDS5[int(sc.argmax())]
    yhat = fuse(a, Pt).argmax(1)
    meta = dict(split=s, base_f1=v[SUB], fused_f1=mf1(yt, yhat), uniform_f1=mf1(yt, fuse(np.ones(K) / K, Pt).argmax(1)))
    for i, g in enumerate(SUB):
        meta[f'alpha_{g}'] = a[i]; meta[f'sharp_{g}'] = Pt[i].max(1).mean()
        for rn in refs: meta[f'{rn}_{g}'] = refs[rn][i]
    for rn in refs: meta[f'order_{rn}'] = order(refs[rn])
    meta_rows.append(meta)
    for an, av in atts(a, Pt, yt).items():
        r = dict(split=s, attribution=an, order=order(av), **{f'v_{g}': x for g, x in zip(SUB, av)})
        for rn, rv in refs.items():
            r[f'ok_{rn}'] = bool(np.array_equal(np.argsort(-av), np.argsort(-rv)))
            r[f'rho_{rn}'] = float(spearmanr(av, rv)[0]); r[f'tau_{rn}'] = float(kendalltau(av, rv)[0])
            r[f'pair_{rn}'] = pair_agree(av, rv)
        rows.append(r)
    # sharpness manipulation: flatten one expert toward uniform (decisions of that expert unchanged), re-search weights
    for k, g in enumerate(SUB):
        for lam in LAMS:
            Pv_m = [flatten(P, lam) if i == k else P for i, P in enumerate(Pv1)]
            Pt_m = [flatten(P, lam) if i == k else P for i, P in enumerate(Pt)]
            sc = search_scores(Pv_m, y1, CANDS5); am = CANDS5[int(sc.argmax())]
            r = dict(split=s, expert=g, lam=lam, fused_f1=mf1(yt, fuse(am, Pt_m).argmax(1)), sharp=Pt_m[k].max(1).mean())
            for an, av in atts(am, Pt_m, yt).items():
                r[f'{an}_self'] = av[k]
                r[f'{an}_ok'] = bool(np.array_equal(np.argsort(-av), np.argsort(-refs['solo'])))
                r[f'{an}_rho'] = float(spearmanr(av, refs['solo'])[0]); r[f'{an}_pair'] = pair_agree(av, refs['solo'])
            man_rows.append(r)
    print(s, 'ok', flush=True)
pd.DataFrame(meta_rows).to_csv(f'{OUT}/meta.csv', index=False)
pd.DataFrame(rows).to_csv(f'{OUT}/main.csv', index=False)
pd.DataFrame(man_rows).to_csv(f'{OUT}/manipulation.csv', index=False)
