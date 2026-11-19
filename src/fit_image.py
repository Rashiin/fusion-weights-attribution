"""Image-level controls on the handcrafted pipeline (needs results/image_controls/image_controls_*.npz, written by image_controls.py).

mode 'group'      : near-duplicate-aware partitions. Every near-duplicate group is assigned to a
                    single split (70/15/15, groups shuffled within their majority class, seed s);
                    then the full pipeline is fitted and cached in the standard cache format.
mode 'background' : colour features recomputed on the leaf only (background removed) replace the
                    colour group; full pipeline fitted on the paper partition and re-splits.
mode 'bgonly'     : test Macro-F1 of a colour SVM trained on background-only colour features
                    (and on leaf-only / original features) per partition -> csv.
usage: python fit_image.py <npz_dir> <mode> <seeds...>
"""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

import sys, os, time, itertools, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, _HERE)
from common import GROUPS, feature_matrix, load_all, make_svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

NPZ, MODE = sys.argv[1], sys.argv[2]
train, val, test, columns = load_all()
pooled = pd.concat([train, val, test], ignore_index=True)
y = pooled['label_id'].values; n_tr, n_va = len(train), len(val)
rel = np.array([Path(p).parts[-2] + '/' + Path(p).parts[-1] for p in pooled['path']])
ic = {s: np.load(f'{NPZ}/image_controls_{s}.npz') for s in ('train', 'val', 'test')}
irel = np.concatenate([ic[s]['rel'] for s in ('train', 'val', 'test')])
assert (irel == rel).all(), 'image-control order differs from the feature order'
group = np.concatenate([ic[s]['group'] for s in ('train', 'val', 'test')])
leaf = np.concatenate([ic[s]['color_leaf'] for s in ('train', 'val', 'test')]).astype(float)
bg = np.concatenate([ic[s]['color_bg'] for s in ('train', 'val', 'test')]).astype(float)
X = {g: feature_matrix(pooled, columns, (g,)) for g in GROUPS}
if MODE == 'background':
    X['color'] = leaf
OUTDIR = {'group': f'{_ROOT}/cache/cache_group', 'background': f'{_ROOT}/cache/cache_leaf'}.get(MODE)
if OUTDIR:
    os.makedirs(OUTDIR, exist_ok=True)


def split_idx(seed):
    idx = np.arange(len(y))
    if MODE != 'group':
        if seed == 'paper':
            return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
        a, b = train_test_split(idx, test_size=0.30, stratify=y, random_state=int(seed))
        v, t = train_test_split(b, test_size=0.50, stratify=y[b], random_state=int(seed))
        return a, v, t
    rng = np.random.default_rng(int(seed))
    gl = pd.DataFrame({'g': group, 'y': y}).groupby('g')['y'].agg(lambda s: s.mode().iloc[0])
    sizes = pd.Series(group).value_counts()
    assign = {}
    for c in range(15):
        gs = gl[gl == c].index.to_numpy().copy(); rng.shuffle(gs)
        tot = sizes[gs].sum(); run = 0
        for gid in gs:
            frac = run / tot
            assign[gid] = 0 if frac < 0.70 else (1 if frac < 0.85 else 2)
            run += sizes[gid]
    lab = np.array([assign[g] for g in group])
    return idx[lab == 0], idx[lab == 1], idx[lab == 2]


rows = []
for s in sys.argv[3:]:
    t0 = time.time(); itr, iva, ite = split_idx(s)
    if MODE == 'bgonly':
        r = dict(split=s)
        for name, F in (('original', X['color']), ('leaf_only', leaf), ('background_only', bg)):
            p = make_svm().fit(F[itr], y[itr]).predict(F[ite]); r[name] = f1_score(y[ite], p, average='macro') * 100
        rows.append(r); print(s, r, flush=True); continue
    Xall = np.hstack([X[g] for g in GROUPS])
    out = dict(itr=itr, iva=iva, ite=ite)
    m = make_svm(probability=True).fit(Xall[itr], y[itr])
    out['Pval_base'] = m.predict_proba(Xall[iva]); out['Pte_base'] = m.predict_proba(Xall[ite])
    out['predval_base'] = m.predict(Xall[iva]); out['pred_base'] = m.predict(Xall[ite])
    for g in GROUPS:
        m = make_svm(probability=True).fit(X[g][itr], y[itr])
        out[f'Pval_{g}'] = m.predict_proba(X[g][iva]); out[f'Pte_{g}'] = m.predict_proba(X[g][ite]); out[f'pred_{g}'] = m.predict(X[g][ite])
    for combo in itertools.combinations(GROUPS, 2):
        Xc = np.hstack([X[g] for g in combo]); out['pred_' + '+'.join(combo)] = make_svm().fit(Xc[itr], y[itr]).predict(Xc[ite])
    out['pred_' + '+'.join(GROUPS)] = out['pred_base']
    np.savez_compressed(f'{OUTDIR}/cache_{s}.npz', **out)
    print(s, 'done', round(time.time() - t0), 's', flush=True)
if rows:
    pd.DataFrame(rows).to_csv(f'{_ROOT}/results/background_only.csv', index=False)
