"""Five-subgroup control (K=5): fit, for one partition, Platt-scaled experts on the five feature
subgroups and plain SVMs on every non-empty subset of subgroups (31 fits) for the retraining references.
Usage: python fit_subgroups.py <src_dir> <cache_dir> seed [seed ...]"""
import sys, time, itertools, warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np, pandas as pd
SRC, CACHE = sys.argv[1], sys.argv[2]
sys.path.insert(0, SRC)
from common import load_all, make_svm
from sklearn.model_selection import train_test_split

SUB = ('rgb', 'hsv', 'glcm', 'lbp', 'shape')
PREFIX = dict(rgb='cm_', hsv='hsv_', glcm='glcm_', lbp='lbp_', shape='shape_')
train, val, test, columns = load_all()
pooled = pd.concat([train, val, test], ignore_index=True)
y = pooled['label_id'].values
allcols = [c for g in columns for c in columns[g]]
cols = {s: [c for c in allcols if c.startswith(PREFIX[s])] for s in SUB}
assert sum(len(v) for v in cols.values()) == 87, {s: len(v) for s, v in cols.items()}
X = {s: pooled[cols[s]].values.astype(float) for s in SUB}

def split_idx(seed):
    idx = np.arange(len(y))
    a, b = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    v, t = train_test_split(b, test_size=0.50, stratify=y[b], random_state=seed)
    return a, v, t

for s in sys.argv[3:]:
    seed = int(s); t0 = time.time()
    itr, iva, ite = split_idx(seed)
    out = dict(itr=itr, iva=iva, ite=ite, n_features=np.array([len(cols[g]) for g in SUB]))
    for g in SUB:
        m = make_svm(probability=True).fit(X[g][itr], y[itr])
        out[f'Pval_{g}'] = m.predict_proba(X[g][iva]); out[f'Pte_{g}'] = m.predict_proba(X[g][ite])
        out[f'pred_{g}'] = m.predict(X[g][ite])
    for r in range(2, 6):
        for combo in itertools.combinations(SUB, r):
            Xc = np.hstack([X[g] for g in combo])
            out['pred_' + '+'.join(combo)] = make_svm().fit(Xc[itr], y[itr]).predict(Xc[ite])
    np.savez_compressed(f'{CACHE}/cache_{s}.npz', **out)
    print(s, 'done', round(time.time() - t0), 's', flush=True)
