"""Fit, for one partition: Platt-scaled baseline (all 87 features), three Platt-scaled
group experts, and plain SVMs on the two-group subsets. Cache everything."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

import sys, time, itertools, warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np, pandas as pd
sys.path.insert(0, _HERE)
from common import GROUPS, feature_matrix, load_all, make_svm
from sklearn.model_selection import train_test_split

train, val, test, columns = load_all()
pooled = pd.concat([train, val, test], ignore_index=True)
y = pooled['label_id'].values
n_tr, n_va = len(train), len(val)
X = {g: feature_matrix(pooled, columns, (g,)) for g in GROUPS}
Xall = feature_matrix(pooled, columns)

def split_idx(seed):
    idx = np.arange(len(y))
    if seed == 'paper':
        return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]
    a, b = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    v, t = train_test_split(b, test_size=0.50, stratify=y[b], random_state=seed)
    return a, v, t

for s in sys.argv[1:]:
    seed = s if s == 'paper' else int(s)
    t0 = time.time()
    itr, iva, ite = split_idx(seed)
    out = dict(itr=itr, iva=iva, ite=ite)
    m = make_svm(probability=True).fit(Xall[itr], y[itr])
    out['Pval_base'] = m.predict_proba(Xall[iva]); out['Pte_base'] = m.predict_proba(Xall[ite])
    out['predval_base'] = m.predict(Xall[iva]); out['pred_base'] = m.predict(Xall[ite])
    out['nsv_base'] = np.array(m.named_steps['clf'].support_vectors_.shape[0])
    for g in GROUPS:
        m = make_svm(probability=True).fit(X[g][itr], y[itr])
        out[f'Pval_{g}'] = m.predict_proba(X[g][iva]); out[f'Pte_{g}'] = m.predict_proba(X[g][ite])
        out[f'pred_{g}'] = m.predict(X[g][ite])
    for combo in itertools.combinations(GROUPS, 2):
        Xc = np.hstack([X[g] for g in combo])
        out['pred_' + '+'.join(combo)] = make_svm().fit(Xc[itr], y[itr]).predict(Xc[ite])
    out['pred_' + '+'.join(GROUPS)] = out['pred_base']
    np.savez_compressed(f'{_ROOT}/cache/cache/cache_{s}.npz', **out)
    print(s, 'done', round(time.time() - t0), 's', flush=True)
