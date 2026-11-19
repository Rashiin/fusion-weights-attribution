"""Hyperparameter control: per expert, C and gamma are tuned on V1 (Macro-F1) over a 3x3 grid
instead of being fixed a priori; the tuned experts are refitted with Platt scaling and cached."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

import sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
sys.path.insert(0, _HERE)
from common import GROUPS, feature_matrix, load_all
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
train, val, test, columns = load_all()
pooled = pd.concat([train, val, test], ignore_index=True); y = pooled['label_id'].values
X = {g: feature_matrix(pooled, columns, (g,)) for g in GROUPS}
def mk(Cc, gm, prob=False):
    return Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler()),
                     ('clf', SVC(kernel='rbf', C=Cc, gamma=gm, class_weight='balanced', probability=prob, random_state=42))])
for s in sys.argv[1:]:
    t0 = time.time(); d = np.load(f'{_ROOT}/cache/cache/cache_{s}.npz')
    itr, iva, ite = d['itr'], d['iva'], d['ite']
    v1, v2 = train_test_split(np.arange(len(iva)), test_size=0.5, stratify=y[iva], random_state=0)
    out = dict(itr=itr, iva=iva, ite=ite)
    for g in GROUPS:
        Xg = X[g]; base = 1.0 / Xg.shape[1]
        best = None
        for Cc in (1, 10, 100):
            for mult in (1 / 3, 1, 3):
                p = mk(Cc, base * mult).fit(Xg[itr], y[itr]).predict(Xg[iva[v1]])
                sc = f1_score(y[iva[v1]], p, average='macro')
                if best is None or sc > best[0]:
                    best = (sc, Cc, mult)
        m = mk(best[1], base * best[2], True).fit(Xg[itr], y[itr])
        out[f'Pval_{g}'] = m.predict_proba(Xg[iva]); out[f'Pte_{g}'] = m.predict_proba(Xg[ite]); out[f'pred_{g}'] = m.predict(Xg[ite])
        out[f'hp_{g}'] = np.array([best[1], best[2]])
    for k in ('Pval_base', 'Pte_base', 'pred_base', 'predval_base'):
        out[k] = d[k]
    np.savez_compressed(f'{_ROOT}/cache/cache_tuned/cache_{s}.npz', **out)
    print(s, 'tuned', round(time.time() - t0), 's', {g: out[f'hp_{g}'].tolist() for g in GROUPS}, flush=True)
