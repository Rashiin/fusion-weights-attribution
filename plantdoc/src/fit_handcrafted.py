"""Handcrafted experts on PlantDoc: the paper's classifier (median imputation, standardization, RBF SVM with
C = 10, gamma 'scale', balanced class weights, Platt scaling by internal five-fold cross-validation) on the
color, texture and shape columns, plus the all-feature baseline, over the same 30 partitions as fit_heads.py.
Writes posteriors in the format read by analyze.py.
usage: python fit_handcrafted.py <code_and_data dir> <index.csv> <feat_dir> <out_dir> [n_splits]"""
import glob, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

pkg, index, fd, out = sys.argv[1:5]
n_splits = int(sys.argv[5]) if len(sys.argv) > 5 else 30
sys.path.insert(0, os.path.join(pkg, 'src'))
from common import make_svm

y = pd.read_csv(index).label_id.values
Z = sorted((np.load(f) for f in glob.glob(f'{fd}/feat_*.npz')), key=lambda z: int(z['idx'][0]))
idx = np.concatenate([z['idx'] for z in Z]); assert (idx == np.arange(len(y))).all(), 'missing feature chunks'
X = np.concatenate([z['X'] for z in Z]); assert X.shape[1] == 87
cols = {'color': slice(0, 57), 'texture': slice(57, 79), 'shape': slice(79, 87)}
C = int(y.max()) + 1
os.makedirs(out, exist_ok=True)


def full(P, classes):
    Q = np.zeros((len(P), C)); Q[:, classes] = P
    return Q


for seed in range(n_splits):
    itr, rest = train_test_split(np.arange(len(y)), test_size=0.30, stratify=y, random_state=seed)
    iva, ite = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    res = dict(itr=itr, iva=iva, ite=ite)
    for g, sl in list(cols.items()) + [('base', slice(0, 87))]:
        m = make_svm(probability=True).fit(X[itr, sl], y[itr])
        cl = m.named_steps['clf'].classes_
        res[f'Pval_{g}'] = full(m.predict_proba(X[iva, sl]), cl)
        res[f'Pte_{g}'] = full(m.predict_proba(X[ite, sl]), cl)
    np.savez_compressed(f'{out}/deep_{seed}.npz', **res)
    print(seed, 'ok', flush=True)
