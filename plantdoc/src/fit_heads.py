"""Linear heads on the deep embeddings, 30 stratified 70/15/15 partitions (seeds 0-29), as in the paper's deep
control: StandardScaler + multinomial logistic regression per stream, fitted on the training split.
Inverse regularization strength from the environment variable HEAD_C (default 1); select_heads.py picks it per stream on V1.
usage: python fit_heads.py <index.csv> <emb_dir> <out_dir> [n_splits]"""
import glob, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

index, ed, out = sys.argv[1:4]
n_splits = int(sys.argv[4]) if len(sys.argv) > 4 else 30
os.makedirs(out, exist_ok=True)
y = pd.read_csv(index).label_id.values
Z = sorted((np.load(f) for f in glob.glob(f'{ed}/emb_*.npz')), key=lambda z: int(z['idx'][0]))
idx = np.concatenate([z['idx'] for z in Z]); assert (idx == np.arange(len(y))).all(), 'missing embedding chunks'
X = {s: np.concatenate([z[s] for z in Z]) for s in ('color', 'texture', 'shape')}
C = int(y.max()) + 1
for seed in range(n_splits):
    all_i = np.arange(len(y))
    itr, irest = train_test_split(all_i, test_size=0.30, stratify=y, random_state=seed)
    iva, ite = train_test_split(irest, test_size=0.50, stratify=y[irest], random_state=seed)
    res = dict(itr=itr, iva=iva, ite=ite)
    for s, Xs in X.items():
        head = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=float(os.environ.get("HEAD_C", "1"))))
        head.fit(Xs[itr], y[itr])
        full = lambda P: np.eye(C)[:, head.classes_] @ P.T  # map to all C columns (all classes are in train)
        res[f'Pval_{s}'] = full(head.predict_proba(Xs[iva])).T
        res[f'Pte_{s}'] = full(head.predict_proba(Xs[ite])).T
    np.savez_compressed(f'{out}/deep_{seed}.npz', **res)
    print(seed, 'ok', flush=True)
