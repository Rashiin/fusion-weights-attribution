"""Choose the regularization of each head per stream and partition by negative log-likelihood on V1
(the same V1 halving as analyze.py) and write the selected posteriors in the format read by analyze.py.
usage: python select_heads.py <index.csv> <out_dir> <C=post_dir> [<C=post_dir> ...]"""
import collections, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

index, out = sys.argv[1:3]
Cs = dict(a.split('=', 1) for a in sys.argv[3:])
y = pd.read_csv(index).label_id.values
os.makedirs(out, exist_ok=True)
pick = collections.Counter()
n = len([f for f in os.listdir(next(iter(Cs.values()))) if f.startswith('deep_')])
for s in range(n):
    Z = {c: np.load(f'{d}/deep_{s}.npz') for c, d in Cs.items()}
    z0 = next(iter(Z.values())); iva = z0['iva']; yv = y[iva]
    for z in Z.values():
        assert (z['iva'] == iva).all() and (z['ite'] == z0['ite']).all()
    try:
        v1, _ = train_test_split(np.arange(len(yv)), test_size=0.5, stratify=yv, random_state=0)
    except ValueError:
        v1, _ = train_test_split(np.arange(len(yv)), test_size=0.5, random_state=0)
    res = dict(itr=z0['itr'], iva=iva, ite=z0['ite'])
    for g in ('color', 'texture', 'shape'):
        nll = {c: -np.log(Z[c][f'Pval_{g}'][v1, yv[v1]] + 1e-12).mean() for c in Cs}
        b = min(nll, key=nll.get); pick[(g, b)] += 1
        res[f'Pval_{g}'] = Z[b][f'Pval_{g}']; res[f'Pte_{g}'] = Z[b][f'Pte_{g}']
    np.savez_compressed(f'{out}/deep_{s}.npz', **res)
print(sorted(pick.items()))
