"""Merge the embedding chunks into the cache format read by export_deep_posteriors.py."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

import glob, numpy as np
fs = sorted(glob.glob(f'{_ROOT}/cache/embeddings/emb_*.npz'), key=lambda f: int(f.split('_')[-2]))
Z = [np.load(f, allow_pickle=True) for f in fs]
paths = np.concatenate([z['paths'] for z in Z])
print(len(paths), [f.split('/')[-1] for f in fs])
import os; os.makedirs(f'{_ROOT}/cache/deep_embeddings', exist_ok=True)
for s in ('color', 'texture', 'shape'):
    X = np.concatenate([z[s] for z in Z]).astype(np.float32)
    np.savez_compressed(f'{_ROOT}/cache/deep_embeddings/deep_moe_embeddings_{s}.npz', X=X, paths=paths)
    print(s, X.shape, float(X.mean()))
