"""The paper's 87 handcrafted features (color 57, texture 22, shape 8) for a slice of the PlantDoc index.
The extractors are imported unchanged from legacy/run_timing.py of the code package.
usage: python extract_handcrafted.py <code_and_data dir> <index.csv> <out_dir> <start> <stop> [PlantDoc root]"""
import os, sys, time
import numpy as np, pandas as pd

pkg, index, out, start, stop = sys.argv[1:6]
start, stop = int(start), int(stop)
root = sys.argv[6] if len(sys.argv) > 6 else ''
sys.path.insert(0, os.path.join(pkg, 'src'))
sys.path.insert(0, os.path.join(pkg, 'legacy'))
from run_timing import full_vector

df = pd.read_csv(index)
stop = min(stop, len(df))
t0 = time.time()
X = np.stack([full_vector(os.path.join(root, f)) for f in df.file.values[start:stop]]).astype(np.float64)
os.makedirs(out, exist_ok=True)
np.savez_compressed(f'{out}/feat_{start}_{stop}.npz', idx=np.arange(start, stop), X=X)
print('done', X.shape, round(time.time() - t0), 's', flush=True)
