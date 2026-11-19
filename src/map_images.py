"""Map every image of the paper's partition to its file in the GitHub PlantVillage mirror,
and check identity by recomputing the colour moments stored in the repository's feature CSVs."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

import os, sys, glob
import numpy as np, pandas as pd, cv2, scipy.stats as st
ROOT = sys.argv[1]
REPO = _ROOT
cmap = {'Pepper__bell___Bacterial_spot': 'Pepper,_bell___Bacterial_spot', 'Pepper__bell___healthy': 'Pepper,_bell___healthy',
        'Potato___Early_blight': 'Potato___Early_blight', 'Potato___Late_blight': 'Potato___Late_blight', 'Potato___healthy': 'Potato___healthy',
        'Tomato_Bacterial_spot': 'Tomato___Bacterial_spot', 'Tomato_Early_blight': 'Tomato___Early_blight', 'Tomato_Late_blight': 'Tomato___Late_blight',
        'Tomato_Leaf_Mold': 'Tomato___Leaf_Mold', 'Tomato_Septoria_leaf_spot': 'Tomato___Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato__Target_Spot': 'Tomato___Target_Spot',
        'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus': 'Tomato___Tomato_mosaic_virus',
        'Tomato_healthy': 'Tomato___healthy'}
rows = []
for split in ('train', 'val', 'test'):
    df = pd.read_csv(f'{REPO}/data/splits/{split}.csv'); df['split'] = split
    df['file'] = [os.path.join(ROOT, cmap[l], os.path.basename(p)) for p, l in zip(df.path, df.label)]
    rows.append(df)
pool = pd.concat(rows, ignore_index=True)
pool['exists'] = [os.path.exists(f) for f in pool.file]
print('found', pool.exists.sum(), 'of', len(pool))
miss = pool[~pool.exists]
if len(miss):
    # try case-insensitive match
    idx = {}
    for d in set(cmap.values()):
        for f in os.listdir(os.path.join(ROOT, d)):
            idx[(d, f.lower())] = os.path.join(ROOT, d, f)
    for i in miss.index:
        k = (cmap[pool.at[i, 'label']], os.path.basename(pool.at[i, 'path']).lower())
        if k in idx: pool.at[i, 'file'] = idx[k]; pool.at[i, 'exists'] = True
    print('after case-insensitive', pool.exists.sum())
pool.to_csv(f'{_ROOT}/cache/image_map.csv', index=False)
# identity check on a random sample: RGB moments (first 9 colour features)
col = pd.concat([pd.read_csv(f'{REPO}/data/features/color_features_{s}.csv') for s in ('train', 'val', 'test')]).set_index('path')
rng = np.random.default_rng(0); samp = pool[pool.exists].sample(300, random_state=0)
err = []
for p, f in zip(samp.path, samp.file):
    img = cv2.resize(cv2.imread(f), (256, 256)); rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
    feats = []
    for c in range(3):
        ch = rgb[..., c].reshape(-1); feats += [ch.mean(), ch.std(), st.skew(ch)]
    ref = col.loc[p, [c for c in col.columns if c.startswith('cm_')]].values.astype(float)
    err.append(np.abs(np.array(feats) - ref).max())
err = np.array(err); print('max abs diff of colour moments over 300 images:', err.max(), 'median', np.median(err), 'n>1e-3:', (err > 1e-3).sum())
