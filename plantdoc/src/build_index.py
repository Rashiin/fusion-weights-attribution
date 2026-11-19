"""Index the PlantDoc images (train and test folders pooled), drop unreadable files and classes too small
for a stratified 70/15/15 split with a V1/V2 halving of validation.
usage: python build_index.py <PlantDoc repo dir> <out csv> [min_per_class]"""
import os, sys
import cv2
import pandas as pd

root, out = sys.argv[1], sys.argv[2]
min_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
rows, bad = [], 0
for part in ('train', 'test'):
    for cls in sorted(os.listdir(os.path.join(root, part))):
        d = os.path.join(root, part, cls)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(EXT):
                continue
            p = os.path.join(d, f)
            img = cv2.imread(p)
            if img is None or img.size == 0:
                bad += 1
                continue
            rows.append(dict(file=p, label=cls.strip(), origin=part))
df = pd.DataFrame(rows)
counts = df.label.value_counts()
print('readable', len(df), 'unreadable', bad, 'classes', df.label.nunique())
print('dropped classes:', counts[counts < min_n].to_dict())
df = df[df.label.isin(counts[counts >= min_n].index)].reset_index(drop=True)
names = sorted(df.label.unique())
df['label_id'] = df.label.map({n: i for i, n in enumerate(names)})
df.to_csv(out, index=False)
print('kept', len(df), 'images,', len(names), 'classes; smallest class', df.label.value_counts().min())
