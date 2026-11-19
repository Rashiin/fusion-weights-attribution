# -*- coding: utf-8 -*-
"""Summarize the image-level controls (near-duplicate-aware partitions, leaf-only colour, background-only colour)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

import json, os, re
import numpy as np, pandas as pd

O = f'{_ROOT}/results'
os.makedirs(f'{O}/image', exist_ok=True)
summ = open(f'{_ROOT}/results/image_controls/near_duplicate_summary.txt').read()
num = lambda key: int(re.search(key + r':\s*(\d+)', summ).group(1))
groups, covered = [int(x) for x in re.search(r'near-duplicate groups \(size>1\): (\d+) covering (\d+)', summ).groups()]
cross, mixed, leak = num('groups spanning more than one split'), num('groups with conflicting labels'), num('test images with a near-duplicate in train/val')
largest = num('largest group')
BG = pd.read_csv(f'{O}/background_only.csv', dtype={'split': str}); BGr = BG[BG.split != 'paper']
n_bg = len(BGr)


def kk(df, an, ref='solo'):
    g = df[(df.objective == 'macroF1') & (df.attribution == an) & (df.split != 'paper')]
    return int(g[f'ok_{ref}'].sum()), len(g)


S = {}
S['method'] = ("Near-duplicate photographs are found with a 64-bit perceptual hash and a 64-bit difference hash computed on "
               "grayscale thumbnails; two images are near-duplicates when both Hamming distances are at most 6 bits, and "
               "near-duplicates are merged into groups. The analysis is repeated on partitions in which every group lies "
               "entirely in one split (70/15/15 within each class). To test whether the color evidence comes from the "
               "background, the 57 color features are recomputed on the leaf only and on the background only, using a leaf "
               "mask from Otsu thresholding of saturation; a color classifier trained on background-only features measures "
               "how much class information the background carries, and the whole analysis is repeated with the leaf-only "
               "color expert.")
S['dup_sentence'] = (f"Perceptual hashing finds {groups} groups of near-duplicate photographs covering {covered} images "
                     f"(largest group {largest}; {mixed} with conflicting labels); in the fixed partition {leak} test images "
                     "have a near-duplicate in the training or validation split.")
txt = ''
if os.path.exists(f'{O}/group/main.csv'):
    Gm = pd.read_csv(f'{O}/group/main.csv', dtype={'split': str}); GM = pd.read_csv(f'{O}/group/meta.csv', dtype={'split': str})
    kr, n = kk(Gm, 'raw'); km, _ = kk(Gm, 'margin_true'); kp, _ = kk(Gm, 'margin_pseudo')
    txt += (f"**Near-duplicates.** On {n} near-duplicate-aware partitions the baseline reaches {GM.base_f1.mean():.2f} ± "
            f"{GM.base_f1.std():.2f}% Macro-F1, the raw weights are consistent with the single-group criterion in {kr}/{n} "
            f"partitions and the margin contribution in {km}/{n} ({kp}/{n} without labels). ")
txt += (f"**Background.** A color classifier trained on background pixels only reaches {BGr.background_only.mean():.2f} ± "
        f"{BGr.background_only.std():.2f}% Macro-F1 (the first {n_bg} partitions), against {BGr.leaf_only.mean():.2f} ± {BGr.leaf_only.std():.2f}% "
        f"on leaf pixels only and {BGr.original.mean():.2f} ± {BGr.original.std():.2f}% on the whole image. ")
if os.path.exists(f'{O}/leaf/main.csv'):
    Lm = pd.read_csv(f'{O}/leaf/main.csv', dtype={'split': str}); LM = pd.read_csv(f'{O}/leaf/meta.csv', dtype={'split': str})
    kr, n = kk(Lm, 'raw'); km, _ = kk(Lm, 'margin_true')
    LMr = LM[LM.split != 'paper']
    txt += (f"With the leaf-only color expert ({LMr.solo_color.mean():.2f}% Macro-F1, sharpness {LMr.sharp_color.mean():.3f}) the raw "
            f"weights are consistent in {kr}/{n} partitions and the margin contribution in {km}/{n}.")
S['text'] = txt
json.dump(S, open(f'{O}/image/summary.json', 'w'), indent=1, ensure_ascii=False)
print(txt)
