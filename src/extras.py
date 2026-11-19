# -*- coding: utf-8 -*-
"""Build extra.json: text fragments for the robustness controls whose data may arrive later."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

import json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest

OUT = f'{_ROOT}/results'
X = {}


def okcount(df, obj, an, ref='solo'):
    g = df[(df.objective == obj) & (df.attribution == an) & (df.split != 'paper')]
    return int(g[f'ok_{ref}'].sum()), len(g)


def mcn(df, obj, an, ref='solo'):
    b = df[(df.objective == obj) & (df.attribution == 'raw') & (df.split != 'paper')].set_index('split')[f'ok_{ref}']
    e = df[(df.objective == obj) & (df.attribution == an) & (df.split != 'paper')].set_index('split')[f'ok_{ref}'].loc[b.index]
    w = int((~b.astype(bool) & e.astype(bool)).sum()); l = int((b.astype(bool) & ~e.astype(bool)).sum())
    return binomtest(w, w + l, 0.5, alternative='greater').pvalue if w + l else 1.0


def p(x): return 'p < 10^{−4}' if x < 1e-4 else f'p = {x:.4f}' if x < 0.01 else f'p = {x:.3f}'


# ---- tuned hyperparameters
if os.path.exists(f'{OUT}/tuned/main.csv'):
    T = pd.read_csv(f'{OUT}/tuned/main.csv', dtype={'split': str}); TM = pd.read_csv(f'{OUT}/tuned/meta.csv', dtype={'split': str})
    TMr = TM[TM.split != 'paper']
    kr, n = okcount(T, 'macroF1', 'raw'); km, _ = okcount(T, 'macroF1', 'margin_true'); kp, _ = okcount(T, 'macroF1', 'margin_pseudo')
    hp = {g: TMr[f'hp_{g}'].value_counts().idxmax() for g in ('color', 'texture', 'shape')}
    X['tuned_text'] = (
        f"**Tuned hyperparameters.** When *C* and the kernel width are tuned per expert on V1 ({n} partitions), the experts "
        f"reach {TMr.solo_color.mean():.2f}, {TMr.solo_texture.mean():.2f} and {TMr.solo_shape.mean():.2f}% Macro-F1 and their "
        f"sharpness is {TMr.sharp_color.mean():.3f}, {TMr.sharp_texture.mean():.3f} and {TMr.sharp_shape.mean():.3f}, so the "
        f"sharpness gap is not an artifact of fixed hyperparameters. The raw weights are consistent with the single-group "
        f"criterion in {kr}/{n} partitions and the margin contribution in {km}/{n} ({kp}/{n} without labels; exact McNemar "
        f"{p(mcn(T, 'macroF1', 'margin_true'))}).")
    X['tuned'] = dict(raw=kr, margin=km, pseudo=kp, n=n)

# ---- image-level controls
if os.path.exists(f'{OUT}/image/summary.json'):
    I = json.load(open(f'{OUT}/image/summary.json'))
    X['image_controls_method'] = I['method']
    X['image_text'] = I['text']
    X['dup_sentence'] = I.get('dup_sentence', '')
    X.setdefault('supp_extra', []).extend(I.get('supp', []))

# ---- deep experts
if os.path.exists(f'{OUT}/deep/summary.json'):
    D = json.load(open(f'{OUT}/deep/summary.json'))
    X['deep_method'] = D['method']; X['deep_text'] = D['text']; X['deep_env'] = D.get('env', '')
    X.setdefault('supp_extra', []).extend(D.get('supp', []))

json.dump(X, open(f'{OUT}/extra.json', 'w'), indent=1, ensure_ascii=False)
print('extra.json keys:', list(X))
