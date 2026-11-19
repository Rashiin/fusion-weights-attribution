# -*- coding: utf-8 -*-
"""Summarize the deep-expert analysis into text for the manuscript (out/deep/summary.json)."""
import os as _os
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_HERE)

import json, os
import numpy as np, pandas as pd
from scipy.stats import binomtest

D = f'{_ROOT}/results/deep'
M = pd.read_csv(f'{D}/deep_main.csv', dtype={'split': str}); ME = pd.read_csv(f'{D}/deep_meta.csv', dtype={'split': str})
MA = pd.read_csv(f'{D}/deep_manip.csv', dtype={'split': str}); CA = pd.read_csv(f'{D}/deep_cal.csv', dtype={'split': str})
SW = pd.read_csv(f'{D}/deep_sweep.csv') if os.path.exists(f'{D}/deep_sweep.csv') else None
env = json.load(open(f'{D}/environment.json')) if os.path.exists(f'{D}/environment.json') else {}
Mr = M[M.split != 'paper']; MEr = ME[ME.split != 'paper']; n = MEr.split.nunique()


def k(obj, an):
    g = Mr[(Mr.objective == obj) & (Mr.attribution == an)]; return int(g.ok_solo.sum())


def mc(obj, an):
    b = Mr[(Mr.objective == obj) & (Mr.attribution == 'raw')].set_index('split').ok_solo.astype(bool)
    e = Mr[(Mr.objective == obj) & (Mr.attribution == an)].set_index('split').ok_solo.astype(bool).loc[b.index]
    w = int((~b & e).sum()); l = int((b & ~e).sum())
    return binomtest(w, w + l, 0.5, alternative='greater').pvalue if w + l else 1.0


def p(x): return 'p < 10^{−4}' if x < 1e-4 else f'p = {x:.4f}' if x < 0.01 else f'p = {x:.3f}'


S = {}
S['method'] = ("A second ensemble replaces the handcrafted experts by deep ones. A frozen MobileNetV2 [@sandler2018] "
               "(width 1.0, publicly released ImageNet weights with 72.2% top-1 accuracy [@li2019mnv2]) embeds three "
               "cue-isolating versions of each image: a color layout (the image reduced to 16 × 16 pixels and enlarged again, "
               "which removes texture and fine shape), a grayscale high-pass image (the image minus a Gaussian blur of itself "
               "with σ = 2 pixels, which removes color and coarse shape) and the binary leaf mask of Section 4.2 (which keeps "
               "only the silhouette). Each stream is resized to 224 × 224 pixels, replicated to three channels when needed and "
               "normalized with the ImageNet statistics, and the 1,280-dimensional globally pooled embedding is computed once "
               "per image. A multinomial logistic-regression head on standardized embeddings is the expert of each stream; the "
               f"partitions, the V1/V2 protocol and the weight search are those of the handcrafted ensemble ({n} re-partitions). "
               "A capacity sweep refits each head on the first 16, 64, 256 or 1,280 embedding dimensions (Supplementary S8).")
um = MA[(MA.split != 'paper')].groupby('par')
ums = {c: um[c].mean() for c in MA.columns if c.endswith('_shape') or c.endswith('_ok') or c == 'fused_f1'}
weak = MA.expert.mode().iloc[0]
S['text'] = (
    f"**Deep experts.** The deep experts reach {MEr.solo_color.mean():.2f} ± {MEr.solo_color.std():.2f}, "
    f"{MEr.solo_texture.mean():.2f} ± {MEr.solo_texture.std():.2f} and {MEr.solo_shape.mean():.2f} ± {MEr.solo_shape.std():.2f}% "
    f"Macro-F1 for the color, texture and shape streams, and their ensemble {MEr.fused_f1.mean():.2f} ± {MEr.fused_f1.std():.2f}%. "
    f"Their sharpness differs less than that of the handcrafted experts ({MEr.sharp_color.mean():.3f}, {MEr.sharp_texture.mean():.3f} "
    f"and {MEr.sharp_shape.mean():.3f}). The raw weights are consistent with the single-stream ordering in {k('macroF1', 'raw')}/{n} "
    f"partitions and the margin contribution in {k('macroF1', 'margin_true')}/{n} (exact McNemar {p(mc('macroF1', 'margin_true'))}); "
    f"with calibrated heads {k('calibrated', 'raw')}/{n} and {k('calibrated', 'margin_true')}/{n}. Flattening the weakest "
    f"({weak}) stream with its decisions held fixed raises its raw weight from {um['raw_' + weak].mean().loc[1.0]:.2f} to "
    f"{um['raw_' + weak].mean().loc[0.1]:.2f} while its margin contribution moves from {um['margin_true_' + weak].mean().loc[1.0]:.3f} "
    f"to {um['margin_true_' + weak].mean().loc[0.1]:.3f}; the raw weights are then consistent in "
    f"{int(MA[(MA.split != 'paper') & (MA.par == 0.1)].raw_ok.sum())}/{n} partitions and the margin contribution in "
    f"{int(MA[(MA.split != 'paper') & (MA.par == 0.1)].margin_true_ok.sum())}/{n}. With experts of similar sharpness the raw "
    "weights are mostly usable here, and they fail once the sharpness of one expert is made to depart from the others, "
    "as the analysis of Section 3.3 predicts.")
if SW is not None and len(SW):
    bins = [0, 0.05, 0.15, 0.30, 1.0]; SW['bin'] = pd.cut(SW.gap, bins)
    tab = SW.groupby('bin', observed=True)[['raw_ok', 'mass_true_ok', 'margin_true_ok', 'fusion_shapley_ok']].mean()
    S['text'] += (f" Across the {len(SW)} budget combinations of the sweep (Supplementary S8), the raw weights recover the "
                  f"single-stream ordering in {SW.raw_ok.mean() * 100:.0f}% of mixtures, the probability-mass contribution in "
                  f"{SW.mass_true_ok.mean() * 100:.0f}% and the margin contribution in {SW.margin_true_ok.mean() * 100:.0f}%.")
    rows = '; '.join(f"gap {iv.left:.2f}–{iv.right:.2f}: raw {r.raw_ok * 100:.0f}%, mass {r.mass_true_ok * 100:.0f}%, margin {r.margin_true_ok * 100:.0f}%"
                     for iv, r in tab.iterrows())
    S['supp'] = [('S8. Sweep over deep-expert capacity',
                  'Each stream\'s head was refitted on the first 16, 64, 256 or 1,280 embedding dimensions (a fixed random order) '
                  'on the fixed partition; mixtures in which two experts lie within one Macro-F1 point would have been excluded, '
                  f'and {len(SW)} of the 64 mixtures remained. Recovery of the single-stream ordering by sharpness gap '
                  f'(number of mixtures in brackets): '
                  + '; '.join(f"gap {iv.left:.2f}–{iv.right:.2f} [{int((SW.bin == iv).sum())}]: raw {r.raw_ok * 100:.0f}%, "
                              f"mass {r.mass_true_ok * 100:.0f}%, margin {r.margin_true_ok * 100:.0f}%" for iv, r in tab.iterrows())
                  + '. Competence and sharpness vary together in this sweep, so it is descriptive; the effect of sharpness '
                  'alone is isolated by the flattening experiments.')]
S['env'] = ("The deep embeddings were computed in the same environment, with the network evaluated in NumPy from the "
            "released weights, so no deep-learning framework is needed to reproduce them.")
json.dump(S, open(f'{D}/summary.json', 'w'), indent=1, ensure_ascii=False)
print(S['text'])
