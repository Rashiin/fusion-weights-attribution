"""
Export the class posteriors of the three deep experts (linear heads on the cached frozen
MobileNetV2 embeddings) so the revised attribution analysis can be run with exactly the
protocol used for the handcrafted experts. No torch and no images are needed: the
embeddings cached by run_deep_moe.py are read directly.

Same heads as run_deep_moe.fit_expert (StandardScaler + LogisticRegression(max_iter=1000,
tol=1e-3, C=1.0, random_state=42)), same partitions (the paper split, then stratified
70/15/15 re-splits with seeds 0..N-1, identical to the handcrafted re-splits because both
pools are ordered by path within train/val/test).

Usage (from src/):
    python export_deep_posteriors.py --cache ../cache/deep_embeddings --splits 30
Output: ../cache/deep_posteriors/deep_<partition>.npz (one small file per partition) and
        ../cache/deep_posteriors/deep_sweep_paper.npz
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "cache" / "deep_posteriors"
STREAMS = ("color", "texture", "shape")
SEED = 42


def find_splits():
    for cand in (HERE / "data" / "splits", HERE.parent / "data" / "splits"):
        if (cand / "train.csv").exists():
            return cand
    raise FileNotFoundError("data/splits/train.csv not found next to or above this script")


def load_pool():
    frames = []
    for split in ("train", "val", "test"):
        f = pd.read_csv(find_splits() / f"{split}.csv").sort_values("path")
        f["split"] = split
        frames.append(f)
    pool = pd.concat(frames, ignore_index=True)
    pool["rel"] = [Path(p).parts[-2] + "/" + Path(p).parts[-1] for p in pool["path"]]
    return pool


def load_embeddings(cache, pool):
    emb = {}
    for s in STREAMS:
        z = np.load(Path(cache) / f"deep_moe_embeddings_{s}.npz", allow_pickle=True)
        rel = [Path(str(p)).parts[-2] + "/" + Path(str(p)).parts[-1] for p in z["paths"]]
        if len(rel) != len(pool) or rel != pool["rel"].tolist():
            raise ValueError(f"cached '{s}' embeddings are not in the pool order")
        emb[s] = np.asarray(z["X"], dtype=np.float64)
    return emb


def fit_expert(Xtr, ytr, Xrest):
    scaler = StandardScaler().fit(Xtr)
    head = LogisticRegression(max_iter=1000, tol=1e-3, C=1.0, random_state=SEED).fit(scaler.transform(Xtr), ytr)
    return [head.predict_proba(scaler.transform(X)) for X in Xrest]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="folder holding deep_moe_embeddings_*.npz")
    ap.add_argument("--splits", type=int, default=30)
    ap.add_argument("--budgets", default="16,64,256,1280")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pool = load_pool()
    y = pool["label_id"].to_numpy()
    emb = load_embeddings(args.cache, pool)
    (OUT / "environment.json").write_text(json.dumps(dict(
        python=sys.version.split()[0], platform=platform.platform(), numpy=np.__version__,
        sklearn=sklearn.__version__, pandas=pd.__version__), indent=2))
    parts = [("paper", np.where(pool["split"] == "train")[0], np.where(pool["split"] == "val")[0],
              np.where(pool["split"] == "test")[0])]
    for seed in range(args.splits):
        idx = np.arange(len(y))
        tr, rest = train_test_split(idx, train_size=0.70, stratify=y, random_state=seed)
        va, te = train_test_split(rest, train_size=0.50, stratify=y[rest], random_state=seed)
        parts.append((str(seed), tr, va, te))
    for name, tr, va, te in parts:
        t0 = time.time()
        out = dict(itr=tr.astype(np.int32), iva=va.astype(np.int32), ite=te.astype(np.int32))
        for s in STREAMS:
            pv, pt = fit_expert(emb[s][tr], y[tr], [emb[s][va], emb[s][te]])
            out[f"Pval_{s}"] = pv.astype(np.float32)
            out[f"Pte_{s}"] = pt.astype(np.float32)
        np.savez_compressed(OUT / f"deep_{name}.npz", **out)
        print(f"partition {name}: {time.time() - t0:.0f} s", flush=True)
    name, tr, va, te = parts[0]
    rng = np.random.default_rng(SEED)
    order = {s: rng.permutation(emb[s].shape[1]) for s in STREAMS}
    budgets = [int(b) for b in args.budgets.split(",")]
    sweep = dict(budgets=np.array(budgets))
    for s in STREAMS:
        for d in budgets:
            cols = order[s][:d]
            pv, pt = fit_expert(emb[s][tr][:, cols], y[tr], [emb[s][va][:, cols], emb[s][te][:, cols]])
            sweep[f"Pval_{s}_{d}"] = pv.astype(np.float32)
            sweep[f"Pte_{s}_{d}"] = pt.astype(np.float32)
            sweep[f"solo_{s}_{d}"] = np.array(f1_score(y[te], pt.argmax(1), average="macro") * 100)
        print(f"sweep heads fitted for stream '{s}'", flush=True)
    np.savez_compressed(OUT / "deep_sweep_paper.npz", **sweep)
    print(f"done; files in {OUT}")


if __name__ == "__main__":
    main()
