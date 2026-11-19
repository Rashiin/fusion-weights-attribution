"""
Image-level controls for the leaf-disease paper. Needs the raw PlantVillage images.

1. Near-duplicates: 64-bit DCT perceptual hash (pHash) and 64-bit difference hash (dHash) for
   every image; two images are near-duplicates when BOTH Hamming distances are <= HAMMING.
   Near-duplicates are merged into groups (union-find) so the analysis can be repeated with
   group-aware partitions in which no group crosses train/validation/test.
2. Background: the 57 colour features of the paper are recomputed on the leaf only and on the
   background only (Otsu threshold on HSV saturation, largest component, holes filled).

Usage (from src/):
    python image_controls.py --image-root /path/to/PlantVillage
(--image-root is the folder that contains the 15 class folders, e.g. Tomato__Target_Spot)

Output: ../results/image_controls/image_controls_{train,val,test}.npz and near_duplicate_summary.txt
Runtime: about 5-15 minutes.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "results" / "image_controls"
HAMMING = 6
HSV_BINS, EPS = 16, 1e-8
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def find_splits():
    for cand in (HERE / "data" / "splits", HERE.parent / "data" / "splits"):
        if (cand / "train.csv").exists():
            return cand
    raise FileNotFoundError("data/splits/train.csv not found next to or above this script")


def phash(gray):
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    flat = cv2.dct(small)[:8, :8].ravel()
    return np.packbits(flat > np.median(flat[1:])).view(">u8")[0]


def dhash(gray):
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return np.packbits((small[:, 1:] > small[:, :-1]).ravel()).view(">u8")[0]


def leaf_mask(bgr):
    s = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[..., 1], (5, 5), 0)
    _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    if cnts:
        cv2.drawContours(filled, [max(cnts, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
    return filled > 0


def skew(x):
    sd = x.std()
    return 0.0 if sd == 0 else float(((x - x.mean()) ** 3).mean() / sd ** 3)


def colour_features(bgr, sel):
    if sel.sum() < 50:
        sel = np.ones(sel.shape, bool)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[sel].astype(np.float64) / 255.0
    mom = []
    for c in range(3):
        ch = rgb[:, c]
        mom += [ch.mean(), ch.std(), skew(ch)]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[sel]
    hists = [np.histogram(hsv[:, i], bins=HSV_BINS, range=r)[0] for i, r in enumerate([(0, 180), (0, 256), (0, 256)])]
    return np.concatenate([mom, *[h / (h.sum() + EPS) for h in hists]]).astype(np.float32)


def popcount64(x):
    return _POP[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1)


def groups_from_hashes(ph, dh, thr):
    n = len(ph)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, 256):
        a = slice(start, min(start + 256, n))
        dp = popcount64(ph[a, None] ^ ph[None, :])
        dd = popcount64(dh[a, None] ^ dh[None, :])
        ii, jj = np.where((dp <= thr) & (dd <= thr))
        for i, j in zip(ii + start, jj):
            if i < j:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
    return np.array([find(i) for i in range(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", required=True)
    args = ap.parse_args()
    root = Path(args.image_root)
    splits = find_splits()
    OUT.mkdir(parents=True, exist_ok=True)
    frames, store = [], {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(splits / f"{split}.csv").sort_values("path").reset_index(drop=True)
        rel = [str(Path(p).parts[-2]) + "/" + str(Path(p).parts[-1]) for p in df["path"]]
        leaf, bg, frac, ph, dh = [], [], [], [], []
        for k, r in enumerate(rel):
            img = cv2.imread(str(root / r))
            if img is None:
                raise RuntimeError(f"cannot read {root / r}")
            img = cv2.resize(img, (256, 256))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ph.append(phash(gray)); dh.append(dhash(gray))
            m = leaf_mask(img); frac.append(m.mean())
            leaf.append(colour_features(img, m)); bg.append(colour_features(img, ~m))
            if k % 2000 == 0:
                print(f"{split}: {k}/{len(rel)}", flush=True)
        store[split] = dict(rel=np.array(rel), label_id=df["label_id"].to_numpy(np.int16),
                            color_leaf=np.vstack(leaf), color_bg=np.vstack(bg),
                            leaf_fraction=np.array(frac, np.float32),
                            phash=np.array(ph, np.uint64), dhash=np.array(dh, np.uint64))
        frames.append(pd.DataFrame({"rel": rel, "label": df["label"], "split": split}))
    allf = pd.concat(frames, ignore_index=True)
    ph = np.concatenate([store[s]["phash"] for s in ("train", "val", "test")])
    dh = np.concatenate([store[s]["dhash"] for s in ("train", "val", "test")])
    g = groups_from_hashes(ph, dh, HAMMING)
    allf["group"] = g
    allf["size"] = allf.groupby("group")["rel"].transform("size")
    start = 0
    for s in ("train", "val", "test"):
        n = len(store[s]["rel"])
        store[s]["group"] = g[start:start + n].astype(np.int32)
        start += n
        np.savez_compressed(OUT / f"image_controls_{s}.npz", **store[s])
    multi = allf[allf["size"] > 1]
    cross = int(multi.groupby("group")["split"].nunique().gt(1).sum())
    mixed = int(multi.groupby("group")["label"].nunique().gt(1).sum())
    leak = allf[(allf.split == "test") & allf.group.isin(allf.loc[allf.split != "test", "group"])]
    txt = (f"images: {len(allf)}\nhamming threshold: {HAMMING}\n"
           f"near-duplicate groups (size>1): {multi.group.nunique()} covering {len(multi)} images\n"
           f"largest group: {int(allf['size'].max())}\n"
           f"groups spanning more than one split: {cross}\ngroups with conflicting labels: {mixed}\n"
           f"test images with a near-duplicate in train/val: {len(leak)}\n"
           f"leaf fraction mean: {np.concatenate([store[s]['leaf_fraction'] for s in store]).mean():.3f}\n")
    (OUT / "near_duplicate_summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
