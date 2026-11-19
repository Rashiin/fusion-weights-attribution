"""Compute frozen-MobileNetV2 embeddings of the three cue-isolating streams for every image of the partition
(same stream construction as src/run_deep_moe.py; NumPy backbone, see mobilenetv2_numpy.py).
usage: python embed_streams.py <image_map.csv> <weights.pth> <out_dir> [start] [stop]"""
import sys, os, time
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mobilenetv2_numpy import MobileNetV2

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)


def stream_color(bgr, grid=16):
    small = cv2.resize(bgr, (grid, grid), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)


def stream_texture(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    high = gray - cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)
    high = high - high.min(); peak = high.max()
    high = high / peak * 255.0 if peak > 1e-6 else np.zeros_like(high)
    out = cv2.resize(high.astype(np.uint8), (IMAGE_SIZE, IMAGE_SIZE))
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def stream_shape(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (25, 25, 25), (95, 255, 255))
    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    out = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def to_input(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return (rgb - MEAN) / STD


def ordered_pool(image_map):
    m = pd.read_csv(image_map)
    parts = [m[m.split == s].sort_values('path') for s in ('train', 'val', 'test')]
    return pd.concat(parts, ignore_index=True)


if __name__ == '__main__':
    imap, wpath, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    pool = ordered_pool(imap)
    start = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    stop = int(sys.argv[5]) if len(sys.argv) > 5 else len(pool)
    net = MobileNetV2(wpath)
    B = 32; emb = {s: [] for s in ('color', 'texture', 'shape')}; t0 = time.time()
    for b0 in range(start, stop, B):
        files = pool.file.values[b0:min(b0 + B, stop)]
        batch = {s: [] for s in emb}
        for f in files:
            bgr = cv2.imread(f)
            if bgr.shape[:2] != (256, 256):
                bgr = cv2.resize(bgr, (256, 256))
            batch['color'].append(to_input(stream_color(bgr)))
            batch['texture'].append(to_input(stream_texture(bgr)))
            batch['shape'].append(to_input(stream_shape(bgr)))
        for s in emb:
            emb[s].append(net.embed(np.stack(batch[s])))
        done = min(b0 + B, stop) - start
        if (b0 - start) // B % 20 == 0:
            print(f'{done}/{stop - start} images, {done / (time.time() - t0):.1f} img/s', flush=True)
    np.savez_compressed(f'{out}/emb_{start}_{stop}.npz', paths=pool.path.values[start:stop],
                        **{s: np.concatenate(v).astype(np.float32) for s, v in emb.items()})
    print('done', time.time() - t0, flush=True)
