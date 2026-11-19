"""Frozen-MobileNetV2 embeddings of the paper's three cue-isolating streams, for a slice of the PlantDoc index.
Stream construction and backbone are imported unchanged from the paper's code package (src/embed_streams.py,
src/mobilenetv2_numpy.py).
usage: python embed.py <paper src dir> <index.csv> <weights.pth> <out_dir> <start> <stop> [PlantDoc root]
Relative paths in index.csv are resolved against the PlantDoc root (the folder holding train/ and test/)."""
import os, sys, time
import numpy as np, pandas as pd, cv2

src, index, wpath, out, start, stop = sys.argv[1:7]
start, stop = int(start), int(stop)
root = sys.argv[7] if len(sys.argv) > 7 else ''
sys.path.insert(0, src)
from embed_streams import stream_color, stream_texture, stream_shape, to_input
from mobilenetv2_numpy import MobileNetV2

cv2.setNumThreads(1)
df = pd.read_csv(index)
stop = min(stop, len(df))
net = MobileNetV2(wpath)
emb = {s: [] for s in ('color', 'texture', 'shape')}
t0, B = time.time(), 16
for b0 in range(start, stop, B):
    batch = {s: [] for s in emb}
    for f in df.file.values[b0:min(b0 + B, stop)]:
        bgr = cv2.imread(os.path.join(root, f))
        if bgr.shape[:2] != (256, 256):
            bgr = cv2.resize(bgr, (256, 256), interpolation=cv2.INTER_AREA)
        batch['color'].append(to_input(stream_color(bgr)))
        batch['texture'].append(to_input(stream_texture(bgr)))
        batch['shape'].append(to_input(stream_shape(bgr)))
    for s in emb:
        emb[s].append(net.embed(np.stack(batch[s])))
    done = min(b0 + B, stop) - start
    print(f'{done}/{stop - start} {done / (time.time() - t0):.1f} img/s', flush=True)
os.makedirs(out, exist_ok=True)
np.savez_compressed(f'{out}/emb_{start}_{stop}.npz', idx=np.arange(start, stop),
                    **{s: np.concatenate(v).astype(np.float32) for s, v in emb.items()})
print('done', round(time.time() - t0), 's', flush=True)
