"""Frozen ImageNet MobileNetV2 (width 1.0) in NumPy, for the deep-expert control.

PyTorch is not needed: the publicly released weights of
https://github.com/d-li14/mobilenetv2.pytorch (pretrained/mobilenetv2_1.0-0c6065bc.pth, 72.19% ImageNet top-1)
are read from the legacy PyTorch serialization format with the standard library, batch normalization is folded
into the convolutions, and the network is evaluated in float32 (NHWC layout) up to the 1,280-dimensional
global-average-pooled embedding.
"""
import pickle
import struct
from collections import OrderedDict

import numpy as np

_DT = {'FloatStorage': (np.float32, 4), 'LongStorage': (np.int64, 8), 'IntStorage': (np.int32, 4),
       'DoubleStorage': (np.float64, 8), 'HalfStorage': (np.float16, 2)}


class _Storage:
    def __init__(self, typ, key, size):
        self.typ, self.key, self.size, self.data = typ, key, size, None


class _Tensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)

    def array(self):
        dt, isz = _DT[self.storage.typ]
        base = self.storage.data
        if len(self.size) == 0:
            return base[self.offset:self.offset + 1].reshape(())
        return np.lib.stride_tricks.as_strided(base[self.offset:], shape=self.size,
                                               strides=[s * isz for s in self.stride]).copy()


def _rebuild_tensor(storage, offset, size, stride, *rest):
    return _Tensor(storage, offset, size, stride)


class _Unpickler(pickle.Unpickler):
    def __init__(self, f, storages):
        super().__init__(f); self.storages = storages

    def find_class(self, module, name):
        if module == 'torch._utils' and name in ('_rebuild_tensor_v2', '_rebuild_tensor'):
            return _rebuild_tensor
        if module == 'torch' and name.endswith('Storage'):
            return name
        if module == 'collections' and name == 'OrderedDict':
            return OrderedDict
        raise pickle.UnpicklingError(f'refusing to load {module}.{name}')

    def persistent_load(self, pid):
        typ, cls, key, location, size, view = pid
        if key not in self.storages:
            self.storages[key] = _Storage(cls if isinstance(cls, str) else str(cls), key, size)
        return self.storages[key]


def load_legacy_state_dict(path):
    storages = {}
    with open(path, 'rb') as f:
        pickle.load(f); pickle.load(f); pickle.load(f)  # magic number, protocol version, system info
        obj = _Unpickler(f, storages).load()
        keys = pickle.load(f)
        for key in keys:
            st = storages[key]; dt, isz = _DT[st.typ]
            (n,) = struct.unpack('<q', f.read(8))
            st.data = np.frombuffer(f.read(n * isz), dtype=dt)
    if 'state_dict' in obj:
        obj = obj['state_dict']
    return OrderedDict((k.replace('module.', ''), v.array()) for k, v in obj.items())


CFG = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]


def _fold(sd, conv, bn, eps=1e-5):
    w = sd[conv + '.weight'].astype(np.float32)
    g, b, m, v = (sd[bn + s].astype(np.float32) for s in ('.weight', '.bias', '.running_mean', '.running_var'))
    scale = g / np.sqrt(v + eps)
    return w * scale[:, None, None, None], (b - m * scale).astype(np.float32)


class MobileNetV2:
    """Inference-only MobileNetV2; input NHWC float32 (ImageNet-normalized RGB), output N x 1280."""

    def __init__(self, path):
        sd = load_legacy_state_dict(path)
        self.layers = []
        w, b = _fold(sd, 'features.0.0', 'features.0.1')
        self.stem = (w.transpose(2, 3, 1, 0).reshape(27, -1), b)  # (kh*kw*cin, cout)
        idx, cin = 1, 32
        for t, c, n, s in CFG:
            for i in range(n):
                stride = s if i == 0 else 1
                p = f'features.{idx}.conv.'
                blk = dict(stride=stride, identity=(stride == 1 and cin == c))
                if t == 1:
                    dw = _fold(sd, p + '0', p + '1'); pw = _fold(sd, p + '3', p + '4')
                    blk.update(expand=None, dw=(dw[0][:, 0].transpose(1, 2, 0), dw[1]), proj=(pw[0][:, :, 0, 0].T, pw[1]))
                else:
                    ex = _fold(sd, p + '0', p + '1'); dw = _fold(sd, p + '3', p + '4'); pw = _fold(sd, p + '6', p + '7')
                    blk.update(expand=(ex[0][:, :, 0, 0].T, ex[1]), dw=(dw[0][:, 0].transpose(1, 2, 0), dw[1]),
                               proj=(pw[0][:, :, 0, 0].T, pw[1]))
                self.layers.append(blk); idx += 1; cin = c
        w, b = _fold(sd, 'conv.0', 'conv.1')
        self.head = (w[:, :, 0, 0].T, b)
        self.fc = (sd['classifier.weight'].astype(np.float32).T, sd['classifier.bias'].astype(np.float32))

    @staticmethod
    def _pw(x, wb, act=True):
        n, h, w_, c = x.shape
        y = x.reshape(-1, c) @ wb[0] + wb[1]
        if act:
            np.clip(y, 0, 6, out=y)
        return y.reshape(n, h, w_, -1)

    @staticmethod
    def _dw(x, wb, stride):
        k, b = wb  # k: 3 x 3 x C
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        n, h, w_, c = x.shape
        ho, wo = (h - 1) // stride + 1, (w_ - 1) // stride + 1
        y = np.zeros((n, ho, wo, c), np.float32); tmp = np.empty_like(y)
        for i in range(3):
            for j in range(3):
                np.multiply(xp[:, i:i + stride * (ho - 1) + 1:stride, j:j + stride * (wo - 1) + 1:stride, :], k[i, j], out=tmp)
                y += tmp
        y += b
        np.clip(y, 0, 6, out=y)
        return y

    def _stem(self, x):
        n, h, w_, _ = x.shape
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
        ho, wo = h // 2, w_ // 2
        cols = [xp[:, i:i + 2 * (ho - 1) + 1:2, j:j + 2 * (wo - 1) + 1:2, :] for i in range(3) for j in range(3)]
        patches = np.concatenate(cols, axis=3)  # n x ho x wo x 27, order (kh, kw, cin)
        y = patches.reshape(-1, 27) @ self.stem[0] + self.stem[1]
        np.clip(y, 0, 6, out=y)
        return y.reshape(n, ho, wo, -1)

    def embed(self, x):
        x = self._stem(x.astype(np.float32))
        for blk in self.layers:
            inp = x
            if blk['expand'] is not None:
                x = self._pw(x, blk['expand'])
            x = self._dw(x, blk['dw'], blk['stride'])
            x = self._pw(x, blk['proj'], act=False)
            if blk['identity']:
                x = x + inp
        x = self._pw(x, self.head)
        return x.mean(axis=(1, 2))

    def logits(self, emb):
        return emb @ self.fc[0] + self.fc[1]
