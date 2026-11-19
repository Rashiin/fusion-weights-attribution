"""Shared functions for the analysis (attributions, fusion search, calibration, references)."""
import math, itertools
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import spearmanr

G = ('color', 'texture', 'shape')
C = 15
N_CAND = 4000
CANDS = np.random.default_rng(42).dirichlet([1, 1, 1], size=N_CAND)


def mf1(yt, yp):
    """Macro-F1 (%) over classes present in yt or yp."""
    cm = np.bincount(yt * C + yp, minlength=C * C).reshape(C, C)
    tp = np.diag(cm).astype(float); ps = cm.sum(0); rs = cm.sum(1)
    pr = np.divide(tp, ps, out=np.zeros(C), where=ps > 0)
    rc = np.divide(tp, rs, out=np.zeros(C), where=rs > 0)
    f = np.divide(2 * pr * rc, pr + rc, out=np.zeros(C), where=(pr + rc) > 0)
    return f[(rs + ps) > 0].mean() * 100


def fuse(a, Ps):
    s = sum(ai * P for ai, P in zip(a, Ps))
    return s / s.sum(1, keepdims=True)


def search_scores(Ps, yv, cands=CANDS):
    stack = np.stack(Ps)
    sc = np.empty(len(cands))
    for i in range(0, len(cands), 250):
        fused = np.einsum('ak,knc->anc', cands[i:i + 250], stack).argmax(2)
        for j, row in enumerate(fused):
            sc[i + j] = mf1(yv, row)
    return sc


def sharpen(P, tau):
    """Probability temperature P^(1/tau), renormalised; preserves each row's argmax."""
    L = np.log(P + 1e-12) / tau
    L -= L.max(1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(1, keepdims=True)


def fit_temp(P, yy):
    L = np.log(P + 1e-12)
    def nll(T):
        Z = L / T; Z = Z - Z.max(1, keepdims=True)
        return -(Z[np.arange(len(yy)), yy] - np.log(np.exp(Z).sum(1))).mean()
    return minimize_scalar(nll, bounds=(0.05, 20), method='bounded').x


def ece(P, yy, bins=15):
    conf = P.max(1); corr = P.argmax(1) == yy; e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return e


def entropy(P):
    return (-(P * np.log(P + 1e-12)).sum(1)).mean()


# ---------------- attributions ----------------
def norm(v):
    v = np.asarray(v, float)
    return v / np.abs(v).sum()


def mass_contrib(a, Pl, ref, balanced=False):
    """Probability-mass contribution: alpha_k * E[P_k(ref|x)]."""
    idx = np.arange(len(ref))
    if balanced:
        v = [ai * np.mean([P[idx, ref][ref == c].mean() for c in range(C) if (ref == c).any()]) for ai, P in zip(a, Pl)]
    else:
        v = [ai * P[idx, ref].mean() for ai, P in zip(a, Pl)]
    return norm(v)


def competitor(a, Pl, ref):
    F = fuse(a, Pl).copy(); idx = np.arange(len(ref)); F[idx, ref] = -1
    return F.argmax(1)


def margin_contrib(a, Pl, ref, balanced=False):
    """Margin contribution: alpha_k * E[P_k(ref|x) - P_k(c'|x)], c' = strongest fused competitor of ref.
    The fused margin is exactly the sum of these terms; uniform probability mass cancels."""
    idx = np.arange(len(ref)); comp = competitor(a, Pl, ref)
    if balanced:
        v = [ai * np.mean([(P[idx, ref] - P[idx, comp])[ref == c].mean() for c in range(C) if (ref == c).any()])
             for ai, P in zip(a, Pl)]
    else:
        v = [ai * (P[idx, ref] - P[idx, comp]).mean() for ai, P in zip(a, Pl)]
    return norm(v)


def margin_pred(a, Pl):
    """Per-decision margin contribution of the fused prediction over its runner-up (label-free)."""
    F = fuse(a, Pl); top2 = np.argsort(-F, 1)[:, :2]; idx = np.arange(len(F))
    return norm([ai * (P[idx, top2[:, 0]] - P[idx, top2[:, 1]]).mean() for ai, P in zip(a, Pl)])


def margin_terms(a, Pl, ref):
    """Per-instance margin terms (n x k) for class ref against its fused competitor."""
    idx = np.arange(len(ref)); comp = competitor(a, Pl, ref)
    return np.stack([ai * (P[idx, ref] - P[idx, comp]) for ai, P in zip(a, Pl)], 1)


def fusion_shapley(a, Pl, yt):
    """Shapley value of each expert for the fused Macro-F1, removing experts from the trained fusion
    (remaining weights renormalised; empty coalition = 0)."""
    k = len(Pl); val = {(): 0.0}
    for r in range(1, k + 1):
        for S in itertools.combinations(range(k), r):
            w = np.array([a[i] for i in S]); P = sum(wi * Pl[i] for wi, i in zip(w / w.sum(), S))
            val[S] = mf1(yt, P.argmax(1))
    phi = []
    for i in range(k):
        oth = [j for j in range(k) if j != i]; t = 0.0
        for r in range(k):
            for S in itertools.combinations(oth, r):
                wgt = math.factorial(r) * math.factorial(k - 1 - r) / math.factorial(k)
                t += wgt * (val[tuple(sorted(S + (i,)))] - val[tuple(sorted(S))])
        phi.append(t)
    return norm(phi)


def all_attributions(a, Pl, yt, pseudo=None, with_shapley=True):
    out = {'raw': norm(a),
           'mass_true': mass_contrib(a, Pl, yt), 'mass_true_bal': mass_contrib(a, Pl, yt, True),
           'margin_true': margin_contrib(a, Pl, yt), 'margin_true_bal': margin_contrib(a, Pl, yt, True),
           'margin_fused': margin_pred(a, Pl)}
    if pseudo is not None:
        out['mass_pseudo'] = mass_contrib(a, Pl, pseudo)
        out['margin_pseudo'] = margin_contrib(a, Pl, pseudo)
        out['margin_pseudo_bal'] = margin_contrib(a, Pl, pseudo, True)
    if with_shapley:
        out['fusion_shapley'] = fusion_shapley(a, Pl, yt)
    return out


def order_str(v):
    return ''.join(G[i][0].upper() for i in np.argsort(-np.asarray(v)))


def order_ok(v, ref):
    return bool(np.array_equal(np.argsort(-np.asarray(v)), np.argsort(-np.asarray(ref))))


def rho(v, ref):
    return float(spearmanr(v, ref)[0])


# ---------------- smooth fusion objectives ----------------
def w_loglinear(Pv, yv):
    L = [np.log(p + 1e-9) for p in Pv]
    def loss(z):
        w = np.exp(z); Z = sum(wi * l for wi, l in zip(w, L)); Z = Z - Z.max(1, keepdims=True)
        return -(Z[np.arange(len(yv)), yv] - np.log(np.exp(Z).sum(1))).mean()
    best = min((minimize(loss, np.array(z0, float), method='Nelder-Mead',
                         options=dict(xatol=1e-6, fatol=1e-10, maxiter=4000))
                for z0 in ([0, 0, 0], [0, -1, -1], [-1, 0, 0])), key=lambda r: r.fun)
    return np.exp(best.x)


def w_convex(Pv, yv, kind):
    onehot = np.eye(C)[yv]
    def loss(z):
        a = np.exp(z - z.max()); a /= a.sum(); P = sum(ai * p for ai, p in zip(a, Pv))
        if kind == 'nll':
            return -np.log(P[np.arange(len(yv)), yv] + 1e-12).mean()
        return ((P - onehot) ** 2).sum(1).mean()
    best = min((minimize(loss, np.array(z0, float), method='Nelder-Mead',
                         options=dict(xatol=1e-6, fatol=1e-9, maxiter=4000))
                for z0 in ([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])), key=lambda r: r.fun)
    a = np.exp(best.x - best.x.max())
    return a / a.sum()


# ---------------- reference criteria from subset models ----------------
def references(v):
    """v: dict mapping tuple-of-groups -> test Macro-F1 of a model retrained on that subset (() -> 0)."""
    solo = np.array([v[(g,)] for g in G]); full = v[G]
    logo = np.array([full - v[tuple(h for h in G if h != g)] for g in G])
    shap = []
    for g in G:
        oth = [h for h in G if h != g]; phi = 0.0
        for r in range(3):
            for S in itertools.combinations(oth, r):
                w = math.factorial(r) * math.factorial(2 - r) / 6
                phi += w * (v[tuple(h for h in G if h in S or h == g)] - v[tuple(h for h in G if h in S)])
        shap.append(phi)
    return {'solo': solo, 'logo': logo, 'shapley': np.array(shap)}
