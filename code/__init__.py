import numpy as np
import random
import time

import minhash

random.seed(74)


class Hash:
    def __init__(self, dom):
        self.perm = list(range(dom))
        random.shuffle(self.perm)

    def __call__(self, col):
        h, v = min((self.perm[v], v) for v in col)
        return v


def jac(x, y):
    v = len(set(x) & set(y))
    return v / (len(x) + len(y) - v)


def make_tables(x, hs, dom):
    # Make a table for a given hash function and m, returning the k-value
    ktbs = np.array(
        [np.searchsorted(sorted(h.perm[xi] for xi in x), h.perm) for h in hs],
        dtype=np.int32)
    mtbs = np.array([h.perm for h in hs], dtype=np.int32)
    x = set(x)
    ctab = np.array([m in x for m in range(dom)])
    return ktbs, mtbs, ctab


def estimate(method, x, ys, ysz, dom, estimates=None):
    if method == 'sym':
        start = time.time()
        hashes = [h(q) for h in hs]
        t1 = time.time() - start

        start = time.time()
        estimates = (ys == hashes).mean(axis=1)
        t2 = time.time() - start

    else:
        if estimates is None:
            estimates = np.zeros(N, dtype=np.float32)  # Space for storing results
        start = time.time()
        ktbs, mtbs, ctab = make_tables(x, hs, dom)
        t1 = time.time() - start

        start = time.time()
        if type(method) == tuple:
            _, newton, typ = method
            minhash.query2(ys, len(x), dom, ysz, newton, typ, mtbs, ktbs, ctab, estimates)
        if method == 'mle':
            minhash.query_mle(ys, len(x), dom, ysz, mtbs, ktbs, ctab, estimates)
        t2 = time.time() - start

    return estimates, t1, t2
