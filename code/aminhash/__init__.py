import numpy as np
import random
import time

from . import minhash

random.seed(74)


class Hash:
    def __init__(self, dom):
        self.perm = list(range(dom))
        random.shuffle(self.perm)

    def __call__(self, col):
        h, v = min((self.perm[v], v) for v in col)
        return v


def jac(x, y):
    v = len(x.intersection(y))
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


def estimate(method, x, ys, ysz, dom, hs, estimates=None):
    N = len(ys)

    if method == 'sym':
        start = time.time()
        hashes = [h(x) for h in hs]
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


def estimate_bottomk(method, x, ys, ysz, dom, h, estimates=None):
    N = len(ys)

    if estimates is None:
        estimates = np.zeros(N, dtype=np.float32)  # Space for storing results
    start = time.time()
    xh = np.array(sorted(h.perm[xi] for xi in x), dtype=np.int32)
    t1 = time.time() - start

    start = time.time()
    typ = {'mikkel': 0, 'minner': 1, 'newton': 2, 'revdiv': 3, 'mle': 4}[method]
    minhash.bottomk(ys, len(x), dom, ysz, xh, typ, estimates)
    t2 = time.time() - start

    return estimates, t1, t2


class DiscretizedLogistic:
    def __init__(self, ps):
        #assert ps[0] != 0 and ps[-1] != 1 # We don't need those
        cl = min(len(ps), int(len(ps)**.5)+10)
        km = KMeans(n_clusters=cl)
        km.fit(np.log(1/ps[x]-1).reshape(-1,1))

        # It would be nice if labels were ordered by center.
        centers = self.km.centers_[:,0]
        reorder = centers.argsort()
        self.centers = centers[reorder]
        self.labels = km.labels_[reorder]

        # Make inverse lists
        self.buckets = [[] for _ in range(n_clusters)]
        for i, l in enumerate(self.km.labels_):
            self.buckets[l].append(i)
        self.cws = [(self.km.centers_[l], len(b)) for l, b in enumerate(self.buckets)]

    def instance(self, s, v):
        ''' f(ps[s:], t) - v '''
        self.km.labels_[s] # Cluster of first index in suffix
        label = self.km.labels_[s]
        bucket_c = self.km.centers_[label]
        bucket_w = sum(i >= s for i in self.buckets[label])
        cws = [(bucket_c, bucket_w)] + self.cws[s+1:]
        def inner(t):
            return sum(w/(1+exp(c-t)) for c, w in cws)
        return inner

from sklearn.cluster import KMeans
from collections import Counter
from scipy.optimize import bisect
def estimate_weighted(ps, x, ys, ysz):
    assert ps == sorted(ps)
    u = len(ps)
    nx = len(x)

    fx = DiscretizedLogistic([ps[i] for i in x], n_clusters=100)
    x_set = set(x)
    fu = DiscretizedLogistic([ps[i] for i in range(u) if i not in x_set], n_clusters=100)

    estimates = []
    for y, ny in zip(ys, ysz):
        s = y[-1] + 1
        l = np.log(1/ps[s-1]-1) # Largest included logit
        k = len(y)
        m = sum(i for i in x if i < s)
        c = sum(i for i in y if i in x_set)

        v0 = c + max(0, s-k-m + nx+ny-u)
        v1 = min(ny-k, nx-m)+c
        if v0 == v1:
            estimates.append(v0)
            continue

        f1 = fx.instance(m, v-c)
        f2 = fu.instance(s-m, ny-k-v+c)
        f3 = fx.instance(0, v)
        f4 = fu.instance(0, ny-v)
        def inner(v):
            t1 = bisect(f1, -30, 30)
            t2 = bisect(f2, -30, 30)
            t3 = bisect(f3, -30, 30)
            t4 = bisect(f4, -30, 30)
            return t1+t4-t2-t3
        v0, v1 = v0+1e-2, v1-1e-2
        t0, t1 = inner(v0), inner(v1)
        if t0 > 0:
            estimates.append(v0)
        elif t1 < 0:
            estimates.append(v1)
        else:
            v = bisect(inner, v0, v1, xtol=1e-2)
            estimates.append(v)

