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

def is_sorted(xs):
    return list(xs) == sorted(xs)


class DiscretizedLogistic2:
    def __init__(self, ps, threshold=10):
        self.threshold = threshold
        ps = np.array(ps).clip(1e-10, 1-1e-10)
        ups, self.counts = np.unique(ps, return_counts=True)
        self.logits = np.log(1/ups-1)
        self.logits.sort() # Note, we change the order
        self.cumsum = [0]
        for c in self.counts:
            self.cumsum.append(self.cumsum[-1] + c)

    def instance(self, s):
        ''' f(ps[s:], t) '''
        # Note: ps[s:] <=> logits[:n-s]
        def inner_v(v):
            def inner_t(t):
                res = 0
                i0 = np.searchsorted(self.logits[:len(self.logits)-s], t)
                # We include the surrounding sigmoids with multiplicity.
                # Might even consider approximating those by the taylor
                # approximation 1/2 + (t - l)/4 + ...
                Th = self.threshold
                for i in range(i0, len(self.logits)-s):
                    if self.logits[i] - t > Th:
                        break
                    res += self.counts[i]/(1 + np.exp(self.logits[i] - t))
                for i in range(i0-1, -1, -1):
                    if self.logits[i] - t < -Th:
                        # We Assume the small sigmoids are 1 (as exp(l-t) will be small)
                        res += self.cumsum[i+1]
                        break
                    res += self.counts[i]/(1 + np.exp(self.logits[i] - t))
                return res - v
            return inner_t
        return inner_v



class DiscretizedLogistic:
    def __init__(self, ps, cl=10):
        #assert ps[0] != 0 and ps[-1] != 1 # We don't need those
        #cl = min(len(ps), int(len(ps)**.33)+10)
        cl = min(len(np.unique(ps)), cl)
        ps = np.array(ps).clip(1e-10, 1-1e-10)
        km = KMeans(n_clusters=cl)
        km.fit(np.log(1/ps-1).reshape(-1,1))

        # It would be nice if labels were ordered by center.
        centers = km.cluster_centers_[:,0]

        reorder = centers.argsort()
        self.centers = centers[reorder]
        #assert is_sorted(self.centers)
        inverse_reorder = np.argsort(reorder)
        self.labels = inverse_reorder[km.labels_][::-1]
        #assert is_sorted([self.centers[l] for l in self.labels])

        # Make inverse lists
        self.buckets = [[] for _ in range(cl)]
        for i, l in enumerate(self.labels):
            self.buckets[l].append(i)
<<<<<<< HEAD
        self.cws = [(self.centers[l], len(b)) for l, b in enumerate(self.buckets)]
        self.cumsum = [0] # cumsum[s] = sum_{i<s} len(buckets[i])
        for bucket in self.buckets:
            self.cumsum.append(self.cumsum[-1] + len(bucket))
        

    def instance(self, s):
        ''' f(ps[s:], t) '''
        # Note, a suffix in ps means a prefix in logistics
        pre = len(self.labels) -  s # Size of prefix to use
        if pre == 0:
            return lambda v: (lambda t: -v)
        label = self.labels[pre-1] # Label of last included
        c0 = self.centers[label]
        w0 = pre - self.cumsum[label]
        #assert w0 == sum(i < pre for i in self.buckets[label])
        def inner_v(v):
            def inner_t(t):
                return -v + w0/(1 + np.exp(c0 - t)) \
                          + sum(w/(1 + np.exp(c - t)) for c, w in self.cws[:label])
            return inner_t
        return inner_v

=======
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
        def prime(t):
            return .5 * sum(w/(1+np.cosh(c-t)) for c, w in cws)
        return inner, prime
>>>>>>> Better outputs

from sklearn.cluster import KMeans
from collections import Counter
from scipy.optimize import bisect
def estimate_weighted(ps, x, ys, ysz):
    assert is_sorted(ps)
    assert is_sorted(x)
    u = len(ps)
    nx = len(x)


    start = time.time()
    fx = DiscretizedLogistic([ps[i] for i in x])
    x_set = set(x)
    fu = DiscretizedLogistic([ps[i] for i in range(u) if i not in x_set])
    time1 = time.time() - start

    #print(f'{u=}', len([ps[i] for i in x]), len([ps[i] for i in range(u) if i not in x_set]))

    start = time.time()
    estimates = []
    for i, (y, ny) in enumerate(zip(ys, ysz)):
        print(f'{i}/{len(ys)}', end='\r')
        y = y[:ny] # y is padded to length K. Remove that.
        s = y[-1] + 1
        k = len(y)
        m = sum(1 for i in x if i < s)
        c = sum(1 for i in y if i in x_set)

        v0 = c + max(0, s-k-m + nx+ny-u)
        v1 = min(ny-k, nx-m)+c
        if v0 == v1:
            estimates.append(v0/(nx+ny-v0))
            continue

        f1, fp1 = fx.instance(m, v-c)
        f2, fp2 = fu.instance(s-m, ny-k-v+c)
        f3, fp3 = fx.instance(0, v)
        f4, fp4 = fu.instance(0, ny-v)
        def inner(v, comp_prime=False):
            t1 = bisect(f1, -30, 30)
            t2 = bisect(f2, -30, 30)
            t3 = bisect(f3, -30, 30)
            t4 = bisect(f4, -30, 30)
            val = t1+t4-t2-t3
            if comp_prime:
                prime = -fp1(t1) - fp2(t2) + fp3(t3) + fp4(t4)
                return val, prime
            return val
        if False:
            v0, v1 = v0+1e-2, v1-1e-2
            t0, t1 = inner(v0), inner(v1)
            if t0 > 0:
                estimates.append(v0)
            elif t1 < 0:
                estimates.append(v1)
            else:
                v = bisect(inner, v0, v1, xtol=1e-2)
                estimates.append(v)
        else:
            v = c*(nx + ny)/(m + k)
            v = np.clip(v, v0, v1)
            val, prime = inner(v)
            v -= val/prime
            v = np.clip(v, v0, v1)
            estimates.append(v)

    t2 = time.time() - start
    return np.array(estimates), t1, t2
