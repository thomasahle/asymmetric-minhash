import numpy as np
import random
import time

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
    from . import minhash

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
    from . import minhash
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
    def __init__(self, ps, cl=100):
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
        self.cws = [(self.centers[l], len(b)) for l, b in enumerate(self.buckets)]
        self.cumsum = [0] # cumsum[s] = sum_{i<s} len(buckets[i])
        for bucket in self.buckets:
            self.cumsum.append(self.cumsum[-1] + len(bucket))
        

    def instance(self, s):
        ''' f(ps[s:], t) '''
        # Note, a suffix in ps means a prefix in logistics
        pre = len(self.labels) -  s # Size of prefix to use
        if pre == 0:
            return (lambda v: (lambda t: -v)), (lambda t: 0)
        label = self.labels[pre-1] # Label of last included
        c0 = self.centers[label]
        w0 = pre - self.cumsum[label]
        #assert w0 == sum(i < pre for i in self.buckets[label])
        def inner_v(v):
            def inner_t(t):
                return -v + w0/(1 + np.exp(c0 - t)) \
                          + sum(w/(1 + np.exp(c - t)) for c, w in self.cws[:label])
            return inner_t
        # Prime doesn't care about v
        def prime(t):
            return .5 * (w0/(1 + np.cosh(c0 - t))
                         + sum(w/(1+np.cosh(c - t)) for c, w in self.cws[:label]))
        return inner_v, prime

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

        f1, fp1 = fx.instance(m)
        f2, fp2 = fu.instance(s-m)
        f3, fp3 = fx.instance(0)
        f4, fp4 = fu.instance(0)
        def inner(v, comp_prime=False):
            def test(f, name):
                th = 100
                res1, res2 = f(-th), f(th)
                if np.sign(res1) == np.sign(res2):
                    print(res1, res2, name)
                    print(f(-2*th), f(2*th))
                    print((u,nx,ny,v),(s,m,k,c), name)
                    if abs(res1) < abs(res2):
                        return -th
                    else: return th
                return bisect(f, -th, th, xtol=1e-4)
            t1 = test(f1(v-c), 1)
            t2 = test(f2(ny-k-v+c), 2)
            t3 = test(f3(v), 3)
            t4 = test(f4(ny-v), 4)
            val = - t1 + t2 + t3 - t4
            if comp_prime:
                fp1t1, fp2t2 = fp1(t1), fp2(t2)
                fp3t3, fp4t4 = fp3(t3), fp4(t4)
                #print(fp1t1, fp2t2, fp3t3, fp4t4)
                if fp1t1 == fp3t3 == 0:
                    prime = -1/fp2t2 + 1/fp4t4
                elif fp2t2 == fp4t4 == 0:
                    prime = -1/fp1t1 + 1/fp3t3
                elif any(fpt == 0 for fpt in [fp1t1, fp2t2, fp3t3, fp4t4]):
                    return val, None
                else:
                    #print((u,nx,ny,v), (s,m,k,c), fp1t1, fp2t2, fp3t3, fp4t4)
                    prime = -1/fp1t1 - 1/fp2t2 + 1/fp3t3 + 1/fp4t4
                # This value is usually (always?) negative
                #assert prime < 0
                #print(prime)
                return val, prime
            return val
        if False:
            v0, v1 = v0+1e-2, v1-1e-2
            t0, t1 = inner(v0), inner(v1)
            if t0 > 0:
                estimates.append(v0/(nx+ny-v0))
            elif t1 < 0:
                estimates.append(v1/(nx+ny-v1))
            else:
                v = bisect(inner, v0, v1, xtol=1e-2)
                estimates.append(v/(nx+ny-v))
        else:
            v = c*(nx + ny)/(m + k)
            for _ in range(5):
                v = np.clip(v, v0+1e-1, v1-1e-1)
                val, prime = inner(v, comp_prime=True)
                if prime is not None:
                    #val2, prime2 = inner(v+1e-1, comp_prime=True)
                    #if abs((val2-val)/1e-1 - prime) > .1:
                    #    print(f'{(val2-val)/1e-1=}, {prime=}, {prime2=}')
                    #print(v, val, prime, val/prime, val2, prime2, (val2-val)/1e-1)
                    v -= val/prime
            v = np.clip(v, v0, v1)
            estimates.append(v/(nx+ny-v))

    time2 = time.time() - start
    return np.array(estimates), time1, time2
