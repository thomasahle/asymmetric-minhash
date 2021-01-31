import numpy as np
import minhash as c_minhash
from scipy.special import binom
import scipy.optimize as opt
import bisect
import random
import os.path
import time
random.seed(74)


# Result on the complete dataset using a previous method:
# Computing normal minhash recall
#    1@10: 0.1499
# Computing asym minhash recall
#    1@10: 0.1611
#    def mle_fast(x, y, k, contains):
#       v = x/(k+1) if contains else 0
#       j = v/(x+y-v)
#       return j
# With the fast newton=1 method we get
#    1@10: 0.1821

# TODO: We should use this as the basis for the Newton method
# rather than Jaccard, since its simpler, and appears to work better on its own.
# Assuming $u$ is very big
def mle_fast(x, y, k, contains):
    if contains:
        v = x/(k+1)
    else: v = 0
    j = v/(x+y-v)
    return j

class Hash:
    def __init__(self, dom):
        self.perm = list(range(dom))
        random.shuffle(self.perm)
    def __call__(self, col):
        h, v = min((self.perm[v], v) for v in col)
        return v

R1 = 1
R2 = 10
fil = '/home/jovt/simfilter/data/ssjoin-data/netflix/netflix-dedup-raw-noone.txt'

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-N', type=int, default=None, help='Number of datapoints')
parser.add_argument('-M', type=int, default=100, help='Number of queries')
parser.add_argument('-K', type=int, default=30, help='Number of minhashes')
parser.add_argument('--newton', type=int, default=1, help='Number of Newton steps in some methods')
parser.add_argument('--method', type=str, help='Whether to use simpler formula')
parser.add_argument('--nosym', action='store_true', help='Dont test normal minhash')
parser.add_argument('--type', type=int, default=0, help='Extra parameter for comb7')
args = parser.parse_args()
N, M, K = args.N, args.M, args.K

print('Loading data')
data = [list(map(int, line.split())) for line in open(fil)]
if N is None:
    N = len(data) - M
print('Shuffling')
random.shuffle(data)
data, qs = data[:N], data[N:N+M]
dom = max(v for y in data+qs for v in y) + 1

print(f'Brute forcing')
cache_file = f'answers_{N}_{M}_{R1}.npy'
if os.path.exists(cache_file):
    answers = np.load(cache_file)
else:
    answers = []
    def jac(x, y):
        v = len(set(x) & set(y))
        return v / (len(x) + len(y) - v)
    for q in qs:
        js = np.array([jac(q, y) for y in data])
        top = np.argpartition(-js, R1)[:R1]
        answers.append(top)
    answers = np.array(answers)
    np.save(cache_file, answers)

print('Making hash functions')
hs = [Hash(dom) for _ in range(K)]

print('Hashing')
hash_file = f'hashes_{N}_{M}_{K}.npy'
if os.path.exists(hash_file):
    size_db = np.load(hash_file)
    db = np.ascontiguousarray(size_db[:, 1:]).astype(np.int32)
    sizes = np.ascontiguousarray(size_db[:, 0]).astype(np.int32)
else:
    db = np.array([[h(y) for h in hs] for y in data], dtype=np.int32)
    sizes = np.array([len(y) for y in data], dtype=np.int32)
    np.save(hash_file, np.hstack((sizes[:, np.newaxis], db)))

print('Sample hashes:')
print(db[0])
print(db[1])

if not args.nosym:
    print('Computing normal minhash recall')
    total = 0
    for q, answer in zip(qs, answers):
        minhash = [h(q) for h in hs]
        estimates = (db == minhash).mean(axis=1)
        candiates = np.argpartition(-estimates, R2)[:R2]
        r = len(set(candiates) & set(answer)) / R1
        total += r
    print(f'{R1}@{R2}: {total/len(qs)}')

print(f'Computing asym minhash recall, {args.method=}')
if False:
    def estimate(x, y_sz, ms):
        x = set(x)
        res = 0
        for h, m in zip(hs, ms):
            if m in x:
                z = sum(int(h.perm[v] < h.perm[m]) for v in x)
                res += mle(len(x), y_sz, z)
        return res / len(ms)

    total = 0
    for i, (q, answer) in enumerate(zip(qs, answers)):
        print(f'{i}/{len(qs)}', end='\r', flush=True)
        minhash = [h(q) for h in hs]
        estimates = np.array([estimate(q, sz, ms) for sz, ms in zip(sizes, db)])
        candiates = np.argpartition(-estimates, R2)[:R2]
        r = len(set(candiates) & set(answer)) / R1
        total += r
    print(f'{R1}@{R2}: {total/len(qs)}')

# At N=10^4, M=10^2
# Computing normal minhash recall
# 1@10: 0.22
# Computing asym minhash recall
# 1@10: 0.33
def estimate(x, ms, hs, ysz):
    res = 0
    for h, m in zip(hs, ms):
        k = sum(int(h.perm[xi] < h.perm[m]) for xi in x)
        res += mle(len(x), ysz, k, h.perm[m], dom, m in x)
    return res/len(ms)

# This version doesn't seem to be working well...
# At N=10^4, M=10^2
# Computing normal minhash recall
# 1@10: 0.22
# Computing asym minhash recall, args.method='fast'
# 1@10: 0.09
def estimate_fast(x, ms, hs, ysz):
    res = 0
    for h, m in zip(hs, ms):
        k = sum(int(h.perm[xi] < h.perm[m]) for xi in x)
        res += mle_fast(len(x), ysz, k, m in x)
    return res/len(ms)

# At N=10^4, M=10^2
# Computing normal minhash recall
# 1@10: 0.22
# Computing asym minhash recall, args.method='comb'
# 1@10: 0.28
def estimate_combined(x, ms, hs, ysz):
    xsz = len(x)
    # We could sum (m-k)/(u-x-y+v) instead. Wouldn't actually be pretty nice
    mu = sum(h.perm[m]/dom for h, m in zip(hs, ms))
    totk = sum(int(h.perm[xi] < h.perm[m]) for h, m in zip(hs, ms) for xi in x)
    n1 = sum(int(m in x) for m in ms)
    n2 = len(ms) - n1
    # Now solve:
    # -k/(x - v) + mu + n1/v - n2/(y - v) == 0
    # Which we rewrite as
    cs = [n1*xsz*ysz,
          -(n1+n2)*xsz - (totk+n1-mu*xsz)*ysz,
          totk+n1+n2-mu*(xsz+ysz),
          mu][::-1]
    rs = np.roots(cs)
    #print(rs, xsz, ysz)
    v = rs[2]
    return v/(xsz+ysz-v)

def logbinom(n, k):
    if k == 0 or k == n:
        return 0
    if k < 0 or k > n:
        return -np.inf
    assert n >= 0
    res = 0
    for _ in range(k):
        res += np.log(n/k)
        n -= 1
        k -= 1
    return res

# Used to test how close our approximate estimator is to the "real thing"
def precise(x, y, ks, ms, contains, verbose=False):
    if verbose:
        print(f'{x=}, {y=}, {ks=}, {ms=}, {contains=}')
    bestv, bestp = -1, None
    for v in range(min(x,y)+1):
        p = 0
        den = logbinom(x, v) + logbinom(dom-x, y-v)
        if verbose:
            print(f'{den=} {logbinom(x,v)=} {logbinom(dom-x, y-v)=}')
        for k, m, con in zip(ks, ms, contains):
            if not con:
                p += logbinom(x-k, v)
                p += logbinom(dom-m-1-(x-k), y-v-1)
            else:
                p += logbinom(x-k-1, v-1)
                p += logbinom(dom-m-(x-k), y-v)
            p -= den
        if verbose:
            print(v, p)
        if bestp is None or p > bestp:
            bestv, bestp = v, p
    return bestv

# Computing normal minhash recall
# 1@10: 0.22
# Computing asym minhash recall, args.method='comb2'
# 1@10: 0.27
def estimate_combined2(x, ysz, ms, ktbs):
    xsz = len(x)
    # Actually, this sort of table summing is exactly what we can do fast
    # with avx? Well, at least if we hash the minhash values down into
    # 4 bits...
    # TODO: Are harmonic means actually better here? They just often seem to be...
    tm = sum(h.perm[m] for h, m in zip(hs, ms))
    tk = sum(ktb[m] for ktb, m in zip(ktbs, ms))
    n1 = sum(int(m in x) for m in ms)
    n2 = len(ms) - n1
    # Stabalize by taking the average?
    #n = len(ms)
    #tk, tm, n1, n2 = tk/n, tm/n, n1/n, n2/n
    # We don't need that for comb2, since it already corresponds to n->infty

    # Now solve:
    # -k/(x - v) + (m-k)/(u-x-y+v) + n1/v - n2/(y - v) == 0
    # Which we rewrite as
    cs = [tm + n1 + n2,
          (n1+n2)*(dom-2*xsz) - tm*xsz + tk*(dom-ysz) - (tm+2*n1+n2)*ysz,
          (n1+n2)*xsz*(-dom+xsz) - (tk+n1)*dom*ysz + (tm+3*n1+n2)*xsz*ysz + (tk+n1)*ysz**2,
          -n1*xsz*ysz*(-dom+xsz+ysz)]
 
    # Let's see if Newton's method is faster than roots
    # Well, it's not when x-hashing is done for every datababse point.
    # To get performance here we should pass the k minhashes of x.
    #poly = np.poly1d(cs)
    #jest = sum(h(x) == m for h, m in zip(hs,ms)) / len(ms)
    #vest = jest/(jest+1) * (xsz + ysz)
    #pd = poly.deriv()
    #for _ in range(3):
    #vest = vest - poly(vest) / pd(vest)
    #v = vest

    #print(rs, xsz, ysz)
    rs = np.roots(cs)
    v = sorted(rs)[1]

    # Code for testing the values are good
    #mps = [h.perm[m] for h, m in zip(hs, ms)]
    #ks = [ktb[m] for ktb, m in zip(ktbs, ms)]
    #cons = [(m in x) for m in ms]
    #vreal = precise(xsz, ysz, ks, mps, cons)
    #if not np.isclose(vreal, v):
    #if abs(vreal - v) > 10:
    #    precise(xsz, ysz, ks, mps, cons, verbose=True)
    #    print(f'{tm=}, {tk=}, {n1=}, {n2=}, {xsz=}, {ysz=}, {dom=}, {vreal=}, {rs=}')

    #if n1 == 0:
    #    if not np.isclose(v, 0):
    #        vreal = precise(xsz, ysz, ks, ms, cons)
    #        print(f'{tm=}, {tk=}, {n1=}, {n2=}, {xsz=}, {ysz=}, {dom=}, {vreal=}, {rs=}')

    #if n1 == 30:
    #    if not (tk == 0 and np.isclose(v,0) or np.isclose(v, min(xsz,ysz))):
    #        vreal = precise(x, y, ks, ms, contains)
    #        print(f'{tm=}, {tk=}, {n1=}, {n2=}, {xsz=}, {ysz=}, {dom=}, {vreal=}, {rs=}')


    if not (-1e-2 <= v <= min(xsz,ysz)+1e-2):
        return jest
        print(f'Bad {v=}: {xsz=} {ysz=} {tm=} {tk=} {n1=} {n2=}')
        print(rs)
        # vest = jest/(jest+1)*(xsz+ysz)
        # vs = [vest]
        # for _ in range(3):
        #     vest = vest - poly(vest) / pd(vest)
        #     vs.append(vest)
        # print(vs)
    return v/(xsz+ysz-v)


# Newton around standard estimate.
# This is about twice as fast as the root based method.
# Probably with more to win from translation to Cython.
# Computing normal minhash recall
# 1@10: 0.22
# With 1 Newton: 1@10: 0.31
# With 2 Newton: 1@10: 0.27
# With 4 Newton: 1@10: 0.28
def estimate_combined4(x, ysz, ms, ktbs, xhs):
    xsz = len(x)
    tm = sum(h.perm[m] for h, m in zip(hs, ms))
    tk = sum(ktb[m] for ktb, m in zip(ktbs, ms))
    b = sum(int(m in x) for m in ms)
    a = len(ms) - b

    # Actually we can also do
    #ham0 = [xh == m for xh, m in zip(xhs, ms)]
    #ham1 = [(ktb[m]==0 and m in x) for ktb, m in zip(ktbs, ms)]
    #ham = sum(ham0)
    #if ham0 != ham1:
    #    print(ham0, ham1)

    #ham = sum(xh == m for xh, m in zip(xhs, ms))
    ham = sum((ktb[m]==0 and m in x) for ktb, m in zip(ktbs, ms))

    j0 = ham/(a+b)
    v = j0*(xsz+ysz)/(1+j0)
    #v = max(min(v, ysz-1, xsz-1), 1)
    v = min(v, ysz, xsz)
    for _ in range(args.newton):
        # This is using the simplified m/u term.
        # It assumes v is clamped in [1, min(x,y)-1].
        #nwtn = tm/dom + b/(v) - a/(ysz-v) - tk/(xsz-v)
        #nwtn /= b/(v)**2 + a/(ysz-v)**2 + tk/(xsz-v)**2

        # This version usea a +1 to fix division by zero problems.
        # It also makes it possible to take the expectation and many other things.
        # It has about the same performance as the normal version.
        nwtn = tm/dom + b/(v+1) - a/(ysz-v+1) - tk/(xsz-v+1)
        nwtn /= b/(v+1)**2 + a/(ysz-v+1)**2 + tk/(xsz-v+1)**2

        # For very large sets we can include all the terms:
        # nwtn = (tm-tk)/(dom-xsz-ysz+v) + b/v - a/(ysz-v) - tk/(xsz-v)
        # nwtn /= (tm-tk)/(dom-xsz-ysz+v)**2 + b/v**2 + a/(ysz-v)**2 + tk/(xsz-v)**2
        # But it doesn't seem to help anything.

        # Let's try without tm/dom at all
        #nwtn = b/v - a/(ysz-v) - tk/(xsz-v)
        #nwtn /= b/v**2 + a/(ysz-v)**2 + tk/(xsz-v)**2
        # Nah, that's bad. Gives 0.18 (from 0.31 before)

        # There are some more "standard" methods to try,
        # like using the expected newton correction term,
        # rather than the particular one. I don't think we
        # need that though, since our f' is about as simple as f.

        v += nwtn
    v = max(min(v, xsz, ysz), 0)
    return v/(xsz+ysz-v)

# Single newton on each term.
# Then summing the terms.
# TODO: If we are only doing a single term, maybe just use the log-version
# of the MLE? That one acts very nicely under differentiation
def estimate_combined5(x, ysz, ms, ktbs):
    xsz = len(x)
    res = 0
    for h, ktb, m in zip(hs, ktbs, ms):
        tm = h.perm[m]
        tk = ktb[m]
        b = int(m in x)
        a = 1 - b

        j0 = b/(a+b)
        v0 = j0*(xsz+ysz)/(1+j0)
        v0 = max(min(v0, ysz-1, xsz-1), 1)
        cor = tm/dom + b/v0 - a/(ysz-v0) - tk/(xsz-v0)
        cor /= b/v0**2 + a/(ysz-v0)**2 + tk/(xsz-v0)**2
        v = max(min(v0 + cor, xsz, ysz), 0)
        res += v/(xsz+ysz-v)
    return res/len(ms)



def estimate_combined3(x, ms, hs, ysz):
    xsz = len(x)
    tk = sum(int(h.perm[xi] < h.perm[m]) for xi in x for h, m in zip(hs, ms))
    tm = sum(h.perm[m] for h, m in zip(hs, ms))
    n1 = sum(int(m in x) for m in ms)
    n2 = len(ms) - n1
    # Stabalize by taking the average?
    n = len(ms)
    tk, tm, n1, n2 = tk/n, tm/n, n1/n, n2/n
    
    cs = [tm + n1 + n2,
          (n1+n2)*(dom-2*xsz) + tk*(n1+n2+dom-ysz) -(2*n1+n2)*ysz - tm*(n1+xsz+ysz),
          tm*n1*xsz - (n1+n2)*(dom-xsz)*xsz + n1*(tm-dom)*ysz + (tm + 3*n1+n2)*xsz*ysz + n1*ysz**2
+ tk * (n2*dom - n1*xsz - n2*xsz - (n1+n2+dom)*ysz + ysz**2),
          -n1*xsz*ysz*(-tk+tm-dom+xsz+ysz)]
    
    # Let's see if Newton's method is faster than roots
    poly = np.poly1d(cs)
    jest = sum(h(x) == m for h, m in zip(h,ms)) / len(ms)
    vest = jest/(jest+1) * (xsz + ysz)
    vest2 = vest - poly(vest) / poly.deriv()(vest)
    v = vest2

    #rs = np.roots(cs)
    #print(rs, xsz, ysz)
    # It appears this can get jk
    # TODO: Sort roots like comb2
    #v = abs(rs[2])

    return v/(xsz+ysz-v)



estimates = np.zeros(N, dtype=np.float32)

t1, t2 = 0, 0
total = 0
for i, (q, answer) in enumerate(zip(qs, answers)):
    print(f'{i}/{len(qs)} r~{total/i if i > 0 else 0}', end='\r', flush=True)
    if i % 400 == 0:
        print()
    start = time.time()

    x = set(q)
    # Make a table for a given hash function and m, returing the k-value
    ktbs = []
    for h in hs:
        #hxs = sorted([h.perm[xi] for xi in x])
        #ktbs.append([bisect.bisect_left(hxs, h.perm[m]) for m in range(dom)])
        ktbs.append(np.searchsorted(sorted(h.perm[qi] for qi in q), h.perm))

    t1 += time.time() - start
    start = time.time()

    if args.method == 'fast':
        estimates = np.array([estimate_fast(q, ms, hs, ysz) for ms, ysz in zip(db, sizes)])
    elif args.method == 'opt':
        estimates = np.array([estimate(q, ms, hs, ysz) for ms, ysz in zip(db, sizes)])
    elif args.method == 'comb':
        estimates = np.array([estimate_combined(q, ms, hs, ysz) for ms, ysz in zip(db, sizes)])
    elif args.method == 'comb2':
        estimates = np.array([estimate_combined2(x, ysz, ms, ktbs) for ms, ysz in zip(db, sizes)])
    elif args.method == 'comb4':
        xhs = [h(x) for h in hs]
        estimates = np.array([estimate_combined4(x, ysz, ms, ktbs, xhs) for ms, ysz in zip(db, sizes)])
    elif args.method in ['comb6', 'comb7', 'comb8']:
        ctab = np.array([m in x for m in range(dom)])
        mtab = np.array([h.perm for h in hs], dtype=np.int32)
        if args.method == 'comb6':
            c_minhash.query(db, len(x), dom, sizes, args.newton,
                            mtab, np.array(ktbs, dtype=np.int32), np.array(ctab),
                            estimates)
        elif args.method == 'comb7':
            c_minhash.query2(db, len(x), dom, sizes, args.newton, args.type,
                            mtab, np.array(ktbs, dtype=np.int32), np.array(ctab),
                            estimates)
        elif args.method == 'comb8':
            c_minhash.query_mle(db, len(x), dom, sizes,
                            mtab, np.array(ktbs, dtype=np.int32), np.array(ctab),
                            estimates)
    elif args.method == 'comb5':
        estimates = np.array([estimate_combined5(x, ysz, ms, ktbs) for ms, ysz in zip(db, sizes)])
    elif args.method == 'comb3':
        estimates = np.array([estimate_combined3(q, ms, hs, ysz) for ms, ysz in zip(db, sizes)])

    t2 += time.time() - start

    candiates = np.argpartition(-estimates, R2)[:R2]
    r = len(set(candiates) & set(answer)) / R1
    total += r
print(f'{R1}@{R2}: {total/len(qs)}')
print(f'Time preparing: {t1=}, Time searching: {t2=}')
