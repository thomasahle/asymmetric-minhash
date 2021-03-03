import random
import functools
from math import log

@functools.lru_cache(10**8)
def p(u, nx, ny, v):
    ''' Probably that independent samples by p end up with (nx, nu, v) profile'''
    if not (0 <= v <= min(nx,ny)
            and max(nx,ny) <= u
            and u-nx-ny+v >= 0): return 0
    if u == 0: return 1
    pi = ps[-u]
    return pi**2     * p(u-1, nx-1, ny-1, v-1) + \
           pi*(1-pi) * p(u-1, nx, ny-1, v) + \
           pi*(1-pi) * p(u-1, nx-1, ny, v) + \
           (1-pi)**2 * p(u-1, nx, ny, v)

def p1(pps, n):
    @functools.lru_cache(10**6)
    def p1_inner(u, n):
        if not 0 <= n <= u: return 0
        if u == 0: return 1
        pi = pps[-u]
        return pi * p1_inner(u-1, n-1) + (1-pi) * p1_inner(u-1, n)
    return p1_inner(len(pps), n)

#print(p1([.1,.1,.1,.7], 1))
#print(p1([.25]*4, 1))

def sample(u, nx, ny, v):
    assert u >= max(nx,ny) and min(nx,ny) >= v >= 0
    assert u-nx-ny+v >= 0
    if u == 0:
        return [], []
    pi = ps[-u]
    p11 = pi**2     * p(u-1, nx-1, ny-1, v-1)
    p10 = pi*(1-pi) * p(u-1, nx-1, ny, v)
    p01 = pi*(1-pi) * p(u-1, nx, ny-1, v)
    p00 = (1-pi)**2 * p(u-1, nx, ny, v)
    #t = p11+p10+p01+p00
    #print([p11/t, p10/t, p01/t, p00/t])
    #print([p11, p10, p01, p00])
    # Should divide probs by p(u,nx,ny,v), but it happens automatically
    x, y = random.choices(((1,1),(1,0),(0,1),(0,0)), [p11,p10,p01,p00])[0]
    xtail, ytail = sample(u-1, nx-x, ny-y, v-x*y)
    return [x]+xtail, [y]+ytail

def estimate(u, nx, ny, X, Y):
    s = max(Y)+1
    c = sum(x in Y for x in X)
    m = sum(1 for x in X if x < s)
    pa = [p for i, p in enumerate(ps) if i >= s and i not in X]
    pb = [p for i, p in enumerate(ps) if i >= s and i in X]
    pc = [p for i, p in enumerate(ps) if i not in X]
    pd = [p for i, p in enumerate(ps) if i in X]
    # ta = sum(pa)
    # pa = [p/ta for p in pa]
    # tb = sum(pb)
    # pb = [p/tb for p in pb]
    #print(len(pa), len(pb), u-s-(nx-m), nx-m)
    assert len(pa) == u-s-(nx-m)
    assert len(pb) == nx-m
    #print('smkc', s, m, k, c)
    #print(pa, pb)
    #for v in range(min(nx,ny)+1):
        #print(v, p1(pa, ny-k-(v-c)) * p1(pb, v-c))
    #print(sum(pa), sum(p1(pa, k) for k in range(len(pa)+1)))
    #print(sum(pb), sum(p1(pb, k) for k in range(len(pb)+1)))
    return max(range(c, c+min(nx-m,ny-k)+1),
            # What is the proability that we get the required remaining
            # samples from X or U\X?
            key=lambda v: p1(pa, ny-k-(v-c)) / p1(pc, ny-v)
                        * p1(pb, v-c) / p1(pd, v)
            )

def est_classic(u, nx, ny, X, Y):
    s = max(Y)+1
    c = sum(x in Y for x in X)
    m = sum(1 for x in X if x < s)
    return c*(nx+ny)/(k+m)

u = 100
nx = 30
ny = 30
k = 10
alpha = .3

# for _ in range(10):
#     xs, ys = sample(len(ps), nx, ny, 10)
#     print(sum(x and y for x, y in zip(xs,ys)))

import sys
mode = sys.argv[1]


ps = [alpha/x**(alpha+1) for x in range(1, u+1)]
tot = sum(ps)
ps = [p/tot for p in ps]
if mode == 'classic':
    random.shuffle(ps)
if mode == 'mle':
    #ps = ps[::-1] # Smallest first
    #ps = ps
    random.shuffle(ps)
#p.cache_clear()

total_err = 0
for v in range(min(nx,ny)+1):
    vest = 0
    mse = 0
    reps = 1000
    for _ in range(reps):
        if mode == 'classic':
            random.shuffle(ps)
        xs, ys = sample(len(ps), nx, ny, v)
        X = [i for i in range(u) if xs[i]]
        Y = [i for i in range(u) if ys[i]]
        assert len(X) == nx
        assert len(Y) == ny
        assert len(set(X)&set(Y)) == v
        if mode == 'classic':
            est = est_classic(u, nx, ny, X, Y[:k])
        if mode == 'mle':
            est = estimate(u, nx, ny, X, Y[:k])
        vest += est
        mse += (v - est)**2
    print(v, vest/reps, mse/reps)
    total_err += mse/reps
print('Total error:', total_err)
# cnts = [0]*len(ps)
# for _ in range(10000):
#     xs, ys = sample(len(ps), 2, 2, 1)
#     for i in range(len(ps)):
#         cnts[i] += xs[i]
# print([c/10000 for c in cnts])
