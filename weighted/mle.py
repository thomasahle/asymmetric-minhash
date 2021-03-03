from math import log, exp
from sklearn.cluster import KMeans
import random
import numpy as np
from scipy.optimize import newton, bisect
from collections import Counter

def discrete_logistic(ps, k):
    #k = len(ps)
    km = KMeans(n_clusters=min(k,len(ps)))
    logits = [np.log((1-p)/p) for p in ps]
    km.fit(np.array(logits).reshape(-1,1))
    cnt = Counter(km.labels_)
    centers = km.cluster_centers_
    def inner(t):
        return sum(w/(1+exp(-t+centers[l])) for l,w in cnt.most_common())
    def fprime(t):
        return .5*sum(w/(1+np.cosh(t-centers[l])) for l,w in cnt.most_common())
    return np.mean(logits), inner, fprime



def mle(ps, X, u, s, ny, k, c):
    m = sum(1 for i in X if i < s)
    nx = len(X)
    print(f'{c=}, {s=}, {k=}, {m=}')
    print(f'{nx-m=}, {u-s-nx+m=}, {nx=}, {u-nx=}')
    v0 = c + max(0, s-k-m + nx+ny-u)
    v1 = min(ny-k, nx-m)+c
    if v0 == v1: return v0
    print(f'{v0=}, {v1=}')
    
    ps = np.array(ps)
    X = np.array(X)
    ps1 = ps[[i for i in X if i >= s]]
    ps2 = ps[[i for i in range(s, u) if i not in X]]
    #print(len([i for i in range(s, u) if i not in X]), ny, k, c, u-s-(nx-m))
    ps3 = ps[X]
    ps4 = ps[[i for i in range(u) if i not in X]]
    cl = 10
    l1, kp1, kpp1 = discrete_logistic(ps1, cl)
    l2, kp2, kpp2 = discrete_logistic(ps2, cl)
    l3, kp3, kpp3 = discrete_logistic(ps3, cl)
    l4, kp4, kpp4 = discrete_logistic(ps4, cl)
    #vs = np.linspace(c, min(ny-k, nx-m)+c)
    def inner(v):
        #print(v-c, ny-k-c+v, v, ny-v)
        #print(kp1(-30)-(v-c), kp1(30)-(v-c))
        #print(kp2(-30)-(ny-k-v+c), kp2(30)-(ny-k-v+c))
        #print(kp3(-30)-(v), kp3(30)-(v))
        #print(kp4(-30)-(ny-v), kp4(30)-(ny-v))
        t1 = bisect((lambda t: kp1(t)-(v-c)), -30, 30)
        t2 = bisect((lambda t: kp2(t)-(ny-k-v+c)), -30, 30)
        t3 = bisect((lambda t: kp3(t)-v), -30, 30)
        t4 = bisect((lambda t: kp4(t)-(ny-v)), -30, 30)
        #print(v-c, ny-k-v+c, v, ny-v)
        assert len(ps1) >= v-c and len(ps2) >= ny-k-v+c
        assert len(ps3) >= v and len(ps4) >= ny-v
        #t1 = newton((lambda t: kp1(t)-(v-c)), l1, fprime=kpp1, tol=1e-5)
        #t2 = newton((lambda t: kp2(t)-(ny-k-v+c)), l2, fprime=kpp2, tol=1e-5)
        #t3 = newton((lambda t: kp3(t)-v), l3, fprime=kpp3, tol=1e-5)
        #t4 = newton((lambda t: kp4(t)-(ny-v)), l4, fprime=kpp4, tol=1e-5)
        #print([l1, l2, l3, l4], t1, t2, t3, t4)
        return t1+t4-t2-t3
    assert len(ps1) == nx-m
    assert len(ps2) == u-s-(nx-m)
    assert len(ps3) == nx
    assert len(ps4) == u-nx
    v0, v1 = v0+1e-2, v1-1e-2
    t0 = inner(v0)
    t1 = inner(v1)
    print(f'{t0=}, {t1=}')
    if t0 > 0: return v0
    if t1 < 0: return v1
    v = bisect(inner, v0, v1, xtol=1e-1)
    mikkel = c*(nx+ny)/(m+k)
    return v

mse = 0
for i in range(10000):
    u = 30
    ps = [random.random()**2 for _ in range(u)]
    ps.sort()
    #X = random.sample(range(u), nx)
    #Y = random.sample(range(u), ny)
    #X.sort()
    #Y.sort()
    X = [i for i in range(u) if random.random() < ps[i]]
    Y = [i for i in range(u) if random.random() < ps[i]]
    #Y = [i for i in Y if i not in X]
    #Y = X
    assert Y == sorted(Y)
    nx = len(X)
    ny = len(Y)
    K = min(nx, ny)//3
    if K == 0: continue
    print(f'{u=}, {nx=}, {ny=}, {K=}')
    s = Y[K-1]+1
    c = len(set(X) & set(Y[:K]))
    real_v = len(set(X) & set(Y))
    guess_v = mle(ps, X, u, s, ny, K, c)
    print(f'{real_v=}', f'{guess_v=}')
    mse += (real_v-guess_v)**2
    print(mse/(i+1))
