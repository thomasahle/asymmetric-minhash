import argparse
import numpy as np
import random
import math
import bisect
import statistics
import functools

def p1(pps, n):
    @functools.lru_cache(10**6)
    def p1_inner(u, n):
        if not 0 <= n <= u: return 0
        if u == 0: return 1
        pi = pps[-u]
        return pi * p1_inner(u-1, n-1) + (1-pi) * p1_inner(u-1, n)
    return p1_inner(len(pps), n)

def sampler(ps):
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
        x, y = random.choices(((1,1),(1,0),(0,1),(0,0)), [p11,p10,p01,p00])[0]
        xtail, ytail = sample(u-1, nx-x, ny-y, v-x*y)
        return [x]+xtail, [y]+ytail

    return sample


def pars(X, Y):
    nx, k = len(X), len(Y)
    c = len(set(X) & set(Y))
    s = max(Y)+1
    m = sum(1 for xi in X if xi < s)
    return nx, c, k, m, s


class MLE_Old:
    def __init__(self, u):
        self.fac_table = [0]
        for i in range(1, u+1):
            self.fac_table.append(self.fac_table[-1] + np.log(i))

    def bi(self, n, k):
        return self.fac_table[n] - self.fac_table[n-k] - self.fac_table[k]

    def estimate(self, ps, nx, ny, X, Y):
        nx, c, k, m, s = pars(X, Y)
        u = len(ps)
        lo, hi = max(c,c-k-m+nx+ny+s-u), c+min(nx-m, ny-k)
        bi = self.bi
        return max(range(lo, hi+1), key=lambda v:
            bi(nx-m,v-c) - bi(nx,v) + bi(u-s-(nx-m),ny-v-(k-c)) - bi(u-nx,ny-v))


class MLE:
    def estimate(self, ps, nx, ny, X, Y):
        u = len(ps)
        s = max(Y)+1
        c = sum(x in Y for x in X)
        m = sum(1 for x in X if x < s)
        k = len(Y)
        pa = [p for i, p in enumerate(ps) if i >= s and i not in X]
        pb = [p for i, p in enumerate(ps) if i >= s and i in X]
        pc = [p for i, p in enumerate(ps) if i not in X]
        pd = [p for i, p in enumerate(ps) if i in X]
        assert len(pa) == u-s-(nx-m)
        assert len(pb) == nx-m
        return max(range(c, c+min(nx-m,ny-k)+1),
                # What is the proability that we get the required remaining
                # samples from X or U\X?
                key=lambda v: p1(pa, ny-k-(v-c)) / p1(pc, ny-v)
                            * p1(pb, v-c) / p1(pd, v)
                )


class Classic:
    def estimate(self, ps, nx, ny, X, Y):
        nx, c, k, m, s = pars(X, Y)
        return c*(nx+ny)/(k+m)


def main():
    nx = 30
    ny = 30
    u = 100
    K = 10
    N = 100
    estimators = [Classic(), MLE(), MLE_Old(u)]
    labels = ['Mikkel', 'MLE', 'MLE Classic']
    estimates = [[[] for _ in range(len(estimators))] for v in range(min(nx,ny)+1)]
    for ii in range(N):
        print(f'{ii}/{N}')
        ps = sorted(random.random()**2 for _ in range(u))
        sample = sampler(ps)
        for v in range(min(nx,ny)+1):
            print(f'{v=}', end='\r')
            xs, ys = sample(u, nx, ny, v)
            X = [i for i in range(u) if xs[i]]
            Y = [i for i in range(u) if ys[i]]
            estimates[v][1].append(estimators[1].estimate(ps, nx, ny, X, Y[:K]))

            perm = np.arange(u)
            np.random.shuffle(perm)
            Xr, Yr = perm[X], perm[Y]
            Xr.sort(); Yr.sort()
            estimates[v][0].append(estimators[0].estimate(ps, nx, ny, Xr, Yr[:K]))
            estimates[v][2].append(estimators[2].estimate(ps, nx, ny, Xr, Yr[:K]))
        print()

    js = []
    series = [[] for _ in estimators]
    for v in range(min(nx,ny)+1):
        #j = v/(nx+ny-v)
        #js.append(j)
        js.append(v)
        for i, es in enumerate(estimates[v]):
            #series[i].append(K*statistics.variance(es, j))
            series[i].append(K*statistics.variance(es, v))

    import matplotlib.pyplot as plt
    for ss, label in zip(series, labels):
        print(ss)
        plt.plot(js, ss, label=label)
    plt.legend()
    plt.xlabel('Overlap')
    plt.ylabel('Variance')
    #plt.ylabel('Mean Squared Error')

    plt.show()
    #print('Writing to', args.out)
    #plt.savefig(args.out, dpi=600)


if __name__ == '__main__':
    main()
