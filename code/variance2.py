import argparse
import numpy as np
import random
import math
import bisect
import statistics
import functools


parser = argparse.ArgumentParser()
parser.add_argument('-R', type=int, default=100, help='Number of repetitions')
parser.add_argument('-K', type=int, default=10, help='Bottom-K')
parser.add_argument('--show', action='store_true')
subparsers = parser.add_subparsers()

parser_a = subparsers.add_parser('data', help='Usinig a dataset')
parser_a.add_argument('dataset', type=str, choices=['netflix','flickr','dblp'])
parser_a.add_argument('-N', type=int, default=None, help='Limit to N ponits of dataset')

parser_b = subparsers.add_parser('gen', help='Generated data')
parser_b.add_argument('-U', type=int, default=100, help='Universe siize')
parser_b.add_argument('-X', type=int, default=30, help='Size of X')
parser_b.add_argument('-Y', type=int, default=30, help='Size of Y')
parser_a.add_argument('--dist', type=str, default='uniform')


def p1(pps, n):
    @functools.lru_cache(10**6)
    def p1_inner(u, n):
        if not 0 <= n <= u: return 0
        if u == 0: return 1
        pi = pps[-u]
        return pi * p1_inner(u-1, n-1) + (1-pi) * p1_inner(u-1, n)
    return p1_inner(len(pps), n)


def make_table_full(ps):
    u = len(ps)
    p = np.zeros(u+1, u+1, dtype=np.float32)
    p[0, 0] = 1
    for i in range(u):
        p[i+1, 0] = (1-ps[i]) * p[i, 0]
        p[i+1, 1:] = ps[i] * p[i, :-1] + (1-ps[i]) * p[i, 1:]
    return p


def make_table(ps):
    u = len(ps)
    p_old = np.zeros(u+1, dtype=np.float32)
    p_new = np.zeros(u+1, dtype=np.float32)
    p_old[0] = 1
    for i in range(u):
        p_new[:] = (1-ps[i]) * p_old
        p_new[1:] += ps[i] * p_old[:-1]
        p_old, p_new = p_new, p_old
    return p_old


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
        v = max(range(lo, hi+1), key=lambda v:
            bi(nx-m,v-c) - bi(nx,v) + bi(u-s-(nx-m),ny-v-(k-c)) - bi(u-nx,ny-v))
        return v/(nx+ny-v)


class MLE:
    def estimate(self, ps, nx, ny, X, Y):
        nx, c, k, m, s = pars(X, Y)
        u = len(ps)

        v0 = c + max(0, s-k-m + nx+ny-u)
        v1 = min(ny-k, nx-m)+c
        if v0 == v1:
            return c/(nx+ny-c)

        pa = [p for i, p in enumerate(ps) if i >= s and i not in X]
        pb = [p for i, p in enumerate(ps) if i >= s and i in X]
        pc = [p for i, p in enumerate(ps) if i not in X]
        pd = [p for i, p in enumerate(ps) if i in X]
        assert len(pa) == u-s-(nx-m)
        assert len(pb) == nx-m
        ta = make_table(pa)
        tb = make_table(pb)
        tc = make_table(pc)
        td = make_table(pd)
        assert len(ta) == len(pa)+1
        # print(len(ta), len(tb), len(tc), len(td))
        # print(len(pa), len(pb), len(pc), len(pd))
        # print((u, nx, ny), (s,m,k,c))
        if u < 200:
            assert all(np.isclose(p, p1(pa,k)) for k, p in enumerate(ta))
        v = max(range(v0, v1+1),
                key=lambda v: ta[ny-k-(v-c)] / tc[ny-v] * tb[v-c] / td[v])
        return v/(nx+ny-v)


class Classic:
    def estimate(self, ps, nx, ny, X, Y):
        nx, c, k, m, s = pars(X, Y)
        v = c*(nx+ny)/(k+m)
        return v/(nx+ny-v)


def main(args):
    nx = args.X
    ny = args.Y
    u = args.U
    K = args.K
    N = args.R
    estimators = [Classic(), MLE(), MLE_Old(u)]
    labels = ['Mikkel', 'MLE', 'MLE Classic']
    estimates = [[[] for _ in range(len(estimators))] for v in range(min(nx,ny)+1)]
    for ii in range(N):
        print(f'{ii}/{N}')
        if args.dist == 'uniform':
            ps = sorted(random.random() for _ in range(u))
        elif args.dist == 'squared':
            ps = sorted(random.random()**2 for _ in range(u))
        elif m := re.match('zipf-(\d+\.?\d*)', args.dist):
            a = float(m.groups(1))
            ps = sorted(1/k**a for k in range(1, u+1))
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

    if args.show;
        plt.show()
    fn = '_'.join(f'{k}={v}' for k, v in vars(args).items() if type(v) in [int,str])+'.png'
    print('Writing to', fn)
    plt.savefig(fn, dpi=600)

from aminhash import datasets
from collections import Counter, defaultdict
import random

def main_data(args):
    print('Loading data')
    data, dom = datasets.load(args.dataset, verbose=True, trim=args.N)
    N = len(data)
    K = args.K
    print(f'{N=}, {dom=}')

    print('Counting')
    cnt = Counter(tok for X in data for tok in X)
    ps = np.array([cnt[i] for i in range(dom)]) / N
    order = ps.argsort() # 0 -> 5 means 5 is smallest.
    inv_order = np.zeros(dom, dtype=int)
    inv_order[order] = np.arange(dom) # Now 5 will be 0 and so on

    estimators = [Classic(), MLE(), MLE_Old(dom)]
    labels = ['Mikkel', 'MLE', 'MLE Classic']
    estimates = [[] for _ in estimators]
    for ii in range(args.R):
        print(f'{ii}/{args.R}')
        X, Y = random.sample(data, 2)
        X = sorted(inv_order[t] for t in X)
        Y = sorted(inv_order[t] for t in Y)
        nx, ny = len(X), len(Y)
        v = len(set(X) & set(Y))
        j = v/(nx+ny-v)

        estimates[1].append((j, estimators[1].estimate(ps, nx, ny, X, Y[:K])))

        perm = np.arange(dom)
        np.random.shuffle(perm)
        X, Y = np.array(X), np.array(Y)
        Xr, Yr = perm[X], perm[Y]
        Xr.sort(); Yr.sort()
        estimates[0].append((j, estimators[0].estimate(ps, nx, ny, Xr, Yr[:K])))
        estimates[2].append((j, estimators[2].estimate(ps, nx, ny, Xr, Yr[:K])))

    import matplotlib.pyplot as plt
    for data, label in zip(estimates, labels):
        js, vals = map(np.array, zip(*data))
        n_buckets = int(args.R**.5)
        _, edges = np.histogram(js, n_buckets)
        labels = np.digitize(js, edges, right=True)
        #print(len(edges), n_buckets, len(set(labels)))

        series = []
        for i in range(n_buckets+1):
            idx = labels == i
            errs = js[idx] - vals[idx]
            series.append(np.sum(errs**2) / sum(idx))
        plt.plot(edges, series, label=label)

    plt.legend()
    plt.xlabel('Overlap')
    plt.ylabel('Mean Squared Error')

    if args.show:
        plt.show()
    fn = '_'.join(f'{k}={v}' for k, v in vars(args).items() if type(v) in [int,str])+'.png'
    print('Writing to', fn)
    plt.savefig(fn, dpi=600)


if __name__ == '__main__':
    parser_a.set_defaults(main=main_data)
    parser_b.set_defaults(main=main)
    args = parser.parse_args()
    args.main(args)
