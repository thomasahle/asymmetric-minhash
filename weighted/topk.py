import heapq
import argparse
import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def h(state, j):
    random.seed(state ^ j)
    return random.randrange(2**32)

def topk(X, k, length, seeds, aggregate='max'):
    q = [(0, seed, 0, ()) for seed in seeds]
    while q and k > 0:
        v, state, l, string = heapq.heappop(q) # Smallest
        if l == length:
            yield state, v, string
            k -= 1
            continue
        # TODO: Better generate these in increasing order
        # and add a pointer back to the coroutine
        # generatnig them for when we need more.
        for j in X:
            #state1 = state ^ h(j, len(s)) # h_i(s, j)
            state1 = h(state, j) # h_i(s, j)
            if aggregate == 'max':
                v1 = max(v, state1)
            elif aggregate == 'sum':
                v1 = v + state1
            elif aggregate == 'xor':
                v1 = v ^ state1
            heapq.heappush(q, (v1, state1, l+1, string+(j,)))

def count_treshold(X, l, Ys, th, state, v, aggregate='max'):
    ''' Returns number of Xs <= th and number of those that are in Ys '''
    if l == 0:
        if state in Ys:
            return 1, 1
        return 0, 1
    res_c, res_m = 0, 0
    for j in X:
        state1 = h(state, j)
        if aggregate == 'max': v1 = max(v, state1)
        elif aggregate == 'sum': v1 = v + state1
        elif aggregate == 'xor': v1 = v ^ state1
        if v1 <= th:
            c, m = count_treshold(X, l-1, Ys, th, state1, v1, aggregate)
            res_c, res_m = res_c+c, res_m+m
    return res_c, res_m


def estimate(X, Ys, Yv, ny, l, seeds, aggregate='max'):
    k = len(Ys)
    s = Yv[-1]
    c, m = 0, 0
    for seed in seeds:
        ci, mi = count_treshold(X, l, set(Ys), s, seed, 0, aggregate)
        c, m = c+ci, m+mi
    nx = len(X)
    vl = c * (nx**l + ny**l) / (k + m)
    v = vl ** (1/l)
    return v
    #return v/(nx + ny - v)

def sample(u, nx, ny, v):
    X = random.sample(range(u), nx)
    Y = set(random.sample(X, v))
    X = set(X)
    while len(Y) != ny:
        y = random.choice(range(u))
        if y not in Y and y not in X:
            Y.add(y)
    assert len(X) == nx
    assert len(Y) == ny
    assert len(X&Y) == v
    return X, Y



parser = argparse.ArgumentParser()
#parser.add_argument('dataset', type=str, choices=['netflix','flickr','dblp'])
parser.add_argument('-U', type=int, default=1000)
parser.add_argument('-X', type=int, default=100)
parser.add_argument('-Y', type=int, default=100)
parser.add_argument('-K', type=int, default=10)
parser.add_argument('-R', type=int, default=100)
parser.add_argument('-Nv', type=int, default=50)
parser.add_argument('-Mv', type=int, default=None)
parser.add_argument('-L', type=int, default=3)
parser.add_argument('-W', type=int, default=10, help='Number of independent restarts')
args = parser.parse_args()




def main():
    u, nx, ny = args.U, args.X, args.Y
    K = args.K
    reps = args.R
    fn = '_'.join(f'{k}={v}' for k, v in vars(args).items() if type(v) in [int,str])+'.png'
    print(fn)

    #_, Y = sample(u, nx, ny, 0)
    #merged = []
    #for _ in range(K):
    #    seed = random.randrange(2**32)
    #    print('Seed', seed)
    #    for state, val, string in topk(Y, K, 3, seed, aggregate='max'):
    #        print(state, val, string)
    #        merged.append((val, state, string))
    #merged.sort()
    #print('Merged')
    #for val, state, string in merged[:K]:
    #    print(state, val, string)

    #return

    series = defaultdict(list)
    vs = np.linspace(0, args.Mv or min(nx, ny), dtype=int, num=args.Nv)
    for v in vs:
        for i in range(reps):
            print(v, f'{i}/{reps}')
            X, Y = sample(u, nx, ny, v)
            for length in range(1, args.L+1):
                # xor is fucking stupid because it's not monotone
                #for aggregate in ['max', 'sum', 'xor']:
                for w in [1, 8, 30]:
                    for aggregate in ['max']:
                        seeds = [random.randrange(2**32) for _ in range(w)]
                        Ys, Yv, _strs = zip(*topk(Y, K, length, seeds, aggregate=aggregate))
                        est = estimate(X, Ys, Yv, ny, length, seeds, aggregate)
                        series[f'{length=}, {aggregate=}, {w=}'].append(est)
    for label, ests in series.items():
        #print(label, ests)
        est_ar = np.array(ests).reshape(-1, reps)
        mse = np.mean((est_ar - vs[:,None])**2, axis=1)
        #print(mse)
        plt.plot(vs, mse, label=label)
    plt.legend()
    plt.savefig(fn, dpi=600)
    plt.show()

main()
