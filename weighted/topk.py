import heapq
import random
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

def h(state, j):
    random.seed(state ^ j)
    return random.randrange(2**32)

def topk(X, k, length, seed, aggregate='max'):
    q = [(0, seed, 0)]
    while q and k > 0:
        v, state, l = heapq.heappop(q) # Smallest
        if l == length:
            yield state, v
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
            heapq.heappush(q, (v1, state1, l+1))

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


def estimate(X, Ys, Yv, ny, l, seed, aggregate='max'):
    k = len(Ys)
    s = Yv[-1]
    c, m = count_treshold(X, l, set(Ys), s, seed, 0, aggregate)
    nx = len(X)
    vl = c * (nx**l + ny**l) / (k + m)
    v = vl ** (1/l)
    return v
    #return v/(nx + ny - v)

def sample(u, nx, ny, v):
    X = random.sample(range(10**8), 10**4)
    Y = set(random.sample(X, v))
    while len(Y) != ny:
        y = random.choice(range(10**8))
        if y not in Y:
            Y.add(y)
    return X, Y

def main():
    u, nx, ny = 10**5, 10**3, 10**3
    K = 10
    reps = 10

    series = defaultdict(list)
    vs = np.linspace(0, min(nx, ny), dtype=int, num=3)
    for v in vs:
        for i in range(reps):
            print(v, f'{i}/{reps}')
            X, Y = sample(u, nx, ny, v)
            seed = random.randrange(2**32)
            for length in [1,2,3]:
                # xor is fucking stupid because it's not monotone
                for aggregate in ['max', 'sum', 'xor']:
                    Ys, Yv = zip(*topk(X, K, length, seed, aggregate=aggregate))
                    est = estimate(X, Ys, Yv, ny, length, seed, aggregate)
                    series[f'{length=}, {aggregate=}'].append(est)
    for label, ests in series.items():
        est_ar = np.array(ests).reshape(-1, reps)
        mse = np.mean((est_ar - vs)**2, axis=1) / reps
        plt.plot(vs, mse, label=label)
    plt.legend()
    print('Savinig to', fn:='out.png')
    plt.savefig(fn, dpi=600)
    plt.show()

main()
