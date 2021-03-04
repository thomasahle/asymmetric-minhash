import random
import numpy as np

from . import estimate_weighted, DiscretizedLogistic, DiscretizedLogistic2


def test_dlog2():
    U = 30
    ps = sorted(random.random() for _ in range(U))
    for th in [5, 100]:
        f = DiscretizedLogistic2(ps, threshold=th)
        for suffix in range(U+1):
            for t in np.linspace(-10, 10):
                est = f.instance(suffix)(0)(t)
                real = sum(1/(1+(1-p)/p*np.exp(-t)) for p in ps[suffix:])
                # Since threshold=nif, the estimate should be exact
                if th == 100:
                    assert np.isclose(est, real)
                else:
                    #assert np.isclose(est, real, rtol=1e-1, atol=1e-1)
                    assert abs(est - real) < 1e-1


def test_dlog():
    U = 10
    ps = sorted(random.random() for _ in range(U))
    f = DiscretizedLogistic(ps, cl=U)
    for suffix in range(U+1):
        for t in np.linspace(-10, 10):
            func, prime = f.instance(suffix)
            est = func(0)(t)
            real = sum(1/(1+(1-p)/p*np.exp(-t)) for p in ps[suffix:])
            # Since n_clusters = U the estimate should be exact
            assert np.isclose(est, real)


def test_weighted():
    random.seed(1)
    mse = 0
    reps = 10
    U = 30
    for i in range(reps):
        ps = sorted(random.random()**2 for _ in range(U))
        X = [i for i in range(U) if random.random() < ps[i]]
        Y = [i for i in range(U) if random.random() < ps[i]]
        nx = len(X)
        ny = len(Y)
        K = min(nx, ny)//3
        if K == 0: continue

        guess_j, _, _ = estimate_weighted(ps, X, [Y[:K]], [ny])
        guess_v = guess_j/(1+guess_j) * (nx + ny)
        real_v = len(set(X) & set(Y))
        mse += (real_v-guess_v)**2
    assert mse/reps < 2.5
