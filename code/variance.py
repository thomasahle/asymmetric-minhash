import argparse
import numpy as np
import random
import math
import bisect
import statistics

def sample(u, x, y, v):
    xs = set(random.sample(range(u), x))
    vs = random.sample(xs, v)
    ys = random.sample(set(range(u)) - xs, y - v)
    return min(vs + ys), xs

class MLE:
    def __init__(self, u):
        self.log_facs = [0]
        for i in range(1, u+1):
            self.log_facs.append(self.log_facs[-1] + math.log(i))

    def estimate(self, u, x, y, xss, rs):
        def lb(n, k):
            if not 0 <= k <= n:
                return float('-inf')
            return self.log_facs[n] - self.log_facs[n-k] - self.log_facs[k]
        bestv, bestp = 0, float('-inf')
        ms = [bisect.bisect_left(sorted(xs), r) for xs, r in zip(xss, rs)]
        cs = [r in xs for xs, r in zip(xss, rs)]
        #for v in range(max(cs), min(x, y, x-max(ms)) + 1):
        for v in range(max(cs), min(x, y) + 1):
            den = lb(x, v) + lb(u-x, y-v)
            p = 0
            for r, m, c in zip(rs, ms, cs):
                if c:
                    p += lb(x-m-1, v-1) + lb(u-r-(x-m), y-v)
                else:
                    p += lb(x-m, v) + lb(u-r-(x-m)-1, y-v-1)
                p -= den
            #print(v, p)
            if p > bestp:
                bestv, bestp = v, p
        #print(range(max(cs), min(x, y, x - max(ms)) + 1))
        #print(max(cs), x, y, x-max(ms))
        return bestv/(x+y-bestv)

class MinHash:
    def estimate(self, u, x, y, xss, rs):
        return sum(min(xs) == r for xs, r in zip(xss, rs))/len(rs)

class Minner:
    def estimate(self, u, nx, ny, xss, rs):
        M, C = 0, 0
        for xs, r in zip(xss, rs):
            C += int(r in xs)
            M += sum(x < r for x in xs)
        v = C * min(nx/(C + M), ny/len(rs))
        return v/(nx + ny - v)




def main():
    x = 30
    y = 30
    u = 500
    K = 30
    N = 100
    estimators = [MinHash(), Minner(), MLE(u)]
    labels = ['Classic MinHash', 'Minner Estimator', 'Maximum Likelihood']
    js = []
    series = [[] for _ in estimators]
    for v in range(min(x,y)+1):
        print(v)
        estimates = [[] for _ in range(len(estimators))]
        for _ in range(N):
            rs, xss = zip(*[sample(u, x, y, v) for _ in range(K)])
            for i, e in enumerate(estimators):
                estimates[i].append(e.estimate(u, x, y, xss, rs))
        j = v/(x+y-v)
        js.append(j)
        for i, es in enumerate(estimates):
            #print(es)
            series[i].append(K*statistics.variance(es, j))
            #series[i].append(K*((np.array(es)-j)**2).mean())
            #series[i].append(np.array(es).mean())

    import matplotlib.pyplot as plt
    for ss, label in zip(series, labels):
        print(ss)
        plt.plot(js, ss, label=label)
    plt.legend()
    plt.xlabel('Variance')
    plt.ylabel('Jaccard Simmilarity')

    plt.show()
    #print('Writing to', args.out)
    #plt.savefig(args.out, dpi=600)


if __name__ == '__main__':
    main()
