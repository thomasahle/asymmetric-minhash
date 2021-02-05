import argparse
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


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--out', type=str, default='synvar.png', help='Output filename')
parser.add_argument('-N', type=int, default=None, help='Number of repetitions')
parser.add_argument('-U', type=int, default=500, help='Universe Size')
parser.add_argument('-X', type=int, default=30, help='X size')
parser.add_argument('-Y', type=int, default=30, help='Y size')
parser.add_argument('-K', type=int, default=30)
args = parser.parse_args()


def main():
    x = args.X
    y = args.Y
    u = args.U
    K = args.K
    N = args.N
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
        js.append(v/(x+y-v))
        for i, es in enumerate(estimates):
            series[i].append(statistics.variance(es))

    import matplotlib.pyplot as plt
    for ss, label in zip(series, labels):
        plt.plot(js, ss, label=label)
    plt.legend()
    plt.xlabel('Variance')
    plt.ylabel('Jaccard Simmilarity')

    #plt.show()
    print('Writing to', args.out)
    plt.savefig(args.out, dpi=600)


if __name__ == '__main__':
    main()
