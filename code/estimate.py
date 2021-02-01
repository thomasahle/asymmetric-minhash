import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from scipy.stats import describe, scoreatpercentile

from aminhash import datasets, Hash, estimate, jac, tasks

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('plot_type', type=str, choices=['scatter', 'histogram', 'variance'])
parser.add_argument('--data', type=str, default='netflix', choices=datasets.files.keys())
parser.add_argument('--out', type=str, default=None, help='Output filename')
parser.add_argument('-N', type=int, default=None, help='Database size to estimate')
parser.add_argument('-M', type=int, default=1, help='Number of queries to estimate')
parser.add_argument('-K', type=int, default=30, help='Number of minhashes')
parser.add_argument('--percentile', type=float, default=15.9, help='Percentile to display on variance plot')
parser.add_argument('--method', type=str, help='Whether to use simpler formula')
parser.add_argument('--newton', type=int, default=1, help='Number of Newton steps in some methods')
parser.add_argument('--type', type=int, default=0, help='Extra parameter for comb7')
args = parser.parse_args()


print('Loading data')
data, dom = datasets.load(args.data, verbose=True, trim=args.N)
print('Domain:', dom)

N = len(data)
M = args.M
data, qs = data[:N-M], data[N-M:N]
N -= M

methods = [
    ('sym', 'r')
    ,(('fast', 0, 10), 'g')
    #,(('fast', 1, 10), 'orange')
    ,('mle', 'b')
]

def repr_method(method):
    if type(method) == tuple:
        method, newtons, typ = method
        return f'{method}_{newtons=}'
    return method

all_hs = {}
def compute_estimates(method, k):
    print(repr_method(method), '...')
    
    if k not in all_hs:
        print(f'Making hash functions at {k=}')
        all_hs[k] = [Hash(dom) for _ in range(k)]

    print('Hashing')
    hash_file = f'hashes_{args.data}_{N}_{k}.npy'
    global inner_hashing
    def inner_hashing(hs, y):
        return [len(y)] + [h(y) for h in hs]
    size_db = tasks.run(inner_hashing, data, args=(all_hs[k],), cache_file=hash_file,
                        verbose=True, chunksize=10000, report_interval=1000, perc=True)
    db = np.ascontiguousarray(size_db[:, 1:]).astype(np.int32)
    sizes = np.ascontiguousarray(size_db[:, 0]).astype(np.int32)

    global inner
    def inner(method, db, data, sizes, dom, hs, q):
        x = set(q)
        estimates, _, _ = estimate(method, x, db, sizes, dom, hs)
        return list(estimates) + [jac(x,y) for y in data]
    est_jacs = tasks.run(inner, qs, args=(method, db, data, sizes, dom, all_hs[k]),
              cache_file=f'est_{repr_method(method)}_{N=}_{k=}.npy',
              verbose=True, processes=5, report_interval=1)

    return est_jacs[:, :N].reshape(-1), est_jacs[:, N:].reshape(-1)

fig, ax = plt.subplots()

if args.plot_type == 'scatter':
    # This requires a newer version of matplotlib
    #plt.axline((0,0), (1,1), linestyle='--')
    for method, col in methods:
        mests, mjacs = compute_estimates(method, args.K)
        sample = np.random.choice(len(mests), 3000, replace=False)
        ax.scatter(mjacs[sample], mests[sample], label=method,
                   s=2, facecolors='none', edgecolors=col, alpha=.6)
    ax.legend()
    ax.set_aspect('equal', 'box')
    ax.grid(True)


if args.plot_type == 'histogram':
    series = []
    colors = []
    labels = []
    for method, col in methods:
        mests, mjacs = compute_estimates(method, args.K)
        vals = mests - mjacs
        print(describe(vals))
        print('5 percentile:', scoreatpercentile(vals, 5))
        print('10 percentile:', scoreatpercentile(vals, 10))
        print('-stdv:', scoreatpercentile(vals, 50-34.1))
        print('+stdv:', scoreatpercentile(vals, 50+34.1))
        print('90 precentile:', scoreatpercentile(vals, 90))
        print('95 precentile:', scoreatpercentile(vals, 95))

        vals = vals[~np.isclose(vals, 0)]
        series.append(vals)
        colors.append(col)
        labels.append(repr_method(method))

    ax.hist(res, int(len(series[0])**.3), density=True, fill=False,
            histtype='step', color=colors, label=labels)
    ax.legend(prop={'size': 10})
    ax.grid(True)

if args.plot_type == 'variance':
    # I want a plot where the x-axis is K
    # Like in Mikkel's article: https://dl.acm.org/doi/pdf/10.1145/2488608.2488655
    # Maybe with fewer Ks and the mean instead of a single experiment.

    for i, ((method, col), line) in enumerate(zip(methods, ('-', '--', '-.'))):
        lows, means, highs = [], [], []
        ks = range(1, args.K+1)
        for k in ks:
            mests, mjacs = compute_estimates(method, k)
            vals = mests - mjacs
            lows.append(scoreatpercentile(vals, args.percentile))
            means.append(vals.mean())
            highs.append(scoreatpercentile(vals, 100-args.percentile))
        ax.fill_between(ks, lows, highs, alpha=.2, color=f'C{i}')
        ax.plot(ks, means, line, label=repr_method(method), color=f'C{i}')
        ax.plot(ks, lows, line, color=f'C{i}', alpha=1, linewidth=.5)
        ax.plot(ks, highs, line, color=f'C{i}', alpha=1, linewidth=.5)

    ax.legend(prop={'size': 10})
    ax.grid(True)
    plt.xlabel('# MinHash functions')
    plt.ylabel('Estimate − True Jaccard Simmilarity')

if args.out is None:
    out_file = f'plot_{args.data}_{N=}_{M=}_{args.K=}_{args.plot_type}.png'
else: out_file = args.out
print('Writing to', out_file)
plt.savefig(out_file, dpi=300)

