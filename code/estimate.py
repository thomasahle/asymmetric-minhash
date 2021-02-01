import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from scipy.stats import describe, scoreatpercentile

from aminhash import datasets, Hash, estimate, jac, tasks

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('plot_type', type=str, choices=['scatter', 'histogram'])
parser.add_argument('--data', type=str, default='netflix', choices=datasets.files.keys())
parser.add_argument('--output', type=str, default='out.png', help='Output filename')
parser.add_argument('-N', type=int, default=None, help='Database size to estimate')
parser.add_argument('-M', type=int, default=1, help='Number of queries to estimate')
parser.add_argument('-K', type=int, default=30, help='Number of minhashes')
parser.add_argument('--method', type=str, help='Whether to use simpler formula')
parser.add_argument('--newton', type=int, default=1, help='Number of Newton steps in some methods')
parser.add_argument('--type', type=int, default=0, help='Extra parameter for comb7')
args = parser.parse_args()


print('Loading data')
data, dom = datasets.load(args.data, verbose=True, trim=args.N)
print('Domain:', dom)

N = len(data)
M = args.M
K = args.K
data, qs = data[:N-M], data[N-M:N]
N -= M

print('Making hash functions')
hs = [Hash(dom) for _ in range(K)]

print('Hashing')
hash_file = f'hashes_{args.data}_{N}_{K}.npy'
def inner_hashing(hs, y):
    return [len(y)] + [h(y) for h in hs]
size_db = tasks.run(inner_hashing, data, args=(hs,), cache_file=hash_file,
                    verbose=True, chunksize=10000, report_interval=1000, perc=True)
db = np.ascontiguousarray(size_db[:, 1:]).astype(np.int32)
sizes = np.ascontiguousarray(size_db[:, 0]).astype(np.int32)


methods = [
    ('sym', 'r'),
    (('fast', 0, 10), 'b'),
    (('fast', 1, 10), 'g'),
    ('mle', 'orange')]


fig, ax = plt.subplots()

if args.plot_type == 'scatter':
    # This requires a newer version of matplotlib
    #plt.axline((0,0), (1,1), linestyle='--')
    for method, col in methods:
        print(method, '...')
        reals = []
        estis = []
        for x in qs:
            estimates, _, _ = estimate(method, x, db, sizes, dom, hs)
            estis += list(estimates)
            reals += [jac(x, y) for y in data]
        #print(reals)
        #print(estis)
        ax.scatter(reals, estis, label=method, s=5, facecolors='none', edgecolors=col)
    ax.legend()
    ax.grid(True)

def repr_method(method):
    if type(method) == tuple:
        method, newtons, typ = method
        return f'{method}_{newtons=}'
    return method

if args.plot_type == 'histogram':


    # This requires a newer version of matplotlib
    #plt.axline((0,0), (1,1), linestyle='--')
    
    res = []
    colors = []
    labels = []
    for method, col in methods:
        print(repr_method(method), '...')
        def inner(method, db, data, sizes, dom, hs, q):
            x = set(q)
            estimates, _, _ = estimate(method, x, db, sizes, dom, hs)
            jacs = [jac(x,y) for y in data]
            return [e - j for e, j in zip(estimates, jacs)]
        ests = tasks.run(inner, qs, args=(method, db, data, sizes, dom, hs),
                  cache_file=f'est_{repr_method(method)}_{N=}_{K=}.npy',
                  verbose=True, processes=5, report_interval=1).reshape(-1)
        ests = ests[~np.isclose(ests, 0)]
        print(describe(ests))
        print('5 percentile:', scoreatpercentile(ests, 5))
        print('-stdv:', scoreatpercentile(ests, 50-34.1))
        print('+stdv:', scoreatpercentile(ests, 50+34.1))
        print('95 precentile:', scoreatpercentile(ests, 95))
        res.append(ests)
        colors.append(col)
        labels.append(repr_method(method))

    ax.hist(res, int(len(res[0])**.3), density=True, fill=False,
            histtype='step', color=colors, label=labels)
    ax.legend(prop={'size': 10})
    ax.grid(True)


print('Writing to', args.output)
plt.savefig(args.output)

