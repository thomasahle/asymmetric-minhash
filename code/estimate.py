import numpy as np
import random
import argparse
import matplotlib.pyplot as plt

from aminhash import datasets, Hash, estimate, jac

parser = argparse.ArgumentParser(description='Process some integers.')
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

print('Making hash functions')
hs = [Hash(dom) for _ in range(K)]

print('Hashing')
db = np.array([[h(y) for h in hs] for y in data], dtype=np.int32)
sizes = np.array([len(y) for y in data], dtype=np.int32)

fig, ax = plt.subplots()
# This requires a newer version of matplotlib
#plt.axline((0,0), (1,1), linestyle='--')
for method, col in [
        ('sym', 'r'),
        (('fast', 0, 10), 'b'),
        (('fast', 1, 10), 'g')
    #, 'mle']:
    ]:
    print(method, '...')
    reals = []
    estis = []
    for x in qs:
        estimates, _, _ = estimate(method, x, db, sizes, dom, hs)
        estis += list(estimates)
        reals += [jac(x, y) for y in db]
    #print(reals)
    #print(estis)
    ax.scatter(reals, estis, label=method, s=5, facecolors='none', edgecolors=col)

ax.legend()
ax.grid(True)

print('Writing to', args.output)
plt.savefig(args.output)

