import numpy as np
import random
import argparse
import matplotlib.pyplot as plt

from . import datasets, Hash, estimate, jac

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data', type=str, default='netflix', choices=datasets.files.keys())
parser.add_argument('--output', type=str, default='out.png', help='Output filename')
parser.add_argument('-N', type=int, default=None, help='Number of pairs to estimate')
parser.add_argument('-K', type=int, default=30, help='Number of minhashes')
parser.add_argument('--method', type=str, help='Whether to use simpler formula')
parser.add_argument('--newton', type=int, default=1, help='Number of Newton steps in some methods')
parser.add_argument('--type', type=int, default=0, help='Extra parameter for comb7')
args = parser.parse_args()

N, K = args.N, args.K

print('Loading data')
data, dom = datasets.load(args.data)
xys = [random.sample(data, 2) for _ in range(N)]

print('Making hash functions')
hs = [Hash(dom) for _ in range(K)]

fig, ax = plt.subplots()
for method in ['sym', ('fast', 0, 10), ('fast', 1, 10), 'mle']:
    reals = []
    estis = []
    for x, y in xys:
        db = np.array([[h(y) for h in hs]], dtype=np.int32)
        sizes = np.array([len(y)], dtype=np.int32)
        estimates, _, _ = estimate(method, x, db, sizes, dom)
        estis.append(estimates[0])
        reals.append(jac(x, y))
    ax.scatter(reals, estis, label=method)

ax.legend()
ax.grid(True)
plt.savefig(args.output)

