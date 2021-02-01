import numpy as np
import random
import os.path
import argparse
import multiprocessing
from functools import partial

from aminhash import datasets, estimate, Hash, jac

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data', type=str, default='netflix', choices=datasets.files.keys())
parser.add_argument('-N', type=int, default=None, help='Number of datapoints')
parser.add_argument('-M', type=int, default=100, help='Number of queries')
parser.add_argument('-K', type=int, default=30, help='Number of minhashes')
parser.add_argument('--newton', type=int, default=1, help='Number of Newton steps in some methods')
parser.add_argument('--method', type=str, default='fast', help='Whether to use simpler formula')
parser.add_argument('--type', type=int, default=10, help='Extra parameter for comb7')
args = parser.parse_args()

M, K = args.M, args.K
R = 10

if args.method == 'fast':
    args.method = ('fast', args.newton, args.type)

print('Loading data')
data, dom = datasets.load(args.data, verbose=True, trim=args.N)
N = len(data)
data, qs = data[:N-M], data[N-M:N]

print(f'Brute forcing')
cache_file = f'tresholds_{args.data}_{N}_{M}_{R}.npy'
if os.path.exists(cache_file):
    answers = np.load(cache_file)
else:
    answers = []
    pool = multiprocessing.Pool(20)
    for i, q in enumerate(qs):
        print(f'{i}/{len(qs)}', end='\r', flush=True)
        x = set(q)
        #def f(x, y): return jac(x, y)
        js = pool.map(partial(jac, set(q)), data)
        js = np.array(js)
        #js = np.array([jac(x, y) for y in data])
        # We store the smallest acceptable similarity, rather than
        # the acutal indicies, since duplicate similarities would
        # otherwise lead to wrong reported results.
        tr = -np.partition(-js, R)[R-1]
        answers.append(tr)
    answers = np.array(answers)
    np.save(cache_file, answers)

print('Making hash functions')
hs = [Hash(dom) for _ in range(K)]

print('Hashing')
hash_file = f'hashes_{N}_{M}_{K}.npy'
if os.path.exists(hash_file):
    size_db = np.load(hash_file)
    db = np.ascontiguousarray(size_db[:, 1:]).astype(np.int32)
    sizes = np.ascontiguousarray(size_db[:, 0]).astype(np.int32)
else:
    db = np.array([[h(y) for h in hs] for y in data], dtype=np.int32)
    sizes = np.array([len(y) for y in data], dtype=np.int32)
    np.save(hash_file, np.hstack((sizes[:, np.newaxis], db)))

print('Sample hash:')
print(db[0])

print(f'Computing recall with method {args.method}, {args.type}')

estimates = np.zeros(N, dtype=np.float32)  # Space for storing results
t1, t2 = 0, 0
total = 0
for i, (q, threshold) in enumerate(zip(qs, answers)):
    print(f'{i}/{len(qs)} r~{total / i if i > 0 else 0}', end='\r', flush=True)
    if i % 400 == 0:
        print()
    estimates, t1, t2 = estimate(args.method, q, db, sizes, dom, hs, estimates)
    guess = np.argmax(estimates)
    realj = jac(set(q), db[guess])
    total += int(realj > threshold or np.isclose(realj, threshold))
print(f'{R1}@{R2}: {total / len(qs)}')
print(f'Time preparing: {t1=}, Time searching: {t2=}')
