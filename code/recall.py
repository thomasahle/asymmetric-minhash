import numpy as np
import os.path
import argparse
import random
import collections

from aminhash import datasets, estimate, estimate_bottomk, estimate_weighted, Hash, jac, tasks

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data', type=str, default='netflix', choices=datasets.files.keys())
parser.add_argument('-N', type=int, default=None, help='Number of datapoints')
parser.add_argument('-M', type=int, default=100, help='Number of queries')
parser.add_argument('-K', type=int, default=30, help='Number of minhashes')
parser.add_argument('-R', type=int, default=10, help='Recall@R')
parser.add_argument('--bottomk', action='store_true', help='Use bottom-k instead of k-minhash')
parser.add_argument('--weighted', action='store_true', help='Use bottom-k with probability estimates')
parser.add_argument('--method', type=str, default='fast', help='Whether to use simpler formula')
parser.add_argument('--newton', type=int, default=0, help='Number of Newton steps in some methods')
parser.add_argument('--type', type=int, default=10, help='Extra parameter for comb7')
args = parser.parse_args()

M, K = args.M, args.K
# We only do 1-recall@R. That is, we compare the top R returned by
# the estimator with the single true nearest neighbour. This makes
# sense in the setting where we use the sketch distance as a "first pass"
# through the data.
R = args.R

if args.method == 'fast':
    args.method = ('fast', args.newton, args.type)

print('Loading data')
data, dom = datasets.load(args.data, verbose=True, trim=args.N)
N = len(data)
data, qs = data[:N-M], data[N-M:N]
N -= M
print(f'{N=}, {dom=}')

print(f'Brute forcing')
cache_file = f'thresholds_{args.data}_{N}.npy'
def inner_brute(ys, q):
    x = set(q)
    js = np.array([jac(x, y) for y in ys])
    return np.max(js)
# TODO: This will pickle and copy data to every process. Can we maybe use
# shared memory instead?
answers = tasks.run(inner_brute, qs, args=(data,), cache_file=cache_file,
                    verbose=True, processes=40, report_interval=1)

print('Making hash functions')
random.seed(74)
if args.bottomk:
    hs = [Hash(dom) for _ in range(1)]
    cnt = collections.Counter(tok for x in data for tok in x)
    for j, (i, _) in enumerate(reversed(cnt.most_common())):
        hs[0].perm[i] = j
elif args.weighted:
    cnt = collections.Counter(tok for x in data for tok in x)
    hs = [Hash(dom) for _ in range(1)]
    for j, (i, _) in enumerate(reversed(cnt.most_common())):
        hs[0].perm[i] = j
    ps = np.array([cnt[tok] for tok in range(dom)]) / len(data)
    ps.sort()
else:
    hs = [Hash(dom) for _ in range(K)]

print('Hashing')
hash_file = f'hashes_{args.data}_botk={args.bottomk or args.weighted}_{N=}_{K=}.npy'
def inner_hashing(hs, y):
    if args.bottomk or args.weighted:
        botk = sorted(hs[0].perm[yi] for yi in y)[:K]
        # We pad with values outside of the universe.
        botk += [len(hs[0].perm)]*(K-len(botk))
        assert len(botk) == K
        return [len(y)] + botk
    return [len(y)] + [h(y) for h in hs]
size_db = tasks.run(inner_hashing, data, args=(hs,), cache_file=hash_file,
                    verbose=True, chunksize=10000, report_interval=1000, perc=True)
db = np.ascontiguousarray(size_db[:, 1:]).astype(np.int32)
sizes = np.ascontiguousarray(size_db[:, 0]).astype(np.int32)

print('Sample hashes:')
print(db)
print(f'Computing recall@{R} with method {args.bottomk=} {args.weighted=} {args.method}, {args.type}')

estimates = np.zeros(N, dtype=np.float32)  # Space for storing results
t1, t2 = 0, 0
total = 0
for i, (q, threshold) in enumerate(zip(qs, answers)):
    #print(f'{i}/{len(qs)} r~{total / i if i > 0 else 0}', end='\r', flush=True)
    print(f'{i}/{len(qs)} r~{total / i if i > 0 else 0}')
    if i % 400 == 0:
        print()
    if args.bottomk:
        estimates, t1, t2 = estimate_bottomk(args.method, q, db, sizes, dom, hs[0], estimates)
    elif args.weighted:
        hq = sorted(hs[0].perm[qi] for qi in q)
        estimates, t1, t2 = estimate_weighted(ps, hq, db, sizes)
    else:
        estimates, t1, t2 = estimate(args.method, q, db, sizes, dom, hs, estimates)
    guesses = np.argpartition(-estimates, R)[:R]
    realj = max(jac(set(q), data[g]) for g in guesses) # brute force the guesses
    total += int(realj >= threshold or np.isclose(realj, threshold))
    print(f'est: {realj=}, {threshold=}')
print(f'recall@{R}: {total / len(qs)}')
print(f'Time preparing: {t1=}, Time searching: {t2=}')

