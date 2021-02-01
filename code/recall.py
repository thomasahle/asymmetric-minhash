import numpy as np
import os.path
import argparse

from aminhash import datasets, estimate, Hash, jac, tasks

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data', type=str, default='netflix', choices=datasets.files.keys())
parser.add_argument('-N', type=int, default=None, help='Number of datapoints')
parser.add_argument('-M', type=int, default=100, help='Number of queries')
parser.add_argument('-K', type=int, default=30, help='Number of minhashes')
parser.add_argument('--method', type=str, default='fast', help='Whether to use simpler formula')
parser.add_argument('--newton', type=int, default=0, help='Number of Newton steps in some methods')
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
N -= M
print(f'{N=}, {dom=}')

print(f'Brute forcing')
cache_file = f'thresholds_{args.data}_{N}_{M}_{R}.npy'
def inner_brute(ys, q):
    x = set(q)
    js = np.array([jac(x, y) for y in ys])
    # We store the smallest acceptable similarity, rather than
    # the acutal indicies, since duplicate similarities would
    # otherwise lead to wrong reported results.
    return -np.sort(np.partition(-js, R))[R - 1]
# TODO: This will pickle and copy data to every process. Can we maybe use
# shared memory instead?
answers = tasks.run(inner_brute, qs, args=(data,), cache_file=cache_file,
                    verbose=True, processes=40, report_interval=1)

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
    realj = jac(set(q), data[guess])
    total += int(realj > threshold or np.isclose(realj, threshold))
    #print(f'est: {estimates[guess]}, {realj=}, {threshold=}')
print(f'recall@{R}: {total / len(qs)}')
print(f'Time preparing: {t1=}, Time searching: {t2=}')

