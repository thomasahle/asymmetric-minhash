import numpy as np
import random
import os.path
import argparse

from .aminhash import datasets, estimate, Hash, jac

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data', type=str, default='netflix', choices=datasets.files.keys())
parser.add_argument('-N', type=int, default=None, help='Number of datapoints')
parser.add_argument('-M', type=int, default=100, help='Number of queries')
parser.add_argument('-K', type=int, default=30, help='Number of minhashes')
parser.add_argument('--newton', type=int, default=1, help='Number of Newton steps in some methods')
parser.add_argument('--method', type=str, help='Whether to use simpler formula')
parser.add_argument('--type', type=int, default=0, help='Extra parameter for comb7')
args = parser.parse_args()

N, M, K = args.N, args.M, args.K
R1 = 1
R2 = 10

if args.method == 'fast':
    args.method = ('fast', args.newton, args.type)

print('Loading data')
data, dom = datasets.load(args.data)
random.shuffle(data)
data, qs = data[:N], data[N:N + M]

print(f'Brute forcing')
cache_file = f'answers_{N}_{M}_{R1}.npy'
if os.path.exists(cache_file):
    answers = np.load(cache_file)
else:
    answers = []
    for q in qs:
        js = np.array([jac(q, y) for y in data])
        top = np.argpartition(-js, R1)[:R1]
        answers.append(top)
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

print('Computing recall with method {args.method}, {args.type}')

estimates = np.zeros(N, dtype=np.float32)  # Space for storing results
t1, t2 = 0, 0
total = 0
for i, (q, answer) in enumerate(zip(qs, answers)):
    print(f'{i}/{len(qs)} r~{total / i if i > 0 else 0}', end='\r', flush=True)
    if i % 400 == 0:
        print()
    estimates, t1, t2 = estimate(args.method, q, db, sizes, dom, hs, estimates)
    candidates = np.argpartition(-estimates, R2)[:R2]
    r = len(set(candidates) & set(answer)) / R1
    total += r
print(f'{R1}@{R2}: {total / len(qs)}')
print(f'Time preparing: {t1=}, Time searching: {t2=}')
