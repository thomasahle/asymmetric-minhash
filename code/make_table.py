import os.path
import sys, re
import collections

data, algs = set(), set()
ks = collections.defaultdict(set)
for fn in sys.argv[1:]:
    dat, k, alg = re.match(r'recall_(.+?)_(\d+)_(.+?)\.out', fn).groups()
    data.add(dat)
    ks[dat].add(int(k))
    algs.add(alg)
data = sorted(data)
algs = sorted(algs)

for data in data:
    print(f'%{data}')
    for alg in ['K'] + algs:
        print(alg.rjust(4), end='')
        if alg == algs[-1]:
            print(' \\\\')
        else: print(' & ', end='')
    for k in sorted(ks[data]):
        print(str(k).rjust(4), end=' & ')
        vals = []
        for alg in algs:
            fn = f'recall_{data}_{k}_{alg}.out'
            if not os.path.isfile(fn):
                continue
            line = open(fn).readlines()[-2]
            if 'recall' in line:
                _, acc = line.split()
                vals.append(float(acc))
            else:
                vals.append(-1)
        for i, v in enumerate(vals):
            if v < 0:
                print('.'.rjust(6), end=' ')
            elif v == max(vals):
                print('\\textbf{', f'{v:.4f}'.rjust(6), end='} ')
            else:
                print(f'{v:.4f}'.rjust(6), end=' ')
            if i != len(vals)-1:
                print('& ', end='')
        print('\\\\')
    print()
