import os.path
for data in ['netflix', 'flickr', 'dblp']:
    if data in ('netflix', 'dblp'):
        ks = [1, 10, 30, 100, 400, 500]
    else: ks = [1, 5, 10, 20, 30]
    for k in ks:
        print(str(k).rjust(4), end=' & ')
        algs = ['sym', 'mle', 'fast_n0', 'fast_n1', 'fast_n8']
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
