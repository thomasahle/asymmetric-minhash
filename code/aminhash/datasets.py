import random


files = dict(
    netflix = '/home/jovt/simfilter/data/ssjoin-data/netflix/netflix-dedup-raw-noone.txt',
    flickr = '/home/jovt/simfilter/data/ssjoin-data/flickr/flickr-dedup-raw-noone.txt',
)


def load(dataset, verbose=False, trim=None):
    with open(files[dataset]) as file:
        lines = file.readlines()
        random.shuffle(lines)
        if trim:
            lines = lines[:trim]
        data = []
        for i, line in enumerate(lines):
            if verbose and i % 100 == 0:
                print(f'{i/len(lines):.1%}', end='\r', flush=True)
            data.append(list(map(int, line.split())))
        if verbose:
            print('100%')
        dom = max(v for y in data for v in y) + 1
        return data, dom
