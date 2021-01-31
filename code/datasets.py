files = dict(
    netflix = '/home/jovt/simfilter/data/ssjoin-data/netflix/netflix-dedup-raw-noone.txt',
    flickr = '/home/jovt/simfilter/data/ssjoin-data/flickr/flickr-dedup-raw-noone.txt',
)


def load(dataset):
    file = open(files[dataset])
    data = [list(map(int, line.split())) for line in file]
    dom = max(v for y in data for v in y) + 1
    return data, dom