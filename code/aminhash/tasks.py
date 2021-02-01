import numpy as np
import multiprocessing
import os.path
import functools


def run(f, tasks, args=(), cache_file=None, verbose=False, processes=10):
    if cache_file and os.path.exists(cache_file):
        return np.load(cache_file)

    res = []
    with multiprocessing.Pool(processes=processes) as pool:
        for i, v in enumerate(pool.imap(functools.partial(f, args), tasks)):
            if verbose:
                print(f'{i}/{len(tasks)}', end='\r', flush=True)
            res.append(v)

    res = np.array(res)
    if cache_file:
        np.save(res, cache_file)
    return res
