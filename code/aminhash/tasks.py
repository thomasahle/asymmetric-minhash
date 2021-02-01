import numpy as np
import multiprocessing
import os.path
import functools


def run(f, tasks, args=(), cache_file=None, verbose=False, processes=10, chunksize=100,
            report_interval=1, perc=False):
    if cache_file and os.path.exists(cache_file):
        if verbose:
            print(f'Using cached file {cache_file}')
        return np.load(cache_file)

    res = []
    if verbose:
        print(f'Starting {processes=}.')
    with multiprocessing.Pool(processes=processes) as pool:
        for i, v in enumerate(pool.imap(functools.partial(f, *args), tasks,
                              chunksize=chunksize)):
            if verbose and i % report_interval == 0:
                if perc:
                    print(f'{i/len(tasks):.1%}', end='\r', flush=True)
                else:
                    print(f'{i}/{len(tasks)}', end='\r', flush=True)
            res.append(v)

    res = np.array(res)
    if cache_file:
        np.save(cache_file, res)
    return res
