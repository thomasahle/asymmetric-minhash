import numpy as np
import random
import itertools
from math import exp, log, prod

def H(p):
    if p == 0 or p == 1: return 0
    return p*log(1/p) + (1-p)*log(1/(1-p))

def f(ps, s):
    return sum(p**2*(1-p) for p in ps[s:])
    #return sum(p*(1-p) for p in ps[s:])

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-N', type=int)
parser.add_argument('-K', type=int)
parser.add_argument('-T', type=float, default=0)
parser.add_argument('-A', type=float, default=1)
args = parser.parse_args()

N = args.N
K = args.K
pps = [random.random() for _ in range(N)]
#pps = [(i+1)/(N+1) for i in range(N)]
#alpha = args.A
#pps = [alpha/(i+1)**(alpha+1) for i in range(N)]
#print(pps)
#tilt = args.T
#pps = [p*exp(tilt)/(1-p+p*exp(tilt)) for p in pps]
# for p in pps:
#     assert 0 < p < 1
pps.sort()
print(pps)
print('Mean:', sum(pps))

def method1():
    score = 0
    for ps in itertools.permutations(pps):
        sc = -1/f(ps, 0)
        pks = np.poly1d([0,1])
        for s in range(0, N):
            if s+1 >= K:
                # Probability that there are K-1 "good" entries in the first S-1
                # variables, and the Sth variable in particular is good too.
                p = pks[K-1] * ps[s]
                # Maybe s+1, but then we get zero division...
                # This is Mikkel's usual "Don't include the last value" horse.
                sc += p * 1/f(ps, s)
            # Add sth entry to polynomium
            pks *= np.poly1d([ps[s],1-ps[s]])
        if sc > score:
            print(' '*80, end='\r')
            print(sc, [pps.index(p) for p in ps], end='\r')
            score = sc
    print()

def plogp(p):
    if p in [0,1]: return 0
    return p * log(1/p)

def ent_dp(ps):
    ''' Returns a table dp, such that dp[n][k] = entropy of bottom-k sample
        on the n-prefix of ps. '''
    N = len(ps)
    h = [[0]*(n+1) for n in range(N+1)]
    f = [[0]*(n+1) for n in range(N+1)]
    f[0][0] = 1
    for n in range(N):
        f[n+1][0] = (1-ps[n])*f[n][0]
        h[n+1][0] = (1-ps[n])*h[n][0] + plogp(1-ps[n])*f[n][0]
        for k in range(1, n+1):
            f[n+1][k] = ps[n]*f[n][k-1] + (1-ps[n])*f[n][k]
            h[n+1][k] = ps[n]*h[n][k-1] + (1-ps[n])*h[n][k] \
                      + plogp(ps[n])*f[n][k-1] \
                      + plogp(1-ps[n])*f[n][k]
        f[n+1][n+1] = ps[n]*f[n][n]
        h[n+1][n+1] = ps[n]*h[n][n] + plogp(ps[n])*f[n][n]
    return f, h

if True:
    ps = [random.random() for _ in range(10)]
    fdp, hdp = ent_dp(ps)
    for n in range(len(ps)):
        for k in range(n+1):
            naiive_h = 0
            naiive_f = 0
            for sub in itertools.combinations(ps[:n], k):
                term = prod(sub) * prod(1-p for p in ps[:n] if p not in sub)
                naiive_h += term * log(1/term)
                naiive_f += term
            assert np.isclose(hdp[n][k], naiive_h)
            assert np.isclose(fdp[n][k], naiive_f)
            assert fdp[n][k] * log(1/fdp[n][k]) <= hdp[n][k]+1e-5 # Jensen
        # Test additivity of entropy
        row_h = sum(plogp(p)+plogp(1-p) for p in ps[:n])
        assert np.isclose(sum(hdp[n]), row_h)
        # Probabilities just sum to 1
        assert np.isclose(sum(fdp[n]), 1)

def method_ent():
    ''' Maximize E[sum_{i<=S} H(pi)] '''
    score, best = 0, None
    for ps in itertools.permutations(pps):
        sc = 0
        fdp, hdp = ent_dp(ps)
        check = 0
        # The cases in which we get a full bottom-k sketch
        for s in range(K-1, N):
            sc += ps[s] * (log(1/ps[s]) * fdp[s][K-1] + hdp[s][K-1])
            #sc += ps[s] * log(1/ps[s]) * fdp[s][K-1] # If I only use this I get (7, 0, 1, 2, ...)
            #sc += ps[s] * hdp[s][K-1] # Using only this is enough for (0, 1, 2, ...)
            check += ps[s] * fdp[s][K-1]
        # There is also the case that we just never get K values...
        #for k in range(K):
        #    sc += hdp[N][k]
        #    check += fdp[N][k]
        # Make sure we have included every event
        #assert np.isclose(check, 1)
        if sc > score:
            #print(' '*80, end='\r')
            #print(sc, [pps.index(p) for p in ps], end='\r')
            print(sc, [pps.index(p) for p in ps])
            best = ps
            score = sc
    ps = best
    print()
    print(pps)
    print(ps)
    fdp, hdp = ent_dp(ps)
    # print('fdp')
    # for row in fdp:
    #     print(row)
    # print('hdp')
    # for row in hdp:
    #     print(row)
    for s in range(K-1, N-1):
        print(f'{s=}', ps[s+1] * (log(1/ps[s+1]) * fdp[s][K-1] + hdp[s][K-1]))
    # There is also the case that we just never get K values...
    for k in range(K):
        print(f'{k=}', hdp[N][k])

#method1()
method_ent()
