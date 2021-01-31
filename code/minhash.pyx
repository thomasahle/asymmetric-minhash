#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True, language_level=3
from libcpp cimport bool
import numpy as np
cimport numpy as np
from libc.math cimport log, log1p

cpdef void query(int[:,::1] data, int x, int u, int[::1] ysz, int newtons,
        int[:,::1] mtab, int[:,::1] ktab, bool[::1] ctab, float[::1] out) nogil:
    cdef:
        int n = data.shape[0]
        int dim = data.shape[1]
        int i, j, dat
        float m, k, a, b, ham, y
        float jac, v, nwtn

    for i in range(n):
        y = ysz[i]
        m, k, b, ham = 0, 0, 0, 0
        for j in range(dim):
            dat = data[i, j]
            m += mtab[j][dat]
            k += ktab[j][dat]
            b += ctab[dat]
            ham += ktab[j][dat] == 0 and ctab[dat] == True
        a = dim - b

        #print(f'x={x}, y={y}, m={m}, k={k}, b={b}, a={a}, ham={ham}, j={ham/dim}')

        jac = ham/dim
        v = min(jac/(1+jac)*(x+y), x, y) # 0.1821
        for j in range(newtons):
            nwtn = m/u + b/(v+1) - a/(y-v+1) - k/(x-v+1)
            nwtn /= b/(v+1)**2 + a/(y-v+1)**2 + k/(x-v+1)**2
            v += nwtn

        v = max(min(v, x, y), 0)
        out[i] = v/(x+y-v)
        #out[i] = jac

# Using a stirling approach
cdef double ent(int a, int b) nogil:
    ''' Returns a H(a/b) '''
    if b < 0 or b > a: return log(0) # -inf
    if a == 0 or b == 0 or a == b: return 0
    return b * log(a/<double>b) + (a-b) * log(a/<double>(a-b)) + log(a/<double>(b*(a-b)))/2

# Using division
cdef double ent2(int a, int b) nogil:
    ''' Returns log Binom(a,b) '''
    if b < 0 or b > a: return log(0) # -inf
    if a == 0 or b == 0 or a == b: return 0
    cdef:
        double res = 0
        int i
    for i in range(b):
        res += log((a-i)/<double>(b-i))
    return res

cdef:
    int i, max_table = 10**5
    double[::1] ftable = np.zeros(max_table)
for i in range(1, max_table):
    ftable[i] = log(i) + ftable[i-1]

# Using table lookups
cdef double ent3(int a, int b) nogil:
    ''' Returns log Binom(a,b) '''
    if b < 0 or b > a: return log(0) # -inf
    if a == 0 or b == 0 or a == b: return 0
    return ftable[a] - ftable[a-b] - ftable[b]

cpdef void query_mle(int[:,::1] data, int x, int u, int[::1] ysz,
        int[:,::1] mtab, int[:,::1] ktab, bool[::1] ctab, float[::1] out) nogil:
    cdef:
        int n = data.shape[0]
        int dim = data.shape[1]
        int i, j, dat
        int m, k, y
        int v, best_v
        double best_logp, logp

    for i in range(n):
        y = ysz[i]
        best_v = 0
        best_logp = log(0)
        # TODO: Make a tertiarray search instead of linear
        for v in range(min(x,y)+1):
            logp = 0
            for j in range(dim):
                dat = data[i, j]
                m = mtab[j][dat]
                k = ktab[j][dat]
                if not ctab[dat]:
                    logp += ent3(x-k, v) + ent3(u-m-1-(x-k), y-v-1)
                else:
                    logp += ent3(x-k-1, v-1) + ent3(u-m-(x-k), y-v)
                logp -= ent3(x, v) + ent3(u-x, y-v)
            #print(v, x, y, logp)
            if logp > best_logp:
                best_v, best_logp = v, logp
        #print(best_v, best_logp)
        out[i] = best_v/<double>(x+y-best_v)

cpdef void query2(int[:,::1] data, int x, int u, int[::1] ysz, int newtons, int type,
        int[:,::1] mtab, int[:,::1] ktab, bool[::1] ctab, float[::1] out) nogil:
    cdef:
        int n = data.shape[0]
        int dim = data.shape[1]
        int i, j, dat
        float m, k, a, b, y, h
        float jac, v, nwtn

    for i in range(n):
        y = ysz[i]
        m, k, b, ham = 0, 0, 0, 0
        for j in range(dim):
            dat = data[i, j]
            m += mtab[j][dat]
            k += ktab[j][dat]
            b += ctab[dat]
        a = dim - b

        h = m/u
        if type ==   0: v = y - a/h # 0.0685
        elif type == 1: v = x - k/h # 0.1462
        elif type == 2: v = (a*x+k*y)/(a+k) # 0.019
        elif type == 3: v = b * x / (b + k) # 0.1818
        elif type == 4: v = b * y / (a + b) # 0.1718
        elif type == 5: v = b/(a/y + k/x - h) # 0.0459
        elif type == 6: v = b * (x/(b+k)+y/(a+b))/2 # 0.1858
        elif type == 7: v = b * 2/((b+k)/x+(a+b)/y) # 0.1885
        elif type == 8: v = b * (x+y)/(a+b+k) # 0.1537
        elif type == 9: v = b * (x+y)/(a+2*b+k) # 0.1854
        elif type == 10: v = b * min(x/(b+k), y/(a+b)) # 0.1921 <-- Nice! newton=0: 0.1948!!!
        elif type == 11: v = min(b*x/(b+k), b*y/(a+b), x-k/h) # 0.1835
        elif type == 12: v = x-(y+1)*k/dim # 0.0243
        elif type == 13: v = min(x-(y+1)*k/dim, b*y/dim) # 0.1638
        v = max(min(v, x, y), 0)
        for j in range(newtons):
            nwtn = m/u + b/(v+1) - a/(y-v+1) - k/(x-v+1)
            nwtn /= b/(v+1)**2 + a/(y-v+1)**2 + k/(x-v+1)**2
            v += nwtn

        v = max(min(v, x, y), 0)
        out[i] = v/(x+y-v)
        #out[i] = jac

