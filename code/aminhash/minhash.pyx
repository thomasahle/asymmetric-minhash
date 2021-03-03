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
    int i, max_table = 10**6 # Flickr has N=810,660
    double[::1] ftable = np.zeros(max_table)
for i in range(1, max_table):
    ftable[i] = log(i) + ftable[i-1]

# Using table lookups
cdef double ent3(int a, int b) nogil:
    ''' Returns log Binom(a,b) '''
    if b < 0 or b > a: return log(0) # -inf
    if a == 0 or b == 0 or a == b: return 0
    return ftable[a] - ftable[a-b] - ftable[b]

cpdef float mle(int[::1] data, int x, int y, int u,
                int[:,::1] mtab, int[:,::1] ktab, bool[::1] ctab) nogil:
    cdef:
        int dim = data.shape[0]
        int i, j, dat
        int m, k
        int v, best_v
        double best_logp, logp
    best_v = 0
    best_logp = log(0)
    # TODO: Make a tertiarray search instead of linear
    for v in range(min(x,y)+1):
        logp = 0
        for j in range(dim):
            dat = data[j]
            m = mtab[j][dat]
            k = ktab[j][dat]
            if not ctab[dat]:
                logp += ent3(x-k, v) + ent3(u-m-1-(x-k), y-v-1)
            else:
                logp += ent3(x-k-1, v-1) + ent3(u-m-(x-k), y-v)
            logp -= ent3(x, v) + ent3(u-x, y-v)
        if logp > best_logp:
            best_v, best_logp = v, logp
    return best_v/<double>(x+y-best_v)

cpdef void query_mle(int[:,::1] data, int x, int u, int[::1] ysz,
        int[:,::1] mtab, int[:,::1] ktab, bool[::1] ctab, float[::1] out) nogil:
    cdef:
        int n = data.shape[0]
        int i, y

    for i in range(n):
        y = ysz[i]
        out[i] = mle(data[i], x, ysz[i], u, mtab, ktab, ctab)

cpdef void query2(int[:,::1] data, int x, int u, int[::1] ysz, int newtons, int type,
        int[:,::1] mtab, int[:,::1] ktab, bool[::1] ctab, float[::1] out) nogil:
    cdef:
        int n = data.shape[0]
        int dim = data.shape[1]
        int i, j, dat
        float m, k, a, b, y, h, im
        float jac, v, nwtn

    for i in range(n):
        y = ysz[i]
        m, k, b, ham, im = 0, 0, 0, 0, 0
        for j in range(dim):
            dat = data[i, j]
            m += mtab[j][dat]
            im += u/<double>(1+mtab[j][dat])
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
        elif type == 14: v = b * min(x/(b+k), u/m) 
        elif type == 15: v = b * min(x/(b+k), im/<double>dim/<double>dim) 
        v = max(min(v, x, y), 0)
        for j in range(newtons):
            nwtn = m/u + b/(v+1) - a/(y-v+1) - k/(x-v+1)
            nwtn /= b/(v+1)**2 + a/(y-v+1)**2 + k/(x-v+1)**2
            v += nwtn

        v = max(min(v, x, y), 0)
        out[i] = v/(x+y-v)
        #out[i] = jac

cpdef void bottomk(int[:,::1] data, int nx, int u, int[::1] nys, int[::1] xh,
                   int type, float[::1] out) nogil:
    cdef:
        int n = data.shape[0]
        int k = data.shape[1]
        int i, j, ry, jx, ny, c, s, m
        double v, ss, mm, kk, uu, cc, nnx, nny, v0
        int a, b, mid
        int best_v, vt
        double best_logp, logp
        double d1, d2, d3, d4

    for i in range(n):
        ny = nys[i]
        jx, c = -1, 0
        for j in range(k):
            ry = data[i, j]
            #while jx+1 < nx and xh[jx+1] <= ry:
            #    jx += 1
            a, b = 0, 1
            while jx+b < nx and xh[jx + b] <= ry:
                b *= 2
            while a+1 != b:
                mid = (a + b) // 2
                if jx+mid < nx and xh[jx+mid] <= ry:
                    a = mid
                else: b = mid
            jx += a

            if xh[jx] == ry:
                c += 1
        s = ry+1 # s the number of 'squares' we can see
        m = jx+1 # m is the number of x hashes < s
        if ry == u: # y data is padded with [u]s
            v = c # If we saw all of y, v is simply c
        elif type == 0:
            v = c * (nx + ny)/<double>(m + k)
        elif type == 1:
            v = c * min(nx/<double>(m+1), ny/<double>k)
        elif type == 2:
            v = c * min(nx/<double>(m+1), ny/<double>k)
            for _ in range(8):
                d1, d2, d3, d4 = v, nx-v, ny-v, u-nx-ny+v
                if d1 > 0 and d2 > 0 and d3 > 0 and d4 > 0:
                    h = (m-c)/d2 + (k-c)/d3 - (c - k - m + s)/d4 - c/d1
                    d = (m-c)/d2**2 + (k-c)/d3**2 + (c - k - m + s)/d4**2 + c/d1**2
                    v -= h/d
                v = min(max(v, c), c+nx-m, c+ny-k)
        elif type == 3:
            if c == 0: v = 0
            elif c == m: v = nx
            elif c == k: v = ny
            elif c == m+k-s: v = nx+ny-u
            else:
                v = c * min(nx/<double>(m+1), ny/<double>k)
                for _ in range(8):
                    h = c*(c - k - m + s)*(nx - v)*(-ny + v) + (c - k)*(c - m)*v*(-nx - ny + u + v)
                    d = c*((c - m)*u + s*(nx + ny - 2*v)) - k*(c*u + m*(nx + ny - u - 2*v)) 
                    if d != 0:
                        v -= h/d
                    v = min(max(v, c), c+nx-m, c+ny-k)
        else:
            best_v, best_logp = 0, log(0)
            for vt in range(c, c+min(nx-m,ny-k)+1):
                logp = ent3(nx-m, vt-c) - ent3(nx, vt)
                logp += ent3(u-s-(nx-m), ny-vt-(k-c)) - ent3(u-nx, ny-vt)
                if logp > best_logp:
                    best_v, best_logp = vt, logp
            v = best_v

        out[i] = v/<double>(nx + ny - v)
