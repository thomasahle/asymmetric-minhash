import random
import numpy as np
from math import ceil, floor, log

def sample(u, nx, ny, v, k):
    x = random.sample(range(u), nx)
    assert len(x) == nx
    sx = set(x)
    y_in = random.sample(x, v)
    y_out = set()
    while len(y_out) != ny-v:
        yi = random.randrange(u)
        if yi in sx or yi in y_out:
            continue
        y_out.add(yi)
    y_all = y_in + list(y_out)
    #print(len(y_in), len(y_out))
    #print(len(y_all), len(x), len(set(x) & set(y_all)), ny, nx, v)
    assert len(y_all) == ny and len(x) == nx and len(set(x) & set(y_all)) == v
    y = sorted(y_all)[:k]
    return x, y

def est_classic(x, y, ny, u, *a):
    nx, k = len(x), len(y)
    xy = sorted(set(x + y))[:k]
    s = max(xy)+1
    est_union = (k-1)*u/(s-1)
    return nx + ny - est_union

def est_classic_2(x, y, ny, u, *a):
    nx, k = len(x), len(y)
    s = max(y)+1
    m = sum(1 for xi in x if xi < s)
    c = len(set(x) & set(y))
    k2 = m + k - c
    est_union = k2 * u / s
    return nx + ny - est_union

def est_mikkel(x, y, ny, u, *a):
    nx, k = len(x), len(y)
    c = len(set(x) & set(y))
    return c * ny / k

def est_thomas(x, y, ny, u, *a):
    nx, k = len(x), len(y)
    c = len(set(x) & set(y))
    s = max(y)+1
    m = sum(1 for xi in x if xi < s)
    return c * nx / (m + 1)

def est_min(x, y, ny, u, *a):
    return min(est_mikkel(x, y, ny, u), est_thomas(x, y, ny, u))

def est_mikkel_2(x, y, ny, u, *a):
    nx, k = len(x), len(y)
    c = len(set(x) & set(y))
    s = max(y)+1
    m = sum(1 for xi in x if xi < s)
    #j = c/(m+k-c)
    #est_v = j/(1+j) * (nx + ny)
    est_v = c * (nx + ny) / (m + k)
    return est_v

def est_dif_xy(x, y, ny, u, *a):
    nx, k = len(x), len(y)
    c = len(set(x) & set(y))
    s = max(y)+1
    m = sum(1 for xi in x if xi < s)
    est_xy = (m-c)*u/s
    est_v = nx - est_xy
    return est_v

def est_dif_yx(x, y, ny, u, *a):
    nx, k = len(x), len(y)
    c = len(set(x) & set(y))
    s = max(y)+1
    m = sum(1 for xi in x if xi < s)
    est_yx = (k-c)*u/s
    est_v = ny - est_yx
    return est_v

def pars(x, y):
    nx, k = len(x), len(y)
    c = len(set(x) & set(y))
    s = max(y)+1
    m = sum(1 for xi in x if xi < s)
    return nx, c, k, m, s

def est_mle(x, y, ny, u, v_dontuse):
    nx, c, k, m, s = pars(x, y)
    # Solve the equation
    # ((c - m + nx - v) (c - k + ny - v) v (-nx - ny + u + v))/((c - v) (c - k - m + nx + ny + s - u - v) (-nx + v) (-ny + v)) == 1
    c0 = -(c**2*nx*ny) + c*k*nx*ny + c*m*nx*ny - c*nx**2*ny - c*nx*ny**2 - c*nx*ny*s + c*nx*ny*u
    c1 = -(k*m*nx) + k*nx**2 - k*m*ny + 2*c*nx*ny + m*ny**2 + c*nx*s + c*ny*s + nx*ny*s + c**2*u - c*k*u - c*m*u + k*m*u - k*nx*u - m*ny*u
    c2 = k*m - k*nx - m*ny - c*s - nx*s - ny*s - c*u + k*u + m*u
    c3 = s
    rs = np.roots([c3, c2, c1, c0])
    v = best_root(rs, max(c,c-k-m+nx+ny+s-u), c+min(nx-m, ny-k))
    if v is None:
        print('mle', rs)
        print(f'{v_dontuse=}, {c=}, {s=}, {k=}, {m=}, {nx=}, {ny=}, {u=}')
        return sorted(rs)[1]
        #return c*ny/k
    return v

fac_table = [0]
for i in range(1, 10**4):
    fac_table.append(fac_table[-1] + log(i))
def bi(n, k):
    return fac_table[n] - fac_table[n-k] - fac_table[k]

def est_mle_real(x, y, ny, u, *a):
    nx, c, k, m, s = pars(x, y)
    lo, hi = max(c,c-k-m+nx+ny+s-u), c+min(nx-m, ny-k)
    return max(range(lo, hi+1), key=lambda v:
        bi(nx-m,v-c) - bi(nx,v) + bi(u-s-(nx-m),ny-v-(k-c)) - bi(u-nx,ny-v))

def best_root(rs, lo, hi):
    if ceil(lo) == floor(hi): return floor(hi)
    rs = np.real_if_close(rs, tol=1)
    rrs = rs[~np.iscomplex(rs)] # Remove complex roots
    for eps in [1, .9, 1e-1, 1e-5, 0, -1e-5, -1e-1, -.9 -1]:
        mids = [r for r in rrs if lo-eps <= r <= hi+eps]
        if mids:
            m0 = mids[0]
            if all(np.isclose(m, m0) for m in mids):
                return m0
    # for eps in [1e-5, 0, -1e-5]:
    #     mids = [r for r in rrs if lo-eps <= r <= hi+eps]
    #     print('mids', mids)

def est_newton(x, y, ny, u, *a):
    v = est_mikkel_2(x, y, ny, u)
    nx, k = len(x), len(y)
    c, s = len(set(x) & set(y)), max(y)+1
    m = sum(1 for xi in x if xi < s)
    for _ in range(8):
        h = -((c - v)*(c - k - m + nx + ny + s - u - v)*(-nx + v)*(-ny + v)) \
                + (c - m + nx - v)*(c - k + ny - v)*v*(-nx - ny + u + v)
        d = 2*c*nx*ny + m*ny**2 + c*nx*s + c*ny*s + nx*ny*s + c**2*u - c*m*u \
                - m*ny*u + k*(-(c*u) + (nx - u)*(nx - 2*v) - m*(nx + ny - u - 2*v)) \
                - 2*((c + nx + ny)*s + m*(ny - u) + c*u)*v + 3*s*v**2
        if d != 0:
            v -= h/d
        v = min(max(v, c), c+nx-m, c+ny-k)
    return v


def est_mle_2(x, y, ny, u, *a):
    nx, k = len(x), len(y)
    c = len(set(x) & set(y))
    s = max(y)+1
    m = sum(1 for xi in x if xi < s)
    s /= u # Svarende til at alle værdier er i [0,1]

    # Solve the equation
    # (c - m + nx - v) (c - k + ny - v) v = -(1 - s) (c - v) (nx - v) (ny - v)
    c0 = c*nx*ny*(1 - s)
    c1 = c**2 - c*k - c*m + k*m + c*nx - k*nx + c*ny - m*ny + nx*ny - c*nx*(1 - s) - c*ny*(1 - s) - nx*ny*(1 - s)
    c2 = -2*c + k + m - nx - ny + c*(1 - s) + nx*(1 - s) + ny*(1 - s)
    c3 = s
    rs = np.roots([c3, c2, c1, c0])
    # TODO: Probably use a root selection procedure similar to the one above.
    return sorted(rs)[1]

def est_div(x, y, ny, u, v_dontuse):
    nx, k = len(x), len(y)
    c = len(set(x) & set(y))
    s = max(y)+1
    m = sum(1 for xi in x if xi < s)
    #if c == 0:
    #    c0 = -(nx*ny*s) - k*nx*(nx - u) - m*ny*(ny - u)
    #    c1 = (nx*s + ny*s + k*(nx - u) + m*(ny - u))
    #    c2 = - s
    #    c3 = 0
    #if c == k: return ny
    #if c == 0: return 0
    #if c == m+k-s: return nx+ny-u
    c0 = -(c*nx*ny*(nx + ny - u))
    c1 = (2*c*nx*ny + nx*ny*s + k*nx*(nx - u) + m*ny*(ny - u))
    c2 = (-(nx*s) - ny*s - k*(nx - u) - m*(ny - u) - c*u)
    c3 = s
    rs = np.roots([c3, c2, c1, c0])
    v = best_root(rs, c, c+min(nx-m, ny-k))
    if v is None:
        print('div', rs, [c, c+min(nx-m, ny-k)])
        print(f'{v_dontuse=}, {c=}, {s=}, {k=}, {m=}, {nx=}, {ny=}, {u=}')
        #print(x, y)
        return sorted(rs)[1]
        #return c*ny/k
    return v

def est_div_2(x, y, ny, u, v_dontuse):
    ''' Using all but the last value of y '''
    nx, k = len(x), len(y)-1
    c = len(set(x) & set(y[:-1]))
    s = max(y)
    m = sum(1 for xi in x if xi < s)
    #if c == 0:
    #    c0 = -(nx*ny*s) - k*nx*(nx - u) - m*ny*(ny - u)
    #    c1 = (nx*s + ny*s + k*(nx - u) + m*(ny - u))
    #    c2 = - s
    #    c3 = 0
    #if c == k: return ny
    #if c == 0: return 0
    #if c == m+k-s: return nx+ny-u
    c0 = -(c*nx*ny*(nx + ny - u))
    c1 = (2*c*nx*ny + nx*ny*s + k*nx*(nx - u) + m*ny*(ny - u))
    c2 = (-(nx*s) - ny*s - k*(nx - u) - m*(ny - u) - c*u)
    c3 = s
    rs = np.roots([c3, c2, c1, c0])
    v = best_root(rs, c, c+min(nx-m, ny-k))
    if v is None:
        print('div2', rs, [c, c+min(nx-m, ny-k)])
        print(f'{v_dontuse=}, {c=}, {s=}, {k=}, {m=}, {nx=}, {ny=}, {u=}')
        #print(x, y)
        return sorted(rs)[1]
        #return c*ny/k
    return v

def est_rev_div(x, y, ny, u, v_dontuse):
    nx, k, c = len(x), len(y), len(set(x) & set(y))
    s = max(y)+1
    m = sum(1 for xi in x if xi < s)
    # Four cases of reverse KL in which we have to give a particular answer:
    if c == 0: return 0
    if c == m: return nx
    if c == k: return ny
    if s-m-k+c==0: return nx+ny-u
    c0 = -(c*nx*ny*(c - k - m + s))
    c1 = (c - k)*(-c + m)*nx + (c - k)*(-c + m)*ny + c*nx*(c - k - m + s) + c*ny*(c - k - m + s) + (-c + k)*(-c + m)*u
    c2 = (-c + k)*(-c + m) - c*(c - k - m + s)
    rs = np.roots([c2, c1, c0])
    v = best_root(rs, c-.9, c+min(nx-m, ny-k)+.9)
    if v is None:
        print('rev div', rs)
        print(f'{v_dontuse=}, {c=}, {s=}, {k=}, {m=}, {nx=}, {ny=}, {u=}')
        return c*ny/k
    return v

def est_div_newton(x, y, ny, u, v_dontuse):
    nx, k = len(x), len(y)
    c = len(set(x) & set(y))
    s = max(y)+1
    m = sum(1 for xi in x if xi < s)
    #v = est_min(x, y, ny, u)
    v = c + min(nx-m,ny-k)/2
    for _ in range(8):
        d1, d2, d3, d4 = v, nx-v, ny-v, u-nx-ny+v
        if all(d > 0 for d in [d1,d2,d3,d4]):
            h = (m-c)/d2 + (k-c)/d3 - (c - k - m + s)/d4 - c/d1
            d = (m-c)/d2**2 + (k-c)/d3**2 + (c - k - m + s)/d4**2 + c/d1**2
            v -= h/d
        v = min(max(v, c), c+nx-m, c+ny-k)
    return v

import matplotlib.pyplot as plt

#estimators = [est_classic, est_mikkel, est_mle]
#estimators = [est_mikkel, est_mle, est_mle_2]
#estimators = [est_min, est_newton, est_mle, est_mikkel_2, est_dif_xy, est_mikkel, est_thomas]
#estimators = [est_min, est_mle, est_p]
#estimators = [est_mle, est_mle_2, est_div]
#estimators = [est_div, est_mle, est_mikkel, est_mikkel_2]
estimators = [est_div, est_div_2, est_mikkel_2, est_mle]

#estimators = [est_newton, est_div_newton, est_mle, est_div, est_mikkel_2, est_min]
u, nx, ny, k = 10000, 300, 200, 5
assert u >= nx+ny
vs = range(min(nx, ny)+1)
js = [v/(nx+ny-v) for v in vs]
reps = 1000
fig, axs = plt.subplots(2,2)
fig.suptitle(f'{u=}, {nx=}, {ny=}, {k=}, {reps=}')
axs[0, 0].set_title('MSE, v')
axs[0, 1].set_title('Mean, v')
axs[1, 0].set_title('MSE, jac')
axs[1, 1].set_title('Mean, jac')
for est in estimators:
    series_mse = []
    series_mean = []
    series_jac = []
    series_jac_var = []
    for v in vs:
        mse, mean, jac, jac_var = 0, 0, 0, 0
        for _ in range(reps):
            x, y = sample(u, nx, ny, v, k)
            est_v = est(x, y, ny, u, v)
            mean += est_v
            mse += (est_v - v)**2
            est_j = est_v/(nx+ny-est_v)
            jac += est_j
            jac_var += (v/(nx+ny-v) - est_j)**2
        series_mse.append(mse / reps)
        series_mean.append(mean / reps)
        series_jac.append(jac / reps)
        series_jac_var.append(jac_var / reps)
    axs[0,0].plot(vs, series_mse, label=est.__name__)
    axs[0,1].plot(vs, series_mean, label=est.__name__)
    axs[1,0].plot(js, series_jac_var, label=est.__name__)
    axs[1,1].plot(js, series_jac, label=est.__name__)
#var = [ ((nx - v)*(ny - v)*v*(-nx - ny + u + v))/ (((nx - v)*(ny - v)*v + (nx - v)*(ny - v)*(-nx - ny + u + v) + (nx - v)*v*(-nx - ny + u + v) + (ny - v)*v*(-nx - ny + u + v))) for v in vs]
var = [((ny*(nx - v)*(ny - v)*(nx + ny - u - v)*v)/(-1 + k)/
        (nx**2*ny + nx*ny*(ny - u - 2*v) + u*v**2)
        if v > 0 and nx-v >0 and ny-v>0 and u-nx-ny+v>0
        else 0)
        for v in vs]
       #(k*(nx*ny*(nx + ny - u) - 2*nx*ny*v + u*v**2)) for v in vs]
axs[0,0].plot(vs, var, label='div analytic, k-1')
var = [((ny*(nx - v)*(ny - v)*(nx + ny - u - v)*v)/(k)/
        (nx**2*ny + nx*ny*(ny - u - 2*v) + u*v**2)
        if v > 0 and nx-v >0 and ny-v>0 and u-nx-ny+v>0
        else 0) for v in vs]
       #(k*(nx*ny*(nx + ny - u) - 2*nx*ny*v + u*v**2)) for v in vs]
axs[0,0].plot(vs, var, label='div analytic, k')
#

plt.legend()
plt.show()

