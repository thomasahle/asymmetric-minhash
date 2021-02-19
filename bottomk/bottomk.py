import random
import numpy as np

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
    s = max(xy)
    est_union = (k-1)*u/(s-1)
    return nx + ny - est_union

def est_mikkel(x, y, ny, u, *a):
    nx, k = len(x), len(y)
    c = len(set(x) & set(y))
    return c * ny / k

def est_thomas(x, y, ny, u, *a):
    nx, k = len(x), len(y)
    c = len(set(x) & set(y))
    s = max(y)
    m = sum(1 for xi in x if xi <= s)
    return c * nx / m

def est_min(x, y, ny, u, *a):
    return min(est_mikkel(x, y, ny, u), est_thomas(x, y, ny, u))

def est_p(x, y, ny, u, *a):
    nx, k = len(x), len(y)
    c = len(set(x) & set(y))
    s = max(y)
    m = sum(1 for xi in x if xi <= s)
    #j = c/(m+k-c)
    #est_v = j/(1+j) * (nx + ny)
    est_v = c * (nx + ny) / (m + k)
    return est_v

def est_mle(x, y, ny, u, v_dontuse):
    nx, k = len(x), len(y)
    c = len(set(x) & set(y))
    s = max(y)
    m = sum(1 for xi in x if xi <= s)
    # Solve the equation
    # ((c - m + nx - v) (c - k + ny - v) v (-nx - ny + u + v))/((c - v) (c - k - m + nx + ny + s - u - v) (-nx + v) (-ny + v)) == 1
    c0 = -(c**2*nx*ny) + c*k*nx*ny + c*m*nx*ny - c*nx**2*ny - c*nx*ny**2 - c*nx*ny*s + c*nx*ny*u
    c1 = -(k*m*nx) + k*nx**2 - k*m*ny + 2*c*nx*ny + m*ny**2 + c*nx*s + c*ny*s + nx*ny*s + c**2*u - c*k*u - c*m*u + k*m*u - k*nx*u - m*ny*u
    c2 = k*m - k*nx - m*ny - c*s - nx*s - ny*s - c*u + k*u + m*u
    c3 = s
    rs = np.roots([c3, c2, c1, c0])
    rs = np.real_if_close(rs, tol=1)
    rrs = rs[~np.iscomplex(rs)] # Remove complex roots
    # Try to find a limited epsilon range that singles out a unique solution
    for eps in [1e-5, 0, -1e-5, 1e-5]:
        mids = [r for r in rrs if c-eps <= r <= c+min(nx-m, ny-k)+eps]
        if len(mids) == 1:
            return mids[0]
    print(rs, v_dontuse, c, s, m)
    return mids[0]
    #srs = sorted(rrs)
    #assert 0 <= srs[1] <= min(nx,ny)
    #print(rs)
    #return srs[1]

def est_newton(x, y, ny, u, *a):
    v = est_p(x, y, ny, u)
    nx, k = len(x), len(y)
    c, s = len(set(x) & set(y)), max(y)
    m = sum(1 for xi in x if xi <= s)
    for _ in range(3):
        h = -((c - v)*(c - k - m + nx + ny + s - u - v)*(-nx + v)*(-ny + v)) + (c - m + nx - v)*(c - k + ny - v)*v*(-nx - ny + u + v)
        d = 2*c*nx*ny + m*ny**2 + c*nx*s + c*ny*s + nx*ny*s + c**2*u - c*m*u - m*ny*u + k*(-(c*u) + (nx - u)*(nx - 2*v) - m*(nx + ny - u - 2*v)) - 2*((c + nx + ny)*s + m*(ny - u) + c*u)*v + 3*s*v**2
        if d != 0:
            v -= h/d
    return v


def est_mle_2(x, y, ny, u):
    nx, k = len(x), len(y)
    c = len(set(x) & set(y))
    s = max(y)
    m = sum(1 for xi in x if xi <= s)
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

import matplotlib.pyplot as plt

#estimators = [est_classic, est_mikkel, est_mle]
#estimators = [est_mikkel, est_mle, est_mle_2]
estimators = [est_min, est_newton, est_mle, est_p]
#estimators = [est_min, est_mle, est_p]
u, nx, ny, k = 1000, 200, 100, 30
vs = range(min(nx, ny)+1)
js = [v/(nx+ny-v) for v in vs]
reps = 1000
fig, axs = plt.subplots(2,2)
fig.suptitle(f'{nx=}, {ny=}, {k=}, {reps=}')
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
plt.legend()
plt.show()

