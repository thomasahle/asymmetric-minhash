import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt
import sys

if sys.argv[1] == 'dblp':
    ks =  np.array([1, 10, 30, 100, 400, 500])
    r1s = np.array([.0036, .0987, .2978, .5736, .8676, .9009])
    r2s = np.array([.0080, .1072, .3524, .6536, .9153, .9385])
if sys.argv[1] == 'netflix':
    ks =  np.array([1, 10, 30, 100, 400, 500])
    r1s = np.array([.0033, .0501, .1474, .3831, .7510, .7942])
    r2s = np.array([.0099, .0623, .1914, .4903, .8338, .8667])
if sys.argv[1] == 'flickr':
    ks =  np.array([1, 5, 10, 20, 30])
    r1s = np.array([.2379, .6256, .7770, .8657, .9108])
    r2s = np.array([.3595, .6688, .8122, .8964, .9301])

# Assuming "positive logistic" model "r = 1 - exp(-a k)"
lt1s = np.log(1/(1-r1s))
lt2s = np.log(1/(1-r2s))
# Assuming normal logistic model r = 1/(1 + exp(-aK))
#lt1s = np.log(r1s/(1-r1s))
#lt2s = np.log(r2s/(1-r2s))
lr = sklearn.linear_model.LinearRegression(fit_intercept=False)
lr.fit(ks.reshape(-1,1), lt1s)
a1 = lr.coef_
lr.fit(ks.reshape(-1,1), lt2s)
a2 = lr.coef_

# for r in np.linspace(.1, .9):
#     k1 = np.log(1/(1-r)) / a1
#     k2 = np.log(1/(1-r)) / a2
#     print(r, k1, k2)

print(1/a1)
print(1/a2)
# It's like having this many more MinHash values.
# E.g. going from 80 to 100 is a 25% improvement.
print((1/a1)/(1/a2)-1)
# It allows this percentage reduction in the number of MinHash values.
# E.g. going from 100 to 80 is a 20% reduction.
print(1-(1/a2)/(1/a1))


if len(sys.argv) > 2 and sys.argv[2] == 'plot':
    fig, ax = plt.subplots()
    plt.scatter(r2s, ks)
    plt.scatter(r1s, ks)
    rs = np.linspace(0,1)
    k1s = np.log(1/(1-rs))/a1
    k2s = np.log(1/(1-rs))/a2
    plt.plot(rs, k2s, label='Best Estimator')
    plt.plot(rs, k1s, label='Classical MinHash')
    ax.legend(prop={'size': 10})
    plt.ylabel('# MinHash functions')
    plt.xlabel('Recall@10')
    fn = 'out.png'
    print(f'Saving to {fn}')
    plt.savefig(fn, dpi=300)
