import aminhash.datasets
import scipy.sparse
from sklearn.decomposition import LatentDirichletAllocation
import sys
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str)
parser.add_argument('--method', type=str, choices=['online', 'batch'])
parser.add_argument('--out', type=str)
parser.add_argument('-K', type=int, default=30)
parser.add_argument('-E', type=int, default=10)
args = parser.parse_args()

dataset = args.dataset
k = args.K
E = args.E
outname = args.out

dblp, _dom = aminhash.datasets.load(dataset)
random.shuffle(dblp)
sz = list(map(len,dblp))
indptr = np.concatenate([[0], np.cumsum(sz)])
m = scipy.sparse.csr_matrix((np.ones(indptr[-1]), np.concatenate(dblp), indptr))

lda = LatentDirichletAllocation(
        n_components=k,
        verbose=True,
        learning_method=args.method,
        max_iter=args.E)
trans = lda.fit_transform(m)
utrans = lda._unnormalized_transform(m)

np.save(outname+'_comp', lda.components_)
np.save(outname+'_exp', lda.exp_dirichlet_component_)
np.save(outname+'_trans', trans)
np.save(outname+'_utrans', utrans)

def test(i):
    words = dblp[i]
    n = len(words)
    res = []
    for tr in [trans, utrans]:
        for comp in [lda.components_, lda.exp_dirichlet_component_]:
            logits = tr[i] @ comp
            guess = np.argpartition(-logits, n)[:n]
            v = len(set(words) & set(guess))
            res.append(v/(2*n-v))
    return res

jss = zip(*(test(i) for i in range(10000)))
for js in jss:
    print(np.mean(js))
