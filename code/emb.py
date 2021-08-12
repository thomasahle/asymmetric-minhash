import argparse
import torch
import torch.nn as nn
from aminhash import datasets
import numpy as np
from softtopk import soft_topk
from sinkhorn import TopK_custom


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['netflix','flickr','dblp'])
parser.add_argument('-N', type=int, default=None)
parser.add_argument('-K', type=int, default=10)
parser.add_argument('-B', type=int, default=100)
parser.add_argument('--sinkhorn', action='store_true')
parser.add_argument('--softmax', action='store_true')
args = parser.parse_args()


print('Loading data')
data, dom = datasets.load(args.dataset, verbose=True, trim=args.N)
np.random.shuffle(data)
N, K = len(data), args.K
print(f'{N=}, {dom=}')

emb = torch.randn(dom, K, requires_grad=True)
#bias = torch.randn(K, requires_grad=True)
with torch.no_grad():
    #torch.nn.init.kaiming_uniform_(emb, a=2.23)
    #torch.nn.init.kaiming_normal_(emb, a=2.23)
    torch.nn.init.sparse_(emb, sparsity=1/K)
    #torch.nn.init.orthogonal_(emb)
    #emb *= 10
    #emb /= K**.5
#model = torch.nn.Sequential(
    #nn.Embedding(dom, K),
#    nn.Linear(dom, K),
#    nn.Linear(K, dom)
#)
#print(model)
lr = 1e-2
optimizer = torch.optim.AdamW([emb], lr=lr)
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

#torch.autograd.detect_anomaly()

roll_avg_j = 0
roll_avg_s = 0
alpha = .95
for t in range(100):

    #BS = int(1/lr+1)
    BS = args.B
    for i in range(0, N, BS):
        loss = 0
        loss_j = 0
        top_j = 0
        loss_s = 0
        for v in data[i : i+BS]:
            hot = torch.zeros(dom)
            hot[v] = 1
            #logits = emb @ (emb.T @ hot)
            logits = emb @ emb[v].sum(axis=0)

            # TODO: Consider handling multiple sets at the same time
            # We can do something like hot[range(BS), labels.T] = 1
            # However, the labels have to have the same size, so we
            # have to preprocess by repeating an element or otherwise.
            # We'd also have to support multiprocessing in the soft-topk.
            # Rewriting it in cython might be nice to do while at it.

            # TODO: Try mean. Since we no longer need extra large logits for our sets
            # (as we needed when there was no coupling between them) it might work better.
            # In particular we should get fewer very large logits for large sets and K.
            # I don't really know how scaling affects the soft-topk. Maybe it makes it
            # less "certain"? Maybe it doesn't do anything at all..
            #logits = emb @ emb[v].mean(axis=0)
            # This is how fasttext does it:
            #logits = emb @ (emb[v] / emb[v].norm(dim=1, keepdim=True)).mean(axis=0)


            #print(logits)
            #ws = torch.ones(dom)
            #ws[v] = -1
            # cross entropy loss
            #loss += torch.log(1+torch.exp(ws * logits)).sum()
            # The torch version is more numercally stable due to pulling out the largest
            # factor (log-sum trick).
            #loss += torch.nn.functional.binary_cross_entropy_with_logits(logits, hot, reduction='sum')
            # The hinge loss is ok, but slightly worse.
            #loss += torch.clamp(1+ws*logits, min=0).sum() # hinge-loss

            # I don't really think I still need this, but just in case
            #logits = logits.clamp(-20, 20)
            #lesum = torch.logsumexp(logits, dim=0)
            #ps = 1/(1+torch.exp(-logits - tilt))
            # For some reason this clamping seems important when we use f1 loss
            #logits = torch.clamp(logits, min=-30, max=30)
            
            if args.sinkhorn:
                ps = TopK_custom(len(v))(logits.reshape(1,-1))[0][0]
                ps = ps.sum(axis=1)
            elif args.softmax:
                ps = torch.softmax(logits, dim=0)
                # Try weighting more like F1. (Note bce already does mean (dividing ben len(v)))
                loss += torch.nn.functional.binary_cross_entropy(ps[v], hot[v])
                # This also works quite well
                #loss += torch.nn.functional.binary_cross_entropy(ps, hot)

                # Try to weigh positive and negative losses the same
                #ws = torch.ones(dom) / (dom-len(v))
                #ws[v] = 1 / len(v)
                #sps = torch.softmax(logits, dim=0)
                #loss += torch.nn.functional.binary_cross_entropy(sps, hot, ws, reduction='sum')
            else:
                ps = soft_topk.apply(logits, len(v))

            #dip = hot @ ps
            dip = ps[v].sum()
            #dip = hot @ torch.exp(logits - lesum)
            j = dip/(2*len(v)) # F1 score
            loss_j -= j

            loss_s += (len(v) - ps.sum())**2
            #loss_s += (len(v) - torch.exp(logits).sum())**2
            
            topn = np.argpartition(-logits.detach().numpy(), len(v))[:len(v)]
            ip = len(set(topn) & set(v))
            top_j += ip/(2*len(v) - ip)
            
        #print(t, i, loss.item() / BS)
        #print(t, i, -loss_j.item() / BS)

        roll_avg_j = top_j/BS + alpha*roll_avg_j
        print(t, i, 'Avg self-jaccard:', roll_avg_j*(1-alpha))
        print(t, i, 'Dif self-jaccard:', -loss_j / BS)
        roll_avg_s = loss_s/BS + alpha*roll_avg_s
        print(t, i, 'RMSD size:', (roll_avg_s*(1-alpha))**.5)
        print(t, i, 'weight avg', (emb**2).mean()**.5)
        #print(t, i, 'bias avg', (bias**2).mean()**.5)

        #if top_j > .1:
            #loss += loss_j * 100
        #loss = loss_j + loss_s/1000000
        loss = loss_j 


        #comb = loss + loss_j*100
    
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        #comb.backward()
        loss.backward()
        #loss_j.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

        print('grad norm', emb.grad.data.norm(2))

        #if (i//BS) % 10 == 0:
            #scheduler.step(loss)

            #tns = torch.tensor(v)
