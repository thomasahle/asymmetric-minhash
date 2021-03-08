import argparse
import torch
import torch.nn as nn
from aminhash import datasets
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['netflix','flickr','dblp'])
parser.add_argument('-N', type=int, default=None)
parser.add_argument('-K', type=int, default=10)
parser.add_argument('-B', type=int, default=100)
args = parser.parse_args()


print('Loading data')
data, dom = datasets.load(args.dataset, verbose=True, trim=args.N)
np.random.shuffle(data)
N, K = len(data), args.K
print(f'{N=}, {dom=}')

#emb = torch.randn(K, dom, requires_grad=True)
#with torch.no_grad():
#    torch.nn.init.kaiming_uniform_(emb, a=2.23)
#    emb /= K**.5
model = torch.nn.Sequential(
    #nn.Embedding(dom, K),
    nn.Linear(dom, K),
    nn.Linear(K, dom)
)
print(model)
lr = 1e-2
#optimizer = torch.optim.AdamW([emb], lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

#torch.autograd.detect_anomaly()

for t in range(100):
    #lr /= 10
    #optimizer.param_groups[0]['lr'] = lr
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
            #logits = emb.T @ (emb @ hot)
            logits = model(hot)
            #logits = model(torch.tensor(v)).sum(axis=0)
            #logits = torch.clamp(logits, min=-30, max=30)
            #print(logits)
            #ws = torch.ones(dom)
            #ws[v] = -1
            #loss += torch.log(1+torch.exp(ws * logits)).sum() # cross entropy loss
            loss += torch.nn.functional.binary_cross_entropy_with_logits(logits, hot, reduction='sum')
            #loss += torch.clamp(1+ws*logits, min=0).sum() # hinge-loss
            ip = (1/(1+torch.exp(-logits[v]))).sum()
            size = (1/(1+torch.exp(-logits))).sum()
            assert size >= ip
            loss_s += (len(v) - size)**2
            loss_j -= ip/(len(v) + size - ip)
            
            topn = np.argpartition(-logits.detach().numpy(), len(v))[:len(v)]
            ip = len(set(topn) & set(v))
            top_j += ip/(2*len(v) - ip)
            
        print(t, i, loss.item() / BS)
        print(t, i, -loss_j.item() / BS)
        print(t, i, top_j / BS)
        print(t, i, (loss_s.item() / BS)**.5)
        #print(t, i, (emb**2).sum())


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

        #if (i//BS) % 10 == 0:
            #scheduler.step(loss)

            #tns = torch.tensor(v)
