import argparse
import torch.nn as nn
from aminhash import datasets


parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['netflix','flickr','dblp'])
parser.add_argument('-N', type=int, default=None)
parser.add_argument('-K', type=int, default=10)
args = parser.parse_args()


print('Loading data')
data, dom = datasets.load(args.dataset, verbose=True, trim=args.N)
N, K = len(data), args.K
print(f'{N=}, {dom=}')

BS = 1000

emb = nn.Embedding(dom, K)
learning_rate = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)


for t in range(100):
    for i in range(0, N, BS):
        loss = 0
        for v in data[i : i+BS]:
            t = torch.tensor(v)
            logits = emb(t).sum(axis=1)
            loss += torch.log(1+torch.exp(-logits[t])) # cross entropy loss
        print(t, i, loss.item())

        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

