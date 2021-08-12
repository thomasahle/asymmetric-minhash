import torch
import numpy as np
from torch.autograd import gradcheck, Function

import random
import time


class test_fun2(Function):
    @staticmethod
    def forward(ctx, xs):
        ctx.save_for_backward(xs)
        x0, x1 = xs
        return torch.tensor([x0 + 2*x1, x0**2 + x1**3])

    @staticmethod
    def backward(ctx, grad_output):
        (x0, x1), = ctx.saved_tensors
        jac = torch.tensor([[1, 2], [2*x0, 3*x1**2]])
        #return grad_output @ jac
        #return jac.T @ grad_output
        # Men det mener jeg jo er det samme som
        #   grad_x <f(x), grad_output>
        w0, w1 = grad_output
        return torch.tensor([w0+2*w1*x0, 2*w0+3*w1*x1**2])

t0, t1, t2 = 0, 0, 0

class soft_topk(Function):
    ''' Computes (sigmoid(x_i + t))_i where t is such that                                    
        sum(sigmoid(x_i + t)) = k '''                                                         

    @staticmethod                                                                             
    def forward(ctx, xs, k):
        global t0, t1, t2

        # First binary search a bit.
        # We use buckets to speed up this part.
        #start = time.time()
        # TODO: What if there is -inf or inf in the xs?
        mi, ma = xs.min(), xs.max()
        #counts = torch.histc(xs, bins=10, min=mi, max=ma)
        #centers = torch.linspace(mi, ma, steps=10, dtype=counts.dtype)
        #t0 += time.time() - start

        #start = time.time()
        a, b = -10-ma, 10-mi
        while b-a > 1:
            t = (a + b) / 2                                                                
            val = (xs + t).sigmoid().sum()                                                 
            #val = (centers + t).sigmoid() @ counts
            if val > k:                                                                       
                b = t                                                                      
            else: a = t                                                                    
        t = (a + b) / 2
        #t1 += time.time() - start

        #start = time.time()
        # Then Newton a bit
        vprime = 1
        while abs(vprime) > 1e-1:
            s = (xs + t).sigmoid()
            t -= (s.sum() - k) / (s*(1-s)).sum()
            vprime = s.sum() - val
            val += vprime
        
        s = (xs + t).sigmoid()
        #t2 += time.time() - start

        ctx.save_for_backward(s)
        return s


    @staticmethod                                                                             
    def forward_old(ctx, xs, k):
        # First binary search a bit
        a, b = -10-xs.max(), 10-xs.min()
        while b-a > 1:
            t = (a + b) / 2                                                                
            val = (xs + t).sigmoid().sum()                                                 
            if val > k:                                                                       
                b = t                                                                      
            else: a = t                                                                    
        t = (a + b) / 2
        # Then Newton a bit
        for _ in range(3):
            s = (xs + t).sigmoid()
            t -= (s.sum() - k) / (s*(1-s)).sum()
        s = (xs + t).sigmoid()
        if not np.isclose(s.sum(), k, atol=1e-3, rtol=1e-3):                                          
            #print(4*(k/n - 1/2) - xs.sum()/n)
            print(t, s.sum(), k, xs.min(), xs.max())                                        
        ctx.save_for_backward(s)
        return s
                                                                                             
    @staticmethod                                                                             
    def backward(ctx, grad_output):                                                           
        """ Returns the gradient of <f(x), grad_output> """
        if not ctx.needs_input_grad[0]:
            print('huh', ctx.needs_input_grad)
            return None, None
        # The none is because we don't need a gradient wrt. k
        #fwd, = ctx.saved_tensors
        #vec = fwd*(1-fwd)
        vec, = ctx.saved_tensors
        vec *= 1-vec # Can we save time by not allocating anything?
        #return vec*(grad_output - grad_output@vec / vec.sum()), None
        grad_output -= grad_output@vec / vec.sum()
        vec *= grad_output
        return vec, None



def test():
    input = torch.randn(2, requires_grad=True, dtype=torch.double)
    print()
    assert gradcheck(test_fun2.apply, input, eps=1e-6, atol=1e-4)

    n = 100
    for i in range(1, 10):
        input = torch.randn(n, requires_grad=True, dtype=torch.double)
        k = random.randrange(n+1)
        print(f'{k=}')

        fwd = soft_topk.apply(input, k).detach().numpy()
        assert np.isclose(fwd.sum(), k, atol=1e-3, rtol=1e-3)

        assert gradcheck(soft_topk.apply, (input, k), eps=1e-6, atol=1e-4)
    print('All good')

def bench():
    n = 17000
    for _ in range(1, 1000):
        input = torch.randn(n, requires_grad=True, dtype=torch.double)
        k = random.randrange(n+1)
        fwd = soft_topk.apply(input, k)
        #print(fwd)
        #if k == 1:
            #print(input.softmax(0))
            #print(fwd - input.softmax(0))
            #print(fwd / input.softmax(0))
        #bwd = soft_topk.backward(input)
        #print(bwd)
        #print()
        #assert gradcheck(soft_topk.apply, (input, k), eps=1e-6, atol=1e-4)
    print(t0, t1, t2)
                                 
if __name__ == '__main__':
    test()
