import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures
import random


words = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0;
itos = {i:s for s, i in stoi.items()}
vocab_size = len(itos)


block_size = 3 # context length: how many characters do we take in to predict the next one

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] #crop and append the next characters
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

Xtr, Ytr = build_dataset(words[:n1])           #80%
Xdev, Ydev = build_dataset(words[n1:n2])     #10%
Xte, Yte = build_dataset(words[n2:])         #10%

# utility function we will use later when comparing manual gradients to PyTorch gradients
def cmp(s, dt, t):
    ex = torch.all(dt == t.grad).item()
    app = torch.allclose(dt, t.grad)
    maxdiff = (dt - t.grad).abs().max().item()
    print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')

#MLP revisited
n_embd = 10  # the dimensionality of the character embedding vectors
n_hidden = 200  # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((vocab_size, n_embd), generator=g)

# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3) / (n_embd * block_size)**0.5
b1 = torch.randn(n_hidden, generator=g) * 0.1

# Layer 2
W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.1
b2 = torch.randn(vocab_size, generator=g) * 0.1

#Batch normalization parameters
bngain = torch.randn((1, n_hidden)) * 0.1 + 1.0
bnbias = torch.randn((1, n_hidden)) * 0.1


parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters)) # number of parameters in total
for p in parameters:
    p.requires_grad = True

# same optimization as last time
batch_size = 32
n = batch_size # a shorter variable also, for convenience

#minibatch construct
ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
Xb, Yb = Xtr[ix], Ytr[ix] # batch X, Y

#forward pass, chunkated into smaller steps that are possible to backward one at a time

emb = C[Xb] # embed the characters into vectors
embcat = emb.view(emb.shape[0], -1) # concatenate the vectors

# Linear layer
hprebn = embcat @ W1 + b1 # hidden layer pre-activation
# BatchNorm layer
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) # note: Bessel's correction (dividing by n-1, not n)
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
# Non linearity
h = torch.tanh(hpreact) # hidden layer
# Linear layer 2
logits = h @ W2 + b2 #output layer
# cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes # subtract max for numerical stability
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()


# PyTorch backward pass
for p in parameters:
    p.grad = None
for t in [logprobs, probs, counts, counts_sum, counts_sum_inv, 
          norm_logits, logit_maxes, logits, h, hpreact, bnraw, 
          bnvar_inv, bnvar, bndiff2, bndiff, hprebn, bnmeani, 
          embcat, emb]:
    t.retain_grad()
loss.backward()
loss

# #update
# lr = 0.1 if i < 100000 else 0.01 #step learning rate decay
# for p in parameters:
#     if p.grad is not None:
#         p.data += -lr * p.grad

# #track stats
# if i % 10000 == 0: # print every once in a while
#     print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
# lossi.append(loss.log10().item()) 


# Exercise 1: backprop through the whole thing manually,
# backpropagating through exactly all of the variables
# as they are defined in the forward pass above, one by one

dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n
dprobs = (1.0 / probs) * dlogprobs
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)


print(cmp('logprobs', dlogprobs, logprobs))
print(cmp('probs', dprobs, probs))
print(cmp('counts_sum_inv', dcounts_sum_inv, counts_sum_inv))
# print(cmp('counts_sum', dcounts_sum, counts_sum))
# print(cmp('counts', dcounts, counts))
# print(cmp('norm_logits', dnorm_logits, norm_logits))
# print(cmp('logit_maxes', dlogit_maxes, logit_maxes))
# print(cmp('logits', dlogits, logits))
# print(cmp('h', dh, h))
# print(cmp('W2', dW2, W2))
# print(cmp('b2', db2, b2))
# print(cmp('hpreact', dhpreact, hpreact))
# print(cmp('bngain', dbngain, bngain))
# print(cmp('bnbias', dbnbias, bnbias))
# print(cmp('bnraw', dbnraw, bnraw))
# print(cmp('bnvar_inv', dbnvar_inv, bnvar_inv))
# print(cmp('bnvar', dbnvar, bnvar))
# print(cmp('bndiff2', dbndiff2, bndiff2))
# print(cmp('bndiff', dbndiff, bndiff))
# print(cmp('bnmeani', dbnmeani, bnmeani))
# print(cmp('hprebn', dhprebn, hprebn))
# print(cmp('embcat', dembcat, embcat))
# print(cmp('W1', dW1, W1))
# print(cmp('b1', db1, b1))
# print(cmp('emb', demb, emb))
# print(cmp('C', dC, C))

# loss = -(a + b + c) / 3 = -1/3a + -1/3b + -1/3c 
# dloss/da = -1/n 