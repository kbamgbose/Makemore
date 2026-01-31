import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


words = open('names.txt', 'r').read().splitlines()


b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1

sorted(b.items(), key = lambda kv: -kv[-1])

N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
stoi = {s:i +1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}


for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1


plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off');

g = torch.Generator().manual_seed(2147483647)
p = N[0].float()
p = p / p.sum()
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g)

N[0]
p = torch.rand(3, generator=g)
p = p / p.sum()

g = torch.Generator().manual_seed(2147483647)

out = []
ix = 0;

P = (N+1).float()
P /= P.sum(1, keepdim=True)

for i in range(5):

    out = []
    ix = 0;

    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    # print(''.join(out))

P.sum(1, keepdim=True).shape


# GOAL: maximize likelihood of the data w.r.t model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimzing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

# log(a*b*c) = log(a) + log(b) + log(c)
# log_likelihood = 0.0
# n = 0

# for w in words[:3]:
# # for w in ["kola"]:
#     chs = ['.'] + list(w) + ['.']
#     for ch1, ch2 in zip(chs, chs[1:]):
#         ix1 = stoi[ch1]
#         ix2 = stoi[ch2]
#         prob = P[ix1, ix2]
#         logprob = torch.log(prob)
#         log_likelihood += logprob
#         n += 1
        # print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

# print(log_likelihood)
# nll = -log_likelihood
# print(f'{nll=}')
# print(f'{nll/n}')


# create the training set of all the bigrams(x, y)
xs, ys = [], []

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs)
ys = torch.tensor(ys)
num = xs.nelement()
print('number of examples: ', num)

# Random initiations of 27 neurons weights, each neuron has 27 inputs
g = torch.Generator().manual_seed(2147483647)
W = torch.randn((27, 27), generator=g, requires_grad=True)

#gradient descent
for k in range(200):
    #forward pass
    xenc = F.one_hot(xs, num_classes=27).float() # (5, 27) @ (27, 1) = (5, 27)
    logits = xenc @ W # log-counts
    counts = (xenc @ W).exp() #equivalent N
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(num), ys].log().mean() + 0.01*(W**2).mean()


    #backward pass
    W.grad = None # set the zero the gradient
    loss.backward() 

    W.data += -50 * W.grad

print(loss.item())


g = torch.Generator().manual_seed(2147483647)

for i in range(5):

    out = []
    ix = 0
    while True:

        xenc = F.one_hot(torch.tensor([ix]), num_classes=27).float() # (5, 27) @ (27, 1) = (5, 27)
        logits = xenc @ W # log-counts
        counts = (xenc @ W).exp() #equivalent N
        p = counts / counts.sum(1, keepdim=True)

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))