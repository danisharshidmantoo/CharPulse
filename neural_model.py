"""
The same bigram relationship as frequency_model.py, but learned by a
single linear layer trained with gradient descent instead of counted
directly. This is the bridge between "counting statistics" and "neural
network" framings of the same problem.
"""

import torch
import torch.nn.functional as F

from dataset import bigrams


def build_dataset(words, stoi):
    """Build the (xs, ys) training tensors: for every bigram in the
    dataset, xs holds the index of the first character and ys holds
    the index of the character that follows it.
    """
    xs, ys = [], []
    for w in words:
        for ch1, ch2 in bigrams(w):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs), torch.tensor(ys)


class NeuralBigramModel:
    """A single (vocab_size, vocab_size) weight matrix trained to
    predict the next character from a one-hot encoded current
    character. Functionally a bigram model, but learned instead of
    counted -- and, unlike the frequency table, it generalizes to
    sequences via gradient-based optimization rather than lookup.
    """

    def __init__(self, vocab_size: int = 27, seed: int = 2147483647):
        self.vocab_size = vocab_size
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

    def probs(self, xs):
        """Forward pass: one-hot encode -> linear layer -> softmax.

        The linear output is interpreted as log-counts; exponentiating
        and row-normalizing it (softmax) turns it into a probability
        distribution over the next character, equivalent in spirit to
        the frequency table N in frequency_model.py.
        """
        xenc = F.one_hot(xs, num_classes=self.vocab_size).float()
        logits = xenc @ self.W
        counts = logits.exp()
        return counts / counts.sum(1, keepdim=True)

    def train(self, xs, ys, epochs: int = 100, lr: float = 50.0, reg: float = 0.01):
        """Train with plain gradient descent.

        `reg` is an L2 penalty on the weights -- it keeps the learned
        distribution smooth instead of collapsing toward one-hot
        certainty, the same role Laplace smoothing plays for the
        frequency model.
        """
        num = xs.nelement()
        losses = []
        for _ in range(epochs):
            probs = self.probs(xs)
            loss = -probs[torch.arange(num), ys].log().mean() + reg * (self.W ** 2).mean()
            losses.append(loss.item())

            self.W.grad = None
            loss.backward()
            self.W.data += -lr * self.W.grad
        return losses

    def generate(self, itos, num_samples: int = 10, seed: int = 2147483647):
        """Sample new names using the trained weights, one character
        at a time, starting and stopping on the '.' token."""
        g = torch.Generator().manual_seed(seed)
        names = []
        for _ in range(num_samples):
            out = []
            ix = 0
            while True:
                p = self.probs(torch.tensor([ix]))
                ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                out.append(itos[ix])
                if ix == 0:
                    break
            names.append("".join(out))
        return names
