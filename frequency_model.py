"""
The frequency-count bigram model: count how often each character
follows another across the dataset, normalize the counts into
probabilities (with add-1 Laplace smoothing), and sample or evaluate
from that distribution.
"""

import torch

from dataset import bigrams


def build_counts(words, stoi, vocab_size: int = 27):
    """Count bigram occurrences into a (vocab_size, vocab_size) tensor.

    N[i, j] is how many times character j followed character i,
    across the whole dataset.
    """
    N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)
    for w in words:
        for ch1, ch2 in bigrams(w):
            N[stoi[ch1], stoi[ch2]] += 1
    return N


def build_probabilities(N, smoothing: float = 1.0):
    """Row-normalize counts into a probability distribution per
    character. `smoothing` adds a constant to every cell first, so no
    bigram ever has exactly zero probability (a bigram unseen in
    training would otherwise make the model assign zero probability,
    and log(0) is undefined, to any word containing it).
    """
    P = (N + smoothing).float()
    P /= P.sum(1, keepdim=True)
    return P


def negative_log_likelihood(P, words, stoi):
    """Average negative log-likelihood of `words` under distribution
    `P` -- lower is better (the model assigns higher probability to
    the real data). This is the standard way to score how well a
    probability model explains a dataset.
    """
    log_likelihood = 0.0
    n = 0
    for w in words:
        for ch1, ch2 in bigrams(w):
            log_likelihood += torch.log(P[stoi[ch1], stoi[ch2]])
            n += 1
    return -log_likelihood / n


def generate(P, itos, num_samples: int = 10, seed: int = 2147483647):
    """Sample new names by walking the Markov chain defined by `P`,
    starting from the '.' token and stopping when '.' is sampled again.
    """
    g = torch.Generator().manual_seed(seed)
    names = []
    for _ in range(num_samples):
        out = []
        ix = 0
        while True:
            p = P[ix]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
        names.append("".join(out))
    return names
