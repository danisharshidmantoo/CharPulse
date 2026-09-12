"""
Trains both bigram models (frequency-based and neural) on names.txt
and prints their negative log-likelihood / loss plus a few generated
samples from each.

Run with:  python train.py
"""

from dataset import load_words, build_vocab
from frequency_model import (
    build_counts,
    build_probabilities,
    negative_log_likelihood,
    generate as generate_frequency,
)
from neural_model import build_dataset, NeuralBigramModel


def main():
    words = load_words("names.txt")
    stoi, itos = build_vocab(words)
    vocab_size = len(stoi)

    # ---- Frequency-based model ----
    N = build_counts(words, stoi, vocab_size)
    P = build_probabilities(N, smoothing=1.0)
    nll = negative_log_likelihood(P, words, stoi)
    print(f"[Frequency model] negative log-likelihood: {nll:.4f}")
    print("[Frequency model] samples:", generate_frequency(P, itos, num_samples=5))

    print()

    # ---- Neural model ----
    xs, ys = build_dataset(words, stoi)
    model = NeuralBigramModel(vocab_size=vocab_size)
    losses = model.train(xs, ys, epochs=100, lr=50.0, reg=0.01)
    print(f"[Neural model] loss after {len(losses)} steps: {losses[-1]:.4f} (started at {losses[0]:.4f})")
    print("[Neural model] samples:", model.generate(itos, num_samples=5))


if __name__ == "__main__":
    main()
