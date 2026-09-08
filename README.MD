# Bigram Neural Network From Scratch

A character-level language model built from scratch using PyTorch.

The project starts with a simple **bigram frequency model** and then implements the same idea using a **neural network**, including the forward pass, negative log-likelihood loss, backpropagation, gradient-based weight updates, and text generation.

## What is a Bigram Model?

A bigram model predicts the next character based only on the current character.

For example, given:

```text
emma
with start/end tokens:

.emma.

the model learns the following character transitions:

. → e
e → m
m → m
m → a
a → .

The model represents these relationships using a 27 × 27 matrix, where each row corresponds to the current character and each column corresponds to the probability of the next character.

Project Structure

The project explores two approaches:

1. Frequency-Based Bigram Model

Character transition counts are collected from the dataset.

These counts are normalized row-wise to obtain probabilities:

P(next_character | current_character)

The resulting probability matrix can then be used to generate new names by sampling from the appropriate row.

2. Neural Network Bigram Model

The same bigram relationships are learned using a simple neural network.

The architecture is essentially:

Character index
      ↓
One-hot encoding
      ↓
Linear layer (27 × 27)
      ↓
Logits
      ↓
Softmax
      ↓
Next-character probabilities

The model is trained using negative log-likelihood (NLL).

Training

For every character pair, the current character is used as the input and the next character as the target.

For example:

Input:  . e m m a
Target: e m m a .

The input characters are converted to integer indices and then one-hot encoded.

The model produces logits:

logits = xenc @ W

which are converted into probabilities using exponentiation and normalization.

The loss is calculated using the probability assigned to the correct next character:

nll = -torch.log(
    probs[torch.arange(len(x)), y]
).mean()
Backpropagation

PyTorch autograd is used to calculate the gradients:

loss.backward()

The weights are then updated using gradient descent:

W.data += -learning_rate * W.grad

and the gradients are reset before the next iteration:

W.grad.zero_()
Regularization

L2 regularization is also explored to discourage excessively large weights:

lambda_ * (W**2).mean()

This encourages smaller weights and generally produces less extreme probability distributions.

Generation

After training, the neural network can generate new names character by character.

The process is:

Start with "."
     ↓
Predict next-character probabilities
     ↓
Sample a character
     ↓
Use sampled character as the next input
     ↓
Repeat
     ↓
Stop when "." is sampled

Example generated output:

['cava.', 'ayama.', '...']
Concepts Covered
Character-level language modeling
Bigram models
Frequency-based probability models
One-hot encoding
Matrix multiplication
Logits
Softmax probabilities
Negative log-likelihood
Cross-entropy
Backpropagation
Gradients
Gradient descent
L2 regularization
Probability-based sampling
Autograd in PyTorch
Technologies
Python
PyTorch
NumPy
Matplotlib
Jupyter Notebook
Goal

The purpose of this project is to understand how a simple language model works from first principles, rather than relying on high-level training abstractions.

The bigram model provides the foundation for understanding how increasingly sophisticated neural language models can learn probability distributions over sequences.