# homemAIde

A language model built from scratch, starting from a single number that knows its own derivative.

The goal isn't to compete with anything. It's to understand every layer of how modern AI works by writing each one by hand, with no framework doing the hard parts.

## Roadmap

1. **Scalar autograd engine**: a `Value` type that records every operation into a computation graph and backpropagates gradients through it. Pure Python.
2. **Neural network library**: neurons, layers, activations, loss functions and gradient descent, built on the engine and trained on real data (Iris flower classification).
3. **Tensor engine**: the same autograd ideas rewritten for n-dimensional arrays on NumPy, with every backward pass derived and written by hand. Trained on MNIST handwritten digits.
4. **Transformer**: embeddings, self-attention, multi-head attention, layer norm and positional encoding, implemented from *Attention Is All You Need*.
5. **Small language model**: a GPT-style model trained on a text corpus on a consumer GPU (RTX 4060), able to generate text.

Gradients are checked against PyTorch at every stage, and PyTorch is used only as a reference, never as a dependency of the engine.

## Status

Stage 1: scalar autograd engine in progress.

## Running

Requires Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```sh
uv run main.py
```
