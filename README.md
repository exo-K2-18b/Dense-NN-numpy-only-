[Dense-NN-numpy-only-README.md](https://github.com/user-attachments/files/32481862/Dense-NN-numpy-only-README.md)
# Dense NN From Scratch (NumPy Only)

A 2-layer fully connected neural network for MNIST digit classification, built entirely in NumPy — no PyTorch, no TensorFlow, no autodiff. Every part of the forward pass, backward pass, and gradient descent update is implemented by hand.

## Why this exists

Most people who can train a neural network have never had to compute a single gradient themselves — the framework does it. This project is the opposite: it proves the math is understood at the level of individual matrix derivatives, not just at the level of `.backward()`.

## Architecture

- Input: 784 (flattened 28×28 MNIST images), normalized to [0, 1]
- Hidden layer: 128 units, ReLU activation
- Output layer: 10 units, softmax activation
- Loss: categorical cross-entropy
- Optimizer: vanilla mini-batch gradient descent (batch size 32, learning rate 0.1)

## What's implemented manually

- Forward pass (`Z1 = X @ W1 + B1`, ReLU, `Z2 = A1 @ W2 + B2`, softmax)
- Cross-entropy loss computation
- Full backward pass: gradients for `W1`, `W2`, `B1`, `B2` derived and coded by hand via the chain rule (no autograd)
- Batched forward/backward pass (vectorized across a batch of 32 examples)
- Full training loop across the entire MNIST training set, run for 175 epochs
- Evaluation on the held-out MNIST test set

## Results

Trained for 175 epochs on the full 60,000-image MNIST training set, evaluated on the 10,000-image test set.

**Test accuracy: 98.02%**

## How to run

```bash
pip install tensorflow numpy
python mnist_mlp.py
```

(TensorFlow is used only to load the MNIST dataset via `tf.keras.datasets.mnist` — no TensorFlow layers or training are used.)

## Known limitations / next steps

- Single hidden layer only — no experimentation yet with deeper architectures
- Fixed learning rate, no scheduling or momentum (e.g. Adam)
- No regularization (dropout, weight decay) — some overfitting is likely at 175 epochs
- No train/validation split during training — evaluation happens only once, on the test set, at the end
