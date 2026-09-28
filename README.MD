# Autodiff and Backpropagation

This directory contains a small educational automatic-differentiation engine. `backprop.py` builds computation graphs, propagates gradients backward, checks those gradients numerically, and trains logistic-regression and neural-network classifiers on simple and XOR datasets.

## Setup

Use Python 3 and install the required packages:

```bash
python3 -m pip install numpy scipy matplotlib
```

## Run

From this directory, run one of the following commands:

```bash
# Check scalar/vector operation gradients
python3 backprop.py grad_check_1

# Check matrix-vector multiplication gradients
python3 backprop.py grad_check_2

# Train logistic regression on the linearly separable dataset
python3 backprop.py logreg --dataset simple

# Train the neural network on either dataset
python3 backprop.py neural --dataset simple
python3 backprop.py neural --dataset xor
```

The gradient-check commands compare backpropagation results with numerical gradients. The training commands print the loss during optimization and the learned parameters at the end.
