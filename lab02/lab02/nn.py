"""Lab 2: two-hidden-layer MLP implemented from scratch in NumPy."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------- 1. data
def load_data(seed=42):
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0            # pixels 0..16 -> 0..1
    y = digits.target
    Y = np.eye(10)[y]                                     # one-hot, shape (N, 10)
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=0.30, stratify=y, random_state=seed)
    return X_tr, X_te, Y_tr, Y_te


# ---------------------------------------------------------------- 2. params
def init_params(W, seed=0, std=0.1):
    # N(0, 0.01) read as N(mean, variance) -> std = sqrt(0.01) = 0.1.
    # (With std = 0.01 the signal vanishes through 3 layers and the net never trains.)
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, std, (64, W)), "b1": np.zeros(W),
        "W2": rng.normal(0, std, (W, W)),  "b2": np.zeros(W),
        "W3": rng.normal(0, std, (W, 10)), "b3": np.zeros(10),
    }


# ---------------------------------------------------------------- 3. forward
def relu(z):
    return np.maximum(0, z)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)                  # numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def forward(p, X):
    z1 = X @ p["W1"] + p["b1"];  a1 = relu(z1)
    z2 = a1 @ p["W2"] + p["b2"]; a2 = relu(z2)
    z3 = a2 @ p["W3"] + p["b3"]; y_hat = softmax(z3)
    cache = (X, z1, a1, z2, a2, y_hat)
    return y_hat, cache


# ---------------------------------------------------------------- 4. loss
def cross_entropy(y_hat, Y):
    return -np.mean(np.sum(Y * np.log(y_hat + 1e-12), axis=1))


# ---------------------------------------------------------------- 5. backward
def backward(p, cache, Y):
    X, z1, a1, z2, a2, y_hat = cache
    B = X.shape[0]
    dz3 = (y_hat - Y) / B                    # softmax + CE combined, mean over batch
    dW3 = a2.T @ dz3;  db3 = dz3.sum(axis=0)
    da2 = dz3 @ p["W3"].T
    dz2 = da2 * (z2 > 0)                     # ReLU derivative
    dW2 = a1.T @ dz2;  db2 = dz2.sum(axis=0)
    da1 = dz2 @ p["W2"].T
    dz1 = da1 * (z1 > 0)
    dW1 = X.T @ dz1;   db1 = dz1.sum(axis=0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def sgd_step(p, g, lr):
    for k in p:
        p[k] -= lr * g[k]


def accuracy(p, X, Y):
    y_hat, _ = forward(p, X)
    return np.mean(y_hat.argmax(1) == Y.argmax(1))