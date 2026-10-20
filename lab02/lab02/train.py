"""Step 6: mini-batch SGD for lr in [0.001, 0.01, 0.1] x width in [16, 64]."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nn import load_data, init_params, forward, cross_entropy, backward, sgd_step, accuracy

EPOCHS, BATCH = 50, 32
LRS, WIDTHS = [0.001, 0.01, 0.1], [16, 64]

X_tr, X_te, Y_tr, Y_te = load_data()
os.makedirs("results", exist_ok=True)


def train(W, lr, seed=0):
    p = init_params(W, seed=seed)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(X_tr))               # shuffle every epoch
        for start in range(0, len(X_tr), BATCH):
            b = order[start:start + BATCH]
            _, cache = forward(p, X_tr[b])
            sgd_step(p, backward(p, cache, Y_tr[b]), lr)
        loss = cross_entropy(forward(p, X_tr)[0], Y_tr)
        history.append(loss)
        print(f"  W={W:<3} lr={lr:<6} epoch {epoch:>2}  train loss {loss:.4f}")
    return p, history


fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
summary = []
for ax, W in zip(axes, WIDTHS):
    for lr in LRS:
        p, hist = train(W, lr)
        test_acc = accuracy(p, X_te, Y_te)
        summary.append((W, lr, hist[-1], test_acc))
        ax.plot(range(1, EPOCHS + 1), hist, label=f"lr={lr} (test acc {test_acc:.1%})")
    ax.set_title(f"Hidden width W={W}")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()
axes[0].set_ylabel("Training cross-entropy loss")
fig.tight_layout()
fig.savefig("results/loss_curves.png", dpi=150)

print("\n W    lr      final train loss   test accuracy")
for W, lr, loss, acc in summary:
    print(f"{W:<4} {lr:<7} {loss:>10.4f}        {acc:>8.2%}")