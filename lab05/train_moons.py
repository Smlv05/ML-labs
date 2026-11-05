"""
Lab 5 - Task 5: train a tiny 2 -> 8 -> 1 model (ReLU hidden, sigmoid output,
BCE loss) on make_moons with our engine and with PyTorch, starting from the
SAME initial weights. Compare losses and training speed (median of 5 runs).
"""
import json
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import make_moons

from engine import MLP, bce, vsum, zero_grad

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

N_SAMPLES, HIDDEN, EPOCHS, LR, RUNS = 100, 8, 200, 0.5, 5

X, y = make_moons(n_samples=N_SAMPLES, noise=0.1, random_state=0)
X = (X - X.mean(axis=0)) / X.std(axis=0)  # standardise features


def init_weights(seed):
    """PyTorch-style uniform init, generated once and shared by both models."""
    rng = np.random.default_rng(seed)
    k1, k2 = 1 / np.sqrt(2), 1 / np.sqrt(HIDDEN)
    return (rng.uniform(-k1, k1, (HIDDEN, 2)), rng.uniform(-k1, k1, HIDDEN),
            rng.uniform(-k2, k2, (1, HIDDEN)), rng.uniform(-k2, k2, 1))


def train_engine(weights):
    model = MLP(*[w.tolist() for w in weights])
    params = model.parameters()
    losses = []
    for _ in range(EPOCHS):
        loss = vsum(bce(model(xi), yi) for xi, yi in zip(X.tolist(), y.tolist())) / N_SAMPLES
        zero_grad(params)
        loss.backward()
        for p in params:                    # plain full-batch gradient descent
            p.data -= LR * p.grad
        losses.append(loss.data)
    return model, losses


def train_torch(weights):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, HIDDEN), torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 1), torch.nn.Sigmoid(),
    ).double()
    with torch.no_grad():
        for param, w in zip(model.parameters(), weights):
            param.copy_(torch.from_numpy(w))
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y.astype(np.float64)).unsqueeze(1)
    opt = torch.optim.SGD(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCELoss()
    losses = []
    for _ in range(EPOCHS):
        loss = loss_fn(model(Xt), yt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return model, losses


def accuracy_engine(model):
    preds = [model(xi).data > 0.5 for xi in X.tolist()]
    return float(np.mean(np.array(preds) == y))


def accuracy_torch(model):
    with torch.no_grad():
        p = model(torch.from_numpy(X)).squeeze(1).numpy()
    return float(np.mean((p > 0.5) == y))


def timed(fn, *args):
    t0 = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t0


if __name__ == "__main__":
    torch.manual_seed(0)
    eng_times, pt_times = [], []
    for run in range(RUNS):
        w = init_weights(seed=run)
        (eng_model, eng_losses), te = timed(train_engine, w)
        (pt_model, pt_losses), tp = timed(train_torch, w)
        eng_times.append(te)
        pt_times.append(tp)
        loss_gap = max(abs(a - b) for a, b in zip(eng_losses, pt_losses))
        print(f"run {run}: engine {te:.3f}s  torch {tp:.3f}s  "
              f"final loss engine={eng_losses[-1]:.6f} torch={pt_losses[-1]:.6f}  "
              f"max loss gap={loss_gap:.2e}")

    med_e, med_p = statistics.median(eng_times), statistics.median(pt_times)
    acc_e, acc_p = accuracy_engine(eng_model), accuracy_torch(pt_model)
    print(f"\nmedian of {RUNS} runs: engine {med_e:.3f}s, torch {med_p:.3f}s "
          f"-> torch is {med_e / med_p:.1f}x faster")
    print(f"accuracy (last run): engine {acc_e:.3f}, torch {acc_p:.3f}")

    report = {
        "setup": {"n_samples": N_SAMPLES, "arch": "2-8-1 ReLU/sigmoid", "loss": "BCE",
                  "epochs": EPOCHS, "lr": LR, "optimizer": "full-batch GD", "dtype": "float64",
                  "runs": RUNS},
        "engine_times_s": eng_times, "torch_times_s": pt_times,
        "engine_median_s": med_e, "torch_median_s": med_p, "speedup_torch": med_e / med_p,
        "final_loss_engine": eng_losses[-1], "final_loss_torch": pt_losses[-1],
        "max_loss_gap_last_run": loss_gap,
        "accuracy_engine": acc_e, "accuracy_torch": acc_p,
    }
    (RESULTS / "moons_training.json").write_text(json.dumps(report, indent=2))

    # figure: loss curves + decision boundary of the engine model
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(eng_losses, label="our engine", lw=2)
    ax1.plot(pt_losses, "--", label="PyTorch", lw=2)
    ax1.set(xlabel="epoch", ylabel="BCE loss", title="Training loss (same init)")
    ax1.legend()

    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 60),
                         np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 60))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.array([eng_model(p).data for p in grid.tolist()]).reshape(xx.shape)
    ax2.contourf(xx, yy, zz, levels=[0, .5, 1], alpha=.3, cmap="coolwarm")
    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k", s=25)
    ax2.set(title=f"Engine decision boundary (acc={acc_e:.2f})")
    fig.tight_layout()
    fig.savefig(RESULTS / "moons_training.png", dpi=120)
    print(f"saved {RESULTS / 'moons_training.json'} and moons_training.png")