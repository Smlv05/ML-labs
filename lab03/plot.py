import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).resolve().parent / "results"
cost = json.loads((RESULTS / "cost.json").read_text())["models"]
train = json.loads((RESULTS / "train.json").read_text())
names = list(cost)
acc = {n: train[n]["test_acc"] * 100 for n in names}


def scatter(ax, xs, xlabel, title, log=False):
    for n in names:
        ax.scatter(xs[n], acc[n], s=70)
        ax.annotate(n, (xs[n], acc[n]), textcoords="offset points", xytext=(6, 4))
    if log:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title(title)
    ax.grid(alpha=0.3)


fig, ax = plt.subplots(figsize=(6, 4))
scatter(ax, {n: cost[n]["params"] for n in names}, "Parameters (log)", "Accuracy vs parameters", log=True)
fig.tight_layout()
fig.savefig(RESULTS / "acc_vs_params.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
scatter(axes[0], {n: cost[n]["latency_b1_ms"] for n in names}, "Latency, batch 1 (ms)", "Accuracy vs latency (b=1)")
scatter(axes[1], {n: cost[n]["latency_b128_ms"] for n in names}, "Latency, batch 128 (ms)", "Accuracy vs latency (b=128)")
fig.tight_layout()
fig.savefig(RESULTS / "acc_vs_latency.png", dpi=150)
print("Saved plots to results/")