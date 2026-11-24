"""Lab 7 - shared utilities: data, models, training, metrics (size, MACs, latency, energy)."""
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
CKPT_DIR = ROOT / "checkpoints"
SIZE_DIR = CKPT_DIR / "size_formats"
for _d in (DATA_DIR, RESULTS_DIR, FIG_DIR, CKPT_DIR, SIZE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DEFAULT_THREADS = torch.get_num_threads()


# ----------------------------------------------------------------------------- reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------- data
def load_fashion_mnist(val_size: int = 5000, seed: int = 42, train_fraction: float = 1.0):
    """Returns dict of tensors: train / val / test. Whole dataset lives in RAM (it is tiny),
    so there is no DataLoader / PIL overhead on CPU. Normalisation stats come from the
    training split only (no leakage into val/test)."""
    train = FashionMNIST(DATA_DIR, train=True, download=True)
    test = FashionMNIST(DATA_DIR, train=False, download=True)

    x_all = train.data.float().div(255.0).unsqueeze(1)
    y_all = train.targets.clone()
    x_test = test.data.float().div(255.0).unsqueeze(1)
    y_test = test.targets.clone()

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x_all), generator=g)
    val_idx, tr_idx = perm[:val_size], perm[val_size:]
    if train_fraction < 1.0:
        tr_idx = tr_idx[: int(len(tr_idx) * train_fraction)]

    x_tr, y_tr = x_all[tr_idx], y_all[tr_idx]
    mean, std = x_tr.mean().item(), x_tr.std().item()
    norm = lambda x: (x - mean) / std
    return {
        "train": (norm(x_tr), y_tr),
        "val": (norm(x_all[val_idx]), y_all[val_idx]),
        "test": (norm(x_test), y_test),
        "stats": (mean, std),
    }


def iterate(tensors, batch_size: int, shuffle: bool, generator=None):
    n = len(tensors[0])
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        b = idx[i : i + batch_size]
        yield tuple(t[b] for t in tensors)


# ----------------------------------------------------------------------------- models
class BigCNN(nn.Module):
    """Oversized baseline / teacher. Default config = 985,290 parameters (~1M)."""

    def __init__(self, ch=(32, 64, 128, 128), hidden=640, num_classes=10):
        super().__init__()
        self.ch, self.hidden = tuple(int(c) for c in ch), hidden
        c1, c2, c3, c4 = self.ch
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),                      # 28x28
            nn.Conv2d(c1, c2, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(), nn.MaxPool2d(2),    # 14x14
            nn.Conv2d(c2, c3, 3, padding=1), nn.BatchNorm2d(c3), nn.ReLU(), nn.MaxPool2d(2),    # 7x7
            nn.Conv2d(c3, c4, 3, padding=1), nn.BatchNorm2d(c4), nn.ReLU(), nn.MaxPool2d(2),    # 3x3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(c4 * 9, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SmallCNN(nn.Module):
    """Compact student. 98,666 parameters (~100k)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),    # 14x14
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),   # 7x7
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),   # 3x3
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2),
            nn.Linear(64 * 9, 128), nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ----------------------------------------------------------------------------- training
@torch.no_grad()
def evaluate(model, split, batch_size: int = 1000) -> float:
    model.eval()
    x, y = split[0], split[1]
    correct = 0
    for i in range(0, len(x), batch_size):
        correct += (model(x[i : i + batch_size]).argmax(1) == y[i : i + batch_size]).sum().item()
    return correct / len(x)


def ce_loss(model, xb, yb, *_):
    return F.cross_entropy(model(xb), yb)


def train_epochs(model, train_tensors, val_split, epochs, lr=1e-3, batch_size=128,
                 loss_fn=ce_loss, after_step=None, seed=42, tag=""):
    """Generic loop. `train_tensors` = (x, y, [extra...]) - extra tensors (e.g. teacher logits)
    are passed to loss_fn. `after_step(model)` is called after every optimizer step
    (used to keep pruning masks fixed)."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(train_tensors[0]) / batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    g = torch.Generator().manual_seed(seed)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        t0, run_loss, n = time.perf_counter(), 0.0, 0
        for batch in iterate(train_tensors, batch_size, shuffle=True, generator=g):
            loss = loss_fn(model, *batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if after_step is not None:
                after_step(model)
            run_loss += loss.item() * len(batch[0])
            n += len(batch[0])
        val_acc = evaluate(model, val_split)
        dt = time.perf_counter() - t0
        history.append({"tag": tag, "epoch": ep, "train_loss": run_loss / n, "val_acc": val_acc, "epoch_s": dt})
        print(f"  [{tag}] epoch {ep}/{epochs}  loss={run_loss / n:.4f}  val_acc={val_acc:.4f}  ({dt:.1f}s)")
    return history


# ----------------------------------------------------------------------------- metrics
def param_stats(model):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    return total, nonzero


def prunable_sparsity(model) -> float:
    ws = [m.weight for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    total = sum(w.numel() for w in ws)
    zeros = sum(int((w == 0).sum()) for w in ws)
    return zeros / total


@torch.no_grad()
def count_macs(model, input_shape=(1, 1, 28, 28)):
    """MACs per single image. dense = what a dense kernel executes;
    effective = MACs with non-zero weights only (ideal sparse kernel)."""
    stats = {"dense": 0, "effective": 0}

    def conv_hook(m, inp, out):
        spatial = out.shape[2] * out.shape[3]
        stats["dense"] += m.weight.numel() * spatial
        stats["effective"] += int((m.weight != 0).sum()) * spatial

    def lin_hook(m, inp, out):
        stats["dense"] += m.weight.numel()
        stats["effective"] += int((m.weight != 0).sum())

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    model.eval()
    model(torch.zeros(input_shape))
    for h in hooks:
        h.remove()
    return stats


def storage_sizes(model, tag: str) -> dict:
    """Writes the same weights in several formats and returns REAL on-disk sizes (bytes)."""
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    sizes = {}

    # 1) dense fp32 (what torch.save gives you by default)
    p = SIZE_DIR / f"{tag}_dense.pt"
    torch.save(sd, p)
    sizes["dense_pt"] = p.stat().st_size

    # 2) PyTorch sparse COO (int64 index per dimension per non-zero!)
    coo = {k: (v.to_sparse() if v.dim() >= 2 else v) for k, v in sd.items()}
    p = SIZE_DIR / f"{tag}_coo.pt"
    torch.save(coo, p)
    sizes["sparse_coo_pt"] = p.stat().st_size

    # 3) CSR with int32 column indices, 4) bitmask + packed non-zero values
    csr, bitmask, dense_np = {}, {}, {}
    for k, v in sd.items():
        a = v.numpy()
        dense_np[k] = a
        if v.dim() >= 2:
            m = a.reshape(a.shape[0], -1)
            nz = m != 0
            csr[k + ".values"] = m[nz].astype(np.float32)
            csr[k + ".col"] = np.nonzero(nz)[1].astype(np.int32)
            csr[k + ".rowptr"] = np.concatenate([[0], np.cumsum(nz.sum(1))]).astype(np.int32)
            csr[k + ".shape"] = np.array(a.shape)
            flat = a.reshape(-1)
            bitmask[k + ".mask"] = np.packbits(flat != 0)
            bitmask[k + ".values"] = flat[flat != 0].astype(np.float32)
            bitmask[k + ".shape"] = np.array(a.shape)
        else:
            csr[k] = a
            bitmask[k] = a
    for name, obj in (("csr_int32", csr), ("bitmask", bitmask)):
        p = SIZE_DIR / f"{tag}_{name}.npz"
        np.savez(p, **obj)
        sizes[f"{name}_npz"] = p.stat().st_size

    # 5) dense weights + generic zip compression (zeros compress well)
    p = SIZE_DIR / f"{tag}_dense_zip.npz"
    np.savez_compressed(p, **dense_np)
    sizes["dense_zip_npz"] = p.stat().st_size
    return sizes


@torch.no_grad()
def measure_latency(model, batch_size=1, threads=1, warmup=20, runs=200):
    """Median and p90 wall-clock latency in ms on CPU."""
    torch.set_num_threads(threads)
    model.eval()
    x = torch.randn(batch_size, 1, 28, 28)
    for _ in range(warmup):
        model(x)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        ts.append(time.perf_counter() - t0)
    torch.set_num_threads(DEFAULT_THREADS)
    ts = np.array(ts) * 1e3
    return float(np.median(ts)), float(np.percentile(ts, 90))


# Analytical energy model (Horowitz, ISSCC 2014, 45 nm), the same source as Reddi Fig. 10.13:
#   fp32 multiply 3.7 pJ + fp32 add 0.9 pJ  -> 4.6 pJ per MAC
#   32-bit DRAM read ~640 pJ (Reddi §10.5 quotes 1.3-2.6 nJ per DRAM access)
E_MAC_PJ = 4.6
E_DRAM_WORD_PJ = 640.0


def energy_uj(macs: int, weight_words: int) -> float:
    """Estimated energy per inference in microjoules: compute + reading every weight once from DRAM."""
    return (macs * E_MAC_PJ + weight_words * E_DRAM_WORD_PJ) * 1e-6