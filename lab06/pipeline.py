"""Lab 6: reproducible, YAML-driven training pipeline (seed, checkpoints, MLflow)."""
import json
import random
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import psutil
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

LAB_DIR = Path(__file__).resolve().parent
RESULTS = LAB_DIR / "results"
sys.path.insert(0, str(LAB_DIR.parent / "lab03"))
from models import MODELS  # noqa: E402  (Lab 3 architectures)

MEAN, STD = 0.2860, 0.3530
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- config
def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else LAB_DIR / path


def load_config(path="config.yaml", overrides=()):
    path = Path(path)
    if not path.exists():
        path = resolve(path)
    cfg = yaml.safe_load(path.read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


# ---------------------------------------------------------------- reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def seed_worker(_worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def epoch_seed(cfg, epoch):
    # shuffle order depends only on (seed, epoch) -> identical after a resume
    return cfg["seed"] * 1000 + epoch


# ---------------------------------------------------------------- builders
def build_loaders(cfg):
    d = cfg["data"]
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    root = resolve(d["root"])
    full = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
    split_g = torch.Generator().manual_seed(cfg["seed"])
    train, val = random_split(full, [len(full) - d["val_size"], d["val_size"]], generator=split_g)

    nw = d["num_workers"]
    shuffle_g = torch.Generator()
    train_dl = DataLoader(
        train, batch_size=d["batch_size"], shuffle=True, generator=shuffle_g,
        num_workers=nw, worker_init_fn=seed_worker, pin_memory=DEVICE == "cuda",
        persistent_workers=nw > 0 and d.get("persistent_workers", True),
    )
    val_dl = DataLoader(val, batch_size=d["eval_batch_size"])
    test_dl = DataLoader(test, batch_size=d["eval_batch_size"])
    return train_dl, val_dl, test_dl, shuffle_g


def build_model(cfg):
    return MODELS[cfg["model"]]()


def build_optimizer(cfg, model):
    t = cfg["train"]
    if t["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"],
                               weight_decay=t["weight_decay"])
    if t["optimizer"] == "adam":
        return torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    raise ValueError(f"unknown optimizer {t['optimizer']}")


def build_scheduler(cfg, opt):
    t = cfg["train"]
    s = t["scheduler"]
    if s == "none":
        return None
    if s == "step":
        return StepLR(opt, step_size=t["step_size"], gamma=t["gamma"])
    if s == "cosine":
        return CosineAnnealingLR(opt, T_max=t["epochs"])
    if s == "plateau":
        return ReduceLROnPlateau(opt, mode="min", factor=t["plateau_factor"],
                                 patience=t["plateau_patience"])
    raise ValueError(f"unknown scheduler {s}")


def setup_mlflow(cfg):
    mlflow.set_tracking_uri(resolve(cfg["mlflow"]["tracking_uri"]).as_uri())
    mlflow.set_experiment(cfg["experiment"])


# ---------------------------------------------------------------- memory
class PeakMemory:
    """Peak memory of this process: CUDA allocator peak on GPU, sampled RSS on CPU."""

    def __init__(self):
        self.proc = psutil.Process()
        if DEVICE == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self.base = self.proc.memory_info().rss
        self.peak = self.base

    def sample(self):
        if DEVICE == "cpu":
            self.peak = max(self.peak, self.proc.memory_info().rss)

    def result(self):
        if DEVICE == "cuda":
            mb = torch.cuda.max_memory_allocated() / 2**20
            return {"peak_mem_mb": mb, "peak_mem_delta_mb": mb}
        return {"peak_mem_mb": self.peak / 2**20, "peak_mem_delta_mb": (self.peak - self.base) / 2**20}


# ---------------------------------------------------------------- loops
def train_epoch(model, loader, opt, loss_fn, accum_steps=1, mem=None):
    model.train()
    opt.zero_grad(set_to_none=True)
    total_loss, correct, n = 0.0, 0, 0
    n_batches = len(loader)
    t0 = time.perf_counter()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        out = model(x)
        loss = loss_fn(out, y)
        (loss / accum_steps).backward()          # gradients add up over accum_steps micro-batches
        if (i + 1) % accum_steps == 0 or i + 1 == n_batches:
            opt.step()
            opt.zero_grad(set_to_none=True)
        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
        if mem is not None:
            mem.sample()
    secs = time.perf_counter() - t0
    return {"train_loss": total_loss / n, "train_acc": correct / n,
            "epoch_s": secs, "throughput": n / secs}


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        total_loss += loss_fn(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(path, model, opt, sched, epoch, history, best_acc, run_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched else None,
        "history": history,
        "best_val_acc": best_acc,
        "mlflow_run_id": run_id,
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                "python": random.getstate()},
    }, tmp)
    tmp.replace(path)                            # atomic: a crash never leaves a half-written file


def load_checkpoint(path, model, opt, sched):
    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state["model"])
    opt.load_state_dict(state["optimizer"])
    if sched and state["scheduler"]:
        sched.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"]["torch"])
    np.random.set_state(state["rng"]["numpy"])
    random.setstate(state["rng"]["python"])
    return state


# ---------------------------------------------------------------- main entry
def run(cfg, resume=False, stop_after=None):
    set_seed(cfg["seed"])
    train_dl, val_dl, test_dl, shuffle_g = build_loaders(cfg)
    model = build_model(cfg).to(DEVICE)
    opt = build_optimizer(cfg, model)
    sched = build_scheduler(cfg, opt)
    loss_fn = nn.CrossEntropyLoss()
    t = cfg["train"]

    ckpt_dir = resolve(cfg["checkpoint"]["dir"]) / cfg["run_name"]
    last_path, best_path = ckpt_dir / "last.pt", ckpt_dir / "best.pt"
    start_epoch, history, best_acc, run_id = 1, [], 0.0, None
    if resume:
        state = load_checkpoint(last_path, model, opt, sched)
        start_epoch = state["epoch"] + 1
        history, best_acc, run_id = state["history"], state["best_val_acc"], state["mlflow_run_id"]
        print(f"Resumed '{cfg['run_name']}' from epoch {state['epoch']} (MLflow run {run_id})")

    setup_mlflow(cfg)
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else cfg["run_name"]) as active:
        run_id = active.info.run_id
        if not resume:
            mlflow.log_params(flatten(cfg))
            mlflow.set_tags({"device": DEVICE, "torch": torch.__version__,
                             "params": sum(p.numel() for p in model.parameters())})
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (ckpt_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
            mlflow.log_artifact(str(ckpt_dir / "config.yaml"))
        else:
            mlflow.set_tag("resumed_from_epoch", start_epoch - 1)

        mem = PeakMemory()
        for epoch in range(start_epoch, t["epochs"] + 1):
            shuffle_g.manual_seed(epoch_seed(cfg, epoch))
            row = train_epoch(model, train_dl, opt, loss_fn, t["accum_steps"], mem)
            val_loss, val_acc = evaluate(model, val_dl, loss_fn)
            lr = opt.param_groups[0]["lr"]
            if isinstance(sched, ReduceLROnPlateau):
                sched.step(val_loss)
            elif sched:
                sched.step()
            row = {"epoch": epoch, **row, "val_loss": val_loss, "val_acc": val_acc, "lr": lr}
            history.append(row)
            mlflow.log_metrics({k: v for k, v in row.items() if k != "epoch"}, step=epoch)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_path)
            save_checkpoint(last_path, model, opt, sched, epoch, history, best_acc, run_id)
            print(f"[{cfg['run_name']}] epoch {epoch}/{t['epochs']} loss={row['train_loss']:.4f} "
                  f"val_acc={val_acc:.4f} lr={lr:.2e} {row['throughput']:.0f} samples/s")
            if stop_after is not None and epoch == stop_after and epoch < t["epochs"]:
                print(f"*** simulated interruption after epoch {epoch} - resume with --resume ***")
                mlflow.set_tag("interrupted_at_epoch", epoch)
                return _write_result(cfg, history, None, mem, interrupted=True)

        _, test_acc = evaluate(model, test_dl, loss_fn)
        result = _write_result(cfg, history, test_acc, mem)
        mlflow.log_metrics({"test_acc": test_acc, "best_val_acc": best_acc,
                            "median_throughput": result["median_throughput"],
                            **result["memory"]})
        mlflow.log_artifact(str(best_path))
        mlflow.log_artifact(str(RESULTS / "runs" / f"{cfg['run_name']}.json"))
        print(f"[{cfg['run_name']}] TEST acc={test_acc:.4f}")
        return result


def _write_result(cfg, history, test_acc, mem, interrupted=False):
    tps = [h["throughput"] for h in history]
    result = {"run_name": cfg["run_name"], "config": cfg, "history": history,
              "test_acc": test_acc, "interrupted": interrupted,
              "median_throughput": float(np.median(tps)) if tps else None,
              "memory": mem.result()}
    out = RESULTS / "runs" / f"{cfg['run_name']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return result
