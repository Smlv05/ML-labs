"""Lab 6 Tasks 3-7: studies. Each training run is a separate MLflow run.

  python studies.py workers     # Task 3: num_workers 0,1,2,4,8
  python studies.py batch       # Task 4: batch 32..512, 5 epochs
  python studies.py schedules   # Task 5: StepLR / Cosine / Plateau x SGD+momentum / Adam, 10 epochs
  python studies.py accum       # Task 6: batch 256 vs 64 x 4 accumulation
  python studies.py resume      # Task 7: stop at epoch 3, resume, compare with uninterrupted run
  python studies.py export      # dump all MLflow runs to results/mlflow_runs.csv
  python studies.py all
Every study writes results/<study>.csv, results/<study>.md (table) and results/<study>.png.
"""
import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from pipeline import (DEVICE, LAB_DIR, RESULTS, build_loaders, build_model,  # noqa: E402
                      build_optimizer, epoch_seed, load_config, set_seed, setup_mlflow)

CONFIG = LAB_DIR / "config.yaml"


# ---------------------------------------------------------------- helpers
def train_run(run_name, overrides=(), flags=()):
    """Run train.py in a fresh process (clean peak-memory measurement per run)."""
    cmd = [sys.executable, str(LAB_DIR / "train.py"), "--config", str(CONFIG), *flags,
           "--set", f"run_name={run_name}", *overrides]
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=LAB_DIR)
    return json.loads((RESULTS / "runs" / f"{run_name}.json").read_text())


def write_table(name, rows):
    RESULTS.mkdir(exist_ok=True)
    cols = list(rows[0])
    with open(RESULTS / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) and abs(v) < 10 else (f"{v:.1f}" if isinstance(v, float) else str(v))  # noqa: E731
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    (RESULTS / f"{name}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def curves(ax, runs, key, labels=None):
    for i, r in enumerate(runs):
        h = r["history"]
        ax.plot([e["epoch"] for e in h], [e[key] for e in h], marker="o", ms=3,
                label=labels[i] if labels else r["run_name"])
    ax.set_xlabel("epoch")
    ax.set_ylabel(key)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


# ---------------------------------------------------------------- Task 3
def study_workers():
    cfg = load_config(CONFIG)
    setup_mlflow(cfg)
    loss_fn = nn.CrossEntropyLoss()
    rows, compute_s = [], None
    print(f"CPU cores: {os.cpu_count()}, device: {DEVICE}")
    for nw in [0, 1, 2, 4, 8]:
        c = copy.deepcopy(cfg)
        c["data"]["num_workers"] = nw
        c["data"]["persistent_workers"] = False      # count worker start-up honestly
        set_seed(c["seed"])
        train_dl, _, _, g = build_loaders(c)

        # (a) data only: iterate the loader, no model
        g.manual_seed(epoch_seed(c, 1))
        t0, data_first, batches = time.perf_counter(), None, []
        for i, (x, y) in enumerate(train_dl):
            if i == 0:
                data_first = time.perf_counter() - t0
            if compute_s is None:
                batches.append((x, y))
        data_s = time.perf_counter() - t0

        # (b) compute only (once): train on batches already in memory
        if compute_s is None:
            model = build_model(c).to(DEVICE)
            opt = build_optimizer(c, model)
            model.train()
            t0 = time.perf_counter()
            for x, y in batches:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                loss_fn(model(x), y).backward()
                opt.step()
            compute_s = time.perf_counter() - t0
            del batches

        # (c) real training epoch = data + compute together
        set_seed(c["seed"])
        model = build_model(c).to(DEVICE)
        opt = build_optimizer(c, model)
        model.train()
        g.manual_seed(epoch_seed(c, 1))
        t0, train_first, n = time.perf_counter(), None, 0
        for i, (x, y) in enumerate(train_dl):
            if i == 0:
                train_first = time.perf_counter() - t0
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x), y).backward()
            opt.step()
            n += y.size(0)
        train_s = time.perf_counter() - t0
        steady_s = train_s - train_first
        data_steady_s = data_s - data_first

        row = {"num_workers": nw, "startup_s": train_first,
               "data_only_steady_s": data_steady_s, "compute_only_s": compute_s,
               "train_epoch_total_s": train_s, "train_epoch_steady_s": steady_s,
               "throughput_total": n / train_s, "throughput_steady": n / steady_s,
               "bottleneck": "data" if data_steady_s > compute_s else "compute"}
        rows.append(row)
        with mlflow.start_run(run_name=f"workers-nw{nw}"):
            mlflow.log_params({"study": "num_workers", "num_workers": nw,
                               "batch_size": c["data"]["batch_size"], "cpu_cores": os.cpu_count()})
            mlflow.log_metrics({k: v for k, v in row.items() if isinstance(v, float)})
        print(row, flush=True)

    write_table("workers", rows)
    nws = [r["num_workers"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.5))
    a1.plot(nws, [r["train_epoch_total_s"] for r in rows], "s-", label="training epoch, total (incl. start-up)")
    a1.plot(nws, [r["train_epoch_steady_s"] for r in rows], "o-", label="training epoch, steady state")
    a1.plot(nws, [r["data_only_steady_s"] for r in rows], "^--", label="data loading only, steady state")
    a1.axhline(compute_s, color="gray", ls=":", label="compute only (data in RAM)")
    a1.set_xlabel("num_workers"); a1.set_ylabel("seconds per epoch"); a1.grid(alpha=0.3)
    a1.legend(fontsize=8); a1.set_title(f"Epoch time vs num_workers ({os.cpu_count()} CPU cores)")
    a2.bar([str(v) for v in nws], [r["startup_s"] for r in rows], color="C3")
    a2.set_xlabel("num_workers"); a2.set_ylabel("seconds until first batch")
    a2.set_title("Worker start-up cost (Windows spawn)"); a2.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(RESULTS / "workers.png", dpi=150)


# ---------------------------------------------------------------- Task 4
def study_batch():
    runs, rows = [], []
    for bs in [32, 64, 128, 256, 512]:
        r = train_run(f"batch-{bs}", [f"data.batch_size={bs}", "train.epochs=5"])
        runs.append(r)
        rows.append({"batch_size": bs, "throughput_samples_s": r["median_throughput"],
                     "peak_mem_mb": r["memory"]["peak_mem_mb"],
                     "peak_mem_delta_mb": r["memory"]["peak_mem_delta_mb"],
                     "epoch_s": sum(h["epoch_s"] for h in r["history"]) / len(r["history"]),
                     "val_acc_ep5": r["history"][-1]["val_acc"], "test_acc": r["test_acc"]})
    write_table("batch", rows)
    bss = [r["batch_size"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(bss, [r["throughput_samples_s"] for r in rows], "o-"); ax[0].set_ylabel("samples / s")
    ax[1].plot(bss, [r["peak_mem_delta_mb"] for r in rows], "o-", color="C1")
    ax[1].set_ylabel("peak memory above baseline (MB)")
    ax[2].plot(bss, [r["val_acc_ep5"] for r in rows], "o-", color="C2", label="val (ep 5)")
    ax[2].plot(bss, [r["test_acc"] for r in rows], "s--", color="C3", label="test")
    ax[2].set_ylabel("accuracy after 5 epochs"); ax[2].legend()
    for a in ax:
        a.set_xscale("log", base=2); a.set_xticks(bss); a.set_xticklabels(bss)
        a.set_xlabel("batch size"); a.grid(alpha=0.3)
    fig.suptitle("Batch-size study (CNN, Adam lr=1e-3, 5 epochs)")
    fig.tight_layout(); fig.savefig(RESULTS / "batch.png", dpi=150)


# ---------------------------------------------------------------- Task 5
def study_schedules():
    runs, rows = [], []
    opts = {"sgd": ["train.optimizer=sgd", "train.lr=0.05", "train.momentum=0.9"],
            "adam": ["train.optimizer=adam", "train.lr=0.001"]}
    for opt_name, opt_over in opts.items():
        for sched in ["step", "cosine", "plateau"]:
            r = train_run(f"sched-{opt_name}-{sched}",
                          [*opt_over, f"train.scheduler={sched}", "train.epochs=10"])
            runs.append(r)
            h = r["history"]
            rows.append({"optimizer": "SGD+momentum" if opt_name == "sgd" else "Adam",
                         "scheduler": sched, "final_train_loss": h[-1]["train_loss"],
                         "final_val_acc": h[-1]["val_acc"], "best_val_acc": max(e["val_acc"] for e in h),
                         "test_acc": r["test_acc"], "final_lr": h[-1]["lr"]})
    write_table("schedules", rows)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    curves(ax[0], runs, "val_acc"); ax[0].set_title("Validation accuracy")
    curves(ax[1], runs, "train_loss"); ax[1].set_title("Training loss")
    curves(ax[2], runs, "lr"); ax[2].set_yscale("log"); ax[2].set_title("Learning rate (log)")
    fig.suptitle("LR schedules x optimizers, equal 10-epoch budget")
    fig.tight_layout(); fig.savefig(RESULTS / "schedules.png", dpi=150)


# ---------------------------------------------------------------- Task 6
def study_accum():
    big = train_run("accum-bs256", ["data.batch_size=256", "train.accum_steps=1", "train.epochs=5"])
    acc = train_run("accum-bs64x4", ["data.batch_size=64", "train.accum_steps=4", "train.epochs=5"])
    rows = []
    for a, b in zip(big["history"], acc["history"]):
        rows.append({"epoch": a["epoch"], "loss_bs256": a["train_loss"], "loss_64x4": b["train_loss"],
                     "val_acc_bs256": a["val_acc"], "val_acc_64x4": b["val_acc"],
                     "abs_diff_val_acc": abs(a["val_acc"] - b["val_acc"]),
                     "throughput_bs256": a["throughput"], "throughput_64x4": b["throughput"]})
    rows.append({"epoch": "test", "loss_bs256": "", "loss_64x4": "", "val_acc_bs256": big["test_acc"],
                 "val_acc_64x4": acc["test_acc"], "abs_diff_val_acc": abs(big["test_acc"] - acc["test_acc"]),
                 "throughput_bs256": "", "throughput_64x4": ""})
    rows.append({"epoch": "peak_mem_delta_mb", "loss_bs256": "", "loss_64x4": "",
                 "val_acc_bs256": big["memory"]["peak_mem_delta_mb"],
                 "val_acc_64x4": acc["memory"]["peak_mem_delta_mb"], "abs_diff_val_acc": "",
                 "throughput_bs256": "", "throughput_64x4": ""})
    write_table("accum", rows)
    labels = ["batch 256", "batch 64 x accum 4"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    curves(ax[0], [big, acc], "train_loss", labels); ax[0].set_title("Training loss")
    curves(ax[1], [big, acc], "val_acc", labels); ax[1].set_title("Validation accuracy")
    fig.suptitle("Gradient accumulation: effective batch 256")
    fig.tight_layout(); fig.savefig(RESULTS / "accum.png", dpi=150)


# ---------------------------------------------------------------- Task 7
def study_resume():
    over = ["train.epochs=6"]
    ref = train_run("resume-reference", over)
    train_run("resume-demo", over, ["--stop-after", "3"])
    res = train_run("resume-demo", over, ["--resume"])
    rows = [{"epoch": a["epoch"], "segment": "before crash" if a["epoch"] <= 3 else "after resume",
             "val_acc_uninterrupted": a["val_acc"], "val_acc_resumed": b["val_acc"],
             "train_loss_uninterrupted": a["train_loss"], "train_loss_resumed": b["train_loss"],
             "abs_diff_loss": abs(a["train_loss"] - b["train_loss"])}
            for a, b in zip(ref["history"], res["history"])]
    write_table("resume", rows)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, key in zip(ax, ["train_loss", "val_acc"]):
        curves(a, [ref, res], key, ["uninterrupted", "interrupted @3 + resumed"])
        a.axvline(3.5, color="red", ls=":", label="crash / resume")
        a.legend(fontsize=8)
    fig.suptitle("Checkpoint recovery: curves continue smoothly")
    fig.tight_layout(); fig.savefig(RESULTS / "resume.png", dpi=150)


def export_runs():
    cfg = load_config(CONFIG)
    setup_mlflow(cfg)
    df = mlflow.search_runs(experiment_names=[cfg["experiment"]])
    keep = [c for c in df.columns if c in ("run_id", "tags.mlflow.runName", "status", "start_time")
            or c.startswith(("metrics.", "params.train.", "params.data.batch_size", "params.data.num_workers"))]
    df[keep].to_csv(RESULTS / "mlflow_runs.csv", index=False)
    print(f"{len(df)} runs -> results/mlflow_runs.csv")


STUDIES = {"workers": study_workers, "batch": study_batch, "schedules": study_schedules,
           "accum": study_accum, "resume": study_resume, "export": export_runs}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("study", choices=[*STUDIES, "all"])
    name = p.parse_args().study
    for key in (STUDIES if name == "all" else [name]):
        STUDIES[key]()
