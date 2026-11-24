"""Lab 7 - Compression: pruning and distillation (Fashion-MNIST, CPU).

Usage (from the repo root, with the lab venv active):
    python lab07/run_lab07.py --quick      # smoke test, ~2-4 min
    python lab07/run_lab07.py              # full run
"""
import argparse
import copy
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import (BigCNN, SmallCNN, CKPT_DIR, DEFAULT_THREADS, FIG_DIR, RESULTS_DIR, count_macs, energy_uj,
                    evaluate, load_fashion_mnist, measure_latency, param_stats, prunable_sparsity,
                    set_seed, storage_sizes, train_epochs)
from pruning import (apply_masks, layer_sparsity, magnitude_masks, mask_keeper,
                     prunable_layers, structured_filter_prune)

SPARSITIES = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95]
FILTER_RATIOS = [0.10, 0.25, 0.50]
SIZE_COLS = ["dense_pt", "sparse_coo_pt", "csr_int32_npz", "bitmask_npz", "dense_zip_npz"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-epochs", type=int, default=8)
    p.add_argument("--ft-epochs", type=int, default=3)
    p.add_argument("--student-epochs", type=int, default=12)
    p.add_argument("--T", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-data-exp", action="store_true", help="skip the 10%%-data distillation experiment")
    p.add_argument("--quick", action="store_true", help="10%% of data, 1 epoch everywhere (smoke test)")
    return p.parse_args()


# ----------------------------------------------------------------------------- helpers
def prunable_count(model):
    return sum(m.weight.numel() for _, m in prunable_layers(model))


def profile(name, model, data, train_samples, supervision, lat_runs):
    """All metrics for one model -> one row."""
    total, nonzero = param_stats(model)
    macs = count_macs(model)
    lat1, lat1_p90 = measure_latency(model, batch_size=1, threads=1, runs=lat_runs)
    lat64, _ = measure_latency(model, batch_size=64, threads=torch.get_num_threads(), runs=max(20, lat_runs // 4))
    sizes = storage_sizes(model, name)
    best_fmt = min(SIZE_COLS, key=lambda c: sizes[c])
    return {
        "model": name,
        "train_samples": train_samples,
        "supervision": supervision,
        "test_acc": evaluate(model, data["test"]),
        "params": total,
        "nonzero_params": nonzero,
        "prunable_sparsity": prunable_sparsity(model),
        "macs_dense_M": macs["dense"] / 1e6,
        "macs_effective_M": macs["effective"] / 1e6,
        "lat_b1_ms": lat1,
        "lat_b1_p90_ms": lat1_p90,
        "lat_b64_ms": lat64,
        **{f"size_{k}_KB": v / 1024 for k, v in sizes.items()},
        "best_format": best_fmt,
        "energy_dense_exec_uJ": energy_uj(macs["dense"], total),
        "energy_ideal_sparse_uJ": energy_uj(macs["effective"], nonzero),
    }


def time_ms(fn, runs, warmup=10):
    import time
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


@torch.no_grad()
def sparse_matmul_benchmark(weight, runs):
    """Micro-benchmark on the biggest layer (fc1, 640x1152): dense GEMM vs sparse kernels."""
    torch.set_num_threads(1)
    rows = []
    for s in [0.0, 0.5, 0.75, 0.9, 0.95, 0.99]:
        W = weight.detach().clone().contiguous()
        k = int(s * W.numel())
        if k:
            W.view(-1)[W.abs().flatten().argsort()[:k]] = 0
        formats = {"dense": W, "coo": W.to_sparse()}
        try:
            formats["csr"] = W.to_sparse_csr()
        except Exception:
            pass
        for bs in (1, 64):
            x = torch.randn(W.shape[1], bs)
            for fmt, Wf in formats.items():
                try:
                    fn = (lambda Wf=Wf: Wf @ x) if fmt == "dense" else (lambda Wf=Wf: torch.sparse.mm(Wf, x))
                    t = time_ms(fn, runs)
                except Exception as e:  # some sparse layouts are not supported on every build
                    print(f"  sparse bench: {fmt} failed: {e}")
                    t = float("nan")
                rows.append({"sparsity": s, "batch": bs, "format": fmt, "ms": t})
    torch.set_num_threads(DEFAULT_THREADS)
    return pd.DataFrame(rows)


def kd_loss(T, alpha):
    """alpha * CE(student, y) + (1 - alpha) * T^2 * KL(teacher_T || student_T).
    Teacher logits are pre-computed once (no augmentation -> they never change), passed as `tb`."""
    def fn(student, xb, yb, tb):
        s = student(xb)
        ce = F.cross_entropy(s, yb)
        kl = F.kl_div(F.log_softmax(s / T, dim=1), F.log_softmax(tb / T, dim=1),
                      reduction="batchmean", log_target=True)
        return alpha * ce + (1 - alpha) * (T ** 2) * kl
    return fn


def ce_only(model, xb, yb, *_):
    return F.cross_entropy(model(xb), yb)


def to_markdown(df, floatfmt=".4g"):
    def f(v):
        return format(v, floatfmt) if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    if args.quick:
        args.baseline_epochs = args.ft_epochs = args.student_epochs = 1
    lat_runs = 30 if args.quick else 200
    suffix = "_quick" if args.quick else ""

    set_seed(args.seed)
    data = load_fashion_mnist(seed=args.seed, train_fraction=0.1 if args.quick else 1.0)
    n_train = len(data["train"][0])
    print(f"train={n_train} val={len(data['val'][0])} test={len(data['test'][0])}  threads={torch.get_num_threads()}")
    histories, profiles = [], []

    # ------------------------------------------------------------- 1. baseline
    print("\n=== 1. Baseline (oversized CNN) ===")
    ckpt = CKPT_DIR / f"baseline{suffix}.pt"
    set_seed(args.seed)
    baseline = BigCNN()
    print(f"baseline params: {param_stats(baseline)[0]:,}")
    if ckpt.exists():
        baseline.load_state_dict(torch.load(ckpt, weights_only=True))
        print(f"loaded {ckpt.name}")
    else:
        histories += train_epochs(baseline, data["train"], data["val"], args.baseline_epochs,
                                  lr=1e-3, seed=args.seed, tag="baseline")
        torch.save(baseline.state_dict(), ckpt)
    profiles.append(profile("baseline", baseline, data, n_train, "hard labels", lat_runs))
    print(f"baseline test acc = {profiles[-1]['test_acc']:.4f}")
    base_prunable = prunable_count(baseline)

    # ------------------------------------------------------------- 2a. unstructured pruning
    print("\n=== 2a. Magnitude pruning: global vs layer-wise ===")
    unstruct_rows, layer_rows = [], []
    cache = {}
    for scope in ("global", "layer"):
        for s in SPARSITIES:
            if s == 0.0 and scope == "layer" and ("global", 0.0) in cache:
                # identical experiment (nothing pruned) -> reuse the global result
                row = dict(next(r for r in unstruct_rows if r["scope"] == "global" and r["target"] == 0.0))
                row["scope"] = "layer"
                unstruct_rows.append(row)
                continue
            m = copy.deepcopy(baseline)
            masks = magnitude_masks(m, s, scope)
            apply_masks(m, masks)
            acc_oneshot = evaluate(m, data["test"])
            histories += train_epochs(m, data["train"], data["val"], args.ft_epochs, lr=3e-4,
                                      after_step=mask_keeper(masks), seed=args.seed, tag=f"{scope}_{s}")
            acc_ft = evaluate(m, data["test"])
            sizes = storage_sizes(m, f"{scope}_{s}")
            lat1, _ = measure_latency(m, batch_size=1, threads=1, runs=lat_runs)
            row = {"scope": scope, "target": s, "sparsity": prunable_sparsity(m),
                   "acc_oneshot": acc_oneshot, "acc_finetuned": acc_ft, "lat_b1_ms": lat1,
                   **{f"{k}_KB": v / 1024 for k, v in sizes.items()}}
            unstruct_rows.append(row)
            cache[(scope, s)] = m
            print(f"  {scope:6s} s={s:.2f}: one-shot {acc_oneshot:.4f} -> fine-tuned {acc_ft:.4f}")
            if s == 0.9:
                for layer, sp in layer_sparsity(m).items():
                    layer_rows.append({"scope": scope, "layer": layer, "sparsity": sp})
    unstruct = pd.DataFrame(unstruct_rows)
    unstruct.to_csv(RESULTS_DIR / "unstructured_pruning.csv", index=False)
    pd.DataFrame(layer_rows).to_csv(RESULTS_DIR / "layer_sparsity_at_90.csv", index=False)

    for key in [("global", 0.5), ("global", 0.9), ("layer", 0.9), ("global", 0.95)]:
        profiles.append(profile(f"{key[0]}_unstructured_{int(key[1]*100)}", cache[key], data,
                                n_train, "hard labels", lat_runs))

    # ------------------------------------------------------------- 2b. structured pruning
    print("\n=== 2b. Structured pruning: remove whole filters by L2 norm ===")
    struct_rows = []
    for r in FILTER_RATIOS:
        m = structured_filter_prune(baseline, r)
        acc_oneshot = evaluate(m, data["test"])
        histories += train_epochs(m, data["train"], data["val"], args.ft_epochs, lr=3e-4,
                                  seed=args.seed, tag=f"structured_{r}")
        acc_ft = evaluate(m, data["test"])
        lat1, _ = measure_latency(m, batch_size=1, threads=1, runs=lat_runs)
        struct_rows.append({"filter_ratio": r, "channels": str(m.ch), "params": param_stats(m)[0],
                            "sparsity": 1 - prunable_count(m) / base_prunable,
                            "acc_oneshot": acc_oneshot, "acc_finetuned": acc_ft, "lat_b1_ms": lat1})
        print(f"  filters -{int(r*100)}% {m.ch}: one-shot {acc_oneshot:.4f} -> fine-tuned {acc_ft:.4f}")
        profiles.append(profile(f"structured_filters_{int(r*100)}", m, data, n_train, "hard labels", lat_runs))
    struct = pd.DataFrame(struct_rows)
    struct.to_csv(RESULTS_DIR / "structured_pruning.csv", index=False)

    # ------------------------------------------------------------- 3. sparse kernels micro-benchmark
    print("\n=== 3. Dense vs sparse matmul on fc1 ===")
    bench = sparse_matmul_benchmark(baseline.classifier[2].weight, runs=max(30, lat_runs))
    bench.to_csv(RESULTS_DIR / "sparse_matmul_benchmark.csv", index=False)
    print(bench.pivot_table(index=["batch", "sparsity"], columns="format", values="ms").round(4))

    # ------------------------------------------------------------- 4. distillation
    print("\n=== 4. Knowledge distillation ===")
    baseline.eval()
    with torch.no_grad():
        x_tr = data["train"][0]
        t_logits = torch.cat([baseline(x_tr[i:i + 1000]) for i in range(0, len(x_tr), 1000)])
    kd_train = (data["train"][0], data["train"][1], t_logits)
    print(f"student params: {param_stats(SmallCNN())[0]:,}")

    def train_student(tag, tensors, use_kd):
        set_seed(args.seed)                 # identical init for plain and KD student
        st = SmallCNN()
        loss = kd_loss(args.T, args.alpha) if use_kd else ce_only
        h = train_epochs(st, tensors, data["val"], args.student_epochs, lr=2e-3,
                         loss_fn=loss, seed=args.seed, tag=tag)
        return st, h

    s_plain, h = train_student("student_plain", kd_train, use_kd=False); histories += h
    s_kd, h = train_student("student_kd", kd_train, use_kd=True); histories += h
    profiles.append(profile("student_plain", s_plain, data, n_train, "hard labels", lat_runs))
    profiles.append(profile("student_distilled", s_kd, data, n_train,
                            f"hard + soft (T={args.T:g}, a={args.alpha:g})", lat_runs))

    distill_rows = [{"student": "plain", "train_fraction": 1.0, "test_acc": profiles[-2]["test_acc"]},
                    {"student": "distilled", "train_fraction": 1.0, "test_acc": profiles[-1]["test_acc"]}]
    if not args.skip_data_exp:
        n10 = len(kd_train[0]) // 10
        sub = tuple(t[:n10] for t in kd_train)          # train split is already shuffled
        for use_kd in (False, True):
            tag = f"student_{'kd' if use_kd else 'plain'}_10pct"
            st, h = train_student(tag, sub, use_kd); histories += h
            distill_rows.append({"student": "distilled" if use_kd else "plain", "train_fraction": 0.1,
                                 "test_acc": evaluate(st, data["test"])})
    distill = pd.DataFrame(distill_rows)
    distill.to_csv(RESULTS_DIR / "distillation.csv", index=False)
    print(distill)

    # ------------------------------------------------------------- 5. tables
    prof = pd.DataFrame(profiles)
    prof.to_csv(RESULTS_DIR / "all_models_profile.csv", index=False)
    pd.DataFrame(histories).to_csv(RESULTS_DIR / "train_history.csv", index=False)

    pillar = pd.DataFrame({
        "Model": prof["model"],
        "DATA: train samples": prof["train_samples"],
        "DATA: supervision": prof["supervision"],
        "MODEL: test acc": prof["test_acc"].round(4),
        "MODEL: params": prof["params"],
        "MODEL: non-zero": prof["nonzero_params"],
        "MODEL: dense size KB": prof["size_dense_pt_KB"].round(1),
        "MODEL: best size KB (format)": [f"{prof.loc[i, 'size_' + prof.loc[i, 'best_format'] + '_KB']:.1f} ({prof.loc[i, 'best_format']})"
                                         for i in prof.index],
        "COMPUTE: dense MMACs": prof["macs_dense_M"].round(2),
        "COMPUTE: non-zero MMACs": prof["macs_effective_M"].round(2),
        "COMPUTE: latency b=1 ms": prof["lat_b1_ms"].round(3),
        "COMPUTE: latency b=64 ms": prof["lat_b64_ms"].round(2),
        "ENERGY: est. uJ dense exec": prof["energy_dense_exec_uJ"].round(1),
        "ENERGY: est. uJ ideal sparse": prof["energy_ideal_sparse_uJ"].round(1),
    })
    pillar.to_csv(RESULTS_DIR / "summary_by_pillar.csv", index=False)
    (RESULTS_DIR / "summary_by_pillar.md").write_text(to_markdown(pillar), encoding="utf-8")
    (RESULTS_DIR / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    print("\n" + to_markdown(pillar))

    make_figures(unstruct, struct, pd.DataFrame(layer_rows), bench, pd.DataFrame(histories), profiles[0])
    print(f"\nDone. Results -> {RESULTS_DIR}, figures -> {FIG_DIR}")


# ----------------------------------------------------------------------------- figures
def make_figures(unstruct, struct, layers, bench, hist, base):
    # (1) sparsity curves - graded evidence
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for scope, c in (("global", "tab:blue"), ("layer", "tab:orange")):
        d = unstruct[unstruct.scope == scope]
        ax.plot(d.sparsity, d.acc_finetuned, "o-", color=c, label=f"{scope} magnitude (fine-tuned 3 ep)")
        ax.plot(d.sparsity, d.acc_oneshot, "o--", color=c, alpha=0.5, label=f"{scope} magnitude (one-shot)")
    ax.plot(struct.sparsity, struct.acc_finetuned, "s-", color="tab:green", label="structured filters (fine-tuned)")
    ax.plot(struct.sparsity, struct.acc_oneshot, "s--", color="tab:green", alpha=0.5, label="structured filters (one-shot)")
    for _, r in struct.iterrows():
        ax.annotate(f"-{int(r.filter_ratio*100)}% filters", (r.sparsity, r.acc_finetuned),
                    textcoords="offset points", xytext=(4, -12), fontsize=8)
    ax.axhline(base["test_acc"], color="grey", lw=1, ls=":", label="dense baseline")
    ax.set_xlabel("fraction of conv/linear weights removed")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs sparsity (Fashion-MNIST)")
    ax.set_ylim(max(0.0, min(unstruct.acc_oneshot.min(), struct.acc_oneshot.min()) - 0.02), 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG_DIR / "sparsity_curves.png", dpi=150); plt.close(fig)

    # (2) where does global pruning cut?
    if len(layers):
        p = layers.pivot(index="layer", columns="scope", values="sparsity")
        ax = p.plot.bar(figsize=(7, 4), rot=30)
        ax.set_ylabel("sparsity"); ax.set_title("Per-layer sparsity at 90% target")
        ax.figure.tight_layout(); ax.figure.savefig(FIG_DIR / "layer_sparsity_90.png", dpi=150); plt.close(ax.figure)

    # (3) latency vs sparsity - honest latency analysis
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for scope, c in (("global", "tab:blue"), ("layer", "tab:orange")):
        d = unstruct[unstruct.scope == scope]
        ax.plot(d.sparsity, d.lat_b1_ms, "o-", color=c, label=f"{scope} unstructured (dense kernels)")
    ax.plot(struct.sparsity, struct.lat_b1_ms, "s-", color="tab:green", label="structured (smaller dense model)")
    ax.set_xlabel("fraction of conv/linear weights removed"); ax.set_ylabel("median latency, batch=1, 1 thread (ms)")
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("CPU latency: zeros are still multiplied")
    fig.tight_layout(); fig.savefig(FIG_DIR / "latency_vs_sparsity.png", dpi=150); plt.close(fig)

    # (4) storage formats vs sparsity
    fig, ax = plt.subplots(figsize=(7, 4.2))
    d = unstruct[unstruct.scope == "global"]
    for col in SIZE_COLS:
        ax.plot(d.sparsity, d[f"{col}_KB"], "o-", label=col)
    ax.set_xlabel("global sparsity"); ax.set_ylabel("file size on disk (KB)")
    ax.set_title("Dense vs sparse storage formats"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "storage_vs_sparsity.png", dpi=150); plt.close(fig)

    # (5) sparse kernel micro-benchmark
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, bs in zip(axes, (1, 64)):
        for fmt in bench.format.unique():
            dd = bench[(bench.batch == bs) & (bench.format == fmt)]
            ax.plot(dd.sparsity, dd.ms, "o-", label=fmt)
        ax.set_title(f"fc1 640x1152 matmul, batch={bs}"); ax.set_xlabel("sparsity"); ax.set_ylabel("ms (median)")
        ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIG_DIR / "sparse_matmul_benchmark.png", dpi=150); plt.close(fig)

    # (6) distillation learning curves
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for tag in sorted(t for t in hist.tag.unique() if str(t).startswith("student")):
        d = hist[hist.tag == tag]
        ax.plot(d.epoch, d.val_acc, "o-" if "10pct" not in tag else "s--", label=tag)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy"); ax.set_title("Student: plain vs distilled")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "distillation_curves.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()