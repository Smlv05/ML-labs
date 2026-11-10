"""Lab 6 Task 2: profile ONE training epoch with torch.profiler.

Each step is split into 4 labelled regions (record_function); the profiler's CPU time
per label gives the share of data loading / forward / backward / optimizer step.
A plain perf_counter timer runs alongside as a cross-check.
  python profile_epoch.py            (add --trace to also save a Chrome trace)
"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.profiler import ProfilerActivity, profile, record_function  # noqa: E402

from pipeline import (DEVICE, RESULTS, build_loaders, build_model, build_optimizer,  # noqa: E402
                      epoch_seed, load_config, set_seed, setup_mlflow)

PHASES = ["data_loading", "forward", "backward", "optimizer_step"]


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--trace", action="store_true")
    p.add_argument("--set", nargs="*", default=[], dest="overrides")
    args = p.parse_args()
    cfg = load_config(args.config, args.overrides)
    cfg["run_name"] = "profile-one-epoch"

    set_seed(cfg["seed"])
    train_dl, _, _, shuffle_g = build_loaders(cfg)
    shuffle_g.manual_seed(epoch_seed(cfg, 1))
    model = build_model(cfg).to(DEVICE)
    opt = build_optimizer(cfg, model)
    loss_fn = nn.CrossEntropyLoss()
    model.train()

    wall = dict.fromkeys(PHASES, 0.0)
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if DEVICE == "cuda" else [])
    it = iter(train_dl)
    t_epoch = time.perf_counter()
    with profile(activities=activities, profile_memory=True) as prof:
        for _ in range(len(train_dl)):
            t = time.perf_counter()
            with record_function("data_loading"):
                x, y = next(it)
                x, y = x.to(DEVICE), y.to(DEVICE)
            sync(); wall["data_loading"] += time.perf_counter() - t; t = time.perf_counter()
            with record_function("forward"):
                loss = loss_fn(model(x), y)
            sync(); wall["forward"] += time.perf_counter() - t; t = time.perf_counter()
            with record_function("backward"):
                loss.backward()
            sync(); wall["backward"] += time.perf_counter() - t; t = time.perf_counter()
            with record_function("optimizer_step"):
                opt.step()
                opt.zero_grad(set_to_none=True)
            sync(); wall["optimizer_step"] += time.perf_counter() - t
    epoch_s = time.perf_counter() - t_epoch

    prof_ms = {e.key: e.cpu_time_total / 1000 for e in prof.key_averages() if e.key in PHASES}
    tot_prof, tot_wall = sum(prof_ms.values()), sum(wall.values())
    rows = [{"phase": ph,
             "profiler_ms": round(prof_ms.get(ph, 0.0), 1),
             "profiler_share_%": round(100 * prof_ms.get(ph, 0.0) / tot_prof, 1),
             "wall_ms": round(1000 * wall[ph], 1),
             "wall_share_%": round(100 * wall[ph] / tot_wall, 1)} for ph in PHASES]

    RESULTS.mkdir(exist_ok=True)
    summary = {"epoch_s": epoch_s, "steps": len(train_dl), "batch_size": cfg["data"]["batch_size"],
               "num_workers": cfg["data"]["num_workers"], "device": DEVICE, "breakdown": rows}
    (RESULTS / "profile_breakdown.json").write_text(json.dumps(summary, indent=2))
    table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=30)
    (RESULTS / "profile_table.txt").write_text(table)

    md = ["| Phase | Profiler CPU time (ms) | Share | Wall time (ms) | Share |", "|---|---|---|---|---|"]
    md += [f"| {r['phase']} | {r['profiler_ms']} | {r['profiler_share_%']}% | {r['wall_ms']} | "
           f"{r['wall_share_%']}% |" for r in rows]
    (RESULTS / "profile_breakdown.md").write_text("\n".join(md) + f"\n\nOne epoch = {epoch_s:.1f} s, "
                                                  f"{len(train_dl)} steps, device={DEVICE}\n")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(PHASES, [r["wall_share_%"] for r in rows], color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    for i, r in enumerate(rows):
        ax.text(i, r["wall_share_%"] + 1, f"{r['wall_share_%']}%", ha="center")
    ax.set_ylabel("% of one epoch")
    ax.set_title(f"Where one training epoch goes (CNN, bs={cfg['data']['batch_size']}, {DEVICE})")
    ax.set_ylim(0, max(r["wall_share_%"] for r in rows) + 10)
    fig.tight_layout()
    fig.savefig(RESULTS / "profile_breakdown.png", dpi=150)

    if args.trace:
        prof.export_chrome_trace(str(RESULTS / "profile_trace.json"))

    setup_mlflow(cfg)
    with mlflow.start_run(run_name=cfg["run_name"]):
        mlflow.log_params({"batch_size": cfg["data"]["batch_size"], "num_workers": cfg["data"]["num_workers"]})
        mlflow.log_metrics({f"share_{r['phase']}": r["wall_share_%"] for r in rows} | {"epoch_s": epoch_s})
        for f in ["profile_breakdown.md", "profile_table.txt", "profile_breakdown.png"]:
            mlflow.log_artifact(str(RESULTS / f))

    print((RESULTS / "profile_breakdown.md").read_text())
    print(table)


if __name__ == "__main__":
    main()
