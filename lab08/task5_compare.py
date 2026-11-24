"""Lab 8 Task 5: one table for baseline, pruned, distilled, dynamic quant, static quant,
pruned+quant, distilled+quant (+ ONNX Runtime row), plots, and a deployment choice
for Cloud / Edge / Mobile / TinyML. All float models come from Lab 7.

Run from lab08/ (after task4_onnx.py):  python task5_compare.py
"""
import json

import matplotlib
import onnxruntime as ort

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (LAB_DIR, RESULTS, check_lab07_full_run, dynamic_quant, evaluate, get_data,  # noqa: E402
                    latency_many, load_model, param_stats, pick_backend, random_train_images,
                    save_json, size_kb, static_quant, system_info, write_table)

MAX_DROP_PP = {"Edge": 1.0, "Mobile": 1.5, "TinyML": 2.0}


def main():
    check_lab07_full_run()
    backend = pick_backend()
    data = get_data()
    test = data["test"]
    calib = random_train_images(data, 500)
    x1 = test[0][:1]

    baseline = load_model("baseline")
    pruned_u = load_model("pruned_unstructured")      # global magnitude 90 %, fine-tuned (Lab 7)
    pruned_s = load_model("pruned_structured")        # 50 % of filters removed, fine-tuned (Lab 7)
    distilled = load_model("distilled")               # SmallCNN student, KD T=4 (Lab 7)

    # (name, model, float source for param counts, precision)
    variants = [
        ("baseline", baseline, baseline, "fp32"),
        ("pruned unstructured 90%", pruned_u, pruned_u, "fp32"),
        ("pruned structured 50% filters", pruned_s, pruned_s, "fp32"),
        ("distilled", distilled, distilled, "fp32"),
        ("dynamic quant", dynamic_quant(baseline), baseline, "int8 linear"),
        ("static quant", static_quant(baseline, calib, backend), baseline, "int8"),
        ("pruned unstructured + static quant", static_quant(pruned_u, calib, backend), pruned_u, "int8"),
        ("pruned structured + static quant", static_quant(pruned_s, calib, backend), pruned_s, "int8"),
        ("distilled + static quant", static_quant(distilled, calib, backend), distilled, "int8"),
    ]

    cands = {name: (model, x1) for name, model, _, _ in variants}
    onnx_path = LAB_DIR / "export" / "baseline.onnx"
    onnx_json = RESULTS / "task4_onnx.json"
    if onnx_path.exists() and onnx_json.exists():        # ORT timed in the same interleaved loop
        so = ort.SessionOptions()
        so.intra_op_num_threads = so.inter_op_num_threads = 1
        sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
        cands["baseline ONNX Runtime"] = (lambda a: sess.run(None, {"input": a}), x1.numpy())
    print("measuring latency (interleaved, takes a few minutes) ...")
    lat = latency_many(cands)
    rows = []
    for name, model, src, precision in variants:
        total, nonzero = param_stats(src)
        raw, zipped = size_kb(model)
        rows.append({"variant": name, "precision": precision, "params": total,
                     "nonzero_params": nonzero, "test_acc": evaluate(model, test),
                     "size_kb": raw, "size_zip_kb": zipped, **lat[name]})

    if "baseline ONNX Runtime" in lat:
        ort_row = json.loads(onnx_json.read_text())["rows"][1]   # accuracy/size from Task 4
        total, nonzero = param_stats(baseline)
        rows.append({"variant": "baseline ONNX Runtime", "precision": "fp32", "params": total,
                     "nonzero_params": nonzero, "test_acc": ort_row["test_acc"],
                     "size_kb": ort_row["size_kb"], "size_zip_kb": float("nan"),
                     **lat["baseline ONNX Runtime"]})

    base = rows[0]
    for r in rows:
        r["delta_acc_pp"] = 100 * (r["test_acc"] - base["test_acc"])
        r["size_ratio"] = base["size_kb"] / r["size_kb"]
        r["speedup_p50"] = base["p50_ms"] / r["p50_ms"]
    df = write_table(rows, "task5_variants")

    # ---- plots: accuracy vs latency and vs size
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, col, label in [(axes[0], "p50_ms", "latency p50, batch 1, 1 thread (ms)"),
                           (axes[1], "size_kb", "model size (KB, log)")]:
        for _, r in df.iterrows():
            marker = "s" if "int8" in r.precision else "o"
            ax.scatter(r[col], 100 * r.test_acc, s=60, marker=marker)
            ax.annotate(r.variant, (r[col], 100 * r.test_acc), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(label)
        ax.set_ylabel("test accuracy (%)")
        ax.grid(alpha=0.3)
    axes[1].set_xscale("log")
    axes[0].set_title("Accuracy vs latency (square = int8)")
    axes[1].set_title("Accuracy vs size")
    fig.tight_layout()
    fig.savefig(RESULTS / "task5_tradeoffs.png", dpi=150)

    # ---- deployment choice (simple, explicit rules; discuss them in report.md)
    pt = df[df.variant != "baseline ONNX Runtime"]
    int8 = pt[pt.precision == "int8"]

    def pick(cands_df, max_drop, sort_cols):
        ok = cands_df[cands_df.delta_acc_pp >= -max_drop]
        if ok.empty:                       # nothing meets the budget -> most accurate candidate
            ok = cands_df.sort_values("test_acc", ascending=False).head(1)
        return ok.sort_values(sort_cols).iloc[0]

    picks = {
        "Cloud": df.sort_values(["test_acc", "p50_ms"], ascending=[False, True]).iloc[0],
        "Edge": pick(pt, MAX_DROP_PP["Edge"], ["p95_ms"]),
        "Mobile": pick(int8, MAX_DROP_PP["Mobile"], ["size_kb", "p95_ms"]),
        "TinyML": pick(int8, MAX_DROP_PP["TinyML"], ["size_kb"]),
    }
    rules = {
        "Cloud": "highest accuracy; latency/size are not binding (fast CPU/GPU, lots of memory)",
        "Edge": f"lowest p95 latency with accuracy drop <= {MAX_DROP_PP['Edge']} pp",
        "Mobile": f"int8 only, smallest size then p95, drop <= {MAX_DROP_PP['Mobile']} pp",
        "TinyML": f"full-integer int8 only (no FPU/PyTorch on MCU), smallest size, "
                  f"drop <= {MAX_DROP_PP['TinyML']} pp",
    }
    lines = ["| platform | chosen variant | test acc | size KB | p95 ms | rule |",
             "|---|---|---|---|---|---|"]
    choice = {}
    for plat, r in picks.items():
        choice[plat] = r.variant
        lines.append(f"| {plat} | {r.variant} | {r.test_acc:.4f} | {r.size_kb:.1f} | "
                     f"{r.p95_ms:.3f} | {rules[plat]} |")
    (RESULTS / "task5_deployment.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    save_json({"backend": backend, "system": system_info(), "deployment": choice,
               "rules": rules}, "task5_deployment")


if __name__ == "__main__":
    main()