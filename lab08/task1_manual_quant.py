"""Lab 8 Task 1: quantize weight tensors by hand (symmetric, zero-point 0) at 8/4/2 bits,
per-tensor vs per-channel scaling, and measure the reconstruction error (MSE).

Run from lab08/:  python task1_manual_quant.py
"""
import copy

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS, evaluate, get_data, load_model, save_json, write_table  # noqa: E402

TENSOR = "features.3.weight"   # main tensor: conv2, 64 filters x 32 x 3 x 3 = 18,432 weights
BITS = (8, 4, 2)


def quantize(w, bits, per_channel):
    """Symmetric quantization: scale = max|W| / qmax, zero-point = 0, q in [-qmax, qmax]."""
    qmax = 2 ** (bits - 1) - 1                    # 127 (8 bit), 7 (4 bit), 1 (2 bit)
    if per_channel:                               # one scale per output channel (dim 0)
        max_abs = w.abs().flatten(1).max(dim=1).values
        scale = (max_abs / qmax).clamp(min=1e-12).view(-1, *[1] * (w.dim() - 1))
    else:                                         # one scale for the whole tensor
        scale = (w.abs().max() / qmax).clamp(min=1e-12)
    zero_point = 0
    q = torch.clamp(torch.round(w / scale) + zero_point, -qmax, qmax).to(torch.int8)
    w_hat = (q.float() - zero_point) * scale      # dequantize (map back to float)
    return q, scale, w_hat


def error_stats(w, w_hat):
    err = w - w_hat
    mse = err.pow(2).mean().item()
    sqnr_db = 10 * np.log10(w.pow(2).sum().item() / max(err.pow(2).sum().item(), 1e-30))
    return mse, sqnr_db, err.abs().max().item()


def main():
    model = load_model("baseline")
    sd = model.state_dict()
    # conv / linear weights only (BatchNorm weights are 1-D and are folded into conv at inference)
    weights = {k: v for k, v in sd.items() if k.endswith("weight") and v.dim() > 1}

    # ---- 1) error per tensor / bit width / granularity
    rows = []
    for name, w in weights.items():
        for bits in BITS:
            for per_channel in (False, True):
                q, scale, w_hat = quantize(w, bits, per_channel)
                mse, sqnr, max_err = error_stats(w, w_hat)
                rows.append({
                    "tensor": name, "shape": "x".join(map(str, w.shape)), "bits": bits,
                    "granularity": "per-channel" if per_channel else "per-tensor",
                    "scale": float(scale.mean()), "levels_used": int(q.unique().numel()),
                    "mse": mse, "sqnr_db": sqnr, "max_abs_err": max_err,
                    "main": name == TENSOR,
                })
    df = write_table(rows, "task1_quant_error")

    # ---- 2) worked int8 example on the main tensor (for the report)
    w = weights[TENSOR]
    q, scale, w_hat = quantize(w, 8, per_channel=False)
    example = {
        "tensor": TENSOR, "shape": list(w.shape),
        "max_abs_w": float(w.abs().max()), "scale": float(scale), "zero_point": 0,
        "w_first8": [round(v, 6) for v in w.flatten()[:8].tolist()],
        "q_first8": q.flatten()[:8].tolist(),
        "w_hat_first8": [round(v, 6) for v in w_hat.flatten()[:8].tolist()],
    }

    # ---- 3) what the error does to accuracy: quantize ALL weight tensors, biases stay FP32
    test = get_data()["test"]
    acc_rows = [{"bits": 32, "granularity": "fp32", "test_acc": evaluate(model, test)}]
    for bits in BITS:
        for per_channel in (False, True):
            m = copy.deepcopy(model)
            new_sd = m.state_dict()
            for name in weights:
                new_sd[name] = quantize(weights[name], bits, per_channel)[2]
            m.load_state_dict(new_sd)
            acc_rows.append({"bits": bits,
                             "granularity": "per-channel" if per_channel else "per-tensor",
                             "test_acc": evaluate(m, test)})
    write_table(acc_rows, "task1_weight_quant_accuracy")
    save_json({"example_int8": example}, "task1_example")

    # ---- 4) figure: MSE vs bits + per-filter range of the main tensor
    main_df = df[df.tensor == TENSOR]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    width = 0.35
    x = np.arange(len(BITS))
    for i, gran in enumerate(["per-tensor", "per-channel"]):
        vals = [main_df[(main_df.bits == b) & (main_df.granularity == gran)].mse.item() for b in BITS]
        axes[0].bar(x + (i - 0.5) * width, vals, width, label=gran)
    axes[0].set_xticks(x, [f"{b}-bit" for b in BITS])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("MSE (log)")
    axes[0].set_title(f"Quantization error of {TENSOR}")
    axes[0].legend()

    per_filter = w.abs().flatten(1).max(dim=1).values.numpy()
    axes[1].bar(range(len(per_filter)), per_filter, color="tab:gray")
    axes[1].axhline(per_filter.max(), color="tab:red", ls="--", label="per-tensor max|W|")
    axes[1].set_xlabel("output channel (filter)")
    axes[1].set_ylabel("max |W|")
    axes[1].set_title("Range differs per filter -> per-channel scales")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "task1_quant_error.png", dpi=150)
    print("saved results/task1_*")


if __name__ == "__main__":
    main()