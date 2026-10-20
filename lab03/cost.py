import copy
import json
import statistics
import time
from pathlib import Path

import torch
from thop import profile
from torch.utils.flop_counter import FlopCounterMode
from torchinfo import summary

from models import MODELS

RESULTS = Path(__file__).resolve().parent / "results"
NUM_THREADS = 4
INPUT = (1, 1, 28, 28)


def count_params(model):
    return summary(model, input_size=INPUT, verbose=0).total_params


def hand_flops(name):
    """Per-image FLOPs = 2 x MACs (Reddi Ch. 4). Bias, ReLU, pooling, softmax, LayerNorm ignored."""
    if name == "mlp":
        return {"fc1": 2 * 784 * 256, "fc2": 2 * 256 * 128, "fc3": 2 * 128 * 10}
    if name == "cnn":
        return {
            "conv1": 2 * 3 * 3 * 1 * 16 * 28 * 28,
            "conv2": 2 * 3 * 3 * 16 * 32 * 14 * 14,
            "fc": 2 * (32 * 7 * 7) * 10,
        }
    s, d, ff = 16, 64, 128
    per_layer = {
        "qkv": 2 * s * d * 3 * d,
        "scores": 2 * s * s * d,
        "attn_v": 2 * s * s * d,
        "out_proj": 2 * s * d * d,
        "ffn": 2 * s * d * ff * 2,
    }
    layers = {"patch_embed": 2 * s * 49 * d}
    for i in (1, 2):
        layers.update({f"L{i}.{k}": v for k, v in per_layer.items()})
    layers["head"] = 2 * d * 10
    return layers


def thop_flops(model):
    m = copy.deepcopy(model).eval()  # deepcopy: thop adds buffers to modules
    macs, _ = profile(m, inputs=(torch.randn(INPUT),), verbose=False)
    return int(2 * macs)


def flopcounter_flops(model):
    # train() disables the fused Transformer fast path so every matmul is dispatched and counted.
    # Dropout does not change FLOPs.
    m = copy.deepcopy(model).train()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(torch.randn(INPUT))
    return int(fc.get_total_flops())


@torch.inference_mode()
def latency_ms(model, batch, runs, warmup=10, device="cpu"):
    model = model.eval().to(device)
    x = torch.randn(batch, 1, 28, 28, device=device)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(runs):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main():
    torch.set_num_threads(NUM_THREADS)
    RESULTS.mkdir(exist_ok=True)
    out = {}
    for name, cls in MODELS.items():
        torch.manual_seed(0)
        model = cls()
        hand = hand_flops(name)
        out[name] = {
            "params": count_params(model),
            "hand_flops_layers": hand,
            "hand_flops": sum(hand.values()),
            "thop_flops": thop_flops(model),
            "flopcounter_flops": flopcounter_flops(model),
            "latency_b1_ms": latency_ms(model, 1, 100),
            "latency_b128_ms": latency_ms(model, 128, 20),
        }
        r = out[name]
        print(f"{name:10s} params={r['params']:>8,}  hand={r['hand_flops']:>10,}  "
              f"thop={r['thop_flops']:>10,}  flopcounter={r['flopcounter_flops']:>10,}  "
              f"b1={r['latency_b1_ms']:.3f} ms  b128={r['latency_b128_ms']:.3f} ms")

    meta = {"device": "cpu", "num_threads": NUM_THREADS, "torch": torch.__version__,
            "protocol": "warmup 10, median of 100 (b=1) / 20 (b=128), inference_mode"}
    (RESULTS / "cost.json").write_text(json.dumps({"meta": meta, "models": out}, indent=2))


if __name__ == "__main__":
    main()