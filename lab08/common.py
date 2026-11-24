"""Lab 8 shared helpers. Models and data come from Lab 7 (BigCNN baseline, pruned, distilled)."""
import copy
import importlib.util
import io
import json
import platform
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

LAB_DIR = Path(__file__).resolve().parent
ROOT = LAB_DIR.parent
LAB07 = ROOT / "lab07"
RESULTS = LAB_DIR / "results"
SEED = 42

# lab07/common.py has the same file name as this one -> load it under another module name
_spec = importlib.util.spec_from_file_location("lab07_common", LAB07 / "common.py")
lab07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lab07)
BigCNN, SmallCNN = lab07.BigCNN, lab07.SmallCNN

# Lab 7 checkpoints (dense state_dicts written by storage_sizes(); pruned zeros are kept)
CKPTS = {
    "baseline": LAB07 / "checkpoints" / "baseline.pt",
    "pruned_unstructured": LAB07 / "checkpoints" / "size_formats" / "global_unstructured_90_dense.pt",
    "pruned_structured": LAB07 / "checkpoints" / "size_formats" / "structured_filters_50_dense.pt",
    "distilled": LAB07 / "checkpoints" / "size_formats" / "student_distilled_dense.pt",
}


def check_lab07_full_run():
    cfg = LAB07 / "results" / "run_config.json"
    if cfg.exists() and json.loads(cfg.read_text()).get("quick"):
        print("WARNING: lab07/results come from a --quick run. Run `python lab07/run_lab07.py` "
              "(full) first, otherwise pruned/distilled checkpoints are 1-epoch models.")


# ---------------------------------------------------------------- backend
def pick_backend():
    """fbgemm for Intel/AMD (x86), qnnpack for Apple Silicon / ARM."""
    arm = platform.machine().lower() in ("arm64", "aarch64")
    backend = "qnnpack" if arm else "fbgemm"
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError(f"{backend} not in {torch.backends.quantized.supported_engines}")
    torch.backends.quantized.engine = backend
    return backend


def system_info():
    return {"torch": torch.__version__, "python": platform.python_version(),
            "machine": platform.machine(), "processor": platform.processor(),
            "quant_engine": torch.backends.quantized.engine, "latency_threads": 1}


# ---------------------------------------------------------------- data (exactly Lab 7's split)
def get_data():
    """{'train': (x, y), 'val': (x, y), 'test': (x, y)} as normalised tensors (Lab 7 loader)."""
    return lab07.load_fashion_mnist(seed=SEED)


def batches(x, y=None, batch_size=1000):
    for i in range(0, len(x), batch_size):
        yield (x[i:i + batch_size], None if y is None else y[i:i + batch_size])


def random_train_images(data, n=500, seed=SEED):
    """n random images from the TRAINING split (never val/test) for calibration."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(data["train"][0]), generator=g)[:n]
    return data["train"][0][idx]


# ---------------------------------------------------------------- models
def bigcnn_from_state(sd):
    """Rebuild BigCNN with the channel counts stored in the state_dict (structured-pruned models)."""
    ch = tuple(sd[f"features.{i}.weight"].shape[0] for i in (0, 3, 7, 11))
    return BigCNN(ch=ch, hidden=sd["classifier.2.weight"].shape[0])


def load_model(key):
    sd = torch.load(CKPTS[key], map_location="cpu", weights_only=True)
    model = SmallCNN() if key == "distilled" else bigcnn_from_state(sd)
    model.load_state_dict(sd)
    return model.eval()


@torch.no_grad()
def evaluate(model, split, batch_size=1000):
    model.eval()
    x, y = split
    correct = sum((model(xb).argmax(1) == yb).sum().item() for xb, yb in batches(x, y, batch_size))
    return correct / len(x)


def param_stats(model):
    total = sum(p.numel() for p in model.parameters())
    nonzero = sum(int((p != 0).sum()) for p in model.parameters())
    return total, nonzero


def size_kb(model):
    """Serialized state_dict size: raw and zlib-compressed (the latter shows sparsity gains)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    raw = buf.getvalue()
    return len(raw) / 1024, len(zlib.compress(raw, 9)) / 1024


@torch.no_grad()
def latency_ms(fn, x, warmup=50, runs=500):
    """Batch-1 latency on one CPU thread: p50 / p95 / mean in milliseconds."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(warmup):
            fn(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn(x)
            times.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(old)
    t = np.array(times)
    return {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "mean_ms": float(t.mean())}

@torch.no_grad()
def latency_many(cands, warmup=30, runs=100, rounds=10):
    """Interleaved batch-1 latency on one thread. cands = {name: (fn, x)}.
    Every round times each candidate in turn (order rotated per round), so changes in CPU
    clock / P-core vs E-core scheduling hit all variants equally. Returns {name: stats}."""
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    names = list(cands)
    times = {n: [] for n in names}
    try:
        for fn, x in cands.values():
            for _ in range(warmup):
                fn(x)
        for r in range(rounds):
            for n in names[r % len(names):] + names[:r % len(names)]:
                fn, x = cands[n]
                for _ in range(runs):
                    t0 = time.perf_counter()
                    fn(x)
                    times[n].append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(old)
    out = {}
    for n, t in times.items():
        t = np.array(t)
        out[n] = {"p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
                  "mean_ms": float(t.mean())}
    return out

# ---------------------------------------------------------------- quantization
def dynamic_quant(model):
    """PTQ dynamic: Linear weights -> int8 ahead of time, activation ranges computed per input."""
    return torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).eval(), {nn.Linear}, dtype=torch.qint8)


def static_quant(model, calib_x, backend, batch_size=50):
    """PTQ static (FX graph mode): fuse conv+bn+relu, observe activations on calib images, convert."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    m = copy.deepcopy(model).eval()
    prepared = prepare_fx(m, get_default_qconfig_mapping(backend), (calib_x[:1],))
    with torch.no_grad():
        for xb, _ in batches(calib_x, batch_size=batch_size):
            prepared(xb)
    return convert_fx(prepared)


def quant_scales(qmodel):
    """Output scale of every quantized module (shows what calibration chose)."""
    out = {}
    for name, mod in qmodel.named_modules():
        s = getattr(mod, "scale", None)
        if name and s is not None and not isinstance(s, nn.Module):
            try:
                out[name] = round(float(s), 6)
            except (TypeError, ValueError):
                pass
    return out


# ---------------------------------------------------------------- output
def _fmt(v):
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def write_table(rows, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    (RESULTS / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return df


def save_json(obj, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=2), encoding="utf-8")