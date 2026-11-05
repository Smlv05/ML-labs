"""
Lab 5 - Task 6:
  (a) inspect the graph of a small PyTorch model (torch.fx graph + autograd graph)
  (b) eager vs torch.compile (fallback: torch.jit.trace) after warm-up
  (c) inspect tensor shape / dtype / stride / contiguity
  (d) measure how non-contiguous views change operation speed
"""
import json
import platform
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
torch.manual_seed(0)


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def bench(fn, *args, warmup=10, iters=100, repeats=7):
    """Median time per call in microseconds (after warm-up)."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn(*args)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            for _ in range(iters):
                fn(*args)
            times.append((time.perf_counter() - t0) / iters)
    return statistics.median(times) * 1e6


# ------------------------------------------------------------------ (a) graphs
def inspect_graphs(model):
    print("=" * 70, "\n(a) torch.fx graph of the forward pass\n" + "=" * 70)
    gm = torch.fx.symbolic_trace(model)
    for node in gm.graph.nodes:
        print(f"  {node.op:<14} {node.name:<10} target={node.target}  args={node.args}")

    print("\n(a) autograd (backward) graph built during the forward pass")
    x = torch.randn(4, 64)
    loss = model(x).sum()

    def walk(fn, depth=0):
        if fn is None:
            return
        extra = ""
        if hasattr(fn, "variable"):  # AccumulateGrad -> a leaf parameter
            extra = f"  (param shape {tuple(fn.variable.shape)})"
        print("  " + "  " * depth + type(fn).__name__ + extra)
        for nxt, _ in fn.next_functions:
            walk(nxt, depth + 1)

    walk(loss.grad_fn)


# ------------------------------------------------------- (b) eager vs compiled
def make_fast_version(model, example):
    """Try torch.compile first; if it is not supported (e.g. no C++ compiler on
    Windows), fall back to torch.jit.trace. Returns (fn, name, first_call_seconds)."""
    try:
        compiled = torch.compile(model, dynamic=False)
        t0 = time.perf_counter()
        with torch.inference_mode():
            compiled(example)  # compilation actually happens on the first call
        return compiled, "torch.compile", time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001
        print(f"  torch.compile not available here ({type(e).__name__}: {str(e)[:150]})")
        try:
            torch._dynamo.reset()
        except Exception:  # noqa: BLE001
            pass
    return make_traced(model, example)


def make_traced(model, example):
    t0 = time.perf_counter()
    with torch.no_grad():
        traced = torch.jit.trace(model, example)
        traced(example)
    return traced, "torch.jit.trace", time.perf_counter() - t0


def compare_execution_modes(model, batch_sizes=(1, 32, 1024)):
    print("\n" + "=" * 70, "\n(b) eager vs compiled/traced (median per call, after warm-up)\n" + "=" * 70)
    rows = []
    for bs in batch_sizes:
        x = torch.randn(bs, 64)
        with torch.inference_mode():
            ref = model(x)
        row = {"batch_size": bs, "eager_us": bench(model, x)}

        fast, mode, first_call = make_fast_version(model, x)
        with torch.inference_mode():
            same = torch.allclose(fast(x), ref, atol=1e-5)
        row.update({"compiled_mode": mode, "compiled_us": bench(fast, x),
                    "compile_first_call_s": first_call, "outputs_match": bool(same)})

        if mode != "torch.jit.trace":  # also show tracing for comparison
            try:
                traced, _, t_first = make_traced(model, x)
                row.update({"traced_us": bench(traced, x), "trace_first_call_s": t_first})
            except Exception as e:  # noqa: BLE001
                print(f"  torch.jit.trace failed: {e}")
        row["speedup_compiled"] = row["eager_us"] / row["compiled_us"]
        # how many calls until the one-off compile cost is paid back
        saved = row["eager_us"] - row["compiled_us"]
        row["break_even_calls"] = (first_call * 1e6 / saved) if saved > 0 else None
        rows.append(row)
        print(f"  batch={bs:<5} eager={row['eager_us']:9.1f}us  {mode}={row['compiled_us']:9.1f}us"
              f"  speedup={row['speedup_compiled']:.2f}x  first call={first_call:.2f}s"
              f"  outputs match={same}")
        if "traced_us" in row:
            print(f"  {'':11}jit.trace={row['traced_us']:9.1f}us")
    return rows


# ------------------------------------------------------- (c) tensor inspection
def describe(name, t):
    info = {"shape": tuple(t.shape), "dtype": str(t.dtype), "stride": t.stride(),
            "contiguous": t.is_contiguous(), "storage_offset": t.storage_offset(),
            "storage_ptr": t.untyped_storage().data_ptr()}
    print(f"  {name:<22} shape={info['shape']!s:<12} dtype={info['dtype']:<14} "
          f"stride={info['stride']!s:<10} contiguous={info['contiguous']}")
    return info


def inspect_tensors(model):
    print("\n" + "=" * 70, "\n(c) tensor shape / dtype / contiguity\n" + "=" * 70)
    x = torch.randn(32, 64)
    info = {
        "x": describe("x", x),
        "fc1.weight": describe("fc1.weight", model.fc1.weight),
        "x.t() (view)": describe("x.t()  (view)", x.t()),
        "x[:, ::2] (view)": describe("x[:, ::2]  (view)", x[:, ::2]),
        "x.t().contiguous()": describe("x.t().contiguous()", x.t().contiguous()),
        "x.double()": describe("x.double()", x.double()),
    }
    print(f"  x.t() shares memory with x: "
          f"{x.t().untyped_storage().data_ptr() == x.untyped_storage().data_ptr()}")
    print(f"  x.t().contiguous() is a new copy: "
          f"{x.t().contiguous().untyped_storage().data_ptr() != x.untyped_storage().data_ptr()}")
    return {k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()
                if kk != "storage_ptr"} for k, v in info.items()}


# --------------------------------------------- (d) cost of non-contiguous views
def contiguity_cost(n=4096):
    print("\n" + "=" * 70, f"\n(d) contiguous vs non-contiguous ({n}x{n} float32)\n" + "=" * 70)
    A = torch.randn(n, n)
    At = A.t()                      # non-contiguous view (no copy)
    At_c = At.contiguous()          # same values, contiguous copy
    S = A[:, ::2]                   # strided slice (non-contiguous)
    S_c = S.contiguous()
    M = torch.randn(1024, 1024)
    kw = dict(warmup=3, iters=10, repeats=5)

    tests = {
        "row sum  .sum(dim=1)": (lambda t: t.sum(dim=1), At_c, At),
        "elementwise  t*2+1": (lambda t: t * 2 + 1, At_c, At),
        "strided slice  .sum()": (lambda t: t.sum(), S_c, S),
        "flatten  .view/.reshape(-1)": (lambda t: t.reshape(-1), At_c, At),
        "matmul 1024 (t @ M)": (lambda t: t @ M, M.clone(), M.t()),
    }
    rows = []
    for name, (fn, contig, noncontig) in tests.items():
        tc, tn = bench(fn, contig, **kw), bench(fn, noncontig, **kw)
        rows.append({"op": name, "contiguous_us": tc, "non_contiguous_us": tn,
                     "slowdown": tn / tc})
        print(f"  {name:<28} contiguous={tc:10.1f}us  non-contiguous={tn:10.1f}us  "
              f"-> {tn / tc:5.2f}x")
    copy_us = bench(lambda t: t.contiguous(), At, **kw)
    print(f"  cost of making it contiguous: At.contiguous() = {copy_us:.1f}us")
    print("  note: reshape(-1) on a non-contiguous tensor must COPY; .view(-1) would raise an error")
    return rows, copy_us


def plot(exec_rows, contig_rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [str(r["batch_size"]) for r in exec_rows]
    xs = range(len(labels))
    ax1.bar([i - 0.2 for i in xs], [r["eager_us"] for r in exec_rows], 0.4, label="eager")
    ax1.bar([i + 0.2 for i in xs], [r["compiled_us"] for r in exec_rows], 0.4,
            label=exec_rows[0]["compiled_mode"])
    ax1.set(xticks=list(xs), xticklabels=labels, xlabel="batch size", ylabel="µs per call (log)",
            yscale="log", title="Eager vs compiled (after warm-up)")
    ax1.legend()
    names = [r["op"] for r in contig_rows]
    ax2.barh(names, [r["slowdown"] for r in contig_rows])
    ax2.axvline(1, color="k", lw=1)
    ax2.set(xlabel="non-contiguous time / contiguous time", title="Cost of non-contiguous views")
    fig.tight_layout()
    fig.savefig(RESULTS / "execution_modes.png", dpi=120)


if __name__ == "__main__":
    print(f"torch {torch.__version__} | python {platform.python_version()} | "
          f"{platform.system()} | threads={torch.get_num_threads()}")
    model = SmallNet().eval()
    inspect_graphs(model)
    exec_rows = compare_execution_modes(model)
    tensor_info = inspect_tensors(model)
    contig_rows, copy_us = contiguity_cost()
    plot(exec_rows, contig_rows)

    report = {"env": {"torch": torch.__version__, "python": platform.python_version(),
                      "os": platform.system(), "threads": torch.get_num_threads()},
              "execution_modes": exec_rows, "tensors": tensor_info,
              "contiguity": contig_rows, "transpose_contiguous_copy_us": copy_us}
    (RESULTS / "execution_modes.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {RESULTS / 'execution_modes.json'} and execution_modes.png")