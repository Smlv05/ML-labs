# Lab 5 – Frameworks and automatic differentiation

Environment: Python 3.11.8, PyTorch 2.3.1+cpu, Windows, 6 threads.

## How to run

```
cd lab05
python engine.py           # tiny demo: graph + gradients of ((a*b + c)**2).relu()
python check_grads.py      # Task 4 -> results/grad_check.json
python train_moons.py      # Task 5 -> results/moons_training.json, moons_training.png
python torch_execution.py  # Task 6 -> results/execution_modes.json, execution_modes.png
```

## 1. Autograd engine (Tasks 1–4)

- `Value` stores `data`, `grad`, `_parents`, `_op` and `_backward` (the local derivative rule).
- `+`, `*`, `**k` and `relu` (plus `sigmoid` and `log` for BCE) each create a new node, so the graph is built during the forward pass (define-by-run, like PyTorch eager).
- `backward()`: topological sort → output grad = 1 → visit nodes in reverse order → every rule uses `+=`, so a node used several times accumulates its gradients.
- Check on `((a*b + c)**2).relu().sum()` with 6-element a, b, c (float64):
  - forward value: engine = torch = 10.8012994576
  - max |engine − PyTorch| = **0.0**
  - max |engine − numerical (central differences, h = 1e-6)| = **1.05e-9**
  - both are under the 1e-6 limit → **PASS**
- Accumulation test: `x*x + 3x + relu(x)` at x = 1.5 gives dy/dx = 7.0, the same as PyTorch → **PASS**.

## 2. make_moons, 2 → 8 → 1 (Task 5)

ReLU hidden layer, sigmoid output, BCE loss. 100 samples, same initial weights for both, full-batch GD, lr 0.5, 200 epochs, float64.

|            | median of 5 runs | final loss (run 4) | accuracy |
| ---------- | ---------------- | ------------------ | -------- |
| our engine | 14.95 s          | 0.164108           | 0.92     |
| PyTorch    | 0.082 s          | 0.164108           | 0.92     |

The loss curves are identical (max gap ≈ 3e-16 in every run), so the engine computes the same gradients as PyTorch.
PyTorch is about **183× faster**. Our engine creates one Python object per scalar operation (about 9 700 graph nodes per epoch), while PyTorch runs whole tensor operations in optimised C++ kernels.
PyTorch's first run took 1.4 s because of one-time start-up costs (library loading and kernel warm-up); using the median removes that outlier.

## 3. Eager vs compiled/traced (Task 6)

Mode used: **torch.jit.trace**. torch.compile failed (`InvalidCxxCompiler: No working C++ compiler found`) because PyTorch 2.3.1 on Windows has no C++ compiler for the inductor backend, so the script fell back to tracing, as the manual allows.

Median time per call after 10 warm-up calls:
| batch | eager (µs) | traced (µs) | speed-up | first call (s) |
|---|---|---|---|---|
| 1 | 47.7 | 28.3 | 1.69× | 0.07 |
| 32 | 68.4 | 55.7 | 1.23× | 0.02 |
| 1024 | 311.0 | 279.3 | 1.11× | 0.02 |

The traced model's outputs matched eager (`allclose` = True) at every batch size.

Graph: torch.fx shows x → fc1 → relu → fc2 → relu → fc3 → output. The autograd graph is SumBackward0 → AddmmBackward0 → ReluBackward0 → …, with AccumulateGrad leaves for the 6 parameters (the weights go through TBackward0).

Tensors: `x` is (32, 64) float32, stride (64, 1), contiguous. `x.t()` is a view with stride (1, 64): it shares memory with x and makes no copy, but it is **not contiguous**. `x[:, ::2]` has stride (64, 2), also not contiguous. `.contiguous()` makes a new copy with stride (32, 1).

Non-contiguous cost (4096×4096 float32):
| operation | contiguous (µs) | non-contiguous (µs) | ratio |
|---|---|---|---|
| row sum `.sum(dim=1)` | 3048 | 10455 | 3.43× slower |
| strided slice `.sum()` | 1535 | 2020 | 1.32× slower |
| `reshape(-1)` | 2.7 | 24806 | must copy (a `view` would fail) |
| elementwise `t*2+1` | 11289 | 11200 | 0.99× (no change) |
| matmul `t @ M` (1024) | 4437 | 4906 | 1.11× (0.78× in an earlier run) |

Making the transposed tensor contiguous costs about 25 ms, about the same as the reshape. Reductions along the "wrong" axis, and anything that needs a flat memory layout, pay for non-contiguity. Elementwise ops barely change, because PyTorch iterates in memory order. Matmul stays about the same and varies between runs, because BLAS takes the transpose as a flag instead of copying.

## When does compilation help? (short note)

Compilation (torch.compile or tracing) records the model as a graph once, then removes Python and dispatcher overhead from every call. torch.compile can also fuse operations into fewer kernels. In my measurements tracing gave **1.69× at batch 1, 1.23× at batch 32 and only 1.11× at batch 1024**.

It helps most when:

1. **Overhead dominates**: small batches and many small operations. At batch 1 the actual maths is tiny, so removing ~19 µs of Python overhead per call matters. At batch 1024 most of the time is spent inside large matrix multiplications that are already optimised C++/BLAS, so the relative gain is small.
2. **The model is called many times with the same shapes**: the one-off cost (0.02–0.07 s here) is only paid back after hundreds to a few thousand calls. A one-off script, or inputs whose shapes keep changing (which forces recompilation), gains nothing.
3. **There are chains of element-wise ops** that a compiler such as torch.compile/inductor can fuse into one kernel, especially on GPUs.

It helps little when the model has data-dependent Python control flow (tracing records only one path), while debugging (eager is easier to inspect), or when the backend is missing. Here, torch.compile could not run at all on Windows without a C++ compiler.
