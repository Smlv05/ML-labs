"""
Lab 5 - Task 4: verify the engine against PyTorch autograd and numerical gradients
on  ((a*b + c)**2).relu().sum().  Max absolute difference must be < 1e-6.
Also checks gradient accumulation (a node used more than once).
"""
import json
from pathlib import Path

import numpy as np
import torch

from engine import Value, vsum

TOL = 1e-6
RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

rng = np.random.default_rng(42)
N = 6
A = rng.normal(size=N)
B = rng.normal(size=N)
C = rng.normal(size=N)


def f_float(a, b, c):
    """Same function with plain floats (used for numerical gradients)."""
    return sum(max((ai * bi + ci) ** 2, 0.0) for ai, bi, ci in zip(a, b, c))


# 1) our engine
a = [Value(x) for x in A]
b = [Value(x) for x in B]
c = [Value(x) for x in C]
out = vsum(((ai * bi + ci) ** 2).relu() for ai, bi, ci in zip(a, b, c))
out.backward()
eng = {k: np.array([v.grad for v in vs]) for k, vs in (("a", a), ("b", b), ("c", c))}

# 2) PyTorch autograd (float64 so the comparison is fair)
ta = torch.tensor(A, dtype=torch.float64, requires_grad=True)
tb = torch.tensor(B, dtype=torch.float64, requires_grad=True)
tc = torch.tensor(C, dtype=torch.float64, requires_grad=True)
tout = ((ta * tb + tc) ** 2).relu().sum()
tout.backward()
pt = {"a": ta.grad.numpy(), "b": tb.grad.numpy(), "c": tc.grad.numpy()}

# 3) numerical gradients: central differences  (f(x+h) - f(x-h)) / 2h
h = 1e-6
num = {}
inputs = {"a": A, "b": B, "c": C}
for name, arr in inputs.items():
    g = np.zeros(N)
    for i in range(N):
        plus = {k: v.copy() for k, v in inputs.items()}
        minus = {k: v.copy() for k, v in inputs.items()}
        plus[name][i] += h
        minus[name][i] -= h
        g[i] = (f_float(**plus) - f_float(**minus)) / (2 * h)
    num[name] = g

forward_diff = abs(out.data - tout.item())
diff_pt = max(np.max(np.abs(eng[k] - pt[k])) for k in "abc")
diff_num = max(np.max(np.abs(eng[k] - num[k])) for k in "abc")

print(f"forward value: engine={out.data:.10f}  torch={tout.item():.10f}  diff={forward_diff:.2e}")
for k in "abc":
    print(f"d/d{k} engine : {np.round(eng[k], 6)}")
    print(f"d/d{k} torch  : {np.round(pt[k], 6)}")
    print(f"d/d{k} numeric: {np.round(num[k], 6)}")
print(f"\nmax |engine - torch|     = {diff_pt:.3e}  -> {'PASS' if diff_pt < TOL else 'FAIL'}")
print(f"max |engine - numerical| = {diff_num:.3e}  -> {'PASS' if diff_num < TOL else 'FAIL'}")

# 4) gradient accumulation test: x is used 3 times -> grads must be added up
x = Value(1.5)
y = x * x + x * 3.0 + x.relu()          # dy/dx = 2x + 3 + 1 = 7
y.backward()
tx = torch.tensor(1.5, dtype=torch.float64, requires_grad=True)
(tx * tx + tx * 3.0 + tx.relu()).backward()
acc_diff = abs(x.grad - tx.grad.item())
print(f"\naccumulation test: engine dy/dx={x.grad}  torch={tx.grad.item()}  "
      f"-> {'PASS' if acc_diff < TOL else 'FAIL'}")

report = {
    "expression": "((a*b + c)**2).relu().sum()",
    "n_elements": N,
    "forward_abs_diff": forward_diff,
    "max_abs_diff_engine_vs_torch": diff_pt,
    "max_abs_diff_engine_vs_numerical": diff_num,
    "numerical_step_h": h,
    "accumulation_test_abs_diff": acc_diff,
    "tolerance": TOL,
    "pass": bool(diff_pt < TOL and diff_num < TOL and acc_diff < TOL),
}
(RESULTS / "grad_check.json").write_text(json.dumps(report, indent=2))
print(f"\nsaved {RESULTS / 'grad_check.json'}")