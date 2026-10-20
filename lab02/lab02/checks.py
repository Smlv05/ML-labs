"""Steps 3-5: shape/softmax check, initial-loss sanity check, numerical gradient check."""
import numpy as np
from nn import load_data, init_params, forward, cross_entropy, backward

X_tr, X_te, Y_tr, Y_te = load_data()
print(f"train {X_tr.shape}, test {X_te.shape}")

# ---------------------------------------------------------------- steps 3 + 4
for W in (16, 64):
    p = init_params(W)
    y_hat, _ = forward(p, X_tr[:32])
    assert y_hat.shape == (32, 10)
    assert np.allclose(y_hat.sum(axis=1), 1.0, atol=1e-6)
    loss = cross_entropy(forward(p, X_tr)[0], Y_tr)
    print(f"W={W}: output shape {y_hat.shape}, rows sum to 1 OK, "
          f"initial loss {loss:.4f} (ln10 = {np.log(10):.4f})")

# ---------------------------------------------------------------- step 5
print("\nGradient check (W=16, batch of 32, eps=1e-5)")
rng = np.random.default_rng(1)
p = init_params(16)            # untrained network, N(0, 0.01) init
Xb, Yb = X_tr[:32], Y_tr[:32]

_, cache = forward(p, Xb)
grads = backward(p, cache, Yb)

def loss_fn():
    return cross_entropy(forward(p, Xb)[0], Yb)

eps = 1e-5
passed = 0
for i in range(10):
    # pick a weight whose gradient is non-zero (zeros come from ReLU units that
    # are inactive for the whole batch; they would "pass" without testing anything)
    while True:
        name = str(rng.choice(["W1", "W2", "W3"]))
        idx = tuple(int(rng.integers(0, s)) for s in p[name].shape)
        if grads[name][idx] != 0:
            break
    old = p[name][idx]
    p[name][idx] = old + eps; lp = loss_fn()
    p[name][idx] = old - eps; lm = loss_fn()
    p[name][idx] = old
    num = (lp - lm) / (2 * eps)
    ana = grads[name][idx]
    rel = abs(ana - num) / max(abs(ana) + abs(num), 1e-12)
    ok = rel < 1e-6
    passed += ok
    print(f"{name}{idx}: analytic {ana:+.6e}  numerical {num:+.6e}  "
          f"rel err {rel:.2e}  {'PASS' if ok else 'FAIL'}")
print(f"{passed}/10 passed -> {'GRADIENT CHECK PASSED' if passed >= 9 else 'FAILED'}")