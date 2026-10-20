"""Step 7: time forward and backward passes separately, median of 100 runs."""
import time
import numpy as np
from nn import load_data, init_params, forward, backward, relu

X_tr, X_te, Y_tr, Y_te = load_data()
RUNS = 100


def median_us(fn):
    fn()                                               # warm-up, not timed
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1e6                      # microseconds


def matmul_flops(B, W):
    fwd = 2 * B * (64 * W + W * W + W * 10)            # X@W1, a1@W2, a2@W3
    bwd = 2 * B * (64 * W + 2 * W * W + 2 * W * 10)    # dW1, dW2, da1, dW3, da2
    return fwd, bwd


print(f"{'W':<4}{'batch':>6}{'fwd us':>9}{'bwd us':>9}{'bwd/fwd':>9}"
      f"{'fwd mm us':>11}{'bwd mm us':>11}{'mm share fwd/bwd':>18}{'FLOP ratio':>12}")
for W in (16, 64):
    p = init_params(W)
    for B in (32, len(X_tr)):
        Xb, Yb = X_tr[:B], Y_tr[:B]
        _, cache = forward(p, Xb)
        _, z1, a1, z2, a2, y_hat = cache
        d10, dW = y_hat - Yb, np.ones_like(z1)

        t_f = median_us(lambda: forward(p, Xb))
        t_b = median_us(lambda: backward(p, cache, Yb))
        # the matrix multiplications alone, same shapes as inside each pass
        mm_f = median_us(lambda: (Xb @ p["W1"], a1 @ p["W2"], a2 @ p["W3"]))
        mm_b = median_us(lambda: (a2.T @ d10, d10 @ p["W3"].T,
                                  a1.T @ dW, dW @ p["W2"].T, Xb.T @ dW))
        ff, fb = matmul_flops(B, W)
        print(f"{W:<4}{B:>6}{t_f:>9.1f}{t_b:>9.1f}{t_b / t_f:>9.2f}"
              f"{mm_f:>11.1f}{mm_b:>11.1f}"
              f"{f'{mm_f / t_f:.0%} / {mm_b / t_b:.0%}':>18}{fb / ff:>12.2f}")

print("\nForward = 3 matmuls; backward = 5 (3 weight grads + 2 activation grads;"
      " no grad w.r.t. the input X).")