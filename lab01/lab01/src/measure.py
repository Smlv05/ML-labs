import os, random, statistics, time, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from memory_profiler import memory_usage
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

SEED = 42
RES = Path(__file__).resolve().parents[1] / "results"

MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED),
}

BUDGETS = {
    "Cloud":  (float("inf"), 100, 500 * 1024),
    "Edge":   (1024, 50, 50 * 1024),
    "Mobile": (256, 20, 10 * 1024),
    "TinyML": (256 / 1024, 10, 100),
}


def peak_mib(fn):
    r = memory_usage((fn, (), {}), interval=0.01, max_usage=True)
    return float(r[0] if isinstance(r, (list, tuple, np.ndarray)) else r)


def rss_mib():
    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    RES.mkdir(exist_ok=True)

    X, y = load_breast_cancer(return_X_y=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED)

    acc_rows, cost_rows = [], []
    for name, make in MODELS.items():
        model = make().fit(X_tr, y_tr)
        acc = round(float(model.score(X_te, y_te)), 4)
        acc_rows.append({"model": name, "test_accuracy": f"{acc:.4f}"})

        make().fit(X_tr, y_tr)
        times = []
        for _ in range(5):
            t0 = time.perf_counter(); make().fit(X_tr, y_tr)
            times.append(time.perf_counter() - t0)
        train_s = statistics.median(times)

        x1 = X_te[:1]
        model.predict(x1)
        lat = []
        for _ in range(100):
            t0 = time.perf_counter(); model.predict(x1)
            lat.append((time.perf_counter() - t0) * 1000)
        infer_ms = statistics.median(lat)

        path = RES / f"model_{name}.joblib"
        joblib.dump(model, path)
        size_b = path.stat().st_size

        base = rss_mib()
        peak_train = peak_mib(lambda: make().fit(X_tr, y_tr))

        def infer_loop():
            for _ in range(100):
                model.predict(x1)
        peak_infer = peak_mib(infer_loop)

        cost_rows.append({
            "model": name,
            "train_time_s_median": round(train_s, 4),
            "infer_ms_median": round(infer_ms, 4),
            "model_bytes": size_b,
            "model_KB": round(size_b / 1024, 2),
            "baseline_rss_MiB": round(base, 1),
            "peak_train_MiB": round(peak_train, 1),
            "peak_infer_MiB": round(peak_infer, 1),
        })

    pd.DataFrame(acc_rows).to_csv(RES / "baseline_accuracy.csv", index=False)
    cost = pd.DataFrame(cost_rows)
    cost.to_csv(RES / "system_costs.csv", index=False)

    fit_rows = []
    for _, r in cost.iterrows():
        peak = max(r.peak_train_MiB, r.peak_infer_MiB)
        for tier, (mem, lat, size) in BUDGETS.items():
            ok_m, ok_l, ok_s = peak <= mem, r.infer_ms_median <= lat, r.model_KB <= size
            fit_rows.append({"model": r.model, "tier": tier,
                             "memory_ok": ok_m, "latency_ok": ok_l,
                             "size_ok": ok_s, "fits": ok_m and ok_l and ok_s})
    fit = pd.DataFrame(fit_rows)
    fit.to_csv(RES / "budget_fit.csv", index=False)

    print(pd.DataFrame(acc_rows).to_string(index=False))
    print(cost.to_string(index=False))
    print(fit.to_string(index=False))


if __name__ == "__main__":
    main()