"""Lab 8 Task 3: post-training STATIC quantization, calibrated on 500 random training images.
Also calibrates on NON-representative sets to show why the calibration subset must look
like real data: one class only, Gaussian noise, all-black images, and noise x10.

Run from lab08/:  python task3_static.py
"""
import torch

from common import (SEED, check_lab07_full_run, evaluate, get_data, latency_many, load_model,
                    pick_backend, quant_scales, random_train_images, save_json, size_kb,
                    static_quant, system_info, write_table)

N_CALIB = 500


def single_class_images(data, cls=0, n=N_CALIB):
    x, y = data["train"]
    idx = torch.nonzero(y == cls).flatten()
    g = torch.Generator().manual_seed(SEED)
    return x[idx[torch.randperm(len(idx), generator=g)[:n]]]


def noise_images(n=N_CALIB, factor=1.0):
    g = torch.Generator().manual_seed(SEED)
    return torch.randn(n, 1, 28, 28, generator=g) * factor


def blank_images(data, n=N_CALIB):
    """All-black images (raw pixel 0) after the same normalisation as the real data."""
    mean, std = data["stats"]
    return torch.full((n, 1, 28, 28), (0.0 - mean) / std)


def main():
    check_lab07_full_run()
    backend = pick_backend()
    print("quantization backend:", backend)
    data = get_data()
    test = data["test"]
    x1 = test[0][:1]
    fp32 = load_model("baseline")

    calib_sets = {
        "random 500 train images (required)": random_train_images(data, N_CALIB),
        "500 images of class 0 only": single_class_images(data, cls=0),
        "500 Gaussian-noise inputs": noise_images(),
        "500 Gaussian-noise inputs x10": noise_images(factor=10.0),
        "500 all-black images": blank_images(data),
    }

    models = {"fp32 baseline": ("-", fp32)}
    scales = {}
    for i, (calib_name, calib_x) in enumerate(calib_sets.items()):
        sq = static_quant(fp32, calib_x, backend)
        models[f"static int8 #{i + 1}"] = (calib_name, sq)
        scales[calib_name] = quant_scales(sq)
        print(f"{calib_name}: output scales {scales[calib_name]}")
    lat = latency_many({key: (m, x1) for key, (_, m) in models.items()})

    rows = []
    for key, (calib_name, m) in models.items():
        raw, _ = size_kb(m)
        rows.append({"variant": key.split(" #")[0], "calibration": calib_name,
                     "test_acc": evaluate(m, test), "size_kb": raw, **lat[key]})

    base = rows[0]
    for r in rows:
        r["delta_acc_pp"] = 100 * (r["test_acc"] - base["test_acc"])
        r["size_ratio"] = base["size_kb"] / r["size_kb"]
        r["speedup_p50"] = base["p50_ms"] / r["p50_ms"]
    write_table(rows, "task3_static")
    save_json({"backend": backend, "system": system_info(), "n_calib": N_CALIB,
               "rows": rows, "output_scales": scales}, "task3_static")


if __name__ == "__main__":
    main()