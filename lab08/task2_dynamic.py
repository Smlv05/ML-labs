"""Lab 8 Task 2: post-training DYNAMIC quantization of the Lab 7 baseline (BigCNN).
Measures accuracy, size and batch-1 latency and records the quantization backend.

Run from lab08/:  python task2_dynamic.py
"""
from common import (check_lab07_full_run, dynamic_quant, evaluate, get_data, latency_many,
                    load_model, pick_backend, save_json, size_kb, system_info, write_table)


def main():
    check_lab07_full_run()
    backend = pick_backend()
    print("quantization backend:", backend)
    test = get_data()["test"]
    x1 = test[0][:1]                              # one real test image, batch size 1

    fp32 = load_model("baseline")
    dq = dynamic_quant(fp32)
    print(dq)                                     # only Linear layers become DynamicQuantizedLinear

    models = {"fp32 baseline": fp32, "dynamic int8": dq}
    lat = latency_many({name: (m, x1) for name, m in models.items()})

    rows = []
    for name, m in models.items():
        raw, _ = size_kb(m)
        rows.append({"variant": name, "test_acc": evaluate(m, test), "size_kb": raw, **lat[name]})
    base = rows[0]
    for r in rows:
        r["delta_acc_pp"] = 100 * (r["test_acc"] - base["test_acc"])
        r["size_ratio"] = base["size_kb"] / r["size_kb"]
        r["speedup_p50"] = base["p50_ms"] / r["p50_ms"]
    write_table(rows, "task2_dynamic")
    save_json({"backend": backend, "system": system_info(), "rows": rows,
               "quantized_modules": [n for n, mod in dq.named_modules()
                                     if "quantized" in type(mod).__module__]}, "task2_dynamic")


if __name__ == "__main__":
    main()