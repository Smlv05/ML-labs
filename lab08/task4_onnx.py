"""Lab 8 Task 4: export the FP32 Lab 7 baseline to ONNX, check outputs match PyTorch within 1e-4
on the whole test set, then measure ONNX Runtime latency (batch 1, one thread).

Run from lab08/:  python task4_onnx.py
"""
import numpy as np
import onnx
import onnxruntime as ort
import torch

from common import (LAB_DIR, batches, check_lab07_full_run, evaluate, get_data, latency_many,
                    load_model, save_json, size_kb, system_info, write_table)

EXPORT = LAB_DIR / "export"
ONNX_PATH = EXPORT / "baseline.onnx"
ATOL = 1e-4


def export(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, torch.randn(1, 1, 28, 28), str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    onnx.checker.check_model(onnx.load(str(path)))       # structural validation


def make_session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def main():
    check_lab07_full_run()
    model = load_model("baseline")
    export(model, ONNX_PATH)
    sess = make_session(ONNX_PATH)

    test = get_data()["test"]
    max_diff, agree, correct = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in batches(*test, batch_size=500):
            ref = model(xb).numpy()
            out = sess.run(None, {"input": xb.numpy()})[0]
            max_diff = max(max_diff, float(np.abs(ref - out).max()))
            agree += int((ref.argmax(1) == out.argmax(1)).sum())
            correct += int((out.argmax(1) == yb.numpy()).sum())
    n = len(test[0])
    passed = max_diff < ATOL
    print(f"max |torch - onnx| = {max_diff:.2e}  ->  {'PASS' if passed else 'FAIL'} (atol {ATOL})")

    x1 = test[0][:1]
    lat = latency_many({"torch": (model, x1),
                        "ort": (lambda a: sess.run(None, {"input": a}), x1.numpy())})
    torch_lat, ort_lat = lat["torch"], lat["ort"]

    rows = [
        {"runtime": "PyTorch eager FP32", "test_acc": evaluate(model, test),
         "size_kb": size_kb(model)[0], **torch_lat},
        {"runtime": "ONNX Runtime FP32", "test_acc": correct / n,
         "size_kb": ONNX_PATH.stat().st_size / 1024, **ort_lat},
    ]
    write_table(rows, "task4_onnx")
    save_json({"onnx_file": ONNX_PATH.name, "opset": 17, "atol": ATOL,
               "max_abs_diff": max_diff, "passed": passed, "argmax_agreement": agree / n,
               "n_test": n, "onnx_version": onnx.__version__, "onnxruntime_version": ort.__version__,
               "system": system_info(), "rows": rows}, "task4_onnx")
    if not passed:
        raise SystemExit("ONNX outputs differ from PyTorch by more than 1e-4")


if __name__ == "__main__":
    main()