# Lab 8 — Quantization and export

Models and data come from **Lab 7** (Fashion-MNIST, Lab 7 split and normalisation, seed 42):
baseline `BigCNN` (985,290 params, `../lab07/checkpoints/baseline.pt`), global unstructured 90 % and
structured 50 %-filter pruned models, and the distilled `SmallCNN` student (98,666 params), loaded from
`../lab07/checkpoints/size_formats/*_dense.pt`. Latency is always batch 1, one CPU thread.

| File                    | Task                                                                                |
| ----------------------- | ----------------------------------------------------------------------------------- |
| `common.py`             | data, model loading, accuracy, size, latency, dynamic/static quantization helpers   |
| `task1_manual_quant.py` | Task 1: manual symmetric quantization 8/4/2 bit, per-tensor vs per-channel, MSE     |
| `task2_dynamic.py`      | Task 2: dynamic PTQ, backend recorded                                               |
| `task3_static.py`       | Task 3: static PTQ, calibration on 500 random train images (+ bad calibration sets) |
| `task4_onnx.py`         | Task 4: ONNX export, check vs PyTorch (atol 1e-4), ONNX Runtime latency             |
| `task5_compare.py`      | Task 5: variant table, trade-off plots, deployment choice                           |
| `report.md`             | answers and discussion                                                              |

## How to run (from `lab08/`, lab01 venv active)
