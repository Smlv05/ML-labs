# Lab 8 — Quantization and export: report

Models from Lab 7 (Fashion-MNIST): baseline BigCNN (4 conv blocks 32-64-128-128 + FC 1152→640→10, 985,290 params),
global unstructured 90 % pruned, structured 50 %-filter pruned (436,266 params), distilled SmallCNN (98,666 params).
Lab 7 test accuracy (full run): baseline 0.9369, unstructured 90 % 0.9303, structured 50 % 0.9252, distilled 0.9256.
Machine: _CPU name_, Windows, PyTorch 2.3.1 CPU, quantization backend **fbgemm** (x86).
All latencies: batch 1, `torch.set_num_threads(1)` / ORT `intra_op_num_threads=1`, 50 warm-up + 500 timed runs.

## Task 1 — Manual quantization (`results/task1_quant_error.md`, `task1_quant_error.png`)

Symmetric quantization, zero-point 0: `scale = max|W| / qmax`, `q = clamp(round(W/scale), -qmax, qmax)`,
`Ŵ = q·scale`, with qmax = 127 / 7 / 1 for 8 / 4 / 2 bits. Main tensor: `features.3.weight`
(conv2, 64×32×3×3 = 18,432 weights); the table also covers every other conv/linear weight.
Worked int8 example: `results/task1_example.json`.

| bits      | MSE per-tensor | MSE per-channel | SQNR dB (tensor / channel) | test acc, all weights quantized (tensor / channel) |
| --------- | -------------- | --------------- | -------------------------- | -------------------------------------------------- |
| 32 (FP32) | –              | –               | –                          | 0.9369                                             |
| 8         | 2.42e-7        | 1.03e-7         | 39.6 / 43.3                | 0.9368 / 0.9371                                    |
| 4         | 7.88e-5        | 3.41e-5         | 14.5 / 18.2                | 0.9316 / 0.9288                                    |
| 2         | 2.10e-3        | 1.47e-3         | 0.3 / 1.8                  | 0.1000 / 0.3437                                    |

- **Bit width.** Fewer bits → larger step (`scale = max|W|/qmax`). Going 8 → 4 bits makes the step
  127/7 ≈ 18× larger, and MSE grows 325× (2.42e-7 → 7.88e-5); SQNR drops 39.6 → 14.5 dB, i.e. ≈ 6.3 dB per bit, matching the ~6 dB/bit rule. 8-bit is lossless for accuracy (±0.02 pp), 4-bit costs 0.5–0.8 pp.
- **2 bits.** Symmetric 2-bit with zero-point 0 has only three levels {−s, 0, +s} (ternary). SQNR ≈ 0–2 dB means the error is as large as the weights themselves: the network collapses to chance (0.10) with per-tensor scales and 0.34 with per-channel scales. 2-bit needs quantization-aware training (Reddi §10.5.5.2), not PTQ.
- **Per-channel vs per-tensor.** Per-channel has lower MSE for every tensor and bit width (2.3× lower for conv2 at 8 and 4 bits). Filters have different max|W| (right panel of the figure); one per-tensor scale is set by the largest filter, so small-range filters use few levels — per-tensor conv2 uses 186 of 255 int8 levels, per-channel uses 252. The largest gain is in the FC layer `classifier.2` (640 rows): 3.2× lower MSE (+5 dB) at 8 bits. This is why per-channel weight scales are standard (Reddi §10.5.5.1, "Granularity").
- **MSE is only a proxy.** At 4 bits per-channel has lower MSE but 0.28 pp lower accuracy (28 of 10,000 images), which is within noise: accuracy depends on which weights get rounded, not only on the average error.

## Task 2 — Dynamic PTQ (`results/task2_dynamic.md`)

Backend: **fbgemm** (x86). `quantize_dynamic(model, {nn.Linear}, qint8)` replaced the two Linear layers with `DynamicQuantizedLinear` (int8 weights, per-tensor); all conv/BN layers stay FP32, because PyTorch dynamic quantization does not support convolutions. Activation ranges are computed per input at run time.

| variant       | test acc | size KB | p50 ms | p95 ms | size ratio | speed-up p50 |
| ------------- | -------- | ------- | ------ | ------ | ---------- | ------------ |
| FP32 baseline | 0.9369   | 3861    | 1.759  | 2.433  | 1.00×      | 1.00×        |
| dynamic int8  | 0.9373   | 1684    | 1.763  | 2.460  | 2.29×      | 1.00×        |

- **Accuracy:** unchanged (+0.04 pp = 4 images, within noise).
- **Size:** 2.29× smaller, not 4×: only the Linear layers (744,330 of 985,290 params, 75 %) are int8.
- **Latency:** no change. The Linear layers are only ~0.74 M of the 37.1 M MACs (2 %); the convolutions, which dominate compute, still run in FP32, and quantizing the activations at run time for every input costs about as much as the int8 matmul saves. Dynamic quantization is a size optimisation for conv-heavy models; it pays off for latency in Linear-dominated models (Transformers, LSTMs).
- **Measurement note:** the first non-interleaved runs gave 0.4×–1.7× "speed-ups" for the same models because the CPU (hybrid P/E cores, power management) changed speed between measurements. All final numbers use interleaved timing (10 rounds × 100 runs per variant, order rotated), on AC power.

## Task 3 — Static PTQ (`results/task3_static.md`)

FX graph mode: conv+BN+ReLU fused, observers inserted, 500 random **training** images passed through for calibration, then converted to int8 (weights per-channel, activations per-tensor with fixed scale/zero-point).

Result with the required calibration: accuracy 0.9363 (−0.06 pp), size 3861 → 992 KB (3.9×),
latency p50 1.646 → 0.668 ms (**2.46×**), p95 1.867 → 0.745 ms. Unlike dynamic quantization, the convolutions (≈ 98 % of MACs) now run as int8. All calibration variants have the same latency (0.665–0.668 ms): calibration changes only the scales, i.e. accuracy, not the amount of compute.

**Why the calibration subset must represent real data.** In static quantization the activation clipping ranges [α, β] are measured once on the calibration set and then frozen for every future input (Reddi §10.5.5.1, "Calibration"). If the calibration data is not like the deployment data, the ranges are wrong: too narrow → real activations saturate (clipped to α/β); too wide → the 256 levels are spread over values that never occur and the resolution for real values is lost. The control runs show this: calibrating on one class only or on Gaussian noise gives different output scales (`output_scales` in `task3_static.json`) and lower accuracy than the random-train calibration. The set must also come from the training split, never from test, otherwise the test score is no longer an unbiased estimate (leakage, Lab 4).

## Task 4 — ONNX export (`results/task4_onnx.md`, `task4_onnx.json`)

`torch.onnx.export` (opset 17, dynamic batch axis, constant folding) → `onnx.checker.check_model` passed. Numerical validation on all 10,000 test images: **max |PyTorch − ORT| = 9.54e-6 < 1e-4 → PASS**, same test accuracy (0.9369), argmax agreement _…_ % (`task4_onnx.json`).

| runtime            | test acc | size KB | p50 ms | p95 ms |
| ------------------ | -------- | ------- | ------ | ------ |
| PyTorch eager FP32 | 0.9369   | 3861    | 1.643  | 1.766  |
| ONNX Runtime FP32  | 0.9369   | 3849    | 0.590  | 0.614  |

ONNX Runtime is **2.8× faster** than PyTorch eager with identical FP32 weights — even faster than PyTorch static int8 (0.668 ms). Reasons: ORT graph optimisation folds BatchNorm into the convolutions and fuses Conv+ReLU, uses optimised MLAS kernels, and runs the whole graph in C++ without per-layer Python dispatch, which is a large share of the time at batch 1 for a model this small. The differences of ~1e-5 come from the different order of floating-point operations after fusion, not from a conversion error.

## Task 5 — Variant comparison (`results/task5_variants.md`, `task5_tradeoffs.png`)

Pruned and distilled models are the fine-tuned Lab 7 checkpoints; all quantized variants use the same 500
calibration images. Latency: batch 1, one thread, interleaved timing (10 rounds × 100 runs per variant).

| variant                            | precision   | params (non-zero) | test acc | Δ pp  | size KB | zipped KB | p50 ms | p95 ms | speed-up |
| ---------------------------------- | ----------- | ----------------- | -------- | ----- | ------- | --------- | ------ | ------ | -------- |
| baseline                           | FP32        | 985,290           | 0.9369   | 0     | 3861    | 3578      | 1.694  | 2.349  | 1.00×    |
| pruned unstructured 90 %           | FP32        | 985,290 (100,065) | 0.9303   | −0.66 | 3861    | 663       | 1.688  | 2.120  | 1.00×    |
| pruned structured 50 % filters     | FP32        | 436,266           | 0.9252   | −1.17 | 1715    | 1587      | 0.794  | 0.922  | 2.13×    |
| distilled (SmallCNN)               | FP32        | 98,666            | 0.9256   | −1.13 | 394     | 362       | 0.372  | 0.892  | 4.56×    |
| dynamic quant                      | int8 Linear | 985,290           | 0.9373   | +0.04 | 1684    | 1471      | 1.778  | 2.662  | 0.95×    |
| static quant                       | int8        | 985,290           | 0.9363   | −0.06 | 992     | 910       | 0.690  | 0.979  | 2.46×    |
| pruned unstructured + static quant | int8        | 985,290 (100,065) | 0.9306   | −0.63 | 992     | 192       | 0.670  | 1.204  | 2.53×    |
| pruned structured + static quant   | int8        | 436,266           | 0.9253   | −1.16 | 453     | 410       | 0.502  | 0.841  | 3.37×    |
| distilled + static quant           | int8        | 98,666            | 0.9244   | −1.25 | 111     | 94        | 0.366  | 0.680  | 4.63×    |
| baseline ONNX Runtime              | FP32        | 985,290           | 0.9369   | 0     | 3849    | –         | 0.607  | 0.790  | 2.79×    |

Observations:

- **Unstructured pruning** removes 90 % of the weights at −0.66 pp but gives **no speed-up** (1.00×) and the same
  dense file size: zeros are still stored and multiplied by dense kernels. It only helps after compression:
  3578 → 663 KB zipped. Combined with int8 it gives the smallest BigCNN file: **192 KB zipped** (18.6× smaller).
- **Structured pruning** and **distillation** make a physically smaller dense network, so they cut latency on any
  hardware: 2.13× and 4.56×, at about −1.1 pp each. The distilled student is the fastest FP32 model.
- **Static int8** alone is the best single optimisation for the baseline: 3.9× smaller, 2.46× faster, −0.06 pp.
  **Dynamic int8** only shrinks the file (2.3×) and does not speed up this conv-dominated model (Task 2).
- **Optimisations stack** (Reddi §10.3.4): structured pruning + int8 gives 8.5× smaller and 3.37× faster than the
  baseline; distillation + int8 gives **34.9× smaller** (111 KB) and 4.63× faster, at −1.25 pp. Quantization
  adds almost no accuracy loss on top of pruning/distillation (≤ 0.12 pp).
- For the tiny student, int8 mostly helps size (394 → 111 KB); latency hardly changes (0.372 → 0.366 ms),
  because at ~2 M MACs the fixed per-call overhead (quantize/dequantize, operator dispatch) dominates.
- **The runtime matters as much as the model**: the unchanged FP32 baseline in ONNX Runtime (0.607 ms) is faster
  than the PyTorch int8 baseline (0.690 ms), thanks to BN folding, operator fusion and no Python dispatch.

## Deployment decision (`results/task5_deployment.md`)

Rules in `task5_compare.py`: Cloud = max accuracy; Edge = lowest p95 within 1 pp; Mobile = int8, smallest size
within 1.5 pp; TinyML = full-integer int8, smallest size within 2 pp.

| Platform                                              | Variant                           | acc    | size   | p95     | Why                                                                                                                                                                                                                                                                                                                                   |
| ----------------------------------------------------- | --------------------------------- | ------ | ------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Cloud**                                             | **baseline FP32 in ONNX Runtime** | 0.9369 | 3.8 MB | 0.79 ms | Size and memory are not limits, so keep full accuracy. The rule picked dynamic quant only because of +0.04 pp (4 images, noise); ORT gives the same accuracy at 2.8× lower latency, and on a server it can batch requests and use a GPU.                                                                                              |
| **Edge** (CPU box, e.g. Raspberry Pi / x86 gateway)   | **static int8 baseline**          | 0.9363 | 992 KB | 0.98 ms | Keeps accuracy within 0.06 pp, 4× less memory and 2.5× faster than FP32. Best next step: run the int8 model in ONNX Runtime for both gains.                                                                                                                                                                                           |
| **Mobile**                                            | **distilled + static int8**       | 0.9244 | 111 KB | 0.68 ms | Smallest download and lowest latency/energy per inference (battery), −1.25 pp is within the 1.5 pp budget. If accuracy matters more than size, static int8 baseline (992 KB, −0.06 pp) is the alternative.                                                                                                                            |
| **TinyML** (MCU, ~256 KB SRAM / ≤ 1 MB flash, no FPU) | **distilled + static int8**       | 0.9244 | 111 KB | –       | The only variant that fits: full-integer int8 (static, so no float range computation at run time), 111 KB weights fit in flash. Every BigCNN variant is ≥ 450 KB in int8 and needs far more activation memory. Dynamic quant is impossible there (float activations). It would be deployed with TFLite Micro / CMSIS-NN, not PyTorch. |
