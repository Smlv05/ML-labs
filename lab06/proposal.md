# Individual Project Proposal — On-device Keyword Spotting

**Student:** Ersultan Ismailov  **Student ID:** _______  **Course:** Machine Learning and Its Applications (Group 01)

## 1. Problem
Detect a small set of spoken voice commands ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
plus *silence* and *unknown* from a 1-second audio clip, running fully **offline on a CPU-only device**
(laptop / Raspberry-Pi-class). Keyword spotting is an always-on workload: it must be small, fast and cheap,
which makes it a good fit for the deployment constraints of this course.

## 2. Dataset
**Google Speech Commands v2** (`torchaudio.datasets.SPEECHCOMMANDS`): ~105k one-second 16 kHz WAV clips, 35 words,
2,600+ speakers, CC-BY 4.0. I use the standard 12-class setup (10 keywords + silence + unknown) and the official
speaker-disjoint `validation_list.txt` / `testing_list.txt` split, so no speaker appears in two splits (no leakage, cf. Lab 4).
Pre-processing: 40 log-Mel features, 30 ms window / 10 ms hop → 40×98 input; augmentation with background noise and ±100 ms time shift.

## 3. Model and baseline
Baseline: small CNN on log-Mel spectrograms (Lab 3 style). Main model: **DS-CNN** (depthwise-separable CNN,
~25–100k parameters). A larger teacher CNN (~1–2 M parameters) is trained for distillation.
Training uses the Lab 6 pipeline (YAML config, MLflow tracking, checkpoints, profiling).

## 4. Success metrics
| Metric | Target |
|---|---|
| Top-1 accuracy, 12-class test set | ≥ 90 % (FP32), ≤ 1.5 pp drop after optimisation |
| Inference latency, batch 1, CPU, ONNX Runtime (p95 over 1,000 runs, incl. feature extraction) | **≤ 50 ms** |
| Model file size (ONNX) | **≤ 5 MB** (goal < 0.5 MB) |
| Secondary: false-accept rate on *unknown*, per-class confusion matrix, energy per inference (CodeCarbon) | reported |

## 5. Deployment constraints
Target: CPU-only, single-threaded ONNX Runtime, batch size 1, streaming input every 1 s.
Hard limits: **latency ≤ 50 ms**, **size ≤ 5 MB**. Served through a small FastAPI endpoint for the demo
and benchmarked with a fixed latency harness (warm-up + 1,000 timed runs, p50/p95/p99).

## 6. Planned optimisation
1. **Quantization (main):** post-training static INT8 quantization (calibrated on 500 training clips), then
   quantization-aware training if accuracy drops > 1.5 pp. Expected ~4× smaller model and lower latency.
2. **Knowledge distillation (secondary):** teacher CNN → DS-CNN student (KL loss, temperature T = 4) to recover
   accuracy lost by shrinking the student.
Each step is compared on the accuracy / latency / size table against the FP32 baseline.

## 7. Plan and risks
Wk 1: data pipeline + baseline · Wk 2: DS-CNN + teacher, MLflow tracking · Wk 3: PTQ / QAT · Wk 4: distillation,
latency benchmark, FastAPI demo · Wk 5: report.
Risks: feature extraction dominating latency (mitigation: precompute Mel filterbank, measure it separately);
INT8 accuracy drop (mitigation: QAT); class imbalance of *unknown* (mitigation: weighted sampling).
