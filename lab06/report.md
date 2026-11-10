# Lab 6 Report — Training pipelines

Machine: _(CPU model, cores, RAM)_ · Device: CPU · PyTorch 2.3.1 · Model: Lab 3 CNN (~?k params)

## 1. YAML-driven pipeline
All hyper-parameters live in `config.yaml`; `train.py --set` overrides single keys. Each run: seed → loaders →
model/optimizer/scheduler → per-epoch MLflow metrics (train loss/acc, val loss/acc, lr, throughput, epoch time) →
`last.pt` every epoch + `best.pt` on best val acc → test accuracy, peak memory and artifacts at the end.
_Insert MLflow UI screenshot._

## 2. Profile of one epoch
_Paste `results/profile_breakdown.md` and `results/profile_breakdown.png`._
Discussion: which phase dominates and why.

## 3. num_workers
_Paste `results/workers.md` + `results/workers.png`._
Discussion: where the bottleneck is (data vs compute), why more workers stop helping on this machine.

## 4. Batch size
_Paste `results/batch.md` + `results/batch.png`._
Discussion: throughput vs memory vs accuracy after a fixed 5-epoch budget.

## 5. LR schedules and optimizers
_Paste `results/schedules.md` + `results/schedules.png`._

## 6. Gradient accumulation
_Paste `results/accum.md` + `results/accum.png`._

## 7. Checkpoint recovery
_Paste `results/resume.md` + `results/resume.png`._

## 8. Project proposal
See `proposal.md`.
