# Lab 6 — Training pipelines and project proposal

Model: **CNN from Lab 3** (best test accuracy 0.900 on FashionMNIST), imported from `../lab03/models.py`.
Data is reused from `../lab03/data`.

| File | Task |
|---|---|
| `config.yaml` | single config object (seed, data, optimizer, scheduler, accumulation, checkpoint, MLflow) |
| `pipeline.py` | seeding, data loaders, model/optimizer/scheduler builders, training loop, checkpoints, MLflow |
| `train.py` | Task 1 entry point (`--set key=value`, `--stop-after N`, `--resume`) |
| `profile_epoch.py` | Task 2: torch.profiler breakdown of one epoch |
| `studies.py` | Tasks 3–7: workers, batch, schedules, accum, resume (+ `export` of MLflow runs) |
| `proposal.md` | Task 8: individual project proposal |
| `report.md` | results and discussion |

## How to run (from `lab06/`, with the lab01 venv active)
```
python train.py                    # Task 1: baseline, 10 epochs
python profile_epoch.py            # Task 2
python studies.py workers          # Task 3
python studies.py batch            # Task 4
python studies.py schedules        # Task 5 (6 runs x 10 epochs - the longest one)
python studies.py accum            # Task 6
python studies.py resume           # Task 7
python studies.py export           # MLflow runs -> results/mlflow_runs.csv
mlflow ui                          # http://127.0.0.1:5000  -> screenshot for the report
```
Outputs: `results/*.md|csv|png`, per-run JSON in `results/runs/`, MLflow store in `mlruns/`,
checkpoints in `checkpoints/<run_name>/` (git-ignored).

## Reproducibility
Fixed seed (Python, NumPy, torch), deterministic algorithms, fixed train/val split generator, and a shuffle
order that depends only on `(seed, epoch)`, so a resumed run sees exactly the same batches as an uninterrupted one.
Every run logs its full flattened config as MLflow params plus `config.yaml` as an artifact.
