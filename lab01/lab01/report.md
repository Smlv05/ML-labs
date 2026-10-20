# Lab 1: Environment and first system measurements

## 1. Goal

In this lab I set up the Python environment and compared two models on the Breast Cancer dataset: logistic regression and random forest. I measured how long they take to train, how fast they predict, how big they are and how much memory they use, then checked which of the four deployment targets (Cloud, Edge, Mobile, TinyML) each model could run on.

## 2. Method

I used Python 3.11.8 in a virtual environment with the pinned packages. I changed pyarrow to 15.0.2, because mlflow 2.14.1 needs pyarrow below 16 and pip refused the original pin. Lab 1 doesn't use pyarrow, so this doesn't affect the results.

I loaded the dataset, split it 70/30 (stratified, random_state=42) and set all seeds to 42. Both models use the settings from the manual. Training time is the median of 5 runs after one warm-up. Prediction time is the median of 100 single-sample predictions. Model size comes from the saved joblib file, and memory from memory_profiler, which measures the whole Python process. Logistic regression gave a convergence warning because the features aren't scaled. I kept the manual's settings anyway. All timings are from my laptop, not from the target devices.

## 3. Results

**Test accuracy** (`results/baseline_accuracy.csv`)

| Model               | Test accuracy |
| ------------------- | ------------- |
| Logistic regression | 0.9415        |
| Random forest       | 0.9357        |

**System cost** (`results/system_costs.csv`)

| Model               | Train time, median (s) | Inference, median (ms) | Size (bytes) | Size (KB) | Peak memory, training (MiB) | Peak memory, inference (MiB) |
| ------------------- | ---------------------- | ---------------------- | ------------ | --------- | --------------------------- | ---------------------------- |
| Logistic regression | 0.4259                 | 0.0569                 | 1055         | 1.03      | 268.8                       | 268.8                        |
| Random forest       | 0.1685                 | 2.5099                 | 290889       | 284.07    | 270.4                       | 270.5                        |

**Budget fit** (`results/budget_fit.csv`)

| Model               | Tier   | Memory OK | Latency OK | Size OK | Fits    |
| ------------------- | ------ | --------- | ---------- | ------- | ------- |
| Logistic regression | Cloud  | yes       | yes        | yes     | **yes** |
| Logistic regression | Edge   | yes       | yes        | yes     | **yes** |
| Logistic regression | Mobile | no        | yes        | yes     | no      |
| Logistic regression | TinyML | no        | yes        | yes     | no      |
| Random forest       | Cloud  | yes       | yes        | yes     | **yes** |
| Random forest       | Edge   | yes       | yes        | yes     | **yes** |
| Random forest       | Mobile | no        | yes        | yes     | no      |
| Random forest       | TinyML | no        | yes        | no      | no      |

**Deployment-fit argument**

- **Cloud and Edge:** both models fit all three budgets. Peak memory is about 270 MiB, which is within the Edge limit of 1024 MiB, and latency and size are far below the limits.
- **Mobile:** both models pass latency (under 3 ms against a 20 ms limit) and size (under 300 KB against 10 MB). They fail only the memory check, and that failure comes from the Python process, not the models. About 270 MiB is mostly the interpreter and the libraries I imported (such as torch). On a phone the model would run in a lighter runtime, so I judge both models deployable to Mobile, but this is an argument and not something I measured.
- **TinyML:** logistic regression passes size (1.03 KB against 100 KB) and latency, and fails memory for the same runtime reason. Its parameters are only 30 coefficients and an intercept, so a port to C could plausibly fit. Random forest fails on size (284 KB against 100 KB), so it cannot run on TinyML as it is.

4. Conclusions
   Accuracy is practically identical between the two models (0.9415 versus 0.9357), representing just a single-sample variation on a 171-sample test set. Consequently, system-cost metrics are essential to differentiate them.

Measuring process-level memory via memory_profiler primarily captures the Python interpreter and imported libraries (~270 MiB) rather than model-specific memory consumption. Future profiling should isolate model allocation to provide accurate resource evaluations.

Surprisingly, logistic regression trained slower than random forest (0.43s vs 0.17s), which is attributable to unscaled features causing the solver to hit its maximum iteration limit. Despite this training penalty, logistic regression remains vastly superior in inference speed (44x faster) and footprint (~275 times smaller).
