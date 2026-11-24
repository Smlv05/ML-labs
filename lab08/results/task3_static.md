| variant | calibration | test_acc | size_kb | p50_ms | p95_ms | mean_ms | delta_acc_pp | size_ratio | speedup_p50 |
|---|---|---|---|---|---|---|---|---|---|
| fp32 baseline | - | 0.9369 | 3861 | 1.646 | 1.867 | 1.685 | 0 | 1 | 1 |
| static int8 | random 500 train images (required) | 0.9363 | 991.8 | 0.6677 | 0.7452 | 0.6835 | -0.06 | 3.893 | 2.464 |
| static int8 | 500 images of class 0 only | 0.9362 | 991.8 | 0.6646 | 0.7223 | 0.6785 | -0.07 | 3.893 | 2.476 |
| static int8 | 500 Gaussian-noise inputs | 0.9369 | 991.8 | 0.6656 | 0.7737 | 0.6912 | 0 | 3.893 | 2.472 |
| static int8 | 500 Gaussian-noise inputs x10 | 0.5762 | 991.8 | 0.665 | 0.7117 | 0.672 | -36.07 | 3.893 | 2.474 |
| static int8 | 500 all-black images | 0.8655 | 991.8 | 0.6653 | 0.7165 | 0.6716 | -7.14 | 3.893 | 2.473 |
