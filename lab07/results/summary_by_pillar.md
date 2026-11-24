| Model | DATA: train samples | DATA: supervision | MODEL: test acc | MODEL: params | MODEL: non-zero | MODEL: dense size KB | MODEL: best size KB (format) | COMPUTE: dense MMACs | COMPUTE: non-zero MMACs | COMPUTE: latency b=1 ms | COMPUTE: latency b=64 ms | ENERGY: est. uJ dense exec | ENERGY: est. uJ ideal sparse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 55000 | hard labels | 0.9369 | 985290 | 985290 | 3861 | 3582.0 (dense_zip_npz) | 37.1 | 37.1 | 1.748 | 27.86 | 801.2 | 801.2 |
| global_unstructured_50 | 55000 | hard labels | 0.9369 | 985290 | 493498 | 3862 | 2062.5 (bitmask_npz) | 37.1 | 23.84 | 1.9 | 29.49 | 801.2 | 425.5 |
| global_unstructured_90 | 55000 | hard labels | 0.9303 | 985290 | 100065 | 3862 | 525.7 (bitmask_npz) | 37.1 | 8.82 | 1.805 | 29.21 | 801.2 | 104.6 |
| layer_unstructured_90 | 55000 | hard labels | 0.8974 | 985290 | 100066 | 3862 | 525.7 (bitmask_npz) | 37.1 | 3.71 | 1.979 | 30.11 | 801.2 | 81.1 |
| global_unstructured_95 | 55000 | hard labels | 0.9189 | 985290 | 50886 | 3862 | 333.6 (bitmask_npz) | 37.1 | 5.4 | 1.858 | 29.58 | 801.2 | 57.4 |
| structured_filters_10 | 55000 | hard labels | 0.9354 | 864855 | 864855 | 3391 | 3145.4 (dense_zip_npz) | 30.34 | 30.34 | 1.777 | 27.89 | 693.1 | 693.1 |
| structured_filters_25 | 55000 | hard labels | 0.9322 | 695802 | 695802 | 2730 | 2532.6 (dense_zip_npz) | 21.05 | 21.05 | 1.355 | 20.87 | 542.1 | 542.1 |
| structured_filters_50 | 55000 | hard labels | 0.9252 | 436266 | 436266 | 1716 | 1590.7 (dense_zip_npz) | 9.52 | 9.52 | 1.136 | 11.78 | 323 | 323 |
| student_plain | 55000 | hard labels | 0.9243 | 98666 | 98666 | 394 | 364.8 (dense_zip_npz) | 1.99 | 1.99 | 0.771 | 5.66 | 72.3 | 72.3 |
| student_distilled | 55000 | hard + soft (T=4, a=0.5) | 0.9256 | 98666 | 98666 | 394.2 | 364.5 (dense_zip_npz) | 1.99 | 1.99 | 0.675 | 4.93 | 72.3 | 72.3 |