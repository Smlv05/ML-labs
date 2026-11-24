| platform | chosen variant | test acc | size KB | p95 ms | rule |
|---|---|---|---|---|---|
| Cloud | dynamic quant | 0.9373 | 1684.0 | 2.662 | highest accuracy; latency/size are not binding (fast CPU/GPU, lots of memory) |
| Edge | static quant | 0.9363 | 991.8 | 0.979 | lowest p95 latency with accuracy drop <= 1.0 pp |
| Mobile | distilled + static quant | 0.9244 | 110.8 | 0.680 | int8 only, smallest size then p95, drop <= 1.5 pp |
| TinyML | distilled + static quant | 0.9244 | 110.8 | 0.680 | full-integer int8 only (no FPU/PyTorch on MCU), smallest size, drop <= 2.0 pp |
