# Validation report - UCI Adult

| split | rows | passed | range violations | duplicates | missing cells |
|---|---|---|---|---|---|
| raw | 48842 | True | 0 | 52 | 6465 |
| train | 39032 | True | 0 | 0 | 5148 |
| test | 9758 | True | 0 | 0 | 1308 |

## Range checks (raw data)

| column | min | max | allowed | bad rows |
|---|---|---|---|---|
| age | 17.0 | 90.0 | [17, 90] | 0 |
| fnlwgt | 12285.0 | 1490400.0 | [1, 2000000] | 0 |
| education-num | 1.0 | 16.0 | [1, 16] | 0 |
| capital-gain | 0.0 | 99999.0 | [0, 99999] | 0 |
| capital-loss | 0.0 | 4356.0 | [0, 4356] | 0 |
| hours-per-week | 1.0 | 99.0 | [1, 99] | 0 |

## Missing values (raw data)

- workclass: 2799
- occupation: 2809
- native-country: 857

## Unseen categories in test

- none

## Leakage checks

- duplicates removed before split: 52
- rows that are in both train and test: 0
- preprocessing is fitted only on train (inside sklearn Pipeline)