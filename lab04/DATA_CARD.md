# Data Card - UCI Adult dataset

Author: Ersultan Ismailov
Course: Machine Learning and Its Applications, Lab 4

## Dataset
- Name: Adult (Census Income), UCI Machine Learning Repository
- Loaded with: fetch_openml(name='adult', version=2, as_frame=True)
- License: CC BY 4.0
- Task: predict if a person earns more than 50K per year (binary classification)
- Rows: 48842 raw, 52 duplicates removed, train 39032, test 9758

## Where the data comes from
The data was extracted from the 1994 US Census database by Barry Becker.
fnlwgt is a census sampling weight, not a feature of the person, so I don't use it.
The income label is self-reported.

## What's in the data
- Classes (%): {'<=50K': 76.1, '>50K': 23.9}
- Sex (%): {'Male': 66.8, 'Female': 33.2}
- Race (%): {'White': 85.5, 'Black': 9.6, 'Asian-Pac-Islander': 3.1, 'Amer-Indian-Eskimo': 1.0, 'Other': 0.8}
- Missing values: {'workclass': 2799, 'occupation': 2809, 'native-country': 857} (in the original file they are marked as "?")

## Processing steps
1. Load data, strip spaces, replace "?" with NaN
2. Validate schema, data types, value ranges and labels
3. Remove exact duplicates
4. Split train/test (80/20, stratified, seed 42) - before any cleaning to avoid leakage
5. New features: capital_net, log of capital gain/loss, has_capital
6. Impute missing values (median for numbers, most frequent for categories)
7. Clip outliers to 0.01-0.99 percentiles
8. Scale numeric columns, one-hot encode categorical columns

Steps 6-8 are fitted only on the train set (sklearn Pipeline).

## Label quality
- 496 training rows (1.27%) look suspicious (model gives their label probability < 0.1). They are saved in label_suspects.csv for checking, I did not delete them.
- 3 groups of rows have the same features but different labels.

## Baseline result
Logistic regression on the test set: accuracy 0.854, F1 0.6624, ROC-AUC 0.9051 (always predicting the majority class gives 0.7606).

## Limitations and bias
- The data is from 1994, so the 50K threshold is in 1994 dollars
- Only US data
- More men than women and most people are White, so a model can be biased
- relationship (Husband/Wife) basically contains sex, so removing sex alone is not enough
- Classes are imbalanced, so accuracy alone is not a good metric

## Reproducibility
Run: python lab04/pipeline.py
Seed: 42
Versions: {'python': '3.11.8', 'numpy': '1.26.4', 'pandas': '2.2.2', 'scikit-learn': '1.5.0', 'pyyaml': '6.0.3', 'matplotlib': '3.9.0', 'joblib': '1.4.2'}
SHA256 hashes are saved in outputs/manifest.json
