# Lab 4 - Data pipeline for UCI Adult dataset
# Ersultan Ismailov
# how to run (from repo folder): python lab04/pipeline.py

import os
import json
import random
import hashlib
import platform
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

SEED = config["seed"]
random.seed(SEED)
np.random.seed(SEED)

target = config["target"]
OUT = os.path.join(BASE_DIR, config["output_dir"])
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)


# ---------- helper functions ----------

def file_hash(path):
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        sha.update(f.read())
    return sha.hexdigest()


def array_hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def save_csv(df, name):
    path = os.path.join(OUT, name)
    # lineterminator so the file is the same on windows and linux
    df.to_csv(path, index=False, lineterminator="\n")
    return path


# new features, only uses values from the same row so no leakage
def add_features(df):
    df = df.copy()
    gain = df["capital-gain"].astype(float)
    loss = df["capital-loss"].astype(float)
    df["capital_net"] = gain - loss
    df["capital_gain_log"] = np.log1p(gain)
    df["capital_loss_log"] = np.log1p(loss)
    df["has_capital"] = ((gain > 0) | (loss > 0)).astype(int)
    return df


# clips outliers, the limits are learned in fit() so only from train data
class Clipper(BaseEstimator, TransformerMixin):
    def __init__(self, low=0.01, high=0.99):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.low, axis=0)
        self.high_ = np.nanquantile(X, self.high, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.low_, self.high_)

    def get_feature_names_out(self, input_features=None):
        return np.array(input_features, dtype=object)


def check_data(df, name, train_categories=None):
    result = {"name": name, "rows": len(df)}

    # schema
    expected = config["numeric_cols"] + config["categorical_cols"] + [target]
    result["missing_columns"] = [c for c in expected if c not in df.columns]
    result["wrong_dtype"] = [c for c in config["numeric_cols"]
                             if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]

    # missing values
    result["missing_values"] = {}
    for c in df.columns:
        n = int(df[c].isna().sum())
        if n > 0:
            result["missing_values"][c] = n

    # ranges
    result["ranges"] = {}
    total_bad = 0
    for col, (low, high) in config["ranges"].items():
        bad = int(((df[col] < low) | (df[col] > high)).sum())
        total_bad += bad
        result["ranges"][col] = {"min": float(df[col].min()), "max": float(df[col].max()),
                                 "allowed": [low, high], "bad_rows": bad}

    # labels
    result["labels"] = sorted(df[target].dropna().unique().tolist())
    labels_ok = set(result["labels"]) == {"<=50K", ">50K"} and df[target].isna().sum() == 0

    result["duplicates"] = int(df.duplicated().sum())

    # categories in test that were not in train
    if train_categories is not None:
        result["unseen_categories"] = {}
        for col in config["categorical_cols"]:
            new_vals = set(df[col].dropna()) - train_categories[col]
            if len(new_vals) > 0:
                result["unseen_categories"][col] = sorted(new_vals)

    result["passed"] = bool(len(result["missing_columns"]) == 0 and len(result["wrong_dtype"]) == 0
                            and total_bad == 0 and labels_ok)
    return result


def build_preprocessing():
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clip", Clipper(config["clip_low"], config["clip_high"])),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("num", num_pipe, config["model_numeric"]),
        ("cat", cat_pipe, config["model_categorical"]),
    ])  # other columns (fnlwgt, education...) are dropped
    return Pipeline([("features", FunctionTransformer(add_features)), ("preprocess", columns)])


def build_model():
    return Pipeline([("prep", build_preprocessing()),
                     ("clf", LogisticRegression(max_iter=2000, random_state=SEED))])


# ---------- 1. load data ----------
print("1. Loading data")
data = fetch_openml(name="adult", version=2, as_frame=True,
                    data_home=os.path.join(BASE_DIR, ".cache"), parser="auto")
df = data.frame.copy()

# categorical columns -> strings, remove spaces, "?" means missing
for col in config["categorical_cols"] + [target]:
    df[col] = df[col].astype("object")
    df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df[col] = df[col].replace("?", np.nan)
df[target] = df[target].str.replace(".", "", regex=False)  # some versions have ">50K."

raw_path = save_csv(df, "raw.csv")
print("raw shape:", df.shape)

# ---------- 2. duplicates + split ----------
# split BEFORE cleaning so test data doesn't influence imputation/clipping/scaling
print("2. Removing duplicates and splitting")
n_duplicates = int(df.duplicated().sum())
df_clean = df.drop_duplicates().reset_index(drop=True)
print("duplicates removed:", n_duplicates)

train_df, test_df = train_test_split(df_clean, test_size=config["test_size"],
                                     stratify=df_clean[target], random_state=SEED)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
train_path = save_csv(train_df, "train.csv")
test_path = save_csv(test_df, "test.csv")
print("train:", len(train_df), "test:", len(test_df))

# ---------- 3. validation ----------
print("3. Validation")
train_categories = {c: set(train_df[c].dropna()) for c in config["categorical_cols"]}
raw_check = check_data(df, "raw")
train_check = check_data(train_df, "train")
test_check = check_data(test_df, "test", train_categories)

# leakage check: same rows in train and test?
overlap = len(train_df.merge(test_df, how="inner"))

with open(os.path.join(OUT, "validation_report.json"), "w") as f:
    json.dump({"raw": raw_check, "train": train_check, "test": test_check,
               "train_test_overlap": overlap}, f, indent=2)

lines = ["# Validation report - UCI Adult", ""]
lines.append("| split | rows | passed | range violations | duplicates | missing cells |")
lines.append("|---|---|---|---|---|---|")
for r in [raw_check, train_check, test_check]:
    bad = sum(v["bad_rows"] for v in r["ranges"].values())
    lines.append(f"| {r['name']} | {r['rows']} | {r['passed']} | {bad} | {r['duplicates']} | "
                 f"{sum(r['missing_values'].values())} |")
lines += ["", "## Range checks (raw data)", "", "| column | min | max | allowed | bad rows |", "|---|---|---|---|---|"]
for col, v in raw_check["ranges"].items():
    lines.append(f"| {col} | {v['min']} | {v['max']} | {v['allowed']} | {v['bad_rows']} |")
lines += ["", "## Missing values (raw data)", ""]
for col, n in raw_check["missing_values"].items():
    lines.append(f"- {col}: {n}")
lines += ["", "## Unseen categories in test", ""]
if test_check["unseen_categories"]:
    for col, vals in test_check["unseen_categories"].items():
        lines.append(f"- {col}: {vals}")
else:
    lines.append("- none")
lines += ["", "## Leakage checks", "",
          f"- duplicates removed before split: {n_duplicates}",
          f"- rows that are in both train and test: {overlap}",
          "- preprocessing is fitted only on train (inside sklearn Pipeline)"]
with open(os.path.join(OUT, "validation_report.md"), "w") as f:
    f.write("\n".join(lines))

if not (raw_check["passed"] and train_check["passed"] and test_check["passed"]) or overlap > 0:
    raise SystemExit("Validation failed, check outputs/validation_report.md")
print("validation passed")

# ---------- 4. preprocessing (fit on train only) ----------
print("4. Preprocessing")
X_train = train_df.drop(columns=[target])
X_test = test_df.drop(columns=[target])
y_train = (train_df[target] == config["positive"]).astype(int)
y_test = (test_df[target] == config["positive"]).astype(int)

prep = build_preprocessing()
prep.fit(X_train)  # only train!
X_train_p = prep.transform(X_train)
X_test_p = prep.transform(X_test)
feature_names = list(prep.named_steps["preprocess"].get_feature_names_out())
print("features after encoding:", len(feature_names))
assert not np.isnan(X_train_p).any() and not np.isnan(X_test_p).any()

np.save(os.path.join(OUT, "X_train.npy"), X_train_p)
np.save(os.path.join(OUT, "X_test.npy"), X_test_p)
np.save(os.path.join(OUT, "y_train.npy"), y_train.values)
np.save(os.path.join(OUT, "y_test.npy"), y_test.values)
joblib.dump(prep, os.path.join(OUT, "preprocessor.joblib"))

# ---------- 5. label quality ----------
# out-of-fold predictions: if the model is very sure the label is wrong -> suspicious
print("5. Label audit")
cv = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=SEED)
proba = cross_val_predict(build_model(), X_train, y_train, cv=cv, method="predict_proba")
p_label = proba[np.arange(len(y_train)), y_train.values]
suspect_mask = p_label < config["suspect_threshold"]

suspects = X_train[suspect_mask].copy()
suspects["label"] = y_train[suspect_mask].values
suspects["prob_of_label"] = p_label[suspect_mask]
suspects = suspects.sort_values("prob_of_label")
save_csv(suspects, "label_suspects.csv")

# same features but different label
tmp = X_train.copy()
tmp["y"] = y_train.values
grouped = tmp.groupby(list(X_train.columns), dropna=False)["y"].nunique()
conflicts = int((grouped > 1).sum())

n_suspects = int(suspect_mask.sum())
pct_suspects = round(n_suspects / len(y_train) * 100, 2)
print(f"suspect labels: {n_suspects} ({pct_suspects}%), conflicting groups: {conflicts}")

# ---------- 6. baseline model ----------
print("6. Baseline model")
model = build_model()
model.fit(X_train, y_train)
test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= 0.5).astype(int)
metrics = {
    "accuracy": round(accuracy_score(y_test, test_pred), 4),
    "majority_class_accuracy": round(max(y_test.mean(), 1 - y_test.mean()), 4),
    "f1": round(f1_score(y_test, test_pred), 4),
    "roc_auc": round(roc_auc_score(y_test, test_proba), 4),
}
print(metrics)
with open(os.path.join(OUT, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

# ---------- 7. plots ----------
print("7. Plots")
df[target].value_counts().plot(kind="bar", title="Class balance")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "class_balance.png"))
plt.close()

missing = df.isna().mean() * 100
missing[missing > 0].plot(kind="barh", title="Missing values (%)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "missing_values.png"))
plt.close()

clipper = prep.named_steps["preprocess"].named_transformers_["num"].named_steps["clip"]
age_idx = config["model_numeric"].index("age")
low, high = clipper.low_[age_idx], clipper.high_[age_idx]
plt.hist(train_df["age"], bins=40, alpha=0.5, label="before")
plt.hist(train_df["age"].clip(low, high), bins=40, alpha=0.5, label="after clipping")
plt.legend()
plt.title("Age before and after clipping")
plt.savefig(os.path.join(FIG, "age_clipping.png"))
plt.close()

# ---------- 8. hashes + reproducibility ----------
print("8. Hashes")
hashes = {
    "config.yaml": file_hash(CONFIG_PATH),
    "raw.csv": file_hash(raw_path),
    "train.csv": file_hash(train_path),
    "test.csv": file_hash(test_path),
    "X_train": array_hash(X_train_p),
    "X_test": array_hash(X_test_p),
    "y_train": array_hash(y_train.values),
    "y_test": array_hash(y_test.values),
}

manifest_path = os.path.join(OUT, "manifest.json")
old_hashes = None
if os.path.exists(manifest_path):
    with open(manifest_path) as f:
        old_hashes = json.load(f)["hashes"]

versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__, "pyyaml": yaml.__version__,
            "matplotlib": matplotlib.__version__, "joblib": joblib.__version__}
with open(manifest_path, "w") as f:
    json.dump({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "seed": SEED,
               "versions": versions, "hashes": hashes}, f, indent=2)

if old_hashes is None:
    print("first run, run again to check reproducibility")
elif old_hashes == hashes:
    print("REPRODUCIBLE: all hashes are the same as last run")
else:
    changed = [k for k in hashes if old_hashes.get(k) != hashes[k]]
    print("hashes changed:", changed)

# ---------- 9. data card ----------
sex = (df["sex"].value_counts(normalize=True) * 100).round(1).to_dict()
race = (df["race"].value_counts(normalize=True) * 100).round(1).to_dict()
classes = (df[target].value_counts(normalize=True) * 100).round(1).to_dict()

card = f"""# Data Card - UCI Adult dataset

Author: Ersultan Ismailov
Course: Machine Learning and Its Applications, Lab 4

## Dataset
- Name: Adult (Census Income), UCI Machine Learning Repository
- Loaded with: fetch_openml(name='adult', version=2, as_frame=True)
- License: CC BY 4.0
- Task: predict if a person earns more than 50K per year (binary classification)
- Rows: {len(df)} raw, {n_duplicates} duplicates removed, train {len(train_df)}, test {len(test_df)}

## Where the data comes from
The data was extracted from the 1994 US Census database by Barry Becker.
fnlwgt is a census sampling weight, not a feature of the person, so I don't use it.
The income label is self-reported.

## What's in the data
- Classes (%): {classes}
- Sex (%): {sex}
- Race (%): {race}
- Missing values: {raw_check['missing_values']} (in the original file they are marked as "?")

## Processing steps
1. Load data, strip spaces, replace "?" with NaN
2. Validate schema, data types, value ranges and labels
3. Remove exact duplicates
4. Split train/test (80/20, stratified, seed {SEED}) - before any cleaning to avoid leakage
5. New features: capital_net, log of capital gain/loss, has_capital
6. Impute missing values (median for numbers, most frequent for categories)
7. Clip outliers to {config['clip_low']}-{config['clip_high']} percentiles
8. Scale numeric columns, one-hot encode categorical columns

Steps 6-8 are fitted only on the train set (sklearn Pipeline).

## Label quality
- {n_suspects} training rows ({pct_suspects}%) look suspicious (model gives their label probability < {config['suspect_threshold']}). They are saved in label_suspects.csv for checking, I did not delete them.
- {conflicts} groups of rows have the same features but different labels.

## Baseline result
Logistic regression on the test set: accuracy {metrics['accuracy']}, F1 {metrics['f1']}, ROC-AUC {metrics['roc_auc']} (always predicting the majority class gives {metrics['majority_class_accuracy']}).

## Limitations and bias
- The data is from 1994, so the 50K threshold is in 1994 dollars
- Only US data
- More men than women and most people are White, so a model can be biased
- relationship (Husband/Wife) basically contains sex, so removing sex alone is not enough
- Classes are imbalanced, so accuracy alone is not a good metric

## Reproducibility
Run: python lab04/pipeline.py
Seed: {SEED}
Versions: {versions}
SHA256 hashes are saved in outputs/manifest.json
"""
with open(os.path.join(BASE_DIR, "DATA_CARD.md"), "w") as f:
    f.write(card)

print("Done! Results are in lab04/outputs, data card in lab04/DATA_CARD.md")