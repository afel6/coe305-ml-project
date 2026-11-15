"""Shared helpers for the COE305 project: data loading, the single split,
metrics, and figure/table saving.  All randomness is seeded with SEED=42."""
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, TAB = ROOT / "data", ROOT / "figures", ROOT / "tables"
SPEND = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]


def set_seeds(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


CHAR_NAMES = {";": "semicolon", "(": "paren", "[": "bracket", "!": "excl", "$": "dollar", "#": "hash"}


def load_spambase():
    """Spambase features + label. The six char_freq_<symbol> columns are renamed
    (e.g. char_freq_[ -> char_freq_bracket) because XGBoost rejects '[', ']' and '<' in names."""
    df = pd.read_csv(DATA / "spambase.csv")
    df.columns = [f"char_freq_{CHAR_NAMES[c[-1]]}" if c.startswith("char_freq_") and c[-1] in CHAR_NAMES else c
                  for c in df.columns]
    return df.drop(columns="spam"), df["spam"].astype(int)


def load_wholesale():
    return pd.read_csv(DATA / "wholesale.csv")


def get_split(X, y):
    """The ONE stratified train/test split used for every classification task."""
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def skf(k=5):
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)


def clf_metrics(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)}


def save_fig(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=150, bbox_inches="tight")


def save_table(df, name, index=False):
    TAB.mkdir(exist_ok=True)
    df.to_csv(TAB / f"{name}.csv", index=index)
    return df
