"""Optional extra (NOT part of the graded protocol): KumoTabular predictions on our exact Spambase split.

KumoTabular (package `structured-data-models`, imported as `sdm`) is a pretrained tabular foundation model.
It does no gradient training on our data: the labelled training rows are passed in as "context" and it predicts
the query rows in one forward pass (in-context learning). It runs locally; no Kumo API is called.

It needs PyTorch, which the graded work does not use. This script therefore lives in extras/ and only writes a CSV;
extras/kumo/compare.py scores that CSV with scikit-learn. Nothing in notebook.ipynb or the report depends on it.

Protocol
  context = the 3,680 training e-mails (features + label)     -> same stratified split, random_state=42
  query   = the 921 test e-mails (features only)              -> labels are never passed to the model
  Known difference from our rule 3: train and test features are encoded together in one TableTensor
  (as in the package's own example), so any per-column normalisation inside sdm sees the test features
  (never the test labels). This is reported in the comparison section.

Output: data/kumo/kumo_predictions.csv with columns row_id, predicted_spam (0/1), prob_spam (0-1)

Setup (separately from requirements.txt):  pip install -r extras/kumo/requirements-kumo.txt
Run from the project root:                  python extras/kumo/kumo_predict.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import sdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from utils import DATA, SEED, load_spambase, get_split

torch.manual_seed(SEED)
np.random.seed(SEED)

X, y = load_spambase()
X_train, X_test, y_train, y_test = get_split(X, y)            # the same split as Tasks 1, 2, 3, 5

n_train = len(X_train)
df = pd.concat([X_train, X_test])
# Placeholder label for the test rows only so the column exists; the query below drops this column,
# so the model never sees it.
df["spam"] = np.concatenate([y_train.values, np.zeros(len(X_test), dtype=int)])
df = df.reset_index(drop=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
table = sdm.TableTensor.from_pandas(
    df=df,
    stypes=sdm.infer_stypes(df, overrides={"spam": "categorical"}),
    device=device,
)
model = sdm.models.KumoTabular(task="classification", device=device)

t0 = time.perf_counter()
with torch.no_grad(), torch.amp.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
    probs = model(
        x_context=table[:n_train].drop_columns("spam"),
        y_context=table[:n_train, "spam"],
        x_query=table[n_train:].drop_columns("spam"),
        num_estimators=8,
    )
elapsed = time.perf_counter() - t0

# The model returns a TableTensor whose numerical columns are named after the classes ("0", "1").
scores = probs.numerical.float().cpu()
names = [n for k, ns in probs.columns.items() if "numerical" in str(k) for n in ns]
print("output columns:", names, "| shape:", tuple(scores.shape))
# Depending on the ensemble path the scores are probabilities or log-probabilities; normalise either way.
is_prob = bool((scores >= 0).all() and torch.allclose(scores.sum(-1), torch.ones(len(scores)), atol=1e-3))
p = (scores if is_prob else scores.softmax(dim=-1)).numpy()
print("scores were", "probabilities" if is_prob else "log-probabilities/logits -> softmax applied")
prob_spam = p[:, names.index("1")]                              # column "1" = spam
assert len(prob_spam) == len(X_test), "unexpected number of predictions"

out = pd.DataFrame({
    "row_id": X_test.index,
    "predicted_spam": (prob_spam >= 0.5).astype(int),
    "prob_spam": prob_spam,
})
path = DATA / "kumo" / "kumo_predictions.csv"
path.parent.mkdir(exist_ok=True)
out.to_csv(path, index=False)
print(f"device={device} | inference {elapsed:.1f}s | predicted spam rate {out.predicted_spam.mean():.4f} -> {path}")
