"""Export our exact train/test split for the optional Kumo comparison (not part of the graded protocol).

Writes
  data/kumo/kumo_train.csv  - 3,680 training e-mails: row_id + 57 features + spam label
  data/kumo/kumo_test.csv   - 921 test e-mails: row_id + 57 features, NO label (so Kumo cannot see the answers)

row_id is the row number in data/spambase.csv, so Kumo's predictions can be joined back to our test set.
Kumo's predictions should be saved as data/kumo/kumo_predictions.csv with columns
  row_id, predicted_spam (0/1) and, if available, prob_spam (0-1).
Run from the project root:  python extras/kumo/export_for_kumo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from utils import DATA, load_spambase, get_split

X, y = load_spambase()
X_train, X_test, y_train, y_test = get_split(X, y)          # the same split as Tasks 1, 2, 3, 5

out = DATA / "kumo"
out.mkdir(exist_ok=True)
train = X_train.copy()
train.insert(0, "row_id", X_train.index)
train["spam"] = y_train.values
test = X_test.copy()
test.insert(0, "row_id", X_test.index)                       # label deliberately left out

train.to_csv(out / "kumo_train.csv", index=False)
test.to_csv(out / "kumo_test.csv", index=False)
print(f"train: {train.shape} spam rate {train.spam.mean():.4f} -> {out / 'kumo_train.csv'}")
print(f"test:  {test.shape} (no label) -> {out / 'kumo_test.csv'}")
