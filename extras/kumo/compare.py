"""Optional extra (NOT part of the graded work): score KumoTabular's predictions against our best Task 5 model.

Uses only scikit-learn / XGBoost / pandas / NumPy / Matplotlib. Inputs:
  data/kumo/kumo_predictions.csv     written by extras/kumo/kumo_predict.py
  tables/t5_best_params.csv          the CV-selected XGBoost setting from Task 5
  tables/t5_test_metrics.csv         used to check that the re-fitted XGBoost matches the notebook exactly
  tables/x2_cost_threshold.csv       XGBoost's cost-optimal threshold (chosen in the notebook on training OOF)
Outputs go to extras/kumo/results/ so the graded figures/ and tables/ folders are untouched.
Run from the project root:  python extras/kumo/compare.py
"""
import ast
import sys
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from utils import SEED, clf_metrics, get_split, load_spambase  # noqa: E402

OUT = ROOT / "extras" / "kumo" / "results"
OUT.mkdir(parents=True, exist_ok=True)
np.random.seed(SEED)

X, y = load_spambase()
X_train, X_test, y_train, y_test = get_split(X, y)
yt = y_test.values

# Best Task 5 model, re-fitted exactly as GridSearchCV refits it in the notebook.
best = pd.read_csv(ROOT / "tables" / "t5_best_params.csv").set_index("model").loc["XGBoost"]
params = ast.literal_eval(best.best_params)
xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                    random_state=SEED, n_jobs=1, **params).fit(X_train, y_train)
p_x, pr_x = xgb.predict(X_test), xgb.predict_proba(X_test)[:, 1]
nb_acc = pd.read_csv(ROOT / "tables" / "t5_test_metrics.csv").set_index("model").loc["XGBoost", "accuracy"]
assert abs(clf_metrics(y_test, p_x)["accuracy"] - nb_acc) < 1e-12, "re-fitted XGBoost does not match the notebook"

kumo = pd.read_csv(ROOT / "data" / "kumo" / "kumo_predictions.csv").set_index("row_id")
assert set(kumo.index) == set(X_test.index), "predictions must cover exactly our test e-mails"
kumo = kumo.loc[X_test.index]
p_k, pr_k = kumo.predicted_spam.values, kumo.prob_spam.values

# Same paired bootstrap as notebook X1 (same seed, 2000 resamples of the 921 test e-mails).
rng = np.random.default_rng(SEED)
idx = rng.integers(0, len(yt), size=(2000, len(yt)))

def boot(p):
    P, T = p[idx], yt[idx]
    tp = ((P == 1) & (T == 1)).sum(1); fp = ((P == 1) & (T == 0)).sum(1); fn = ((P == 0) & (T == 1)).sum(1)
    return {"accuracy": (P == T).mean(1), "f1": 2 * tp / (2 * tp + fp + fn)}

def mcnemar_exact(b, c):
    n, k = b + c, min(b, c)
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

B = {"XGBoost": boot(p_x), "KumoTabular": boot(p_k)}
rows = []
for n, p, pr in [("XGBoost", p_x, pr_x), ("KumoTabular", p_k, pr_k)]:
    m = clf_metrics(y_test, p)
    rows.append({"model": n, **{k: m[k] for k in ["accuracy", "precision", "recall", "f1"]},
                 "accuracy_ci_low": np.percentile(B[n]["accuracy"], 2.5), "accuracy_ci_high": np.percentile(B[n]["accuracy"], 97.5),
                 "f1_ci_low": np.percentile(B[n]["f1"], 2.5), "f1_ci_high": np.percentile(B[n]["f1"], 97.5),
                 "roc_auc": roc_auc_score(y_test, pr), "average_precision": average_precision_score(y_test, pr)})
metrics = pd.DataFrame(rows)

d = B["KumoTabular"]["accuracy"] - B["XGBoost"]["accuracy"]
rk, rx = p_k == yt, p_x == yt
b, c = int((rk & ~rx).sum()), int((~rk & rx).sum())
paired = pd.DataFrame([{"comparison": "KumoTabular - XGBoost", "accuracy_diff": rk.mean() - rx.mean(),
                        "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5),
                        "share_of_resamples_kumo_wins": (d > 0).mean(), "kumo_right_xgb_wrong": b,
                        "kumo_wrong_xgb_right": c, "both_wrong": int((~rk & ~rx).sum()), "mcnemar_p_exact": mcnemar_exact(b, c)}])

cost_tab = pd.read_csv(ROOT / "tables" / "x2_cost_threshold.csv")
t_x = cost_tab.query("model == 'XGBoost' and threshold_rule != 'default 0.5'").threshold.item()
rows = []
for n, pr, t, rule in [("XGBoost", pr_x, 0.5, "default 0.5"), ("XGBoost", pr_x, t_x, "cost-optimal (chosen on training OOF)"),
                       ("KumoTabular", pr_k, 0.5, "default 0.5"), ("KumoTabular", pr_k, t_x, "borrowed from XGBoost (not tuned for Kumo)")]:
    pp = (pr >= t).astype(int)
    fp, fn = int(((pp == 1) & (yt == 0)).sum()), int(((pp == 0) & (yt == 1)).sum())
    rows.append({"model": n, "threshold_rule": rule, "threshold": t, "test_FP": fp, "test_FN": fn,
                 "test_cost_per_1000_emails": 1000 * (10 * fp + fn) / len(yt), "test_accuracy": (pp == yt).mean()})
cost = pd.DataFrame(rows)

for name, df in [("kumo_vs_xgboost_metrics", metrics), ("kumo_paired_test", paired), ("kumo_cost", cost)]:
    df.to_csv(OUT / f"{name}.csv", index=False)
    print(f"\n== {name}\n{df.round(4).to_string(index=False)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
for i, (n, pr) in enumerate([("XGBoost", pr_x), ("KumoTabular", pr_k)]):
    col = ["#2a78d6", "#eb6834"][i]
    fpr, tpr, _ = roc_curve(y_test, pr); prec, rec, _ = precision_recall_curve(y_test, pr)
    axes[0].plot(fpr, tpr, color=col, lw=1.8, label=f"{n} (AUC {roc_auc_score(y_test, pr):.4f})")
    axes[1].plot(rec, prec, color=col, lw=1.8, label=f"{n} (AP {average_precision_score(y_test, pr):.4f})")
axes[0].plot([0, 1], [0, 1], ls=":", color="#52514e", lw=1)
axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC on the test set", xlim=(0, 0.3), ylim=(0.7, 1.005))
axes[1].set(xlabel="recall", ylabel="precision", title="Precision-recall on the test set", xlim=(0.7, 1.005), ylim=(0.7, 1.005))
for ax in axes:
    ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
fig.suptitle("Optional extra (not graded): KumoTabular vs best Task 5 model (XGBoost)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "kumo_roc_pr.png", dpi=150)
print("\nwrote", OUT)
