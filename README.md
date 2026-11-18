# COE305 Machine Learning — Practical Project (Fall 2026–2027)

Students: Mohammed Bendardf (229912053), Abdalraouf Alkabir (2309015482)

## For graders

- Install (Python 3.12 virtual environment, see [Setup and run](#setup-and-run)): `python -m pip install -r requirements.txt -c constraints.txt`; an environment made by `uv venv` has no pip, so there use `uv pip install -r requirements.txt -c constraints.txt`. Here and below `python` means the environment's interpreter: activate the environment first, or write `.venv\Scripts\python.exe` (Windows) / `.venv/bin/python` (Linux/macOS) instead. The pins reproduce the reported numbers; newer packages run but change some of them.
- Run: `python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebook.ipynb`. It succeeded if it ends with `Writing ... bytes to notebook.ipynb` and exit code 0; `KeyError` tracebacks from joblib's resource tracker, during the run and at kernel shutdown, are harmless. The run rewrites `notebook.ipynb` and some tracked CSVs (timing columns, last-digit rounding); `git checkout -- notebook.ipynb figures tables` restores the committed versions.
- Measured runtime of the whole notebook on a laptop CPU (AMD Ryzen AI 9 HX 370, Windows 11): 333 s with the pinned packages (Python 3.12), 348 s for the notebook alone in an empty folder including both downloads, 311 s with the newest packages (Python 3.13). Slower machines take longer.
- No run is needed to see the results: `notebook.ipynb` is committed executed, with every output; `figures/` and `tables/` hold every figure and table it writes; the report is `report/report.pdf`.

## Deliverables

- `notebook.ipynb` — all five tasks, executed with outputs
- `report/report.pdf` — the written report (source: `report/report.md`, built by `report/build_pdf.py`)
- `figures/` (PNG, 150 dpi) and `tables/` (CSV) — every figure and table produced by the notebook

## Setup and run

Python 3.12 is the reference version (`.python-version`); 3.13 also works. Python 3.14 is not supported: the pinned
numpy 2.1.3, pandas 2.2.3, scikit-learn 1.5.2 and matplotlib 3.9.2 have no Python 3.14 wheels. Run every command
from the repository root. The commands call the virtual environment's interpreter directly, so no activation is needed.

1. Create the virtual environment `.venv` with Python 3.12:
   - Windows: `py -3.12 -m venv .venv` (`py` is the Python launcher that the python.org installer adds; without it,
     use the uv route below)
   - Linux/macOS: `python3.12 -m venv .venv` (`python3` works too when `python3 --version` shows 3.12 or 3.13;
     `python` may be missing or point to another version; on Debian/Ubuntu `venv` needs the `python3-venv` package).
     macOS is untested: CI covers Ubuntu and Windows.
2. Install the pinned packages (`constraints.txt` fixes the version of every dependency):
   - Windows: `.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints.txt`
   - Linux/macOS: `.venv/bin/python -m pip install -r requirements.txt -c constraints.txt`
3. Execute the notebook in place:
   - Windows: `.venv\Scripts\python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebook.ipynb`
   - Linux/macOS: `.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebook.ipynb`
4. Optional: rebuild the PDF with `.venv\Scripts\python.exe report/build_pdf.py` (Windows) or
   `.venv/bin/python report/build_pdf.py` (needs Microsoft Edge, Google Chrome or Chromium).

No Python 3.12 installed? [uv](https://docs.astral.sh/uv/) provides one: replace steps 1–2 with
`uv venv --python 3.12` and `uv pip install -r requirements.txt -c constraints.txt`, then run step 3 as above.

**If virtual-environment launchers are blocked** (some managed Windows machines block `.venv\Scripts\python.exe`
and `jupyter-nbconvert.exe` through Application Control), install the requirements into a standalone Python and call it
directly, e.g. with uv: `uv pip install --python <path-to-python.exe> --system -r requirements.txt -c constraints.txt`, then run
`<path-to-python.exe> -m nbconvert ...` as above.

The run succeeded if nbconvert ends with `Writing ... bytes to notebook.ipynb` and exits with code 0. The console
also shows harmless warnings (zmq event loop, unencrypted local kernel). On Windows it may show `KeyError` tracebacks
from joblib's resource tracker during the run and at kernel shutdown: they are harmless, nbconvert still exits with
code 0 and they are not in the notebook.

## Data

`data/spambase.csv` (UCI #94) and `data/wholesale.csv` (UCI #292) are committed. If one is missing, the notebook
fetches it itself: from the UCI repository, then from OpenML, verified by SHA-256 against the committed CSV and saved
to `data/`. Each download attempt is limited to 3 minutes; if every source fails, the cell stops with one line that
names the manual download.

**Notebook alone.** `notebook.ipynb` also runs by itself: put it in an empty folder and run step 3 there with the
interpreter of an environment set up as above. It fetches the data (internet needed) and writes `data/`, `figures/`
and `tables/` next to it.

## Reproducibility

Every random component uses `random_state=42` (`np.random.seed(42)` is also set). Other package versions run too
(tested with Python 3.13, scikit-learn 1.9.1, pandas 3.1.1 and XGBoost 3.4.1; the first code cell then prints a version
warning), but they change some results, for example Gradient Boosting then scores higher than XGBoost on the test set (0.9490 vs 0.9457): install with
`-c constraints.txt` to reproduce the reported numbers. With the pinned packages,
`tables/summary_key_results.csv` (the headline numbers) is identical on Windows and Linux
(Ubuntu) with Python 3.12 and 3.13 (CI) and with the uv route. The other tables agree up to floating-point rounding, except wall-clock
timing columns, which differ on every run, and four tables that differ slightly on Linux: `t5_search_gradient_boosting.csv`
(up to 10% relative on one entry), the `cv_std_acc` column of `t5_best_params.csv` (2.2%), `x2_auc.csv` (about 1e-4) and
`x2_cost_threshold.csv` (3.1%). The 12 tables written by notebook section R (`r_*.csv`, `t3_fs_pca_compressed.csv`, `t4_chosen_k_raw_vs_scaled.csv`, `t5_train_vs_test.csv`, `x3_mailbox_words_removed.csv`), `t4_channel_agreement_by_space.csv`, `x2_errors_by_probability_band.csv`, `x4_bootstrap_stability_capped.csv`, `x2_error_analysis.csv` and `t4_cluster_profiles.csv` have not been compared on Linux. The committed outputs and the report's results, apart from its reproducibility comparisons, come from one Windows run
with Python 3.12.

CI (`.github/workflows/ci.yml`) executes the notebook on Ubuntu and Windows with Python 3.12 and 3.13 and checks the
summary table.

## Layout

```
notebook.ipynb          all tasks; setup and data loading in its first code cells
data/                   spambase.csv, wholesale.csv
figures/  tables/       outputs written by the notebook
report/                 report.md -> report.html, report.pdf (build_pdf.py)
requirements.txt        direct dependencies, pinned (ucimlrepo is not imported by the notebook; it stays so that
                        the tested pin set is unchanged)
constraints.txt         every dependency, pinned (used with -c)
.python-version         3.12
.github/workflows/      CI
AGENTS.md               the "For graders" block, for coding agents
```

## Protocol
- One stratified train/test split (`test_size=0.2`, `random_state=42`) for every classification task (1, 2, 3, 5).
- All model/parameter selection by stratified 5-fold CV on the training set; no choice is ever made on the test set: it gives the final scores, and its other uses, such as the report's test curves (Figures 12 and 14) and Appendix A, are for illustration and analysis only.
- Scaling, feature selection and PCA are steps inside `sklearn` `Pipeline`s, so they are fitted on training folds only.
- K-Means (Task 4) uses only the six spending features; `Channel`/`Region` are dropped (Channel is used only after clustering, post hoc, for interpretation).
