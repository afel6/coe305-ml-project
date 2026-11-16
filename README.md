# COE305 Machine Learning — Practical Project (Fall 2026–2027)

Students: Mohammed Bendardf (229912053), Abdalraouf Alkabir (2309015482)

## For graders

- Install (Python 3.12 virtual environment, see [Setup and run](#setup-and-run)): `python -m pip install -r requirements.txt -c constraints.txt`
- Run: `python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebook.ipynb`
- Measured runtime of the whole notebook on a laptop CPU: 598 s (Windows, Python 3.12), 626 s (Ubuntu on WSL, Python 3.12), 672 s (Windows, Python 3.13, latest packages); 17–26 min when two or three runs share the machine.
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
   - Windows: `py -3.12 -m venv .venv`
   - Linux/macOS: `python3.12 -m venv .venv` (`python3` works too when `python3 --version` shows 3.12 or 3.13;
     `python` may be missing or point to another version; on Debian/Ubuntu `venv` needs the `python3-venv` package)
2. Install the pinned packages (`constraints.txt` fixes the version of every dependency):
   - Windows: `.venv\Scripts\python.exe -m pip install -r requirements.txt -c constraints.txt`
   - Linux/macOS: `.venv/bin/python -m pip install -r requirements.txt -c constraints.txt`
3. Execute the notebook in place:
   - Windows: `.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebook.ipynb`
   - Linux/macOS: `.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebook.ipynb`
4. Optional: rebuild the PDF with `.venv\Scripts\python.exe report/build_pdf.py` (Windows) or
   `.venv/bin/python report/build_pdf.py` (needs Microsoft Edge or Chrome).

No Python 3.12 installed? [uv](https://docs.astral.sh/uv/) provides one: replace steps 1–2 with
`uv venv --python 3.12` and `uv pip install -r requirements.txt -c constraints.txt`, then run step 3 as above.

**If virtual-environment launchers are blocked** (some managed Windows machines block `.venv\Scripts\python.exe`
and `jupyter.exe` through Application Control), install the requirements into a standalone Python and call it
directly, e.g. with uv: `uv pip install --python <path-to-python.exe> --system -r requirements.txt -c constraints.txt`, then run
`<path-to-python.exe> -m jupyter nbconvert ...` as above.

The run succeeded if nbconvert ends with `Writing ... bytes to notebook.ipynb` and exits with code 0. The console
also shows harmless warnings (zmq event loop, unencrypted local kernel). On Windows it may show joblib `KeyError`
tracebacks at kernel shutdown: they are harmless, nbconvert still exits with code 0 and they are not in the notebook.

## Data

`data/spambase.csv` (UCI #94) and `data/wholesale.csv` (UCI #292) are committed. If one is missing, the notebook
fetches it itself: from the UCI repository, then from OpenML, verified by SHA-256 against the committed CSV and saved
to `data/`. Each download attempt is limited to 3 minutes; if every source fails, the cell stops with one line that
names the manual download.

**Notebook alone.** `notebook.ipynb` also runs by itself: put it in an empty folder and run step 3 there with the
interpreter of an environment set up as above. It fetches the data (internet needed) and writes `data/`, `figures/`
and `tables/` next to it.

## Reproducibility

Every random component uses `random_state=42` (`np.random.seed(42)` is also set). With the pinned packages,
`tables/summary_key_results.csv` (the headline numbers) is identical on Windows with Python 3.12 and 3.13, on Linux
(Ubuntu) with Python 3.12 and with the uv route. The other tables reproduce exactly only on the same platform: on
Linux, `t5_search_gradient_boosting.csv` (up to 10% relative on one entry), the `cv_std_acc` column of
`t5_best_params.csv` (2.2%), `x2_auc.csv` (about 1e-4) and `x2_cost_threshold.csv` (3.1%) differ slightly, and
wall-clock timing columns differ on every run. The committed outputs and the report's numbers come from the Windows
Python 3.12 run.

CI (`.github/workflows/ci.yml`) executes the notebook on Ubuntu and Windows with Python 3.12 and 3.13 and checks the
summary table.

## Layout

```
notebook.ipynb          all tasks; setup and data loading in its first code cells
data/                   spambase.csv, wholesale.csv
figures/  tables/       outputs written by the notebook
report/                 report.md -> report.html, report.pdf (build_pdf.py)
requirements.txt        direct dependencies, pinned
constraints.txt         every dependency, pinned (used with -c)
.python-version         3.12
.github/workflows/      CI
AGENTS.md               the "For graders" block, for coding agents
```

## Protocol
- One stratified train/test split (`test_size=0.2`, `random_state=42`) for every classification task (1, 2, 3, 5).
- All model/parameter selection by stratified 5-fold CV on the training set; the test set is used only for final scores.
- Scaling, feature selection and PCA are steps inside `sklearn` `Pipeline`s, so they are fitted on training folds only.
- K-Means (Task 4) uses only the six spending features; `Channel`/`Region` are dropped (Channel is used once, post hoc, for interpretation).
