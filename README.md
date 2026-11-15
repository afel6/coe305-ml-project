# COE305 Machine Learning — Practical Project (Fall 2026–2027)

Students: Mohammed Bendardf (229912053), Abdalraouf Alkabir (2309015482)

Deliverables
- `notebook.ipynb` — all five tasks, executed with outputs
- `report/report.pdf` — the written report (source: `report/report.md`, built by `report/build_pdf.py`)
- `figures/` (PNG, 150 dpi) and `tables/` (CSV) — every figure/table produced by the notebook

## Reproduce

On the authors' Windows machine Python is not on PATH and Windows Application Control blocks virtual-environment
launchers (`.venv\Scripts\python.exe`, `jupyter.exe`). The working route is the uv-managed Python, called directly:

```bash
uv pip install --python "C:\Users\benda\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe" --system --break-system-packages -r requirements.txt
& "C:\Users\benda\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebook.ipynb
& "C:\Users\benda\AppData\Roaming\uv\python\cpython-3.12.14-windows-x86_64-none\python.exe" report\build_pdf.py
```

The run prints harmless warnings (zmq event loop, unencrypted local kernel, joblib `KeyError` cleanup tracebacks);
it succeeded if it ends with `Writing ... bytes to notebook.ipynb`.

On a fresh machine with Python 3.12 installed:

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py       # downloads Spambase (UCI #94) and Wholesale Customers (UCI #292) into data/
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebook.ipynb
python report/build_pdf.py        # optional: rebuild the PDF (needs Microsoft Edge or Chrome)
```

The full notebook takes a few minutes on a laptop CPU. Every random component uses `random_state=42`
(`np.random.seed(42)` is also set), so a second run reproduces the same numbers
(`tables/summary_key_results.csv`).

## Layout

```
notebook.ipynb          all tasks
src/download_data.py    data download (ucimlrepo, falls back to the UCI zip files)
src/utils.py            seeds, data loading, the single stratified split, metrics, save helpers
extras/kumo/            optional, NOT graded: KumoTabular vs our XGBoost (own scripts, pinned
                        requirements-kumo.txt and README); nothing in the notebook or report depends on it
data/                   spambase.csv, wholesale.csv
figures/  tables/       outputs
report/                 report.md -> report.pdf
requirements.txt        pinned versions
```

## Protocol
- One stratified train/test split (`test_size=0.2`, `random_state=42`) for every classification task (1, 2, 3, 5).
- All model/parameter selection by stratified 5-fold CV on the training set; the test set is used only for final scores.
- Scaling, feature selection and PCA are steps inside `sklearn` `Pipeline`s, so they are fitted on training folds only.
- K-Means (Task 4) uses only the six spending features; `Channel`/`Region` are dropped (Channel is used once, post hoc, for interpretation).
