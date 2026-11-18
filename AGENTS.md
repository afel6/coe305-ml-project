# AGENTS.md — COE305 Machine Learning practical project

## For graders

- Install (Python 3.12 virtual environment, see "Setup and run" in README.md): `python -m pip install -r requirements.txt -c constraints.txt`; an environment made by `uv venv` has no pip, so there use `uv pip install -r requirements.txt -c constraints.txt`. Here and below `python` means the environment's interpreter: activate the environment first, or write `.venv\Scripts\python.exe` (Windows) / `.venv/bin/python` (Linux/macOS) instead. The pins reproduce the reported numbers; newer packages run but change some of them.
- Run: `python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebook.ipynb`. It succeeded if it ends with `Writing ... bytes to notebook.ipynb` and exit code 0; `KeyError` tracebacks from joblib's resource tracker, during the run and at kernel shutdown, are harmless. The run rewrites `notebook.ipynb` and some tracked CSVs (timing columns, last-digit rounding); `git checkout -- notebook.ipynb figures tables` restores the committed versions.
- Measured runtime of the whole notebook on a laptop CPU (AMD Ryzen AI 9 HX 370, Windows 11): 333 s with the pinned packages (Python 3.12), 348 s for the notebook alone in an empty folder including both downloads, 311 s with the newest packages (Python 3.13). Slower machines take longer.
- No run is needed to see the results: `notebook.ipynb` is committed executed, with every output; `figures/` and `tables/` hold every figure and table it writes; the report is `report/report.pdf`.

README.md has the full setup (Windows, Linux/macOS, uv), the data sources, the notebook-only route and what
reproduces across platforms.
