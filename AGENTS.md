# AGENTS.md — COE305 Machine Learning practical project

## For graders

- Install (Python 3.12 virtual environment, see "Setup and run" in README.md): `python -m pip install -r requirements.txt -c constraints.txt`
- Run: `python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebook.ipynb`
- Measured runtime of the whole notebook on a laptop CPU: 598 s (Windows, Python 3.12), 626 s (Ubuntu on WSL, Python 3.12), 672 s (Windows, Python 3.13, latest packages); 17–26 min when two or three runs share the machine.
- No run is needed to see the results: `notebook.ipynb` is committed executed, with every output; `figures/` and `tables/` hold every figure and table it writes; the report is `report/report.pdf`.

README.md has the full setup (Windows, Linux/macOS, uv), the data sources, the notebook-only route and what
reproduces across platforms.
