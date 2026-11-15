"""Download the two UCI datasets used in the project into data/.

Spambase (UCI #94)            -> data/spambase.csv
Wholesale Customers (UCI #292) -> data/wholesale.csv

Uses the `ucimlrepo` package; falls back to the raw UCI zip files.
Run from the project root:  python src/download_data.py
"""
import io
import zipfile
import urllib.request
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)

SPAM_ZIP = "https://archive.ics.uci.edu/static/public/94/spambase.zip"
WHOLE_ZIP = "https://archive.ics.uci.edu/static/public/292/wholesale+customers.zip"


def _zip_member(url, suffix):
    with urllib.request.urlopen(url, timeout=60) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    name = next(n for n in z.namelist() if n.endswith(suffix))
    return z.read(name).decode("utf-8")


def get_spambase():
    out = DATA / "spambase.csv"
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=94)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        df = df.rename(columns={df.columns[-1]: "spam"})
    except Exception as e:  # fallback: raw zip (no header in spambase.data)
        print("ucimlrepo failed, using zip:", e)
        names_txt = _zip_member(SPAM_ZIP, "spambase.names")
        cols = [ln.split(":")[0].strip() for ln in names_txt.splitlines()
                if ":" in ln and ("continuous" in ln) and not ln.startswith("|")]
        df = pd.read_csv(io.StringIO(_zip_member(SPAM_ZIP, "spambase.data")),
                         header=None, names=cols + ["spam"])
    assert df.shape == (4601, 58), df.shape
    df.to_csv(out, index=False)
    print("saved", out, df.shape)


def get_wholesale():
    out = DATA / "wholesale.csv"
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=292)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1) \
            if ds.data.targets is not None else ds.data.features
        if "Channel" not in df.columns:
            df = ds.data.original
    except Exception as e:
        print("ucimlrepo failed, using zip:", e)
        df = pd.read_csv(io.StringIO(_zip_member(WHOLE_ZIP, ".csv")))
    cols = ["Channel", "Region", "Fresh", "Milk", "Grocery", "Frozen",
            "Detergents_Paper", "Delicassen"]
    df = df[cols]
    assert df.shape == (440, 8), df.shape
    df.to_csv(out, index=False)
    print("saved", out, df.shape)


if __name__ == "__main__":
    get_spambase()
    get_wholesale()
