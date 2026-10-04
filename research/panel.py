"""Build date x code matrices from the 2019-2024 price files.

Only files up to 2024 are ever read: 2025-2026 prices are a sealed test set
(see data/research_data_v1_20261002/RESEARCH_USE_NOTES.txt).
"""
import glob, os
import numpy as np
import polars as pl

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
CACHE = os.environ.get("FEED_CACHE", "/tmp/feed_cache")
# research runs stop at 2024; only the pre-registered final test sets FEED_LAST_YEAR=2026
LAST_ALLOWED_YEAR = int(os.environ.get("FEED_LAST_YEAR", "2024"))

FIELDS = ["open", "high", "low", "close", "pre_close", "high_limit", "low_limit",
          "volume", "money", "qfq_open", "qfq_high", "qfq_low", "qfq_close",
          "market_cap", "circulating_market_cap", "circulating_cap", "capitalization",
          "roe", "roa"]


def _price_files():
    files = []
    for f in sorted(glob.glob(os.path.join(DATA, "stock_price_v1", "stock_price_*.parquet"))):
        year = int(os.path.basename(f).split("_")[-1].split(".")[0])
        if year <= LAST_ALLOWED_YEAR:
            files.append(f)
    return files


def load():
    """Return (dates ndarray[datetime64[D]], codes ndarray[str], dict field -> float64 [T, N])."""
    os.makedirs(CACHE, exist_ok=True)
    meta = os.path.join(CACHE, "meta.npz")
    if os.path.exists(meta):
        m = np.load(meta, allow_pickle=True)
        dates, codes = m["dates"], m["codes"]
        return dates, codes, {k: np.load(os.path.join(CACHE, f"{k}.npy")) for k in FIELDS}
    df = pl.concat([pl.read_parquet(f, columns=["code", "time"] + FIELDS) for f in _price_files()])
    df = df.with_columns(pl.col("time").str.slice(0, 10).str.to_date().alias("date"))
    dates = np.array(sorted(df["date"].unique().to_list()), dtype="datetime64[D]")
    codes = np.array(sorted(df["code"].unique().to_list()))
    di = {d: i for i, d in enumerate(dates)}
    ci = {c: i for i, c in enumerate(codes)}
    r = df["date"].to_numpy().astype("datetime64[D]")
    row = np.searchsorted(dates, r)
    col = np.array([ci[c] for c in df["code"].to_list()])
    out = {}
    for k in FIELDS:
        a = np.full((len(dates), len(codes)), np.nan)
        a[row, col] = df[k].cast(pl.Float64).to_numpy()
        out[k] = a
        np.save(os.path.join(CACHE, f"{k}.npy"), a)
    np.savez(meta, dates=dates, codes=codes)
    return dates, codes, out
