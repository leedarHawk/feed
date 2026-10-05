"""Price store for the trading system: the published yearly files plus validated daily increments.

* Base   : data/stock_price_v1/stock_price_<year>.parquet (read-only, checksummed in CHECKSUMS.json)
* Daily  : <home>/data/stock_price/stock_price_<yyyymmdd>.parquet, written only by `ingest_prices`
* Tables : <home>/data/reference/<name>.parquet overrides the same table in data/research_data_v1_20261002
* Cache  : <home>/cache/<fingerprint>/ holds the [date x code] matrices in research/panel.py's format, so the
           research loaders run unchanged on the combined data. A new increment changes the fingerprint.

qfq prices in an increment are not trusted (their anchor date differs from the base files). They are rebuilt
by chain-linking: qfq_x[t] = x[t] * qfq_close[t-1] / pre_close[t], which is exact because pre_close is the
exchange's ex-rights reference price (checked on 2019-2026: close/pre_close matches the qfq return to a
median 8e-5, the rounding of qfq prices).
"""
import datetime as dt
import glob
import hashlib
import os
import shutil

import numpy as np
import polars as pl

from .bridge import DATA, activate, research
from .config import HOME

BASE_PRICE = DATA / "stock_price_v1"
BASE_REF = DATA / "research_data_v1_20261002"
REF_TABLES = ("listing", "st_daily_intervals", "dividends", "industry_intervals", "suspensions", "status_intervals",
              "st_events")
RAW_FIELDS = ["open", "high", "low", "close", "pre_close", "high_limit", "low_limit", "volume", "money",
              "market_cap", "circulating_market_cap", "circulating_cap", "capitalization", "roe", "roa"]
QFQ = {"qfq_open": "open", "qfq_high": "high", "qfq_low": "low", "qfq_close": "close"}


class DataError(RuntimeError):
    pass


class Store:
    def __init__(self, home=HOME):
        self.home = home
        self.inc_dir = home / "data" / "stock_price"
        self.ref_dir = home / "data" / "reference"
        self.cache_root = home / "cache"

    # ------------------------------------------------------------------ files
    def base_files(self):
        return sorted(glob.glob(str(BASE_PRICE / "stock_price_*.parquet")))

    def increment_files(self):
        return sorted(glob.glob(str(self.inc_dir / "stock_price_*.parquet")))

    def price_files(self):
        return self.base_files() + self.increment_files()

    def fingerprint(self):
        h = hashlib.sha1()
        for f in self.price_files():
            st = os.stat(f)
            h.update(f"{os.path.basename(f)}|{st.st_size}|{st.st_mtime_ns}".encode())
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------ cache
    def ensure_cache(self, log=print):
        fp = self.fingerprint()
        cdir = self.cache_root / fp
        if not (cdir / "meta.npz").exists():
            log(f"building price cache {fp} from {len(self.price_files())} files ...")
            tmp = self.cache_root / f".{fp}.tmp"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            fields = research().panel.FIELDS
            df = pl.concat([pl.read_parquet(f, columns=["code", "time"] + fields) for f in self.price_files()],
                           how="vertical_relaxed")
            df = df.with_columns(pl.col("time").cast(pl.Utf8).str.slice(0, 10).str.to_date().alias("date"))
            dates = np.array(sorted(df["date"].unique().to_list()), dtype="datetime64[D]")
            codes = np.array(sorted(df["code"].unique().to_list()))
            row = np.searchsorted(dates, df["date"].to_numpy().astype("datetime64[D]"))
            ci = {c: i for i, c in enumerate(codes)}
            col = np.array([ci[c] for c in df["code"].to_list()])
            for k in fields:
                a = np.full((len(dates), len(codes)), np.nan)
                a[row, col] = df[k].cast(pl.Float64).to_numpy()
                np.save(tmp / f"{k}.npy", a)
            np.savez(tmp / "meta.npz", dates=dates, codes=codes)
            del df
            shutil.rmtree(cdir, ignore_errors=True)
            tmp.rename(cdir)
            for old in self.cache_root.iterdir():          # keep only the current cache (each is ~1.5 GB)
                if old.name != fp and not old.name.startswith("."):
                    shutil.rmtree(old, ignore_errors=True)
        return cdir

    def reference_dir(self):
        """Directory with every reference table, live overrides taking precedence over the published ones."""
        out = self.cache_root / "reference"
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        for name in REF_TABLES:
            src = self.ref_dir / f"{name}.parquet"
            if not src.exists():
                src = BASE_REF / f"{name}.parquet"
            if src.exists():
                shutil.copy2(src, out / f"{name}.parquet")     # copies, not links: miniQMT hosts run Windows
        return out

    def activate(self, log=print):
        return activate(self.ensure_cache(log), self.reference_dir())

    def last_date(self):
        f = self.price_files()[-1]
        t = pl.read_parquet(f, columns=["time"])["time"].cast(pl.Utf8).str.slice(0, 10).max()
        return dt.date.fromisoformat(t)

    def stock_names(self, day):
        """code -> stock_name on `day` (names carry the ST / *ST / 退 markers)."""
        day = str(day)[:10]
        for f in reversed(self.price_files()):
            df = pl.read_parquet(f, columns=["code", "time", "stock_name"])
            df = df.filter(pl.col("time").cast(pl.Utf8).str.slice(0, 10) == day)
            if df.height:
                return dict(zip(df["code"].to_list(), df["stock_name"].to_list()))
        return {}

    # ------------------------------------------------------------------ ingestion
    def ingest_prices(self, path, cal=None, log=print):
        """Validate one trading day of prices (same columns as stock_price_v1) and add it to the store.
        Returns (date, warnings). Raises DataError on anything that would corrupt the panel."""
        df = pl.read_parquet(path) if str(path).endswith(".parquet") else pl.read_csv(path, try_parse_dates=False)
        if "time" not in df.columns and "date" in df.columns:
            df = df.rename({"date": "time"})
        missing = [c for c in ["code", "time"] + RAW_FIELDS if c not in df.columns]
        if missing:
            raise DataError(f"{path}: missing columns {missing}")
        df = df.with_columns(pl.col("time").cast(pl.Utf8).str.slice(0, 10).alias("time"))
        days = df["time"].unique().to_list()
        if len(days) != 1:
            raise DataError(f"{path}: one trading day per file, found {sorted(days)[:5]}")
        day = dt.date.fromisoformat(days[0])
        last = self.last_date()
        if day <= last:
            raise DataError(f"{path}: {day} is not after the last stored day {last}")
        if df["code"].n_unique() != df.height:
            raise DataError(f"{path}: duplicate codes")
        warn = []
        if cal is not None and not cal.is_open(day):
            warn.append(f"{day} is not a trading day in the calendar")
        if cal is not None and cal.next_day(last) != day:
            warn.append(f"gap: expected {cal.next_day(last)} after {last}, got {day}")
        df = df.with_columns([pl.col(c).cast(pl.Float64) for c in RAW_FIELDS])
        tr = df.filter(pl.col("close").is_not_null() & (pl.col("volume") > 0))
        bad = tr.filter((pl.col("high") < pl.max_horizontal("open", "close") - 1e-6) |
                        (pl.col("low") > pl.min_horizontal("open", "close") + 1e-6) |
                        (pl.col("close") <= 0) | (pl.col("high_limit") < pl.col("low_limit")) |
                        (pl.col("close") > pl.col("high_limit") + 0.011) | (pl.col("close") < pl.col("low_limit") - 0.011))
        if bad.height:
            raise DataError(f"{path}: {bad.height} rows with inconsistent prices, e.g. {bad['code'].head(5).to_list()}")
        # units as in stock_price_v1: caps in 1e8 CNY, share counts in 1e4 shares, volume in shares, money in CNY
        for cap, shares in (("market_cap", "capitalization"), ("circulating_market_cap", "circulating_cap")):
            ratio = tr.select((pl.col(cap) / (pl.col("close") * pl.col(shares) / 1e4)).median()).item()
            if ratio is None or not 0.95 < ratio < 1.05:
                raise DataError(f"{path}: {cap} / (close x {shares} / 1e4) has median {ratio}; expected 1 "
                                "(caps in 1e8 CNY, shares in 1e4)")
        vwap = tr.filter(pl.col("volume") > 0).select((pl.col("money") / pl.col("volume") / pl.col("close")).median()).item()
        if vwap is None or not 0.8 < vwap < 1.25:
            raise DataError(f"{path}: money / volume / close has median {vwap}; expected ~1 (volume in shares, money in CNY)")

        R = self.activate(log)
        dates, codes, P = R.panel.load()
        lib = R.lib
        prev_n = int((~np.isnan(P["close"][-1])).sum())
        if tr.height < 0.95 * prev_n:
            raise DataError(f"{path}: only {tr.height} traded names vs {prev_n} on {dates[-1]} (incomplete file?)")
        last_qc = lib.ffill(P["qfq_close"])[-1]
        last_c = lib.ffill(P["close"])[-1]
        ci = {c: i for i, c in enumerate(codes)}
        idx = np.array([ci.get(c, -1) for c in df["code"].to_list()])
        known = idx >= 0
        qprev = np.where(known, last_qc[np.maximum(idx, 0)], np.nan)
        cprev = np.where(known, last_c[np.maximum(idx, 0)], np.nan)
        pre = df["pre_close"].to_numpy()
        k = np.where(np.isfinite(qprev) & np.isfinite(pre) & (pre > 0), qprev / pre, 1.0)
        exr = np.isfinite(cprev) & np.isfinite(pre) & (np.abs(pre / cprev - 1) > 1e-3)
        if exr.mean() > 0.05:
            warn.append(f"pre_close differs from the stored close for {exr.sum()} names (ex-rights days?)")
        df = df.with_columns([(pl.col(src) * pl.Series(k)).alias(q) for q, src in QFQ.items()])
        if "stock_name" not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("stock_name"))
        cols = ["code", "time", "stock_name"] + RAW_FIELDS + list(QFQ)
        self.inc_dir.mkdir(parents=True, exist_ok=True)
        out = self.inc_dir / f"stock_price_{day:%Y%m%d}.parquet"
        df.select(cols).write_parquet(out)
        log(f"ingested {day}: {df.height} rows, {tr.height} traded, {int((~known).sum())} new codes")
        return day, warn

    def ingest_reference(self, name, path):
        if name not in REF_TABLES:
            raise DataError(f"unknown table {name}; expected one of {REF_TABLES}")
        new = pl.read_parquet(path)
        base = pl.read_parquet(BASE_REF / f"{name}.parquet", n_rows=1)
        missing = [c for c in base.columns if c not in new.columns]
        if missing:
            raise DataError(f"{path}: missing columns {missing}")
        if new.height < 0.9 * pl.read_parquet(BASE_REF / f"{name}.parquet", columns=[base.columns[0]]).height:
            raise DataError(f"{path}: fewer than 90% of the published rows; a full table is expected")
        self.ref_dir.mkdir(parents=True, exist_ok=True)
        new.select(base.columns).write_parquet(self.ref_dir / f"{name}.parquet")
