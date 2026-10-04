"""Daily market-sentiment indicators from price data (A-share specific: limit-up / limit-down mechanics).
All values for date t use data up to the close of t."""
import numpy as np, polars as pl
import panel


def build(start_year=2019):
    dates, codes, P = panel.load()
    c, h, pc = P["close"], P["high"], P["pre_close"]
    hl, ll = P["high_limit"], P["low_limit"]
    v = ~np.isnan(c)
    with np.errstate(invalid="ignore", divide="ignore"):
        normal = (hl / pc < 1.25) & (ll / pc > 0.75)                   # skip IPO / delisting no-limit days
        r = np.where(v, c / pc - 1, np.nan)
    up = v & normal & np.isclose(c, hl, atol=0.005)
    dn = v & normal & np.isclose(c, ll, atol=0.005)
    touched = v & normal & (h >= hl - 0.005)
    broke = touched & ~up                                               # touched limit-up intraday, failed to hold
    # consecutive limit-up streaks
    streak = np.zeros(c.shape, dtype=np.int16)
    for t in range(len(dates)):
        streak[t] = np.where(up[t], (streak[t - 1] + 1) if t > 0 else 1, 0)
    prev_up = np.zeros_like(up); prev_up[1:] = up[:-1]
    prem = np.array([np.nanmean(r[t][prev_up[t] & v[t]]) if prev_up[t].any() else np.nan for t in range(len(dates))])
    n_up, n_brk = up.sum(1), broke.sum(1)
    df = pl.DataFrame({
        "date": dates.astype("datetime64[ms]"),
        "n_limit_up": n_up, "n_limit_down": dn.sum(1),
        "broke_rate": np.where(n_up + n_brk > 0, n_brk / np.maximum(n_up + n_brk, 1), np.nan),
        "max_streak": streak.max(1).astype(int), "n_streak2plus": (streak >= 2).sum(1),
        "lu_premium": prem,                                             # avg return today of yesterday's limit-ups
        "pct_up": np.nanmean(np.where(v, r > 0, np.nan), 1),
        "turnover_bn": np.where(v, P["money"], 0.0).sum(1) / 1e8,
    })
    # composite: each indicator vs its own trailing 250-day history (z-score), signs so that higher = hotter
    def tz(x, w=250):
        x = np.asarray(x, float); out = np.full(len(x), np.nan)
        for t in range(60, len(x)):
            a = x[max(0, t - w):t]; a = a[~np.isnan(a)]
            if len(a) > 40 and a.std() > 0:
                out[t] = (x[t] - a.mean()) / a.std()
        return out
    comp = np.nanmean(np.stack([tz(df["n_limit_up"]), -tz(df["n_limit_down"]), -tz(df["broke_rate"]),
                                tz(df["max_streak"]), tz(df["lu_premium"]), tz(df["pct_up"]), tz(np.log(df["turnover_bn"]))]), axis=0)
    df = df.with_columns(pl.Series("sentiment", comp).rolling_mean(5).alias("sentiment_5d"))
    return df.filter(pl.col("date").dt.year() >= start_year)
