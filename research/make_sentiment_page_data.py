"""Build the JSON payload for the market-sentiment dashboard (whole A-share market, data through 2026-09-30).
Run with FEED_LAST_YEAR=2026 FEED_CACHE=<full cache> so the panel covers 2025-2026."""
import json, sys, numpy as np, polars as pl
import panel

s = pl.read_csv("results/sentiment_daily_2019_2026.csv", try_parse_dates=True).with_columns(pl.col("date").cast(pl.Date))

# whole-market index: all A shares, circulating-market-cap weighted (weights from the previous close)
dates, codes, P = panel.load()
with np.errstate(invalid="ignore", divide="ignore"):
    r = P["close"] / P["pre_close"] - 1
w = np.vstack([np.full((1, len(codes)), np.nan), P["circulating_market_cap"][:-1]])
ok = ~np.isnan(r) & ~np.isnan(w) & (w > 0)
mret = np.where(ok, r * w, 0).sum(1) / np.maximum(np.where(ok, w, 0).sum(1), 1e-9)
mret[0] = 0.0
allA = 1000 * np.cumprod(1 + mret)
mk = pl.DataFrame({"date": dates.astype("datetime64[D]").astype("datetime64[ms]"), "allA": allA}).with_columns(pl.col("date").cast(pl.Date))
df = s.join(mk, on="date", how="left").sort("date")

# margin financing (SH + SZ combined; days where both exchanges reported)
mg = pl.read_parquet("../data/smallcap_risk_v1_20261004/margin_total_combined_daily.parquet").with_columns(pl.col("date").cast(pl.Date))
df = df.join(mg.select("date", "fin_value", "fin_buy_value"), on="date", how="left")
df = df.with_columns((pl.col("fin_buy_value") / (pl.col("turnover_bn") * 1e8)).alias("fin_buy_ratio"))
num = [c for c in df.columns if c != "date"]
df = df.with_columns([pl.col(c).cast(pl.Float64).fill_nan(None) for c in num])

# rolling thresholds (trailing 500 trading days, at least 250 observations), no look-ahead
s5 = df["sentiment_5d"].to_numpy().astype(float)
T = len(s5)
thr = {q: np.full(T, np.nan) for q in (10, 20, 30, 40, 60, 70, 80, 90)}
rank = np.full(T, np.nan)
for t in range(T):
    win = s5[max(0, t - 500):t]; win = win[~np.isnan(win)]
    if len(win) >= 250:
        for q in thr:
            thr[q][t] = np.quantile(win, q / 100)
        if not np.isnan(s5[t]):
            rank[t] = np.mean(win < s5[t])

# evidence: rolling quintile bucket of the sentiment score vs the next 20 trading days of the all-A index
c = df["allA"].to_numpy().astype(float)
f20 = np.full(T, np.nan); f20[:-20] = c[20:] / c[:-20] - 1
b = np.full(T, np.nan)
for t in range(T):
    if np.isnan(s5[t]) or np.isnan(thr[20][t]):
        continue
    b[t] = 1 + (s5[t] >= thr[20][t]) + (s5[t] >= thr[40][t]) + (s5[t] >= thr[60][t]) + (s5[t] >= thr[80][t])
ev = pl.DataFrame({"date": df["date"], "b": b, "f20": f20}).with_columns(pl.col("b", "f20").fill_nan(None)).drop_nulls()
evidence = {}
for label, cond in (("2020–2024", pl.col("date") < pl.date(2025, 1, 1)), ("2025–2026", pl.col("date") >= pl.date(2025, 1, 1))):
    t_ = ev.filter(cond).group_by("b").agg(pl.len().alias("n"), pl.col("f20").mean().alias("mean"), (pl.col("f20") > 0).mean().alias("hit")).sort("b")
    evidence[label] = [{"b": int(x["b"]), "n": x["n"], "mean": round(x["mean"], 4), "hit": round(x["hit"], 3)} for x in t_.to_dicts()]

def col(name, k):
    return [None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), k) for v in df[name].to_list()]

def arr(a, k):
    return [None if np.isnan(v) else round(float(v), k) for v in a]

rows = {"d": [str(x) for x in df["date"].to_list()],
        "up": col("n_limit_up", 0), "dn": col("n_limit_down", 0), "brk": col("broke_rate", 3), "stk": col("max_streak", 0),
        "prem": col("lu_premium", 4), "pu": col("pct_up", 3), "to": col("turnover_bn", 0), "s5": col("sentiment_5d", 3),
        "mkt": col("allA", 2), "fin": col("fin_value", -6), "fbr": col("fin_buy_ratio", 4),
        "z": {k: col(k, 2) for k in ("z_up", "z_dn", "z_brk", "z_stk", "z_prem", "z_pu", "z_to")},
        "p10": arr(thr[10], 3), "p30": arr(thr[30], 3), "p70": arr(thr[70], 3), "p90": arr(thr[90], 3), "rank": arr(rank, 3)}
mdf = df.group_by_dynamic(pl.col("date").cast(pl.Datetime), every="1mo").agg(
    pl.col("n_limit_up").mean().alias("up"), pl.col("n_limit_down").mean().alias("dn"), pl.col("broke_rate").mean().alias("brk"),
    pl.col("max_streak").max().alias("stk"), pl.col("lu_premium").mean().alias("prem"), pl.col("pct_up").mean().alias("pu"),
    pl.col("turnover_bn").mean().alias("to"), pl.col("fin_buy_ratio").mean().alias("fbr"), pl.col("sentiment_5d").mean().alias("s5"),
    (pl.col("allA").last() / pl.col("allA").first() - 1).alias("mkt"))
r4 = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)
monthly = [{"m": str(x["date"])[:7], **{k: r4(x[k]) for k in ("up", "dn", "brk", "stk", "prem", "pu", "to", "fbr", "s5", "mkt")}} for x in mdf.to_dicts()]
payload = {"asof": str(df["date"][-1]), "rows": rows, "monthly": monthly, "evidence": evidence}
json.dump(payload, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))

# sanity: yearly returns of the self-built all-A index vs published price indices
ix = pl.read_parquet("../data/smallcap_risk_v1_20261004/index_daily.parquet").with_columns(pl.col("date").cast(pl.Date))
def yr(series_df, colname):
    return {y: round(float(g[colname].to_list()[-1] / g[colname].to_list()[0] - 1) * 100, 1)
            for (y,), g in series_df.with_columns(pl.col("date").dt.year().alias("y")).group_by("y", maintain_order=True) if g.height > 100}
print("all-A (self-built, cap-weighted) yearly %:", yr(df.select("date", "allA"), "allA"))
for code, nm in (("000300.XSHG", "HS300"), ("000001.XSHG", "SSE Composite")):
    print(f"{nm:14s} yearly %:", yr(ix.filter(pl.col("index_code") == code).sort("date").select("date", "close"), "close"))
lt = df.tail(1).to_dicts()[0]
print("latest", payload["asof"], "s5", lt["sentiment_5d"], "rolling p10/p90", rows["p10"][-1], rows["p90"][-1], "rank(2y)", rows["rank"][-1],
      "| margin", lt["fin_value"], "fin_buy_ratio", lt["fin_buy_ratio"])
print("evidence", json.dumps(evidence, ensure_ascii=False))
