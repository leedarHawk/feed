"""Build the JSON payload for the sentiment dashboard page (data through 2026-09-30)."""
import json, sys, numpy as np, polars as pl
s = pl.read_csv("results/sentiment_daily_2019_2026.csv", try_parse_dates=True).with_columns(pl.col("date").cast(pl.Date))
ix = pl.read_parquet("../data/smallcap_risk_v1_20261004/index_daily.parquet").with_columns(pl.col("date").cast(pl.Date))
g = ix.filter(pl.col("index_code") == "399303.XSHE").select("date", pl.col("close").alias("gz2000"))
df = s.join(g, on="date", how="left").sort("date")
df = df.with_columns(pl.col(c).fill_nan(None) for c in ("broke_rate", "lu_premium", "sentiment_5d"))
s5 = df["sentiment_5d"].drop_nulls().to_numpy()
pct = {f"p{q}": float(np.quantile(s5, q / 100)) for q in (10, 30, 70, 90)}
latest = df.tail(1).to_dicts()[0]
rank = float(np.mean(s5 < latest["sentiment_5d"]))
r = lambda x, k: None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)
rows = {"d": [str(x) for x in df["date"].to_list()],
        "up": df["n_limit_up"].to_list(), "dn": df["n_limit_down"].to_list(),
        "brk": [r(x, 3) for x in df["broke_rate"].to_list()], "stk": df["max_streak"].to_list(),
        "prem": [r(x, 4) for x in df["lu_premium"].to_list()], "pu": [r(x, 3) for x in df["pct_up"].to_list()],
        "to": [r(x, 0) for x in df["turnover_bn"].to_list()], "s5": [r(x, 3) for x in df["sentiment_5d"].to_list()],
        "gz": [r(x, 2) for x in df["gz2000"].to_list()]}
# monthly table
m = (df.group_by_dynamic(pl.col("date").cast(pl.Datetime), every="1mo").agg(
        pl.col("n_limit_up").mean().alias("up"), pl.col("n_limit_down").mean().alias("dn"), pl.col("broke_rate").mean().alias("brk"),
        pl.col("max_streak").max().alias("stk"), pl.col("lu_premium").mean().alias("prem"), pl.col("pct_up").mean().alias("pu"),
        pl.col("turnover_bn").mean().alias("to"), pl.col("sentiment_5d").mean().alias("s5")))
monthly = [{"m": str(x["date"])[:7], **{k: r(x[k], 4) for k in ("up", "dn", "brk", "stk", "prem", "pu", "to", "s5")}} for x in m.to_dicts()]
# quintile evidence (computed in the research step; reproduced here from the same data)
c = df["gz2000"].to_numpy().astype(float)
f20 = np.full(len(c), np.nan); f20[:-20] = c[20:] / c[:-20] - 1
ev = df.with_columns(pl.Series("f20", f20).fill_nan(None)).drop_nulls(["sentiment_5d", "f20"])
cuts = [float(ev["sentiment_5d"].quantile(q)) for q in (0.2, 0.4, 0.6, 0.8)]
ev = ev.with_columns(pl.col("sentiment_5d").cut(cuts, labels=["1", "2", "3", "4", "5"]).alias("b"))
evidence = {}
for label, cond in (("2019–2024", pl.col("date") < pl.date(2025, 1, 1)), ("2025–2026", pl.col("date") >= pl.date(2025, 1, 1))):
    t = ev.filter(cond).group_by("b").agg(pl.len().alias("n"), pl.col("f20").mean().alias("mean"), (pl.col("f20") > 0).mean().alias("hit")).sort("b")
    evidence[label] = [{"b": x["b"], "n": x["n"], "mean": round(x["mean"], 4), "hit": round(x["hit"], 3)} for x in t.to_dicts()]
payload = {"asof": str(latest["date"]), "pct": {k: round(v, 3) for k, v in pct.items()}, "rank": round(rank, 3),
           "cuts": [round(x, 3) for x in cuts], "rows": rows, "monthly": monthly, "evidence": evidence}
out = sys.argv[1]
json.dump(payload, open(out, "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("asof", payload["asof"], "| latest s5", latest["sentiment_5d"], "| pct", payload["pct"], "| rank", payload["rank"], "| rows", len(rows["d"]))
print("evidence", json.dumps(evidence, ensure_ascii=False))
