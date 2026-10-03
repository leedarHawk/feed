"""Long-side screen: which single factors have alpha in their BEST decile (what a long-only book can own)?"""
import numpy as np, polars as pl, lib
B = lib.build_base(); F = lib.build_factors(B)
print("no-limit stock-days removed from universe:", int((B.no_limit & B.valid).sum()), flush=True)
d = B.dates
s = int(np.searchsorted(d, np.datetime64("2022-01-01")))
tr = slice(60, s - 25); va = slice(s, None)
rows = []
fw = {h: np.where(B.U, B.fwd[h], np.nan) for h in (5, 20)}
for name, f in F.items():
    x = np.where(B.U, f, np.nan)
    rk = lib.rank_rows(x); n = np.sum(~np.isnan(rk), axis=1, keepdims=True)
    q = np.floor(rk / np.maximum(n, 1) * 10)
    for h in (5, 20):
        y = fw[h]; mu = np.nanmean(np.where(~np.isnan(x), y, np.nan), axis=1)
        out = {"factor": name, "h": h}
        for side, k in (("top", 9), ("bot", 0)):
            ex = np.nanmean(np.where(q == k, y, np.nan), axis=1) - mu
            for nm, sl in (("tr", tr), ("va", va)):
                out[f"{side}_{nm}"] = float(np.nanmean(ex[sl]) * 252 / h)
        rows.append(out)
R = pl.DataFrame(rows)
R.write_csv("results/longside_screen_discovery.csv")
pl.Config.set_tbl_rows(40); pl.Config.set_fmt_float("mixed")
for h in (5, 20):
    r = R.filter(pl.col("h") == h)
    # best DECILE on either end (for a 'bot' hit the factor is simply used with flipped sign)
    best = pl.concat([
        r.select(pl.col("factor"), pl.lit("high").alias("own"), pl.col("top_tr").alias("tr"), pl.col("top_va").alias("va")),
        r.select(pl.col("factor"), pl.lit("low").alias("own"), pl.col("bot_tr").alias("tr"), pl.col("bot_va").alias("va"))])
    best = best.filter((pl.col("tr") > 0) & (pl.col("va") > 0)).with_columns(pl.min_horizontal("tr", "va").alias("worst")).sort("worst", descending=True).head(15)
    print(f"\n=== h={h}: factors whose OWN decile beats the universe in BOTH train and valid (annualised excess, no costs) ===")
    print(best.with_columns([pl.col("tr").round(3), pl.col("va").round(3), pl.col("worst").round(3)]))
