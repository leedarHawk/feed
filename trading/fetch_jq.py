"""Fetch one trading day of A-share prices from JoinQuant (jqdatasdk) in the stock_price_v1 layout.

NOT TESTED here (no JoinQuant account in this environment). The output goes through `python -m trading
ingest`, which validates it (one day, coverage, price consistency, market-cap units) before anything uses it.
qfq columns are left empty on purpose: ingest rebuilds them by chain-linking from pre_close.

    JQ_USER=... JQ_PASS=... python -m trading.fetch_jq 2026-10-08 trading/var/inbox/stock_price_20261008.parquet
"""
import os
import sys

import polars as pl

PRICE_FIELDS = ["open", "close", "high", "low", "volume", "money", "high_limit", "low_limit", "pre_close", "paused"]
VAL_FIELDS = ["capitalization", "circulating_cap", "market_cap", "circulating_market_cap"]


def fetch(day):
    import jqdatasdk as jq
    jq.auth(os.environ["JQ_USER"], os.environ["JQ_PASS"])
    secs = jq.get_all_securities(types=["stock"], date=day)
    codes = list(secs.index)
    px = jq.get_price(codes, start_date=day, end_date=day, frequency="daily", fields=PRICE_FIELDS,
                      skip_paused=False, fq=None, panel=False)
    val = jq.get_valuation(codes, end_date=day, count=1, fields=VAL_FIELDS)
    st = jq.get_extras("is_st", codes, start_date=day, end_date=day, df=True)
    px = pl.from_pandas(px).with_columns(pl.col("time").cast(pl.Utf8).str.slice(0, 10))
    assert px["time"].unique().to_list() == [day], "get_price returned another day"
    val = pl.from_pandas(val).with_columns(pl.col("day").cast(pl.Utf8).str.slice(0, 10))
    assert val["day"].unique().to_list() == [day], "get_valuation returned another day"
    names = pl.DataFrame({"code": codes, "stock_name": secs["display_name"].tolist(),
                          "is_st": [bool(st[c].iloc[0]) if c in st else False for c in codes]})
    names = names.with_columns(pl.when(pl.col("is_st") & ~pl.col("stock_name").str.contains("ST"))
                               .then(pl.lit("ST") + pl.col("stock_name")).otherwise(pl.col("stock_name")).alias("stock_name"))
    df = (px.join(val.drop("day"), on="code", how="left").join(names.drop("is_st"), on="code", how="left"))
    # suspended rows: JoinQuant fills prices with the last close; the panel expects them missing
    df = df.with_columns([pl.when(pl.col("paused") == 1).then(None).otherwise(pl.col(c)).alias(c)
                          for c in ("open", "close", "high", "low")]).drop("paused")
    df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("roe"), pl.lit(None, dtype=pl.Float64).alias("roa"))
    return df


if __name__ == "__main__":
    day, out = sys.argv[1], sys.argv[2]
    df = fetch(day)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.write_parquet(out)
    print(f"{day}: {df.height} rows -> {out}")
