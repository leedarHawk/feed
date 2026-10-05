"""Daily increments: validation and qfq chain-linking (slow: builds two price caches in a temp directory)."""
import numpy as np
import polars as pl
import pytest

from trading.data import BASE_PRICE, DataError, Store

pytestmark = pytest.mark.slow


def test_ingest_chain_links_qfq(tmp_path):
    store = Store(tmp_path)
    day0 = pl.read_parquet(BASE_PRICE / "stock_price_2026.parquet").filter(pl.col("time").str.slice(0, 10) == "2026-09-30")
    with pytest.raises(DataError):                                 # not after the last stored day
        p = tmp_path / "dup.parquet"
        day0.write_parquet(p)
        store.ingest_prices(p)
    code = "600000.XSHG"
    inc = day0.with_columns(pl.lit("2026-10-08 00:00:00").alias("time"), pl.col("close").alias("pre_close"),
                            pl.lit(1.0).alias("qfq_close"))          # junk qfq must be ignored
    inc = inc.with_columns(pl.when(pl.col("code") == code).then(pl.col("close") * 0.98).otherwise(pl.col("pre_close"))
                           .alias("pre_close"))                     # an ex-dividend day for one name
    p = tmp_path / "inc.parquet"
    inc.write_parquet(p)
    day, warns = store.ingest_prices(p, log=lambda *a: None)
    assert str(day) == "2026-10-08"
    dates, codes, P = store.activate(log=lambda *a: None).panel.load()
    assert str(dates[-1]) == "2026-10-08"
    j = list(codes).index(code)
    r_qfq = P["qfq_close"][-1, j] / P["qfq_close"][-2, j] - 1
    assert r_qfq == pytest.approx(P["close"][-1, j] / P["pre_close"][-1, j] - 1, abs=1e-12)
    k = np.isfinite(P["qfq_close"][-2]) & np.isfinite(P["close"][-1])
    ratio = P["qfq_close"][-1][k] / P["qfq_close"][-2][k]
    other = np.array([c != code for c in codes])[k]
    assert np.allclose(ratio[other], (P["close"][-1] / P["pre_close"][-1])[k][other])
