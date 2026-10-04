"""Composite scores built from family-level rank averages. Signs come from TRAIN IC only."""
import numpy as np
import lib, backtest

FAMILIES = {
    "lottery":   ["ivol20", "maxr20", "vol10", "skew20"],
    "turnover":  ["turn5", "turnr5_60"],
    "limitup":   ["nlimup20"],
    "intraday":  ["intraday20", "overnight20"],
    "reversal":  ["ret5", "ret20", "ret60"],
}


def train_signs(ics, dates, factors, split="2022-01-01", embargo=25, h=10):
    s = int(np.searchsorted(dates, np.datetime64(split)))
    return {f: float(np.sign(np.nanmean(ics[(f, h)][:max(s - embargo, 0)]))) for f in factors}


def composite(B, F, signs, families=FAMILIES, weights=None, U=None):
    U = B.U if U is None else U
    fam_scores = []
    for fam, names in families.items():
        z = [signs[n] * backtest.zrank(F[n], U) for n in names]
        fam_scores.append((weights or {}).get(fam, 1.0) * np.nanmean(np.stack(z), axis=0))
    return np.nanmean(np.stack(fam_scores), axis=0)


def tilted(B, F, signs, U, tilt, junk_cut=0.3, base_families=FAMILIES):
    """Stage 1: drop the worst `junk_cut` share by the family composite (anti-lottery etc.).
    Stage 2: rank survivors by (1-tilt)*composite + tilt*small-liquidity score (low 20d turnover value)."""
    z = [signs[n] * backtest.zrank(F[n], U) for fams in base_families.values() for n in fams[:1]]
    comp = np.nanmean(np.stack([np.nanmean(np.stack([signs[n] * backtest.zrank(F[n], U) for n in names]), axis=0)
                                for names in base_families.values()]), axis=0)
    comp_r = backtest.zrank(comp, U)                       # [-1, 1]
    small = backtest.zrank(-F["lmoney20"], U)              # +1 = least liquid in universe
    score = (1 - tilt) * comp_r + tilt * small
    score = np.where(comp_r > (2 * junk_cut - 1), score, np.nan)   # junk filter
    return score


def neutralize(f, U, controls):
    """Daily cross-sectional OLS residual of rank(f) on ranks of `controls`, inside U."""
    y = backtest.zrank(f, U)
    Xc = [backtest.zrank(c, U) for c in controls]
    m = U & ~np.isnan(y)
    for x in Xc:
        m &= ~np.isnan(x)
    X = np.stack([m.astype(np.float64)] + [np.where(m, x, 0.0) for x in Xc])
    k = X.shape[0]
    XtX = np.einsum("itn,jtn->tij", X, X) + np.eye(k)[None] * 1e-9
    Xty = np.einsum("itn,tn->ti", X, np.where(m, y, 0.0))
    beta = np.linalg.solve(XtX, Xty[..., None])[..., 0]
    r = y - np.einsum("ti,itn->tn", beta, X)
    r[~m] = np.nan
    return r


def size_neutral(B, F, signs, U, n_buckets=10, junk_cut=0.3):
    """Score = mean of size/liquidity-neutral (-vwapdev20, -maxr60); drop the worst `junk_cut`
    by the neutralised family composite; then convert to within-size-decile percentiles so a
    top-N pick takes ~N/n_buckets names from every size decile."""
    from lib import rank_rows
    ctrl = [F["lmcap"], F["lmoney20"]]
    a = backtest.zrank(neutralize(-F["vwapdev20"], U, ctrl), U)
    b = backtest.zrank(neutralize(-F["maxr60"], U, ctrl), U)
    score = np.nanmean(np.stack([a, b]), axis=0)
    junk = backtest.zrank(neutralize(composite(B, F, signs, U=U), U, ctrl), U)
    score = np.where(junk > 2 * junk_cut - 1, score, np.nan)
    zs = backtest.zrank(F["lmcap"], U)
    bucket = np.clip(np.floor((zs + 1) / 2 * n_buckets), 0, n_buckets - 1)
    out = np.full(score.shape, np.nan, dtype=np.float32)
    for k in range(n_buckets):
        x = np.where(bucket == k, score, np.nan)
        r = rank_rows(x)
        n = np.sum(~np.isnan(r), axis=1, keepdims=True)
        pct = r / np.maximum(n - 1, 1)
        out = np.where(np.isnan(pct), out, pct)
    return out


def dividend_yield_ttm(B, seed=0):
    """TTM cash dividends / market cap, known from the trading day after the implementation notice.
    A tiny random tie-break separates non-payers (all zero)."""
    import os
    import polars as pl
    import lib
    dv = pl.read_parquet(os.path.join(lib.DATA, "dividends.parquet")).filter(
        (pl.col("status") == "实施方案") & pl.col("implementation_pub_date").is_not_null())
    cidx = {c: i for i, c in enumerate(B.codes)}
    T, N = len(B.dates), len(B.codes)
    E = np.zeros((T, N))
    for c, pub, cash in zip(dv["code"].to_list(), dv["implementation_pub_date"].to_list(), dv["cash_total_10k_cny"].to_list()):
        if c in cidx and cash is not None:
            k = int(np.searchsorted(B.dates, np.datetime64(pub, "D"), side="right"))
            if k < T:
                E[k, cidx[c]] += cash
    cs = np.cumsum(E, axis=0)
    ttm = cs - lib.shift(cs, 243)
    ttm[:243] = cs[:243]
    rng = np.random.default_rng(seed)
    return ttm / 1e4 / B.P["market_cap"] + rng.random((T, N)) * 1e-9


def mainboard_scores(B, F, signs, U):
    """V0 (drop worst 30% by composite, rank by small) and V9 (... rank by 0.5 small + 0.5 dividend yield)."""
    comp_r = backtest.zrank(composite(B, F, signs, U=U), U)
    small = backtest.zrank(-F["lmoney20"], U)
    value = backtest.zrank(dividend_yield_ttm(B), U)
    k30 = comp_r > -0.4
    return {"V0": np.where(k30, small, np.nan), "V9": np.where(k30, 0.5 * small + 0.5 * value, np.nan)}
