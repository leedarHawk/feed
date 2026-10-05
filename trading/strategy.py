"""The frozen strategy of research/final_spec.py, evaluated one signal day at a time.

Research (run_final.py) builds the score matrix for all days with the universe
    U[s] = U_base[s] & buyable at the open of s+1
which uses the next open. Live, that is known only after the 09:25 call auction. So a signal day is kept as
raw per-name inputs (U_base row, the 12 composite factors, liquidity, dividend yield) and the score is formed
at the open with the observed buyability: `Market.score_row`. On history this reproduces the research score
matrix exactly (tests/test_strategy.py).
"""
import warnings

import numpy as np
import polars as pl

from .bridge import RESEARCH, research

FACTORS = ("ivol20", "maxr20", "vol10", "skew20", "turn5", "turnr5_60", "nlimup20", "intraday20", "overnight20",
           "ret5", "ret20", "ret60")
MAINBOARD_EXCLUDE = ("688", "689", "300", "301")


def frozen_signs():
    """Factor signs from the TRAIN-period (2019-01 .. 2021-11) 10-day Rank IC, as signals.train_signs."""
    S = pl.read_csv(RESEARCH / "results" / "ic_screen_discovery.csv").filter(pl.col("h") == 10)
    ic = dict(zip(S["factor"].to_list(), S["ic_tr"].to_list()))
    return {f: float(np.sign(ic[f])) for f in FACTORS}


def lean_factors(B):
    """The 13 factors the strategy uses, computed exactly as research/lib.py build_factors (which builds 69
    and peaks at ~10 GB on the full panel). tests/test_strategy.py checks equality with the research code."""
    lib = research().lib
    rsum, rstd, rcorr, rmean, shift = lib.rsum, lib.rstd, lib.rcorr, lib.rmean, lib.shift
    P, r = B.P, B.r
    F = {}
    valid = B.valid.astype(np.float64)
    r0 = np.nan_to_num(r)
    lr0 = np.log1p(np.clip(r0, -0.95, None))
    cum = np.cumsum(lr0, axis=0)

    def cov(k):
        return rsum(valid, k, 1)[1] >= 0.7 * k

    for k in (5, 20, 60):
        f = cum - shift(cum, k)
        f[~cov(k)] = np.nan
        f[:k] = np.nan
        F[f"ret{k}"] = f
    del cum, lr0, r0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mret = np.nanmean(r, axis=1)[:, None] * np.ones_like(r)
    F["vol10"] = rstd(r, 10, max(3, int(0.7 * 10)))
    vol20 = rstd(r, 20, max(3, int(0.7 * 20)))
    c = rcorr(r, mret, 20, int(0.7 * 20))
    F["ivol20"] = vol20 * np.sqrt(np.maximum(1 - c * c, 0))
    del mret, vol20, c
    F["maxr20"] = lib.rmax(r, 20)
    m1, n = rsum(r, 20, int(0.7 * 20))
    m2, _ = rsum(r * r, 20, int(0.7 * 20))
    m3, _ = rsum(r ** 3, 20, int(0.7 * 20))
    nn = np.maximum(n, 1)
    mu, var = m1 / nn, m2 / nn - (m1 / nn) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        sk = (m3 / nn - 3 * mu * m2 / nn + 2 * mu ** 3) / var ** 1.5
    sk[var < 1e-10] = np.nan
    F["skew20"] = sk
    del m1, m2, m3, n, nn, mu, var, sk
    turn_d = P["volume"] / (P["circulating_cap"] * 1e4)
    turn5 = rmean(turn_d, 5, int(0.7 * 5))
    turn60 = rmean(turn_d, 60, int(0.7 * 60))
    F["turn5"] = turn5
    F["turnr5_60"] = turn5 / turn60
    del turn_d, turn60
    F["lmoney20"] = np.log(np.maximum(rmean(P["money"], 20, int(0.7 * 20)), 1.0))
    F["overnight20"] = rsum(B.on, 20, int(0.7 * 20))[0]
    F["intraday20"] = rsum(B.idr, 20, int(0.7 * 20))[0]
    F["nlimup20"] = rsum(B.lim_up_close.astype(np.float64), 20, 1)[0]
    for k, v in F.items():
        v[~np.isfinite(v)] = np.nan
        F[k] = v.astype(np.float32)
    return F


def _stable_noise(dates, codes):
    """Tie-break in [0, 1e-9) that depends only on (date, code), so a decision does not change when the panel
    grows (the research version draws one random matrix of the panel's shape)."""
    import zlib
    cid = np.array([zlib.crc32(c.encode()) for c in codes], dtype=np.uint64)
    did = dates.astype("datetime64[D]").astype(np.int64).astype(np.uint64)
    x = (did[:, None] * np.uint64(0x9E3779B97F4A7C15)) ^ (cid[None, :] * np.uint64(0xBF58476D1CE4E5B9))
    x ^= x >> np.uint64(31)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(29)
    return (x >> np.uint64(11)).astype(np.float64) / float(1 << 53) * 1e-9


def dividend_yield(B, tie_break="stable"):
    """signals.dividend_yield_ttm with a choice of tie-break: "research" reproduces it exactly."""
    import os
    R = research()
    dv = pl.read_parquet(os.path.join(R.lib.DATA, "dividends.parquet")).filter(
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
    ttm = cs - R.lib.shift(cs, 243)
    ttm[:243] = cs[:243]
    if tie_break == "research":
        noise = np.random.default_rng(0).random((T, N)) * 1e-9
    else:
        noise = _stable_noise(B.dates, B.codes)
    return ttm / 1e4 / B.P["market_cap"] + noise


def universe_base(B, adv_floor, mainboard):
    """lib.make_U without the next-open condition: known at the close of the signal day."""
    return (B.valid & ~B.st & ~B.no_limit & B.age_ok & B.is_a[None, :] & (B.adv20 >= adv_floor) & mainboard[None, :])


def mainboard_mask(codes):
    return ~np.isin(np.array([c[:3] for c in codes]), MAINBOARD_EXCLUDE)


class Market:
    """Everything the strategy needs from the panel, shared by all accounts."""

    def __init__(self, adv_floor, min_age_days, tie_break="stable", log=print):
        R = research()
        self.R = R
        dates, _, _ = R.panel.load()
        log(f"loading panel {dates[0]} .. {dates[-1]} ...")
        B = R.lib.build_base(end=str(dates[-1]), min_age_days=min_age_days)
        B.fwd = None
        self.B = B
        self.dates, self.codes = B.dates, B.codes
        self.cidx = {c: i for i, c in enumerate(B.codes)}
        self.mainboard = mainboard_mask(B.codes)
        self.U_base = universe_base(B, adv_floor, self.mainboard)      # the panel's last day keeps its universe
        nxt = np.zeros_like(B.can_buy_open)
        nxt[:-1] = B.can_buy_open[1:]
        self.U_research = self.U_base & nxt          # == lib.make_U(B, adv_floor) & mainboard
        log("crowding state ...")
        self.crowded = R.risk_rules.states(B, self.U_research)["crowded"]
        log("factors ...")
        self.F = lean_factors(B)
        self.dy = dividend_yield(B, tie_break)
        self.signs = frozen_signs()
        self.tie_break = tie_break
        self.adv_floor = adv_floor

    def index(self, day):
        i = int(np.searchsorted(self.dates, np.datetime64(str(day)[:10])))
        if i >= len(self.dates) or self.dates[i] != np.datetime64(str(day)[:10]):
            raise KeyError(f"{day} is not in the panel ({self.dates[0]} .. {self.dates[-1]})")
        return i

    # ---------------------------------------------------------------- exposure / schedule
    def exposure(self, strategy, cal):
        """ex[s] = target gross exposure for trades on the trading day after signal day s."""
        crowd = np.where(self.crowded, 0.5, 1.0)
        ex = self.R.risk_rules.ramp(crowd, strategy.step, every=1)
        if strategy.january:
            ex = ex * self.january_mask(cal)
        return ex

    def january_mask(self, cal):
        T = len(self.dates)
        trade = [str(d) for d in self.dates[1:]] + [str(cal.next_day(str(self.dates[-1])))]
        jan, cache = np.ones(T), {}
        for s, t in enumerate(trade):
            if t[5:7] == "01":
                y = int(t[:4])
                if y not in cache:
                    cache[y] = {str(d) for d in cal.january_exit_days(y)}
                if t in cache[y]:
                    jan[s] = 0.0
        return jan

    def scheduled(self, strategy, t):
        """Is trade-day index t (len(dates) = the day after the panel) a scheduled rebalance day?"""
        t0 = max(int(np.searchsorted(self.dates, np.datetime64(strategy.anchor))), 1)
        return (t - t0) % strategy.rebal == strategy.offset % strategy.rebal

    # ---------------------------------------------------------------- scores
    def signal_row(self, s):
        """Per-name inputs of signal day s; `SignalRow.score` forms the score once the next open is known."""
        return SignalRow(day=str(self.dates[s]), codes=self.codes, U_base=self.U_base[s].copy(),
                         F={n: self.F[n][s].copy() for n in FACTORS + ("lmoney20",)}, dy=self.dy[s].copy(),
                         close=self.B.P["close"][s].copy(), valid=self.B.valid[s].copy(), signs=self.signs)

    def score_row(self, s, can_buy_next, score_kind):
        return self.signal_row(s).score(can_buy_next, score_kind)

    # ---------------------------------------------------------------- market rows
    def open_info(self, t):
        """What is known right after the call auction of trade-day index t."""
        B = self.B
        return dict(open=B.P["open"][t], pre_close=B.P["pre_close"][t], high_limit=B.P["high_limit"][t],
                    low_limit=B.P["low_limit"][t], can_buy=B.can_buy_open[t], can_sell=B.can_sell_open[t])

    def close_info(self, t):
        B = self.B
        return dict(close=B.P["close"][t], valid=B.valid[t])

    def bench_return(self, t):
        """Universe equal-weight return of day t (the research benchmark)."""
        m = self.U_research[t - 1] & ~np.isnan(self.B.r[t])
        return float(np.nanmean(self.B.r[t][m])) if m.any() else 0.0


class SignalRow:
    """Everything needed to score one signal day; small enough to save at the close and load at 09:25."""

    def __init__(self, day, codes, U_base, F, dy, close, valid, signs):
        self.day, self.codes, self.U_base, self.F, self.dy = day, codes, U_base, F, dy
        self.close, self.valid, self.signs = close, valid, signs

    def score(self, can_buy_next, score_kind):
        """Score given which names can be bought at the next open; NaN outside the universe.
        Same arithmetic as signals.mainboard_scores on one row."""
        R = research()
        bt, sg = R.backtest, R.signals
        U = (self.U_base & can_buy_next)[None, :]
        F = {n: self.F[n][None, :] for n in FACTORS}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            comp_r = bt.zrank(sg.composite(None, F, self.signs, U=U), U)
        small = bt.zrank(-self.F["lmoney20"][None, :], U)
        k30 = comp_r > -0.4
        if score_kind == "V0":
            out = np.where(k30, small, np.nan)
        elif score_kind == "V9":
            value = bt.zrank(self.dy[None, :], U)
            out = np.where(k30, 0.5 * small + 0.5 * value, np.nan)
        else:
            raise ValueError(score_kind)
        return out[0], U[0]

    def save(self, path):
        np.savez_compressed(path, day=self.day, codes=self.codes, U_base=self.U_base, dy=self.dy, close=self.close,
                            valid=self.valid, signs=np.array([self.signs[f] for f in FACTORS]),
                            **{f"F_{k}": v for k, v in self.F.items()})

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        F = {k[2:]: z[k] for k in z.files if k.startswith("F_")}
        return cls(day=str(z["day"]), codes=z["codes"], U_base=z["U_base"], F=F, dy=z["dy"], close=z["close"],
                   valid=z["valid"], signs=dict(zip(FACTORS, z["signs"].tolist())))
