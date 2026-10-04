"""Futures-basis stress and small-cap ETF rescue flows as additions to the crowding (C2) and
crowding-or-breakdown (C5) overlays. RULES FIXED BEFORE THE FIRST RUN (committed before execution).
In-sample 2020-04..2024-12; all thresholds are trailing-window percentiles fixed a priori.

  basis stress : IC next-month (expiry_rank 2) annualised basis, 20-day change, below its trailing
                 250-day 5th percentile  (20-day change used because the level is seasonal: dividends)
  ETF rescue   : CSI 1000 ETF family shares (lagged 1 day), 5-day % change above its trailing 250-day
                 95th percentile -> target forced to 1.0 for the next 20 trading days (overrides risk-off)
  execution    : whole book -5%/day toward the target (same as the frozen candidates)

  configs: C2 | C2 or stress | C2 + rescue | C2 or stress + rescue
           C5 | C5 or stress | C5 + rescue | C5 or stress + rescue
"""
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr, risk_data as rd
B = lib.build_base(end="2024-12-31"); F = lib.build_factors(B); U = lib.make_U(B, 1e7)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
S = signals.tilted(B, F, signals.train_signs(ics, z["dates"], allf), U, 1.0)
st = rr.states(B, U)
M = rd.market_series(B.dates)
T = len(B.dates)


def trailing_pct(x, q, win=250, minn=200):
    out = np.full(T, np.nan)
    for t in range(win, T):
        w = x[t - win:t]; w = w[~np.isnan(w)]
        if len(w) >= minn:
            out[t] = np.quantile(w, q)
    return out


b = M["IC_basis_ann_r2"]
db = b - np.r_[np.full(20, np.nan), b[:-20]]
stress = ~np.isnan(db) & (db < trailing_pct(db, 0.05))
e = M["etf_shares_中证1000"]
de = e / np.r_[np.full(5, np.nan), e[:-5]] - 1
spike = ~np.isnan(de) & (de > trailing_pct(de, 0.95))
rescue = np.zeros(T, bool)
for t in np.where(spike)[0]:
    rescue[t:t + 20] = True
A, Z = "2020-04-01", "2024-12-31"
a0 = int(np.searchsorted(B.dates, np.datetime64(A)))


def episodes(flag):
    ep, s0 = [], None
    for t in range(a0, T):
        if flag[t] and s0 is None: s0 = t
        if s0 is not None and (not flag[t] or t == T - 1):
            ep.append(f"{B.dates[s0]}({(t if flag[t] else t - 1) - s0 + 1}d)"); s0 = None
    return ", ".join(ep)


print(f"stress days {stress[a0:].mean():.0%}: {episodes(stress)}")
print(f"rescue spikes {spike[a0:].sum()} days: {episodes(spike)}")
cr, bk = st["crowded"], st["below_ma"]
BASE = {"C2": cr, "C5": cr | bk}
CONF = []
for name, flag in BASE.items():
    for tag, f, resc in (("", flag, False), (" or stress", flag | stress, False), (" + rescue", flag, True), (" or stress + rescue", flag | stress, True)):
        tgt = np.where(f, 0.5, 1.0)
        if resc:
            tgt = np.where(rescue, 1.0, tgt)
        CONF.append((name + tag, rr.ramp(tgt, 0.05, every=1)))


def dd_in(res, a, b_):
    d = res["dates"]; m = (d >= np.datetime64(a)) & (d <= np.datetime64(b_))
    nav = np.cumprod(1 + res["ret"][m]); return float((nav / np.maximum.accumulate(nav) - 1).min())


rows = []
for name, ex in CONF:
    row = {"rule": name}
    for cl, c in (("", bt.Costs()), ("_20bp", bt.Costs(slippage=0.002)), ("_3M", bt.Costs(capital=3e6))):
        res = bt.run(B, S, A, Z, n_hold=100, rebal=20, buffer=3.0, U=U, costs=c, exposure=ex, exposure_trade="rescale")
        m = bt.metrics(res["ret"], res["bench"], res["turn_buy"])
        row[f"ann{cl}"] = m["ann"]
        if cl == "":
            row.update(mdd=m["mdd"], sharpe=m["sharpe"], calmar=m["calmar"], avg_expo=float(ex[a0:a0 + len(res["ret"])].mean()),
                       dd_2022=dd_in(res, "2022-01-01", "2022-12-31"), dd_2024H1=dd_in(res, "2024-01-01", "2024-06-30"),
                       **{f"y{y}": a for y, (a, _) in bt.yearly(res).items()})
    rows.append(row)
    print(f"  {name:24s} ann {row['ann']:+.1%} mdd {row['mdd']:+.1%} sharpe {row['sharpe']:.2f}", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/macro_overlay_2020_2024.csv")
pl.Config.set_tbl_rows(10); pl.Config.set_tbl_cols(18); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(250)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c != "rule"]))
