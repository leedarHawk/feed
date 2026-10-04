"""Step out of the market around the periodic-report deadlines (user idea: sell before annual/Q1 and
semi-annual reports, buy back afterwards). WINDOWS FIXED BEFORE THE FIRST RUN (committed before execution).
No per-stock report dates are available, so windows are built from the statutory deadlines:
Apr 30 (annual + Q1), Aug 31 (semi-annual), and for W4 Jan 31 (annual loss pre-announcements).
d = last trading day on or before the deadline. "Exit k" = flat on trading days d-k+1 .. d.

  W0 none | W1 exit 10, re-enter d+1 | W2 exit 5, re-enter d+1 | W3 exit 10, re-enter d+5
  W4 W1 + January window | W5 W1 but half exposure instead of flat
Base books (main board, non-ST, listed >= 1y, ADV >= 10M, 50 holdings, rebalance 20d, buffer 3x,
crowding overlay whole book -5%/day): V9 small + dividend yield (main) and V0 small only.
Window exposure multiplies the crowding-overlay exposure. In-sample 2020-04..2024-12.
"""
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, z["dates"], allf)
B = lib.build_base(end="2024-12-31", min_age_days=365); F = lib.build_factors(B)
pre = np.array([c[:3] for c in B.codes])
U = lib.make_U(B, 1e7) & (~np.isin(pre, ["688", "689", "300", "301"]))[None, :]
SC = signals.mainboard_scores(B, F, signs, U)
st = rr.states(B, U)
c2 = rr.ramp(np.where(st["crowded"], 0.5, 1.0), 0.05, every=1)
d = B.dates; T = len(d)
yrs = sorted(set((d.astype("datetime64[Y]").astype(int) + 1970).tolist()))


def window(months_days, k_exit, k_after=1, level=0.0):
    """exposure multiplier, indexed by signal day t (applies to trades at t+1)."""
    f = np.ones(T)
    for y in yrs:
        for (m, dd) in months_days:
            dl = np.datetime64(f"{y}-{m:02d}-{dd:02d}")
            i = int(np.searchsorted(d, dl, side="right")) - 1        # last trading day <= deadline
            if i < 0 or d[i].astype("datetime64[M]") != dl.astype("datetime64[M]"):
                continue
            a, b = i - k_exit, i + k_after - 1                        # flat from a+1 .. i+k_after-1 trading days
            f[max(a, 0):min(b, T)] = level
    return f


AQ, H, J = (4, 30), (8, 31), (1, 31)
WIN = {"W0 none": np.ones(T), "W1 exit10 re+1": window([AQ, H], 10), "W2 exit5 re+1": window([AQ, H], 5),
       "W3 exit10 re+5": window([AQ, H], 10, k_after=5), "W4 W1+Jan": window([J, AQ, H], 10),
       "W5 W1 half": window([AQ, H], 10, level=0.5)}
A, Z = "2020-04-01", "2024-12-31"

# diagnostic: base-book (no window) return inside each window, by year
print("diagnostic: V9 book without windows - return during the windows (strategy / universe EW):")
res0 = bt.run(B, SC["V9"], A, Z, n_hold=50, rebal=20, buffer=3.0, U=U, exposure=c2, exposure_trade="rescale")
rd_ = res0["dates"]
for label, md, k, ka in (("Apr annual+Q1, last 10d + d+1..d+4", [AQ], 10, 5), ("Aug semi-annual, last 10d", [H], 10, 1), ("Jan loss warnings, last 10d", [J], 10, 1)):
    f = window(md, k, k_after=ka)
    held_flat = f < 1                                  # signal days -> trading days are the next day
    trade_days = np.zeros(T, bool); trade_days[1:] = held_flat[:-1]
    out = []
    for y in yrs:
        m = trade_days[np.searchsorted(d, rd_)] & (rd_.astype("datetime64[Y]").astype(int) + 1970 == y)
        if m.sum():
            out.append(f"{y}: {np.prod(1 + res0['ret'][m]) - 1:+.1%}/{np.prod(1 + res0['bench'][m]) - 1:+.1%}")
    print(f"   {label}: " + "  ".join(out))
rows = []
for sname in ("V9", "V0"):
    for wname, wf in WIN.items():
        ex = c2 * wf
        row = dict(book=sname, window=wname)
        for cl, c in (("", bt.Costs()), ("_20bp", bt.Costs(slippage=0.002))):
            res = bt.run(B, SC[sname], A, Z, n_hold=50, rebal=20, buffer=3.0, U=U, costs=c, exposure=ex, exposure_trade="rescale")
            m = bt.metrics(res["ret"], res["bench"], res["turn_buy"])
            row[f"ann{cl}"] = m["ann"]
            if cl == "":
                row.update(sharpe=m["sharpe"], mdd=m["mdd"], calmar=m["calmar"], turn=m["turn_ann"],
                           flat_days_per_yr=float((wf < 1).sum() / (T / 243)),
                           **{f"y{y}": a for y, (a, _) in bt.yearly(res).items()})
        rows.append(row)
        print(f"  {sname} {wname:16s} ann {row['ann']:+.1%} mdd {row['mdd']:+.1%} sharpe {row['sharpe']:.2f}", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/earnings_window_2020_2024.csv")
pl.Config.set_tbl_rows(14); pl.Config.set_tbl_cols(18); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(250)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c not in ("book", "window")]))
