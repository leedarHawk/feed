"""Gradual vs one-step de-risking on top of version A (hand score, top 100, rebalance 20d, buffer 3x).
CONFIGS FIXED BEFORE THE FIRST RUN (committed before execution). In-sample 2020-04..2024-12.

  rules : C1 below 120d MA | C2 small-cap crowding | C5 C1 or C2     (risk-off target = 0.5)
  speed : one-step | ramp 0.2 per week | ramp 0.1 per week            (same speed when re-risking)
  trades: exposure changes rescale existing positions (no re-selection); names change only on the
          scheduled 20-day rebalance
"""
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr
B = lib.build_base(end="2024-12-31"); F = lib.build_factors(B); U = lib.make_U(B, 1e7)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
S = signals.tilted(B, F, signals.train_signs(ics, z["dates"], allf), U, 1.0)
st = rr.states(B, U)
RULES = {"C1 MA120": st["below_ma"], "C2 crowd": st["crowded"], "C5 MA|crowd": st["below_ma"] | st["crowded"]}
A, Z = "2020-04-01", "2024-12-31"
a0 = int(np.searchsorted(B.dates, np.datetime64(A))); a1 = int(np.searchsorted(B.dates, np.datetime64(Z), side="right"))
CONF = [("C0 none", "-", None)]
for rn, flag in RULES.items():
    tgt = np.where(flag, 0.5, 1.0)
    CONF += [(rn, "one-step", tgt), (rn, "0.2/wk", rr.ramp(tgt, 0.2)), (rn, "0.1/wk", rr.ramp(tgt, 0.1))]


def dd_in(res, a, b):
    d = res["dates"]; m = (d >= np.datetime64(a)) & (d <= np.datetime64(b))
    nav = np.cumprod(1 + res["ret"][m]); return float((nav / np.maximum.accumulate(nav) - 1).min())


rows = []
for rn, sp, ex in CONF:
    row = {"rule": rn, "speed": sp}
    for cl, cost in (("", bt.Costs()), ("_20bp", bt.Costs(slippage=0.002))):
        res = bt.run(B, S, A, Z, n_hold=100, rebal=20, buffer=3.0, U=U, costs=cost, exposure=ex, exposure_trade="rescale")
        m = bt.metrics(res["ret"], res["bench"], res["turn_buy"])
        row.update({f"ann{cl}": m["ann"]})
        if cl == "":
            e = np.ones(a1 - a0) if ex is None else ex[a0:a1]
            row.update(mdd=m["mdd"], sharpe=m["sharpe"], calmar=m["calmar"], turn=m["turn_ann"],
                       avg_expo=float(e.mean()), expo_moves_per_yr=float((np.diff(e) != 0).sum() / ((a1 - a0) / 252)),
                       dd_2022=dd_in(res, "2022-01-01", "2022-12-31"), dd_2024H1=dd_in(res, "2024-01-01", "2024-06-30"),
                       **{f"y{y}": a for y, (a, b) in bt.yearly(res).items()})
    rows.append(row)
    print(f"  {rn:12s} {sp:8s} ann {row['ann']:+.1%} mdd {row['mdd']:+.1%} sharpe {row['sharpe']:.2f}", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/gradual_derisk_2020_2024.csv")
pl.Config.set_tbl_rows(12); pl.Config.set_tbl_cols(22); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(260)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c not in ("rule", "speed")]))
