"""Daily (not weekly) de-risking on top of version A (hand score, top 100, rebalance 20d, buffer 3x).
CONFIGS FIXED BEFORE THE FIRST RUN (committed before execution). In-sample 2020-04..2024-12.

  rules : C1 below 120d MA | C5 below MA or small-cap crowding        (risk-off target = 0.5)
  modes : one-step rescale (reference) | 0.1/week rescale (previous recommendation)
          daily 2 names/day (exposure step 0.02/day, whole positions; executable with board lots)
          daily 4 names/day (0.04/day, whole positions)
          daily 0.02/day pro-rata rescale (theoretical only: trims are below one lot at small capital)
"""
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr
B = lib.build_base(end="2024-12-31"); F = lib.build_factors(B); U = lib.make_U(B, 1e7)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
S = signals.tilted(B, F, signals.train_signs(ics, z["dates"], allf), U, 1.0)
st = rr.states(B, U)
RULES = {"C1 MA120": st["below_ma"], "C5 MA|crowd": st["below_ma"] | st["crowded"]}
A, Z = "2020-04-01", "2024-12-31"
CONF = [("C0 none", "-", None, "reselect")]
for rn, flag in RULES.items():
    tgt = np.where(flag, 0.5, 1.0)
    CONF += [(rn, "one-step", tgt, "rescale"),
             (rn, "0.1/week", rr.ramp(tgt, 0.1, every=5), "rescale"),
             (rn, "2 names/day", rr.ramp(tgt, 0.02, every=1), "names"),
             (rn, "4 names/day", rr.ramp(tgt, 0.04, every=1), "names"),
             (rn, "0.02/day pro-rata*", rr.ramp(tgt, 0.02, every=1), "rescale")]


def dd_in(res, a, b):
    d = res["dates"]; m = (d >= np.datetime64(a)) & (d <= np.datetime64(b))
    nav = np.cumprod(1 + res["ret"][m]); return float((nav / np.maximum.accumulate(nav) - 1).min())


rows = []
for rn, mode, ex, how in CONF:
    row = {"rule": rn, "mode": mode}
    for cl, cost in (("", bt.Costs()), ("_20bp", bt.Costs(slippage=0.002))):
        res = bt.run(B, S, A, Z, n_hold=100, rebal=20, buffer=3.0, U=U, costs=cost, exposure=ex, exposure_trade=how,
                     return_weights=(cl == ""))
        m = bt.metrics(res["ret"], res["bench"], res["turn_buy"])
        row[f"ann{cl}"] = m["ann"]
        if cl == "":
            nt = (np.diff(res["weights"] > 1e-9, axis=0)).sum(1)          # names entering/leaving per day
            row.update(mdd=m["mdd"], sharpe=m["sharpe"], calmar=m["calmar"], turn=m["turn_ann"],
                       trade_days_per_yr=float((nt > 0).sum() / (len(nt) / 252)), names_traded_per_yr=float(nt.sum() / (len(nt) / 252)),
                       dd_2024H1=dd_in(res, "2024-01-01", "2024-06-30"), **{f"y{y}": a for y, (a, b) in bt.yearly(res).items()})
    rows.append(row)
    print(f"  {rn:12s} {mode:19s} ann {row['ann']:+.1%} mdd {row['mdd']:+.1%} sharpe {row['sharpe']:.2f}", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/daily_derisk_2020_2024.csv")
pl.Config.set_tbl_rows(12); pl.Config.set_tbl_cols(22); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(260)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c not in ("rule", "mode")]))
