"""Combining small-cap crowding with the 120d-MA breakdown. CONFIGS FIXED BEFORE THE FIRST RUN
(committed before execution). In-sample 2020-04..2024-12. Base: version A. Execution for every rule:
whole book -5%/day ramp toward the target (same speed when re-risking).

  C1   breakdown only                 -> 0.5
  C2   crowding only                  -> 0.5
  OR   crowding or breakdown (= C5)   -> 0.5
  AND  crowding and breakdown         -> 0.5
  STEP 1 - 0.25*crowding - 0.25*breakdown   (each signal cuts 25%, both cut 50%)
"""
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr
B = lib.build_base(end="2024-12-31"); F = lib.build_factors(B); U = lib.make_U(B, 1e7)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
S = signals.tilted(B, F, signals.train_signs(ics, z["dates"], allf), U, 1.0)
st = rr.states(B, U)
cr, bk = st["crowded"], st["below_ma"]
TGT = {"C1 breakdown": np.where(bk, 0.5, 1.0), "C2 crowding": np.where(cr, 0.5, 1.0),
       "OR (C5)": np.where(cr | bk, 0.5, 1.0), "AND": np.where(cr & bk, 0.5, 1.0),
       "STEP 25%+25%": 1.0 - 0.25 * cr - 0.25 * bk}
A, Z = "2020-04-01", "2024-12-31"
a0 = int(np.searchsorted(B.dates, np.datetime64(A)))
print("share of days 2020-04..2024-12: crowded %.0f%%  breakdown %.0f%%  both %.0f%%  either %.0f%%" % (
    100 * cr[a0:].mean(), 100 * bk[a0:].mean(), 100 * (cr & bk)[a0:].mean(), 100 * (cr | bk)[a0:].mean()))


def dd_in(res, a, b):
    d = res["dates"]; m = (d >= np.datetime64(a)) & (d <= np.datetime64(b))
    nav = np.cumprod(1 + res["ret"][m]); return float((nav / np.maximum.accumulate(nav) - 1).min())


rows = [] 
for name, tgt in [("C0 none", None)] + list(TGT.items()):
    ex = None if tgt is None else rr.ramp(tgt, 0.05, every=1)
    row = {"rule": name}
    for cl, c in (("", bt.Costs()), ("_20bp", bt.Costs(slippage=0.002)), ("_3M", bt.Costs(capital=3e6))):
        res = bt.run(B, S, A, Z, n_hold=100, rebal=20, buffer=3.0, U=U, costs=c, exposure=ex,
                     exposure_trade="reselect" if ex is None else "rescale")
        m = bt.metrics(res["ret"], res["bench"], res["turn_buy"])
        row[f"ann{cl}"] = m["ann"]
        if cl == "":
            e = np.ones(len(res["ret"])) if ex is None else ex[a0:a0 + len(res["ret"])]
            row.update(mdd=m["mdd"], sharpe=m["sharpe"], calmar=m["calmar"], avg_expo=float(e.mean()),
                       dd_2022=dd_in(res, "2022-01-01", "2022-12-31"), dd_2024H1=dd_in(res, "2024-01-01", "2024-06-30"),
                       **{f"y{y}": a for y, (a, b) in bt.yearly(res).items()})
    rows.append(row)
    print(f"  {name:14s} ann {row['ann']:+.1%} mdd {row['mdd']:+.1%} sharpe {row['sharpe']:.2f}", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/combo_crowding_breakdown_2020_2024.csv")
pl.Config.set_tbl_rows(10); pl.Config.set_tbl_cols(18); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(240)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c != "rule"]))
