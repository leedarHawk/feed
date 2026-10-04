"""Fewer holdings on the main-board universe (main board, non-ST, listed >= 365d, ADV >= 10M).
Configs fixed before the first run. In-sample 2020-04..2024-12. Version A settings otherwise
(rebalance 20d, keep until rank > 3x holdings), overlays executed as whole book -5%/day.

  holdings : 20 | 30 | 50 | 100
  rules    : C0 none | C2 crowding | C2 or basis stress | C5 crowding or breakdown
  check    : 20-name books built from score ranks 1-20, 21-40, ..., 81-100 (no overlay)
"""
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr, risk_data as rd
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, z["dates"], allf)
B = lib.build_base(end="2024-12-31", min_age_days=365); F = lib.build_factors(B)
pre = np.array([c[:3] for c in B.codes])
U = lib.make_U(B, 1e7) & (~np.isin(pre, ["688", "689", "300", "301"]))[None, :]
T = len(B.dates)
S = signals.tilted(B, F, signs, U, 1.0)
st = rr.states(B, U); M = rd.market_series(B.dates)
b = M["IC_basis_ann_r2"]; db = b - np.r_[np.full(20, np.nan), b[:-20]]
q = np.full(T, np.nan)
for t in range(250, T):
    w = db[t - 250:t]; w = w[~np.isnan(w)]
    if len(w) >= 200: q[t] = np.quantile(w, 0.05)
stress = ~np.isnan(db) & (db < q)
cr, bk = st["crowded"], st["below_ma"]
RULES = {"C0 none": None, "C2 crowding": cr, "C2 or stress": cr | stress, "C5 crowd|breakdown": cr | bk}
A, Z = "2020-04-01", "2024-12-31"
rows = []
for n in (20, 30, 50, 100):
    for name, flag in RULES.items():
        ex = None if flag is None else rr.ramp(np.where(flag, 0.5, 1.0), 0.05, every=1)
        row = dict(n=n, rule=name)
        for cl, c in (("", bt.Costs()), ("_20bp", bt.Costs(slippage=0.002)), ("_3M", bt.Costs(capital=3e6))):
            res = bt.run(B, S, A, Z, n_hold=n, rebal=20, buffer=3.0, U=U, costs=c, exposure=ex,
                         exposure_trade="reselect" if ex is None else "rescale", return_weights=(cl == ""))
            m = bt.metrics(res["ret"], res["bench"], res["turn_buy"])
            row[f"ann{cl}"] = m["ann"]
            if cl == "":
                W = res["weights"]; t_ = int(np.searchsorted(B.dates, res["dates"][0])); held = W > 1e-9
                row.update(vol=m["vol"], sharpe=m["sharpe"], mdd=m["mdd"], calmar=m["calmar"], worst_day=float(res["ret"].min()),
                           adv_med_M=float(np.nanmedian(B.adv20[t_:t_ + len(W)][held]) / 1e6),
                           **{f"y{y}": a for y, (a, _) in bt.yearly(res).items()})
        rows.append(row)
        print(f"  n={n:<3d} {name:20s} ann {row['ann']:+.1%} mdd {row['mdd']:+.1%} sharpe {row['sharpe']:.2f}", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/mainboard_holdings_count_2020_2024.csv")
pl.Config.set_tbl_rows(20); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(250)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c not in ("n", "rule")]))
print("\nrank-slice 20-name books, main board, no overlay, 10bp:")
rk = lib.rank_rows(np.where(U, -S, np.nan))
for off in (0, 20, 40, 60, 80):
    res = bt.run(B, np.where(rk >= off, S, np.nan), A, Z, n_hold=20, rebal=20, buffer=3.0, U=U)
    m = bt.metrics(res["ret"])
    print(f"   ranks {off + 1:>3d}-{off + 20:<3d}: ann {m['ann']:+6.1%}  sharpe {m['sharpe']:4.2f}  mdd {m['mdd']:6.1%}   "
          + " ".join(f"{y}:{a:+.0%}" for y, (a, _) in bt.yearly(res).items()))
