"""Restricted universe requested by the user: main board only (exclude STAR 688/689 and ChiNext 300/301),
exclude ST, exclude listings younger than 1 year (365 calendar days, was 180). Everything else unchanged:
ADV >= 10M, version A (hand score, top 100, rebalance 20d, buffer 3x), whole book -5%/day overlays.
Score, universe-EW MA index and holdings are all recomputed inside the restricted universe.
In-sample 2020-04..2024-12. Configs fixed before the first run.

  universes: original (ADV>=10M, listed>=180d, all boards) | main board, listed>=365d
  configs  : C0 none | C2 crowding | C2 or basis stress | C5 crowding or breakdown
"""
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr, risk_data as rd
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, z["dates"], allf)
A, Z = "2020-04-01", "2024-12-31"
rows = []
for uname, age in (("original", 180), ("main board, >=1y", 365)):
    B = lib.build_base(end="2024-12-31", min_age_days=age); F = lib.build_factors(B)
    U = lib.make_U(B, 1e7)
    if uname != "original":
        pre = np.array([c[:3] for c in B.codes])
        main = ~np.isin(pre, ["688", "689", "300", "301"])
        U = U & main[None, :]
    T = len(B.dates)
    a0 = int(np.searchsorted(B.dates, np.datetime64(A)))
    S = signals.tilted(B, F, signs, U, 1.0)
    st = rr.states(B, U)
    M = rd.market_series(B.dates)
    b = M["IC_basis_ann_r2"]; db = b - np.r_[np.full(20, np.nan), b[:-20]]
    q = np.full(T, np.nan)
    for t in range(250, T):
        w = db[t - 250:t]; w = w[~np.isnan(w)]
        if len(w) >= 200: q[t] = np.quantile(w, 0.05)
    stress = ~np.isnan(db) & (db < q)
    cr, bk = st["crowded"], st["below_ma"]
    usize = U[a0:].sum(1).mean()
    print(f"[{uname}] universe {usize:.0f} names/day; breakdown days {bk[a0:].mean():.0%}", flush=True)
    for name, flag in (("C0 none", None), ("C2 crowding", cr), ("C2 or stress", cr | stress), ("C5 crowd|breakdown", cr | bk)):
        ex = None if flag is None else rr.ramp(np.where(flag, 0.5, 1.0), 0.05, every=1)
        row = dict(universe=uname, rule=name, universe_size=float(usize))
        for cl, c in (("", bt.Costs()), ("_20bp", bt.Costs(slippage=0.002)), ("_3M", bt.Costs(capital=3e6))):
            res = bt.run(B, S, A, Z, n_hold=100, rebal=20, buffer=3.0, U=U, costs=c, exposure=ex,
                         exposure_trade="reselect" if ex is None else "rescale", return_weights=(cl == ""))
            m = bt.metrics(res["ret"], res["bench"], res["turn_buy"])
            row[f"ann{cl}"] = m["ann"]
            if cl == "":
                W = res["weights"]; t_ = int(np.searchsorted(B.dates, res["dates"][0]))
                held = W > 1e-9
                row.update(mdd=m["mdd"], sharpe=m["sharpe"], calmar=m["calmar"], bench=m["bench_ann"], turn=m["turn_ann"],
                           names=float(held.sum(1).mean()),
                           adv_med_M=float(np.nanmedian(B.adv20[t_:t_ + len(W)][held]) / 1e6),
                           mcap_med=float(np.nanmedian(B.P["market_cap"][t_:t_ + len(W)][held])),
                           **{f"y{y}": a for y, (a, _) in bt.yearly(res).items()})
        rows.append(row)
        print(f"  {name:20s} ann {row['ann']:+.1%} mdd {row['mdd']:+.1%} sharpe {row['sharpe']:.2f} | bench {row['bench']:+.1%}", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/mainboard_vs_original_2020_2024.csv")
pl.Config.set_tbl_rows(10); pl.Config.set_tbl_cols(22); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(260)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c not in ("universe", "rule")]))
