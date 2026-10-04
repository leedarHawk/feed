"""Is the ADV floor useful? J1 (V9 + January exit, crowding overlay, 50 holdings, 4 staggered sleeves)
at ADV floors 0 / 3M / 5M / 10M / 20M, slippage 10bp and 20bp per side. Score, crowding state and
universe are rebuilt for each floor. In-sample 2020-04..2024-12."""
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, z["dates"], allf)
B = lib.build_base(end="2024-12-31", min_age_days=365); F = lib.build_factors(B)
pre = np.array([c[:3] for c in B.codes]); main = (~np.isin(pre, ["688", "689", "300", "301"]))[None, :]
d = B.dates; T = len(d); jan = np.ones(T)
for y in range(2019, 2025):
    i = int(np.searchsorted(d, np.datetime64(f"{y}-01-31"), side="right")) - 1
    if i >= 10 and str(d[i])[:7] == f"{y}-01":
        jan[i - 10:i] = 0.0
rows = []
for floor in (0.0, 3e6, 5e6, 1e7, 2e7):
    U = lib.make_U(B, floor) & main
    S = signals.mainboard_scores(B, F, signs, U)["V9"]
    ex = rr.ramp(np.where(rr.states(B, U)["crowded"], 0.5, 1.0), 0.05, every=1) * jan
    row = dict(floor=f"{floor/1e6:.0f}M", universe=float(U[250:].sum(1).mean()))
    for cl, c in (("10bp", bt.Costs()), ("20bp", bt.Costs(slippage=0.002))):
        sl = [bt.run(B, S, "2020-04-01", "2024-12-31", n_hold=50, rebal=20, buffer=3.0, U=U, costs=c, exposure=ex,
                     exposure_trade="rescale", rebal_offset=o, return_weights=(cl == "10bp" and o == 0)) for o in (0, 5, 10, 15)]
        m = bt.metrics(np.mean([s["ret"] for s in sl], axis=0))
        row[f"ann_{cl}"] = m["ann"]
        if cl == "10bp":
            row.update(mdd=m["mdd"], sharpe=m["sharpe"])
            W = sl[0]["weights"]; t0 = int(np.searchsorted(d, sl[0]["dates"][0])); held = W > 1e-9
            a = B.adv20[t0:t0 + len(W)][held]
            row.update(adv_p10_M=float(np.nanpercentile(a, 10) / 1e6), adv_med_M=float(np.nanmedian(a) / 1e6),
                       share_below_5M=float(np.mean(a < 5e6)),
                       capacity_1pct_M=float(np.nanpercentile(a, 10) * 0.01 * 50 / 1e6))   # NAV where p10 name hits 1% ADV
    rows.append(row)
    print(f"  floor {row['floor']:>3s}: ann {row['ann_10bp']:+.1%} (20bp {row['ann_20bp']:+.1%}) mdd {row['mdd']:+.1%} sharpe {row['sharpe']:.2f} | "
          f"holdings ADV p10 {row['adv_p10_M']:.1f}M median {row['adv_med_M']:.1f}M, <5M {row['share_below_5M']:.0%} | capacity@1%ADV ~{row['capacity_1pct_M']:.1f}M CNY", flush=True)
pl.DataFrame(rows).write_csv("results/adv_floor_j1_2020_2024.csv")
