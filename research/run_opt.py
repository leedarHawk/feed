"""Optimisation round on 2020-04..2024-12. EVERYTHING HERE IS IN-SAMPLE: 2019-2024 has been used for
design; the only clean test left is the sealed 2025-2026. All 18 configs are reported (no hidden picks).

  score    : hand (two-stage small tilt, spec A) | ens (mean of hand and walk-forward LightGBM ranks)
  turnover : (rebalance 5d, buffer 1.5) baseline | (10d, 3.0) | (20d, 3.0)
  exposure : none | half when universe-EW index < its 120d MA | cash when below (MA length fixed a priori)
"""
import os, time, itertools, numpy as np, polars as pl
import lib, backtest as bt, signals, ml
t0 = time.time()
B = lib.build_base(end="2024-12-31"); F = lib.build_factors(B)
U = lib.make_U(B, 1e7)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, z["dates"], allf)
S_hand = signals.tilted(B, F, signs, U, 1.0)
cache = "/tmp/feed_cache/ml_pred_2020_2024.npy"
if os.path.exists(cache):
    pred = np.load(cache)
else:
    pred, _ = ml.walk_forward(B, F, U, "2020-04-01", log=lambda s: print(s, flush=True))
    np.save(cache, pred)
S_ens = np.where(np.isnan(S_hand), np.nan, np.nanmean(np.stack([bt.zrank(S_hand, U), bt.zrank(pred, U)]), axis=0))
print(f"scores ready ({time.time()-t0:.0f}s)", flush=True)

# universe-EW index and its 120d moving average (known at close t -> applies to trades at t+1)
mr = np.array([np.nanmean(B.r[t][U[t - 1] & ~np.isnan(B.r[t])]) if t > 0 and U[t - 1].any() else 0.0 for t in range(len(B.dates))])
idx = np.cumprod(1 + np.nan_to_num(mr))
ma = lib.rmean(idx[:, None], 120)[:, 0]
risk_on = ~(idx < ma)                          # NaN MA (warm-up) -> risk on
EXPO = {"none": None, "half<MA120": np.where(risk_on, 1.0, 0.5), "cash<MA120": np.where(risk_on, 1.0, 0.0)}
print(f"risk-off share of days 2020-04..2024-12: {np.mean(~risk_on[np.searchsorted(B.dates, np.datetime64('2020-04-01')):]):.0%}", flush=True)

A, Z = "2020-04-01", "2024-12-31"
PER = {"full": (A, Z), "y2024": ("2024-01-01", Z), "y22_24": ("2022-01-01", Z)}
rows = []
for (sn, S), (reb, buf), en in itertools.product((("hand", S_hand), ("ens", S_ens)), ((5, 1.5), (10, 3.0), (20, 3.0)), EXPO):
    row = dict(score=sn, reb=reb, buf=buf, expo=en)
    for cl, c in (("", bt.Costs()), ("_20bp", bt.Costs(slippage=0.002))):
        res = bt.run(B, S, A, Z, n_hold=100, rebal=reb, buffer=buf, U=U, costs=c, exposure=EXPO[en])
        p = bt.by_period(res, PER)
        row.update({f"ann{cl}": p["full"]["ann"], f"sharpe{cl}": p["full"]["sharpe"], f"mdd{cl}": p["full"]["mdd"]})
        if cl == "":
            row.update(ann_2024=p["y2024"]["ann"], mdd_2024=p["y2024"]["mdd"], ann_22_24=p["y22_24"]["ann"],
                       turn=p["full"]["turn_ann"], bench=p["full"]["bench_ann"],
                       **{f"y{y}": a for y, (a, b) in bt.yearly(res).items()})
    rows.append(row)
    print(f"  {sn:4s} reb{reb:<2d} buf{buf} {en:10s} ann {row['ann']:+.1%} mdd {row['mdd']:+.1%} | 20bp ann {row['ann_20bp']:+.1%} ({time.time()-t0:.0f}s)", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/opt_grid_2020_2024.csv")
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(250)
cols = ["score", "reb", "buf", "expo", "ann", "sharpe", "mdd", "ann_20bp", "mdd_20bp", "ann_2024", "mdd_2024", "ann_22_24", "turn"]
print(R.select(cols).with_columns([pl.col(c).round(3) for c in cols[4:]]))
print("yearly:"); print(R.select(["score", "reb", "expo"] + [c for c in R.columns if c.startswith("y20")]).with_columns([pl.col(c).round(2) for c in R.columns if c.startswith("y20")]))
print("done", flush=True)
