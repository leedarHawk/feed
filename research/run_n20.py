"""Concentration test (in-sample, 2020-04..2024-12): version A settings (rebalance 20d, buffer 3x),
holding 20 vs 50 vs 100 names; plus rank-slice portfolios of 20 names to check monotonicity."""
import numpy as np, polars as pl
import lib, backtest as bt, signals
B = lib.build_base(end="2024-12-31"); F = lib.build_factors(B); U = lib.make_U(B, 1e7)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
S = signals.tilted(B, F, signals.train_signs(ics, z["dates"], allf), U, 1.0)
mr = np.array([np.nanmean(B.r[t][U[t - 1] & ~np.isnan(B.r[t])]) if t > 0 and U[t - 1].any() else 0.0 for t in range(len(B.dates))])
idx = np.cumprod(1 + np.nan_to_num(mr)); ma = lib.rmean(idx[:, None], 120)[:, 0]
HALF = np.where(~(idx < ma), 1.0, 0.5)
A, Z = "2020-04-01", "2024-12-31"
PER = {"full": (A, Z), "y2024": ("2024-01-01", Z)}
rows = []
for n in (20, 50, 100):
    for en, ex in (("none", None), ("half<MA120", HALF)):
        row = dict(n=n, expo=en)
        for cl, c in (("", bt.Costs()), ("_20bp", bt.Costs(slippage=0.002))):
            res = bt.run(B, S, A, Z, n_hold=n, rebal=20, buffer=3.0, U=U, costs=c, exposure=ex, return_weights=(cl == "" and en == "none"))
            p = bt.by_period(res, PER)
            row.update({f"ann{cl}": p["full"]["ann"], f"mdd{cl}": p["full"]["mdd"]})
            if cl == "":
                row.update(vol=p["full"]["vol"], sharpe=p["full"]["sharpe"], ann_2024=p["y2024"]["ann"], turn=p["full"]["turn_ann"],
                           **{f"y{y}": a for y, (a, b) in bt.yearly(res).items()})
                if "weights" in res:
                    W = res["weights"]; t0 = int(np.searchsorted(B.dates, res["dates"][0]))
                    a = B.adv20[t0:t0 + len(W)][W > 1e-9]
                    row["adv_med_M"] = float(np.nanmedian(a) / 1e6)
                    row["worst_day"] = float(res["ret"].min())
        rows.append(row)
R = pl.DataFrame(rows)
pl.Config.set_tbl_rows(20); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(250)
num = [c for c in R.columns if c not in ("n", "expo")]
print(R.with_columns([pl.col(c).round(3) for c in num]))
R.write_csv("results/concentration_n20_n50_n100.csv")

# rank slices: 20-name books from ranks [off, off+20) of the same score
print("\nrank-slice 20-name books (no exposure rule, 10bp):")
rk = lib.rank_rows(np.where(U, -S, np.nan))           # 0 = best score
for off in (0, 20, 40, 60, 80):
    Ss = np.where(rk >= off, S, np.nan)
    res = bt.run(B, Ss, A, Z, n_hold=20, rebal=20, buffer=3.0, U=U)
    m = bt.metrics(res["ret"], res["bench"])
    print(f"   ranks {off + 1:>3d}-{off + 20:<3d}: ann {m['ann']:+6.1%}  vol {m['vol']:5.1%}  sharpe {m['sharpe']:4.2f}  mdd {m['mdd']:6.1%}"
          f"   by year {' '.join(f'{y}:{a:+.0%}' for y, (a, b) in bt.yearly(res).items())}")
