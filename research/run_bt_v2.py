import itertools, time, numpy as np, polars as pl
import lib, backtest as bt, signals
t0=time.time()
B = lib.build_base(); F = lib.build_factors(B)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, B.dates, allf)
TR = ("2019-04-01","2021-12-31"); VA = ("2022-01-01","2023-12-29")
rows = []
for adv in (3e7, 1e7):
    U = lib.make_U(B, adv)
    for tilt in (0.5, 1.0):
        S = signals.tilted(B, F, signs, U, tilt)
        for n_hold, rebal in itertools.product((50, 100), (5, 10)):
            res = bt.run(B, S, TR[0], VA[1], n_hold=n_hold, rebal=rebal, U=U)
            per = bt.by_period(res, {"tr": TR, "va": VA})
            rows.append(dict(adv_M=adv/1e6, tilt=tilt, n=n_hold, reb=rebal,
                             ann_tr=per["tr"]["ann"], sh_tr=per["tr"]["sharpe"], mdd_tr=per["tr"]["mdd"], ex_tr=per["tr"]["excess_ann"],
                             ann_va=per["va"]["ann"], sh_va=per["va"]["sharpe"], mdd_va=per["va"]["mdd"], ex_va=per["va"]["excess_ann"],
                             turn=per["tr"]["turn_ann"]))
        print(f"adv {adv/1e6:.0f}M tilt {tilt} done ({time.time()-t0:.0f}s)", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/bt_v2_grid_train_valid.csv")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(20); pl.Config.set_fmt_float("mixed")
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c not in ("adv_M","tilt","n","reb")]).sort("sh_tr", descending=True))
