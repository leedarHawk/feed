import time, numpy as np
import lib, backtest as bt, signals
t=time.time()
B = lib.build_base(); F = lib.build_factors(B)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, B.dates, allf)
print("train-derived signs:", {k:int(v) for k,v in signs.items()}, f"({time.time()-t:.0f}s)")
S = signals.composite(B, F, signs)
PER = {"TRAIN 2019-04..2021-12": ("2019-04-01","2021-12-31"), "VALID 2022-2023": ("2022-01-01","2023-12-29")}
for n_hold, rebal in ((50,5),(50,10),(100,5),(30,5)):
    res = bt.run(B, S, "2019-04-01", "2023-12-29", n_hold=n_hold, rebal=rebal)
    print(f"\n--- composite top{n_hold}, rebalance every {rebal}d, buffer 1.5, ADV>=30M, costs default")
    for k, m in bt.by_period(res, PER).items(): print(f"  {k:24s} {bt.fmt(m)}")
    print("  by year (strategy / universe EW):", {y: f"{a:+.0%}/{b:+.0%}" for y,(a,b) in bt.yearly(res).items()})
