"""ONE-SHOT holdout evaluation on 2024. Specs below were fixed before this script was first run;
nothing is re-tuned after seeing these numbers. 2025-2026 stays sealed (panel.LAST_ALLOWED_YEAR=2024)."""
import time, numpy as np, polars as pl
import lib, backtest as bt, signals, ml
t0 = time.time()
B = lib.build_base(end="2024-12-31"); F = lib.build_factors(B)
assert str(B.dates[-1]) <= "2024-12-31", B.dates[-1]
U = lib.make_U(B, 1e7)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, z["dates"], allf)          # signs from TRAIN 2019-2021 only
S_hand = signals.tilted(B, F, signs, U, 1.0)
pred, imp = ml.walk_forward(B, F, U, "2020-04-01", log=lambda s: print(s, flush=True))
print(f"scores ready {time.time()-t0:.0f}s", flush=True)
SPECS = [("hand", S_hand, 100, 5), ("hand", S_hand, 100, 10), ("ml", pred, 100, 5), ("ml", pred, 100, 10)]
PER = {"DEV-OOS 2020-04..2023-12": ("2020-04-01", "2023-12-29"), "HOLDOUT 2024": ("2024-01-01", "2024-12-31"),
       "FULL 2020-04..2024-12": ("2020-04-01", "2024-12-31")}
navs = {}
for name, S, n, reb in SPECS:
    lab = f"{name}_top{n}_reb{reb}"
    res = bt.run(B, S, "2020-04-01", "2024-12-31", n_hold=n, rebal=reb, U=U)
    res2 = bt.run(B, S, "2020-04-01", "2024-12-31", n_hold=n, rebal=reb, U=U, costs=bt.Costs(slippage=0.002))
    print(f"\n[{lab}]")
    for k, m in bt.by_period(res, PER).items(): print(f"   {k:26s} {bt.fmt(m)}")
    h2 = bt.by_period(res2, {"h": PER["HOLDOUT 2024"]})["h"]
    print(f"   {'HOLDOUT 2024, slippage 20bp':26s} {bt.fmt(h2)}")
    print("   by year (strat/bench):", {y: f"{a:+.0%}/{b:+.0%}" for y, (a, b) in bt.yearly(res).items()})
    d = res["dates"]; m24 = d >= np.datetime64("2024-01-01")
    mon = d[m24].astype("datetime64[M]")
    print("   2024 monthly strat:", " ".join(f"{str(u)[5:]}:{np.prod(1+res['ret'][m24][mon==u])-1:+.0%}" for u in np.unique(mon)))
    print("   2024 monthly bench:", " ".join(f"{str(u)[5:]}:{np.prod(1+res['bench'][m24][mon==u])-1:+.0%}" for u in np.unique(mon)))
    navs[lab] = np.cumprod(1 + res["ret"]); navs["bench_universe_ew"] = np.cumprod(1 + res["bench"]); navs["date"] = d.astype(str)
pl.DataFrame(navs).write_csv("results/nav_2020_2024.csv")
print("\ndone", flush=True)
