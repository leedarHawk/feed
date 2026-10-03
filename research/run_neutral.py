"""Size-neutral long-only books. SPECS FIXED BEFORE THE FIRST RUN (committed before execution):

  V1: universe = ADV>=10M (as before); score = signals.size_neutral; top 100, rebalance 10d, buffer 1.5
  V2: universe = V1 universe restricted to the 1000 largest names by market cap each day; same score
      (re-ranked inside V2); top 100, rebalance 10d, buffer 1.5

vwapdev20 / maxr60 were selected on 2019-2023 data, so 2019-04..2023-12 is IN-SAMPLE for these specs.
2024 is a semi-independent check (decided before looking at these factors in 2024). 2025-2026 sealed.
Benchmarks: each book's own universe, equal-weighted (daily). Reported once; nothing re-tuned after.
"""
import time, numpy as np, polars as pl
import lib, backtest as bt, signals
t0 = time.time()
B = lib.build_base(end="2024-12-31"); F = lib.build_factors(B)
U1 = lib.make_U(B, 1e7)
mc = np.where(U1, B.P["market_cap"], np.nan)
rk = lib.rank_rows(-mc)                                  # 0 = largest
U2 = U1 & (rk < 1000)
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, z["dates"], allf)
PER = {"IN-SAMPLE 2019-04..2023-12": ("2019-04-01", "2023-12-29"), "2024 (semi-indep.)": ("2024-01-01", "2024-12-31")}
navs = {}
for lab, U in (("V1_adv10M", U1), ("V2_top1000cap", U2)):
    S = signals.size_neutral(B, F, signs, U)
    print(f"\n[{lab}] universe size/day {U[250:].sum(1).mean():.0f}   (score built {time.time()-t0:.0f}s)", flush=True)
    for costs, cl in ((bt.Costs(), "slip 10bp"), (bt.Costs(slippage=0.002), "slip 20bp")):
        res = bt.run(B, S, "2019-04-01", "2024-12-31", n_hold=100, rebal=10, U=U, costs=costs, return_weights=(cl == "slip 10bp"))
        for k, m in bt.by_period(res, PER).items():
            print(f"   {cl}  {k:27s} {bt.fmt(m)}")
        if cl == "slip 10bp":
            W = res["weights"]; t_ = int(np.searchsorted(B.dates, res["dates"][0]))
            zs = backtest_size = (lib.rank_rows(np.where(U, B.P["market_cap"], np.nan)) /
                                  np.maximum(np.sum(U, 1, keepdims=True) - 1, 1))[t_:t_ + len(W)]
            port_pct = np.nansum(W * np.nan_to_num(zs), 1) / np.maximum(W.sum(1), 1e-9)
            print(f"   size check: portfolio avg market-cap percentile within its universe {np.nanmean(port_pct):.2f} (0.5 = neutral)")
            print("   by year (strat/bench):", {y: f"{a:+.0%}/{b:+.0%}" for y, (a, b) in bt.yearly(res).items()})
            d = res["dates"]; m24 = d >= np.datetime64("2024-01-01"); mon = d[m24].astype("datetime64[M]")
            print("   2024 monthly strat:", " ".join(f"{str(u)[5:]}:{np.prod(1+res['ret'][m24][mon==u])-1:+.0%}" for u in np.unique(mon)))
            print("   2024 monthly bench:", " ".join(f"{str(u)[5:]}:{np.prod(1+res['bench'][m24][mon==u])-1:+.0%}" for u in np.unique(mon)))
            navs[lab] = np.cumprod(1 + res["ret"]); navs[f"bench_{lab}"] = np.cumprod(1 + res["bench"]); navs["date"] = d.astype(str)
pl.DataFrame(navs).write_csv("results/nav_size_neutral.csv")
print("\ndone", flush=True)
