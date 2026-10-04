"""Run the pre-registered final test defined in final_spec.py. Requires FEED_LAST_YEAR=2026 and a separate
FEED_CACHE so the research cache (<= 2024) is untouched."""
import os
assert os.environ.get("FEED_LAST_YEAR") == "2026", "run with FEED_LAST_YEAR=2026"
import numpy as np, polars as pl
import lib, backtest as bt, signals, risk_rules as rr, final_spec as fs
C = fs.COMMON
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)          # TRAIN-period ICs (frozen signs)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, z["dates"], allf)
B = lib.build_base(end=C["sealed_end"], min_age_days=C["min_age_days"]); F = lib.build_factors(B)
print(f"panel {B.dates[0]}..{B.dates[-1]}, {len(B.dates)} days", flush=True)
pre = np.array([c[:3] for c in B.codes])
U = lib.make_U(B, C["adv_floor"]) & (~np.isin(pre, ["688", "689", "300", "301"]))[None, :]
SC = signals.mainboard_scores(B, F, signs, U)
crowd = np.where(rr.states(B, U)["crowded"], 0.5, 1.0)
d = B.dates; T = len(d); jan = np.ones(T)
for y in range(2019, 2027):
    i = int(np.searchsorted(d, np.datetime64(f"{y}-01-31"), side="right")) - 1
    if i >= 10 and str(d[i])[:7] == f"{y}-01":
        jan[i - 10:i] = 0.0
s0 = np.datetime64(C["sealed_start"]); s1 = np.datetime64(C["sealed_end"])
PER = {"IN-SAMPLE 2020-04..2024-12": (C["run_start"], "2024-12-31"), "SEALED 2025-01..2026-09": (C["sealed_start"], C["sealed_end"]),
       "2025": ("2025-01-01", "2025-12-31"), "2026 YTD": ("2026-01-01", C["sealed_end"])}
rows = []
for vname, v in fs.VERSIONS.items():
    ex = rr.ramp(crowd, v["step"], every=1) * (jan if v["january"] else 1.0)
    for cl, cost in (("10bp", bt.Costs(capital=C["capital"])), ("20bp", bt.Costs(slippage=0.002, capital=C["capital"]))):
        per = {k: [] for k in PER}
        for o in C["offsets"]:
            r = bt.run(B, SC[v["score"]], C["run_start"], C["sealed_end"], n_hold=v["n_hold"], rebal=C["rebal"], buffer=C["buffer"],
                       U=U, costs=cost, exposure=ex, exposure_trade="names", rebal_offset=o)
            for k, m in bt.by_period(r, PER).items():
                per[k].append(m)
        for k, ms in per.items():
            rows.append(dict(version=vname, cost=cl, period=k, ann_mean=float(np.mean([m["ann"] for m in ms])),
                             ann_min=float(min(m["ann"] for m in ms)), ann_max=float(max(m["ann"] for m in ms)),
                             mdd_worst=float(min(m["mdd"] for m in ms)), sharpe_mean=float(np.mean([m["sharpe"] for m in ms])),
                             bench_ann=float(np.mean([m["bench_ann"] for m in ms])), excess_mean=float(np.mean([m["excess_ann"] for m in ms]))))
        print(f"  done {vname} {cl}", flush=True)
R = pl.DataFrame(rows)
R.write_csv("results/final_test_sealed_2025_2026.csv")
# index context (same windows, price index, no dividends)
ix = pl.read_parquet(os.path.join(os.path.dirname(__file__), "..", "data", "smallcap_risk_v1_20261004", "index_daily.parquet"))
ix = ix.with_columns(pl.col("date").cast(pl.Date))
for code, nm in (("000852.XSHG", "CSI 1000"), ("000300.XSHG", "HS300"), ("399303.XSHE", "Guozheng 2000")):
    for k, (a, b) in PER.items():
        s = ix.filter((pl.col("index_code") == code) & (pl.col("date") >= pl.lit(a).str.to_date()) & (pl.col("date") <= pl.lit(b).str.to_date())).sort("date")
        if s.height > 20:
            c = s["close"].to_numpy(); n = s.height
            print(f"  index {nm:14s} {k:28s} ann {(c[-1] / c[0]) ** (252 / n) - 1:+.1%}  mdd {(c / np.maximum.accumulate(c) - 1).min():+.1%}")
pl.Config.set_tbl_rows(40); pl.Config.set_tbl_cols(12); pl.Config.set_fmt_float("mixed"); pl.Config.set_tbl_width_chars(240)
print(R.with_columns([pl.col(c).round(3) for c in R.columns if c not in ("version", "cost", "period")]))
