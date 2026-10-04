"""One-shot final test of the frozen GP selection (results/gp_selected.json), as pre-registered in gp_spec.py.
    FEED_LAST_YEAR=2026 FEED_CACHE=/tmp/feed_cache_full python3 run_gp_final.py
Exports the full panel, evaluates the frozen formulas, reports test-period ICs (each factor and the
composite) and backtests V0 vs V0+GP(lambda) with the CTRL V0-50 settings."""
import json, os, subprocess, sys
assert os.environ.get("FEED_LAST_YEAR") == "2026", "run with FEED_LAST_YEAR=2026"
import numpy as np, polars as pl
import lib, backtest as bt, gp_spec as gs, gp_common as gc

sel = json.load(open("results/gp_selected.json"))
assert sel["chosen"], "nothing was selected"
formulas = [c["expr"] for c in sel["chosen"]]
signs = [c["sign"] for c in sel["chosen"]]
lam = sel["lambda"]
full = "/tmp/gp_data/full"
if not os.path.exists(os.path.join(full, "meta.json")):
    subprocess.run([sys.executable, "gp_export.py", full], check=True)
S = gc.setup(gc.C["sealed_end"])
R = gc.eval_formulas(full, formulas, "/tmp/gp_runs/chosen_full.json", write_dir="/tmp/gp_runs/chosen_full")
facs = [gc.load_factor(S, full, f"/tmp/gp_runs/chosen_full/f{j}.f32") for j in range(len(formulas))]

# sanity: values on dates <= 2024 match the mining export (point-in-time operators)
mine = "/tmp/gp_data/mine"
mm = json.load(open(os.path.join(mine, "meta.json")))
Tm = mm["T"]
cidx = {c: i for i, c in enumerate(S.B.codes)}
cols = [cidx[c] for c in mm["codes"]]
for j in range(len(formulas)):
    fm = np.fromfile(f"/tmp/gp_runs/chosen_mine/f{j}.f32", dtype=np.float32).reshape(mm["N"], Tm).T
    a, b = fm[300:], facs[j][300:Tm, cols]
    both = np.isfinite(a) & np.isfinite(b)
    print(f"  overlap check f{j}: nan mismatch {(np.isfinite(a) != np.isfinite(b)).mean():.1e}, "
          f"max abs diff {np.max(np.abs(a[both] - b[both])):.1e}")

# test-period ICs
meta = json.load(open(os.path.join(full, "meta.json")))
G = gs.SELECT
rows = []
for j, f in enumerate(R["factors"]):
    st = {}
    for sp in ("train", "valid", "test"):
        ic = np.array([0.0 if v is None else v for v in f["splits"][sp]["ic"]])
        st[sp] = (np.nanmean(ic), lib.nw_t(ic, G["nw_lag"]))
    rows.append(dict(factor=formulas[j], sign=signs[j], ic_train=st["train"][0], t_train=st["train"][1],
                     ic_2024=st["valid"][0], t_2024=st["valid"][1], ic_test=st["test"][0], t_test=st["test"][1]))
gp = gc.composite(S, facs, signs)
st = {}
for sp in ("train", "valid", "test"):
    a, b = meta["splits"][sp]
    ic = gc.py_ic(S, gp, range(a, b + 1))
    st[sp] = (np.nanmean(ic), lib.nw_t(ic, G["nw_lag"]))
rows.append(dict(factor="COMPOSITE", sign=1.0, ic_train=st["train"][0], t_train=st["train"][1], ic_2024=st["valid"][0],
                 t_2024=st["valid"][1], ic_test=st["test"][0], t_test=st["test"][1]))
ICT = pl.DataFrame(rows)
ICT.write_csv("results/gp_final_ic.csv")
pl.Config.set_tbl_rows(20); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(70)
print(ICT.with_columns([pl.col(c).round(4) for c in ICT.columns if c not in ("factor",)]))

# backtests
PER = {"IN-SAMPLE 2020-04..2024-12": (gc.C["run_start"], "2024-12-31"),
       "SEALED 2025-01..2026-09": (gc.C["sealed_start"], gc.C["sealed_end"]),
       "2025": ("2025-01-01", "2025-12-31"), "2026 YTD": ("2026-01-01", gc.C["sealed_end"])}
out = []
for name, score in (("V0", gc.overlay(S, gp, 0.0)), (f"V0+GP lam {lam}", gc.overlay(S, gp, lam))):
    for cl, cost in (("10bp", bt.Costs(capital=gc.C["capital"])), ("20bp", bt.Costs(slippage=0.002, capital=gc.C["capital"]))):
        for k, m in gc.run_bt(S, score, gc.C["run_start"], gc.C["sealed_end"], cost, PER).items():
            out.append(dict(version=name, cost=cl, period=k, **m))
        print(f"  done {name} {cl}", flush=True)
BT = pl.DataFrame(out)
BT.write_csv("results/gp_final_backtest.csv")
print(BT.with_columns([pl.col(c).round(3) for c in ("ann", "ann_min", "mdd_worst", "sharpe", "excess")]))

sealed = "SEALED 2025-01..2026-09"
g = lambda v, c="10bp": BT.filter((pl.col("version") == v) & (pl.col("cost") == c) & (pl.col("period") == sealed)).row(0, named=True)
v0, vg = g("V0"), g(f"V0+GP lam {lam}")
ic_ok = st["test"][0] > 0 and st["test"][1] >= gs.TEST_PASS["t_test"]
bt_ok = vg["ann"] > v0["ann"] and vg["mdd_worst"] >= v0["mdd_worst"] - gs.TEST_PASS["mdd_slack"]
print(f"\nPASS composite IC: {ic_ok} (test IC {st['test'][0]:+.4f}, NW t {st['test'][1]:+.2f})")
print(f"PASS backtest   : {bt_ok} (V0+GP ann {vg['ann']:+.1%} mdd {vg['mdd_worst']:+.1%} vs V0 ann {v0['ann']:+.1%} mdd {v0['mdd_worst']:+.1%})")
print("OVERALL:", "PASS" if ic_ok and bt_ok else "FAIL")
