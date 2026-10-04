"""Apply the pre-registered selection (gp_spec.py) to the 4 halls of fame and pick lambda in-sample.
Research cache only (<= 2024).    python3 run_gp_select.py /tmp/gp_data/mine
Writes results/gp_pool.csv (every pooled formula with its statistics and the rule it failed) and
results/gp_selected.json (frozen formulas, signs, lambda) -> the input of run_gp_final.py."""
import json, os, sys
assert os.environ.get("FEED_LAST_YEAR") is None, "selection uses the research cache only"
import numpy as np, polars as pl
import lib, backtest as bt, gp_spec as gs, gp_common as gc

data = sys.argv[1]
G = gs.SELECT
exprs, seen = [], set()
for s in gs.SEEDS:
    for m in json.load(open(f"/tmp/gp_runs/hof_seed{s}.json"))["members"]:
        if m["expr"] not in seen:
            seen.add(m["expr"]); exprs.append(m["expr"])
print(f"pooled {len(exprs)} distinct formulas from {len(gs.SEEDS)} runs", flush=True)
R = gc.eval_formulas(data, exprs, "/tmp/gp_runs/pool_eval.json")
corr = np.array(R["train_corr"])
meta = json.load(open(os.path.join(data, "meta.json")))
yrs = meta["years"]

rows = []
for k, f in enumerate(R["factors"]):
    tr, va = f["splits"]["train"], f["splits"]["valid"]
    ic = np.array([0.0 if v is None else v for v in tr["ic"]]); t_idx = np.array(tr["t"])
    icv = np.array([0.0 if v is None else v for v in va["ic"]])
    mu = np.nanmean(ic); sign = float(np.sign(mu))
    ym = [np.nanmean(ic[(t_idx >= yrs[str(y)][0]) & (t_idx <= yrs[str(y)][1])]) for y in (2020, 2021, 2022, 2023)]
    rows.append(dict(k=k, expr=f["expr"], nodes=f["nodes"], sign=sign, ic_train=mu, icir_train=mu / np.nanstd(ic),
                     t_train=lib.nw_t(ic, G["nw_lag"]), years_same=int(sum(np.sign(y) == sign for y in ym)),
                     ic_2020=ym[0], ic_2021=ym[1], ic_2022=ym[2], ic_2023=ym[3],
                     ic_valid=np.nanmean(icv), t_valid=lib.nw_t(icv, G["nw_lag"]), cover=tr["cover"]))
# 1. dedup at |corr| >= 0.7, keep the better train |t|
order = sorted(range(len(rows)), key=lambda i: -abs(rows[i]["t_train"]))
kept = []
for i in order:
    if all(abs(corr[i, j]) < G["dedup_corr"] for j in kept):
        kept.append(i); rows[i]["status"] = "pooled"
    else:
        rows[i]["status"] = "duplicate"
# 2. filters
for i in kept:
    r = rows[i]
    if abs(r["t_train"]) < G["t_train"]:
        r["status"] = "fail: train t"
    elif r["years_same"] < 4:
        r["status"] = "fail: train years"
    elif not (np.sign(r["ic_valid"]) == r["sign"] and r["t_valid"] * r["sign"] >= G["t_valid"]):
        r["status"] = "fail: 2024"
    else:
        r["status"] = "passed"
# 3. greedy decorrelation
chosen = []
for i in kept:
    if rows[i]["status"] != "passed":
        continue
    if len(chosen) < G["max_factors"] and all(abs(corr[i, j]) < G["max_corr"] for j in chosen):
        chosen.append(i); rows[i]["status"] = "CHOSEN"
    else:
        rows[i]["status"] = "passed, not chosen (corr / cap)"
df = pl.DataFrame(rows).sort(pl.col("t_train").abs(), descending=True)
df.write_csv("results/gp_pool.csv")
print(df.group_by("status").len().sort("len", descending=True))
pl.Config.set_tbl_rows(30); pl.Config.set_tbl_width_chars(250); pl.Config.set_fmt_str_lengths(80)
show = ["expr", "sign", "ic_train", "t_train", "years_same", "ic_valid", "t_valid", "status"]
print(df.filter(pl.col("status").str.starts_with("CHOSEN") | pl.col("status").str.starts_with("passed")).select(show)
      .with_columns([pl.col(c).round(4) for c in ("ic_train", "t_train", "ic_valid", "t_valid")]))
if not chosen:
    json.dump({"chosen": [], "note": "nothing passes the pre-registered rules"}, open("results/gp_selected.json", "w"), indent=1)
    sys.exit("nothing passes -> no overlay test (pre-registered)")
print("chosen-factor train correlation:\n", np.round(corr[np.ix_(chosen, chosen)], 2))

# 4. lambda, in-sample 2020-04 .. 2024-12, 10bp
S = gc.setup("2024-12-31")
wdir = "/tmp/gp_runs/chosen_mine"
gc.eval_formulas(data, [rows[i]["expr"] for i in chosen], "/tmp/gp_runs/chosen_mine.json", write_dir=wdir)
facs = [gc.load_factor(S, data, f"{wdir}/f{j}.f32") for j in range(len(chosen))]
gp = gc.composite(S, facs, [rows[i]["sign"] for i in chosen])
PER = {"IN-SAMPLE 2020-04..2024-12": (gc.C["run_start"], "2024-12-31")}
res = {}
for lam in (0.0,) + gs.LAMBDAS:
    m = gc.run_bt(S, gc.overlay(S, gp, lam), gc.C["run_start"], "2024-12-31", bt.Costs(capital=gc.C["capital"]), PER)
    res[lam] = m[list(PER)[0]]
    print(f"  lam {lam:.2f}: ann {res[lam]['ann']:+.1%} (min {res[lam]['ann_min']:+.1%})  sharpe {res[lam]['sharpe']:.2f}  "
          f"worst mdd {res[lam]['mdd_worst']:+.1%}", flush=True)
lam = max(gs.LAMBDAS, key=lambda x: res[x]["sharpe"])
out = {"chosen": [dict(expr=rows[i]["expr"], sign=rows[i]["sign"], t_train=rows[i]["t_train"], ic_train=rows[i]["ic_train"],
                       t_valid=rows[i]["t_valid"]) for i in chosen],
       "lambda": lam, "in_sample": {str(k): v for k, v in res.items()}}
json.dump(out, open("results/gp_selected.json", "w"), indent=1)
print(f"lambda = {lam} -> results/gp_selected.json")
