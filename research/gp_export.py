"""Export daily matrices for the Rust GP miner (research/gpminer).

    python3 gp_export.py <out_dir>                       # mining data, research cache (<= 2024)
    FEED_LAST_YEAR=2026 FEED_CACHE=/tmp/feed_cache_full python3 gp_export.py <out_dir>   # final test only

Layout: every matrix is float32, stock-major ([n][t], time contiguous), written as raw little-endian.
Terminals are point-in-time and scale-free; qfq price LEVELS are never exported (only returns built
from them). meta.json carries dates, codes, terminal names and the train/valid/test date ranges.

Mining universe M (where the IC is measured): V0's main-board universe after the junk filter,
restricted to the least liquid 30% of the universe (small score zrank(-lmoney20) >= 0.4), which is
where V0 picks its holdings. Cross-sectional operators (cs_rank, cs_z) run inside the tradable
universe U of each day, so they mean the same thing in the mining and the final-test exports.
"""
import json, os, sys
import numpy as np
import lib, backtest, signals

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
last_year = int(os.environ.get("FEED_LAST_YEAR", "2024"))
end = "2024-12-31" if last_year == 2024 else "2026-09-30"
H = 20

B = lib.build_base(end=end, min_age_days=365)
F = lib.build_factors(B)
P = B.P
pre = np.array([c[:3] for c in B.codes])
U = lib.make_U(B, 5e6) & (~np.isin(pre, ["688", "689", "300", "301"]))[None, :]
z = np.load("/tmp/feed_cache/ics_discovery.npz", allow_pickle=True)          # TRAIN-period ICs (frozen signs)
ics = {(k.split("|")[0], int(k.split("|")[1])): z[k] for k in z.files if k != "dates"}
allf = sorted({f for fam in signals.FAMILIES.values() for f in fam})
signs = signals.train_signs(ics, z["dates"], allf)
comp_r = backtest.zrank(signals.composite(B, F, signs, U=U), U)
small = backtest.zrank(-F["lmoney20"], U)
M = U & (comp_r > -0.4) & (small >= 0.4)

cols = np.where(U.any(0))[0]
T, N = len(B.dates), len(cols)
v = B.valid
with np.errstate(invalid="ignore", divide="ignore"):
    hi, lo, cl, money, vol = P["high"], P["low"], P["close"], P["money"], P["volume"]
    lr = np.log1p(np.clip(np.nan_to_num(B.r), -0.95, None))
    term = {
        "ret": B.r,
        "oret": B.on,
        "iret": B.idr,
        "hl": hi / lo - 1.0,
        "ch": cl / hi - 1.0,
        "cl": cl / lo - 1.0,
        "vwapc": (money / vol) / cl - 1.0,
        "turn": vol / (P["circulating_cap"] * 1e4),
        "lmoney": np.log(money),
        "lcap": np.log(P["circulating_market_cap"]),
        "lprice": np.log(cl),
        "uplim": B.lim_up_close.astype(np.float64),
        "dnlim": B.lim_dn_close.astype(np.float64),
        "cumr": np.cumsum(lr, axis=0),
    }


def put(name, a):
    a = np.where(np.isfinite(a), a, np.nan).astype(np.float32)[:, cols]
    np.ascontiguousarray(a.T).tofile(os.path.join(out, name + ".f32"))


for k, a in term.items():
    put(k, np.where(v, a, np.nan))
put("fwd", np.where(M, B.fwd[H], np.nan))
put("small", np.where(M, small, np.nan))
np.ascontiguousarray(M[:, cols].T).astype(np.uint8).tofile(os.path.join(out, "mask.u8"))
np.ascontiguousarray(U[:, cols].T).astype(np.uint8).tofile(os.path.join(out, "univ.u8"))   # cross-sectional ops run inside U

d = B.dates
yr = d.astype("datetime64[Y]").astype(int) + 1970


def span(y0, y1):
    """t-range whose forward window (opens t+1 .. t+1+H) stays inside years y0..y1."""
    idx = np.where((yr >= y0) & (yr <= y1))[0]
    return [int(idx[0]), int(idx[-1] - (H + 1))]


splits = {"train": span(2020, 2023), "valid": span(2024, 2024)}
if last_year > 2024:
    idx = np.where(yr >= 2025)[0]
    splits["test"] = [int(idx[0]), int(T - 1 - (H + 1))]
years = {str(y): [int(np.where(yr == y)[0][0]), int(np.where(yr == y)[0][-1])] for y in range(2019, int(yr[-1]) + 1)}
meta = dict(T=T, N=N, H=H, terminals=list(term), dates=[str(x) for x in d], codes=[str(c) for c in B.codes[cols]],
            splits=splits, years=years, end=end)
json.dump(meta, open(os.path.join(out, "meta.json"), "w"))
nm = M.sum(1)
tr = splits["train"]
print(f"T={T} N={N} end={end} splits={splits}")
print(f"mining universe size: train median {int(np.median(nm[tr[0]:tr[1] + 1]))}, "
      f"min {int(nm[tr[0]:tr[1] + 1].min())}; U median {int(np.median(U.sum(1)[tr[0]:tr[1] + 1]))}")
for k, a in term.items():
    x = np.where(v, a, np.nan)[:, cols]
    print(f"  {k:7s} finite {np.isfinite(x[v[:, cols]]).mean():.4f}")
