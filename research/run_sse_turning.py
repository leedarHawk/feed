"""SSE Composite (上证指数) turning points: how volume, MACD, RSI and Fibonacci behave around tops and
bottoms, and whether their textbook signals catch the turns. Index timing only, no stock selection.

Everything below is fixed before the first run (no tuning):
  turning points  zigzag on closes, theta = 10% (intermediate) and 20% (major); 5%/15% only for Fibonacci
  signals         daily unless marked 周 (weekly bars, fired at the week's last close); each signal is
                  debounced: a repeat within 10 trading days of its previous fire is dropped
    volume        地量 MA5 volume at a 250-day low (bull) / 天量 at a 250-day high (bear)
                  量能由缩转放 MA5 vol crosses above MA60 vol after MA5/MA60 <= 0.75 in the prior 20 days (bull)
                  量能由放转缩 MA5 vol crosses below MA60 vol after MA5/MA60 >= 1.33 in the prior 20 days (bear)
                  量价背离 close at a 120-day high while MA10 vol < 80% of its 120-day max (bear)
                  恐慌放量下跌 volume >= 2x MA20 vol and day return <= -3% (bull)
                  放量大涨 volume >= 2x MA20 vol and day return >= +3% (bull)
    MACD 12/26/9  golden/death cross, split by zero axis; divergence at the cross (sse.macd_divergence)
    RSI (Wilder)  RSI14 entering/leaving <30 and >70; RSI6 entering <20 / >80; weekly RSI14 <30 / >70
    benchmark     the filter rule itself: the zigzag confirmation day (index already 10% off the extreme)
    post hoc      两个「事后组合」(超卖缩量 RSI14<30 & 量比<0.8; 超买放量 RSI14>70 & 量比>1.5) were added AFTER
                  seeing the single-signal results and are labelled as such; treat them as descriptive
  grading
    catch         an event "catches" a turn if a same-kind theta-10% turning point is within +-20 trading
                  days; base = share of all days that are within +-20 days of one (a random signal's score)
    recall        share of turning points with an event within +-20 days; lag/dist = first such event's
                  timing (trading days, negative = before the turn) and close vs the turning-point close
    trade         bull: P(+10% before -5%), bear: P(-10% before +5%) from the signal close (250-day cap);
                  mean forward log returns 20/60/120d; p = share of random same-size day samples at least
                  as good on 60d (ignores clustering, indicative only)
Also printed: indicator paths around turning points, states on the turning-point day (and the reverse
probability), where each indicator makes its own extreme relative to the price turn (symmetric +-40 days),
the theta-20% turning-point table, and Fibonacci retracement / time tests against shifted placebo levels.
Research sample 1997-01..2024-12 is graded on data cut at 2024-12-31. The recent window 2025-01..2026-09 is
graded separately on the full series.
"""
import math
import numpy as np
import polars as pl
import sse

COOL, W = 10, 20
OFFS = (-40, -20, -10, -5, -1, 0, 1, 5, 10, 20, 40)


def indicators(s):
    c, v = s.close, s.volume
    I = {}
    I["dif"], I["dea"], I["hist"] = sse.macd(c)
    I["rsi14"], I["rsi6"] = sse.rsi(c, 14), sse.rsi(c, 6)
    for k in (5, 10, 20, 60, 250):
        I[f"mv{k}"] = sse.sma(v, k)
    I["vr"] = I["mv5"] / I["mv250"]
    I["ret"] = np.r_[np.nan, c[1:] / c[:-1] - 1]
    I["ma20"] = sse.sma(c, 20)
    w = sse.weekly(s)
    wd, we, _ = sse.macd(w.close)
    wr = sse.rsi(w.close, 14)
    I["w"] = dict(idx=w.idx, dif=wd, dea=we, rsi14=wr, close=w.close)
    # weekly RSI of the last completed week, carried on daily rows (causal)
    wr_d = np.full(len(c), np.nan)
    wr_d[w.idx] = wr
    I["wrsi14"] = wr_d
    return I


def cross_both(a, b):
    """First day on which both conditions hold (after a day on which they did not)."""
    x = np.nan_to_num(a, nan=0).astype(bool) & np.nan_to_num(b, nan=0).astype(bool)
    return x & ~np.r_[False, x[:-1]]


def signals(s, I):
    c, n = s.close, len(s.close)
    mv5, mv10, mv20, mv60 = I["mv5"], I["mv10"], I["mv20"], I["mv60"]
    prev_mv20 = np.r_[np.nan, mv20[:-1]]
    q = mv5 / mv60
    qmin = np.r_[np.nan, sse.rmin(np.nan_to_num(q, nan=9.0), 20)[:-1]]
    qmax = np.r_[np.nan, sse.rmax(np.nan_to_num(q, nan=0.0), 20)[:-1]]
    dif, dea = I["dif"], I["dea"]
    gc, dc = sse.cross_up(dif, dea), sse.cross_dn(dif, dea)
    bot, top = sse.macd_divergence(c, dif, dea)
    r14, r6 = I["rsi14"], I["rsi6"]

    def weekly_flag(fn):
        out = np.zeros(n, bool)
        out[I["w"]["idx"][fn]] = True
        return out
    wk = I["w"]
    wgc, wdc = sse.cross_up(wk["dif"], wk["dea"]), sse.cross_dn(wk["dif"], wk["dea"])
    wbot, wtop = sse.macd_divergence(wk["close"], wk["dif"], wk["dea"])
    up10, dn10 = sse.confirm_flags(sse.zigzag(c, 0.10), n)
    with np.errstate(invalid="ignore"):
        S = {
            ("成交量", "地量: MA5量创250日新低"): (mv5 <= sse.rmin(mv5, 250) + 1e-9, "bull"),
            ("成交量", "天量: MA5量创250日新高"): (mv5 >= sse.rmax(mv5, 250) - 1e-9, "bear"),
            ("成交量", "量能由缩转放"): (sse.cross_up(mv5, mv60) & (qmin <= 0.75), "bull"),
            ("成交量", "量能由放转缩"): (sse.cross_dn(mv5, mv60) & (qmax >= 1.33), "bear"),
            ("成交量", "量价背离: 价创120日高、量不足"): ((c >= sse.rmax(c, 120) - 1e-9) & (mv10 < 0.8 * sse.rmax(mv10, 120)), "bear"),
            ("成交量", "恐慌放量下跌(≥2倍量,跌≥3%)"): ((s.volume >= 2 * prev_mv20) & (I["ret"] <= -0.03), "bull"),
            ("成交量", "放量大涨(≥2倍量,涨≥3%)"): ((s.volume >= 2 * prev_mv20) & (I["ret"] >= 0.03), "bull"),
            ("MACD", "金叉(全部)"): (gc, "bull"),
            ("MACD", "零轴下金叉"): (gc & (dif < 0), "bull"),
            ("MACD", "零轴上金叉"): (gc & (dif >= 0), "bull"),
            ("MACD", "底背离金叉"): (bot, "bull"),
            ("MACD", "死叉(全部)"): (dc, "bear"),
            ("MACD", "零轴上死叉"): (dc & (dif > 0), "bear"),
            ("MACD", "零轴下死叉"): (dc & (dif <= 0), "bear"),
            ("MACD", "顶背离死叉"): (top, "bear"),
            ("MACD", "周金叉"): (weekly_flag(wgc), "bull"),
            ("MACD", "周底背离金叉"): (weekly_flag(wbot), "bull"),
            ("MACD", "周死叉"): (weekly_flag(wdc), "bear"),
            ("MACD", "周顶背离死叉"): (weekly_flag(wtop), "bear"),
            ("RSI", "RSI14下穿30(进入超卖)"): (sse.cross_dn(r14, 30), "bull"),
            ("RSI", "RSI14上穿30(离开超卖)"): (sse.cross_up(r14, 30), "bull"),
            ("RSI", "RSI6下穿20"): (sse.cross_dn(r6, 20), "bull"),
            ("RSI", "周RSI14<30"): (weekly_flag(sse.cross_dn(wk["rsi14"], 30)), "bull"),
            ("RSI", "RSI14上穿70(进入超买)"): (sse.cross_up(r14, 70), "bear"),
            ("RSI", "RSI14下穿70(离开超买)"): (sse.cross_dn(r14, 70), "bear"),
            ("RSI", "RSI6上穿80"): (sse.cross_up(r6, 80), "bear"),
            ("RSI", "周RSI14>70"): (weekly_flag(sse.cross_up(wk["rsi14"], 70)), "bear"),
            ("事后组合", "超卖缩量: RSI14<30且量比<0.8"): (cross_both(r14 < 30, I["vr"] < 0.8), "bull"),
            ("事后组合", "超买放量: RSI14>70且量比>1.5"): (cross_both(r14 > 70, I["vr"] > 1.5), "bear"),
            ("基准", "过滤规则: 低点反弹10%确认"): (up10, "bull"),
            ("基准", "过滤规则: 高点回落10%确认"): (dn10, "bear"),
        }
    return {k: (sse.debounce(np.nan_to_num(e, nan=0).astype(bool), COOL), side) for k, (e, side) in S.items()}


def grade(s, I, SIG, win):
    c, n = s.close, len(s.close)
    tp = {th: [p for p in sse.zigzag(c, th) if win[p[0]]] for th in (0.10, 0.20)}
    fp = {"bull": sse.first_passage(c, 0.10, 0.05), "bear": -sse.first_passage(c, 0.05, 0.10)}
    fw = {h: sse.fwd_log(c, h) for h in (20, 60, 120)}
    rows = []
    for (fam, name), (e, side) in SIG.items():
        kind = "B" if side == "bull" else "T"
        ev = np.where(e & win)[0]
        idx10 = [p[0] for p in tp[0.10] if p[1] == kind]
        idx20 = [p[0] for p in tp[0.20] if p[1] == kind]
        base = np.zeros(n, bool)
        for p in idx10:
            base[max(p - W, 0):p + W + 1] = True
        hit, _ = sse.near(ev, idx10, W)
        _, first10 = sse.near(ev, idx10, W)
        _, first20 = sse.near(ev, idx20, W)
        caught20 = first20 >= 0
        sgn = 1 if side == "bull" else -1
        r = dict(family=fam, signal=name, side=side, n=len(ev), per_year=len(ev) / (win.sum() / 244),
                 catch=hit.mean() if len(ev) else np.nan, catch_base=base[win].mean(),
                 recall10=(first10 >= 0).mean() if len(idx10) else np.nan,
                 recall20=caught20.mean() if len(idx20) else np.nan,
                 lag20=float(np.median(first20[caught20] - np.array(idx20)[caught20])) if caught20.any() else np.nan,
                 dist20=float(np.median(c[first20[caught20]] / c[np.array(idx20)[caught20]] - 1)) if caught20.any() else np.nan)
        r["catch_lift"] = r["catch"] / r["catch_base"] if r["catch_base"] > 0 else np.nan
        f = fp[side]
        ok = ev[~np.isnan(f[ev])]
        r["p_win"] = float((f[ok] == 1).mean()) if len(ok) else np.nan
        fb = f[win][~np.isnan(f[win])]
        r["p_win_base"] = float((fb == 1).mean())
        for h, x in fw.items():
            okh = ev[~np.isnan(x[ev])]
            r[f"fwd{h}"] = float(x[okh].mean()) if len(okh) else np.nan
            r[f"fwd{h}_base"] = float(np.nanmean(x[win]))
        okh = ev[~np.isnan(fw[60][ev])]
        obs = sgn * r["fwd60"]
        r["p_rand60"] = sse.rand_p(sgn * fw[60], win, len(okh), obs)
        rows.append(r)
    return pl.DataFrame(rows), tp


def print_grades(G, title):
    print(f"\n{title}")
    print("  信号                              方向   次数  次/年 | 抓转折(±20日) 随机基准  倍数 | 召回10% 召回20%  滞后(日) 距转折点 |"
          " 胜率(10%先于5%) 基准 | 20日均值 60日均值 120日均值 (基准: 20/60/120) | p随机60日")
    for r in G.iter_rows(named=True):
        print(f"  {r['family'][:3]:3s} {r['signal'][:28]:28s} {r['side']:4s} {r['n']:5d} {r['per_year']:5.1f} |"
              f"  {sse_pct(r['catch'])} {sse_pct(r['catch_base'])} {r['catch_lift']:5.2f} |"
              f" {sse_pct(r['recall10'])} {sse_pct(r['recall20'])} {r['lag20']:+7.0f} {sse_pct(r['dist20'], sign=True)} |"
              f"  {sse_pct(r['p_win'])} {sse_pct(r['p_win_base'])} |"
              f" {sse_pct(r['fwd20'], sign=True)} {sse_pct(r['fwd60'], sign=True)} {sse_pct(r['fwd120'], sign=True)}"
              f"  ({sse_pct(r['fwd20_base'], sign=True)}/{sse_pct(r['fwd60_base'], sign=True)}/{sse_pct(r['fwd120_base'], sign=True)}) |"
              f"  {r['p_rand60']:.3f}")


def sse_pct(x, sign=False):
    if x is None or not np.isfinite(x):
        return "    nan"
    return f"{x * 100:+6.1f}%" if sign else f"{x * 100:6.1f}%"


def profiles(s, I, tp, win):
    """Median indicator values at event-time offsets around tops and bottoms, vs all research days."""
    c = s.close
    feats = {"量比 MA5量/MA250量": I["vr"], "RSI14": I["rsi14"], "RSI6": I["rsi6"],
             "MACD柱/收盘 (%)": 100 * I["hist"] / c, "DIF/收盘 (%)": 100 * I["dif"] / c}
    for th in (0.20, 0.10):
        for kind, lab in (("T", "顶"), ("B", "底")):
            idx = np.array([p[0] for p in tp[th] if p[1] == kind])
            print(f"\n  theta={th:.0%} {lab}部 (n={len(idx)}) 指标中位数，按相对转折点的交易日偏移:")
            print(f"    {'指标':22s}" + "".join(f"{o:>8d}" for o in OFFS) + "   全样本中位数  转折日值所处全样本分位")
            for name, x in feats.items():
                vals = []
                for o in OFFS:
                    j = idx + o
                    j = j[(j >= 0) & (j < len(c))]
                    vals.append(np.nanmedian(x[j]))
                allv = x[win & ~np.isnan(x)]
                pct = np.mean([np.mean(allv <= x[i]) for i in idx])
                print(f"    {name:22s}" + "".join(f"{v:8.2f}" for v in vals) + f"   {np.median(allv):10.2f}   {pct:8.0%}")


def conditions(s, I, tp, win):
    """How often a state holds on the turning-point day, how often it holds on any day, and the reverse
    question: on a day the state holds, is a same-kind theta-10% turning point within +-10 days?"""
    c, n = s.close, len(s.close)
    wr = I["wrsi14"].copy()
    wr = np.where(np.isnan(wr), np.nan, wr)
    # carry the last completed weekly RSI forward on daily rows
    idx = np.where(~np.isnan(wr), np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    wr = wr[idx]
    C = {"B": {"RSI14<30": I["rsi14"] < 30, "RSI6<20": I["rsi6"] < 20, "周RSI14<35": wr < 35,
               "量比<0.8 (缩量)": I["vr"] < 0.8, "DIF<0 (零轴下)": I["dif"] < 0,
               "RSI14<30 且 量比<0.8": (I["rsi14"] < 30) & (I["vr"] < 0.8)},
         "T": {"RSI14>70": I["rsi14"] > 70, "RSI6>80": I["rsi6"] > 80, "周RSI14>65": wr > 65,
               "量比>1.3 (放量)": I["vr"] > 1.3, "DIF>0 (零轴上)": I["dif"] > 0,
               "RSI14>70 且 量比>1.5": (I["rsi14"] > 70) & (I["vr"] > 1.5)}}
    print("\n转折点当天的状态（研究样本）：P(状态|转折点) vs P(状态|任意一天)，以及反过来 P(±10日内有10%级别同向转折|状态)")
    print(f"  {'状态':24s} {'20%转折点':>10s} {'10%转折点':>10s} {'任意一天':>9s} | {'看到状态时±10日内有转折':>22s} {'随机一天':>8s}")
    rows = []
    for kind, lab in (("B", "底部"), ("T", "顶部")):
        i10 = np.array([p[0] for p in tp[0.10] if p[1] == kind])
        i20 = np.array([p[0] for p in tp[0.20] if p[1] == kind])
        nearm = np.zeros(n, bool)
        for p in i10:
            nearm[max(p - 10, 0):p + 11] = True
        print(f"  -- {lab}")
        for name, m in C[kind].items():
            m = np.nan_to_num(m, nan=0).astype(bool)
            r = dict(kind=lab, state=name, at_tp20=m[i20].mean(), at_tp10=m[i10].mean(), any_day=m[win].mean(),
                     near_given_state=nearm[win & m].mean() if (win & m).any() else np.nan, near_base=nearm[win].mean())
            rows.append(r)
            print(f"  {name:24s} {r['at_tp20']:10.0%} {r['at_tp10']:10.0%} {r['any_day']:9.0%} | {r['near_given_state']:22.0%} {r['near_base']:8.0%}")
    return pl.DataFrame(rows)


def lead_lag(s, I, tp, w=40):
    """Where, relative to the price turning point, does each indicator make its own extreme? Symmetric
    window [-w, +w] so a randomly placed extreme has median lag 0; negative = the indicator turned first."""
    c = s.close
    feats = {"量比 MA5量/MA250量": I["vr"], "日成交量": s.volume, "RSI14": I["rsi14"], "MACD柱": I["hist"], "DIF": I["dif"]}
    print(f"\n指标极值相对价格转折点的时间差（交易日，对称窗口 ±{w}；负数 = 指标先见顶/底，即量价或指标背离）")
    print(f"  {'':12s} {'指标':18s} {'中位数':>6s} {'25%':>6s} {'75%':>6s} {'领先>5日占比':>12s} {'同步±5日占比':>12s}")
    for th in (0.20, 0.10):
        for kind, lab in (("T", "顶"), ("B", "底")):
            idx = [p[0] for p in tp[th] if p[1] == kind and p[0] - w >= 0 and p[0] + w < len(c)]
            for name, x in feats.items():
                lags = np.array([(np.nanargmax if kind == "T" else np.nanargmin)(x[i - w:i + w + 1]) - w for i in idx])
                print(f"  theta={th:.0%}{lab}(n={len(idx)}) {name:18s} {np.median(lags):6.0f} {np.percentile(lags, 25):6.0f} {np.percentile(lags, 75):6.0f}"
                      f" {np.mean(lags < -5):12.0%} {np.mean(np.abs(lags) <= 5):12.0%}")


def tp_table(s, I, SIG, th, win):
    c, d = s.close, s.date
    pts = sse.zigzag(c, th)
    allpts = [p for p in pts]
    rows = []
    gc_all = np.where(sse.cross_up(I["dif"], I["dea"]))[0]
    dc_all = np.where(sse.cross_dn(I["dif"], I["dea"]))[0]
    for k, (i, kind, conf) in enumerate(allpts):
        if not win[i] or k < 2:
            continue
        p1, p2 = allpts[k - 1][0], allpts[k - 2][0]
        leg, prev = c[i] - c[p1], c[p1] - c[p2]
        cr = gc_all if kind == "B" else dc_all
        after = cr[cr >= i]
        x = after[0] if len(after) else -1
        widx = I["w"]["idx"]
        wk = np.searchsorted(widx, i)
        near_flags = []
        for (fam, name), (e, side) in SIG.items():
            if (side == "bull") == (kind == "B") and fam != "基准":
                if e[max(i - W, 0):i + W + 1].any():
                    near_flags.append(name.split("(")[0].split(":")[0])
        rows.append(dict(date=str(d[i]), kind="顶" if kind == "T" else "底", close=round(c[i], 1),
                         leg_pct=round(100 * (c[i] / c[p1] - 1), 1), leg_days=int(i - p1),
                         fib_ratio=round(abs(leg / prev), 3), confirm_lag=int(conf - i),
                         vr=round(I["vr"][i], 2), rsi14=round(I["rsi14"][i], 1), rsi6=round(I["rsi6"][i], 1),
                         wrsi14=round(float(I["w"]["rsi14"][min(wk, len(widx) - 1)]), 1),
                         macd_cross_lag=int(x - i) if x >= 0 else None,
                         macd_cross_dist=round(100 * (c[x] / c[i] - 1), 1) if x >= 0 else None,
                         signals_pm20=" / ".join(near_flags)))
    return pl.DataFrame(rows)


def fib_tests(s, win, thetas=(0.05, 0.10, 0.15, 0.20)):
    c = s.close
    out = []
    shifts = (-0.09, -0.07, -0.05, 0.0, 0.05, 0.07, 0.09)
    print("\n斐波那契回撤：每段走势 / 前一段走势（点数比）是否在 0.236/0.382/0.5/0.618/0.786 附近扎堆")
    print("  实际/期望 = 落在各水平 ±0.025 内的段数 / 用带宽 0.10 的核密度抹平后应有的段数（5 个水平合计）。")
    print("  对照组 = 把 5 个斐波那契水平整体平移 ±0.05/0.07/0.09；如果斐波那契有效，平移 0 那一列应明显最高")
    print(f"  {'':22s}" + "".join(f"{('平移' + format(d, '+.2f')) if d else '斐波那契':>12s}" for d in shifts))
    for th in thetas:
        pts = sse.zigzag(c, th)
        for basis in ("收盘", "高低点"):
            if basis == "高低点" and th not in (0.05, 0.10):
                continue
            px = [c[i] if basis == "收盘" else (s.high[i] if k == "T" else s.low[i]) for i, k, _ in pts]
            rr = np.array([abs(px[k] - px[k - 1]) / abs(px[k - 1] - px[k - 2]) for k in range(2, len(pts)) if win[pts[k][0]]])
            x = rr[rr <= 3]
            cells = []
            for dlt in shifts:
                o = e = 0.0
                for f in sse.FIB:
                    o += np.sum(np.abs(rr - (f + dlt)) <= 0.025)
                    e += len(x) * sse.kde_mass(x, f + dlt - 0.025, f + dlt + 0.025, 0.10)
                cells.append(f"{o:.0f}/{e:.1f}")
                out.append(dict(theta=th, basis=basis, shift=dlt, observed=o, expected=e, ratio=o / e, n=len(rr)))
            print(f"  theta={th:.0%} {basis} n={len(rr):4d}  " + "".join(f"{v:>12s}" for v in cells))
    # the golden ratio on its own, next to equally spaced non-Fibonacci neighbours, and split in halves
    print("\n  单看 0.618（黄金分割）及其等距邻居：实际/期望 (泊松单侧 p)；邻居不是斐波那契水平，用来判断 0.618 是否特殊")
    pois = lambda k, lam: 1 - sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(int(k)))
    for th in thetas:
        pts = sse.zigzag(c, th)
        for basis in ("收盘", "高低点"):
            px = [c[i] if basis == "收盘" else (s.high[i] if k == "T" else s.low[i]) for i, k, _ in pts]
            rr = np.array([abs(px[k] - px[k - 1]) / abs(px[k - 1] - px[k - 2]) for k in range(2, len(pts)) if win[pts[k][0]]])
            x = rr[rr <= 3]
            cells = []
            for f in (0.50, 0.56, 0.618, 0.68, 0.74):
                o = int(np.sum(np.abs(rr - f) <= 0.025))
                e = len(x) * sse.kde_mass(x, f - 0.025, f + 0.025, 0.10)
                cells.append(f"{f:.3f}: {o:2d}/{e:4.1f} (p {pois(o, e):.2f})")
            print(f"    theta={th:.0%} {basis:3s} " + "   ".join(cells))
    pts = sse.zigzag(c, 0.05)
    for a, b in (("1997-01-01", "2010-12-31"), ("2011-01-01", "2024-12-31")):
        m = win & (s.date >= np.datetime64(a)) & (s.date <= np.datetime64(b))
        rr = np.array([abs(c[pts[k][0]] - c[pts[k - 1][0]]) / abs(c[pts[k - 1][0]] - c[pts[k - 2][0]]) for k in range(2, len(pts)) if m[pts[k][0]]])
        x = rr[rr <= 3]
        o = int(np.sum(np.abs(rr - 0.618) <= 0.025))
        print(f"    theta=5% 收盘 {a[:4]}-{b[:4]}: n={len(rr)}, 0.618 附近 {o}/{len(x) * sse.kde_mass(x, 0.593, 0.643, 0.10):.1f}")

    # hazard: of the pullbacks that reached ratio x, how many stopped in [x, x+0.05)?
    print("\n  条件止跌率：回撤已经到达 x 的走势中，最终停在 [x, x+0.05) 的比例（theta=5%，所有反向段合并）")
    pts = sse.zigzag(c, 0.05)
    rr = np.array([abs(c[pts[k][0]] - c[pts[k - 1][0]]) / abs(c[pts[k - 1][0]] - c[pts[k - 2][0]])
                   for k in range(2, len(pts)) if win[pts[k][0]]])
    grid = np.round(np.arange(0.20, 1.20, 0.05), 3)
    hz = []
    for g in grid:
        reached = rr >= g
        stop = reached & (rr < g + 0.05)
        hz.append((g, reached.sum(), stop.sum() / max(reached.sum(), 1)))
    print("    x     " + " ".join(f"{g:5.2f}" for g, _, _ in hz))
    print("    到达数 " + " ".join(f"{m:5d}" for _, m, _ in hz))
    print("    止跌率 " + " ".join(f"{h:5.0%}" for _, _, h in hz))
    print("    (斐波那契位所在的格: 0.20-0.25 含 0.236, 0.35-0.40 含 0.382, 0.50-0.55 含 0.5, 0.60-0.65 含 0.618, 0.75-0.80 含 0.786)")
    return pl.DataFrame(out)


def fib_time(s, win):
    c = s.close
    fibd = np.array([8, 13, 21, 34, 55, 89, 144, 233])
    scales = (0.80, 0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20)
    print("\n斐波那契时间：每段走势持续的交易日数落在斐波那契数（8,13,21,...,233）±1 日的次数，实际/期望（期望用对数时长的核密度）")
    print("  对照组 = 把这组数整体乘以 0.8~1.2 再取整；如果斐波那契有效，1.00 那一列的实际/期望比应明显最高")
    print(f"  {'':16s}" + "".join(f"{('x' + format(k, '.2f')):>10s}" for k in scales))
    for th in (0.05, 0.10, 0.20):
        pts = sse.zigzag(c, th)
        dur = np.array([pts[k][0] - pts[k - 1][0] for k in range(1, len(pts)) if win[pts[k][0]]])
        ld = np.log(dur)
        cells = []
        for k in scales:
            cen = np.unique(np.round(fibd * k).astype(int))
            o = sum(int(np.sum(np.abs(dur - f) <= 1)) for f in cen)
            e = sum(len(dur) * sse.kde_mass(ld, np.log(f - 1.5), np.log(f + 1.5), 0.25) for f in cen)
            cells.append(f"{o}/{e:.1f}")
        print(f"  theta={th:.0%} n={len(dur):4d}  " + "".join(f"{v:>10s}" for v in cells))
    pts = [p for p in sse.zigzag(c, 0.20) if win[p[0]]]
    fw = np.array([3, 5, 8, 13, 21, 34, 55, 89, 144, 233])
    for kind, lab in (("B", "底到底"), ("T", "顶到顶")):
        ix = np.array([p[0] for p in pts if p[1] == kind])
        wks = [int(round(x / 5)) for x in np.diff(ix)]
        hits = sum(bool(np.any(np.abs(w - fw) <= 1)) for w in wks)
        print(f"  theta=20% {lab}间隔(周): {wks} -> 落在斐波那契数±1周: {hits}/{len(wks)}")


def latest(s, I):
    t = len(s.close) - 1
    print(f"\n最新读数 {s.date[t]} 收盘 {s.close[t]:.1f}: 量比(MA5/MA250) {I['vr'][t]:.2f}, RSI14 {I['rsi14'][t]:.1f}, RSI6 {I['rsi6'][t]:.1f},"
          f" 周RSI14 {I['w']['rsi14'][-1]:.1f}, DIF {I['dif'][t]:+.1f}, DEA {I['dea'][t]:+.1f}, MACD柱 {I['hist'][t]:+.1f},"
          f" 周DIF {I['w']['dif'][-1]:+.1f} 周DEA {I['w']['dea'][-1]:+.1f}")
    for th in (0.10, 0.20):
        pts = sse.zigzag(s.close, th)
        i, k, cf = pts[-1]
        ext = s.close[cf:].max() if k == "B" else s.close[cf:].min()
        print(f"  theta={th:.0%}: 最近确认的转折点 {s.date[i]} {'底' if k == 'B' else '顶'} {s.close[i]:.1f}，"
              f"此后{'最高' if k == 'B' else '最低'}收盘 {ext:.1f}，当前距该极值 {100 * (s.close[t] / ext - 1):+.1f}%")


if __name__ == "__main__":
    pl.Config.set_tbl_rows(80); pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(80)
    # ---------------- research sample, data cut at 2024-12-31
    s = sse.load(end=sse.IS_END)
    win = s.date >= np.datetime64(sse.IS_START)
    I = indicators(s)
    SIG = signals(s, I)
    print(f"上证指数 {s.date[0]} .. {s.date[-1]}；研究样本 {sse.IS_START} .. {sse.IS_END}，{win.sum()} 个交易日")
    G, tp = grade(s, I, SIG, win)
    print(f"转折点数: theta=10% 顶 {sum(p[1] == 'T' for p in tp[0.10])} 底 {sum(p[1] == 'B' for p in tp[0.10])};"
          f" theta=20% 顶 {sum(p[1] == 'T' for p in tp[0.20])} 底 {sum(p[1] == 'B' for p in tp[0.20])}")
    print_grades(G, "信号评分（研究样本 1997-2024）")
    G.write_csv("results/sse_turning_signals_1997_2024.csv")
    print("\n转折点附近的指标形态（研究样本）")
    profiles(s, I, tp, win)
    CO = conditions(s, I, tp, win)
    CO.write_csv("results/sse_turning_conditions_1997_2024.csv")
    lead_lag(s, I, tp)
    T20 = tp_table(s, I, SIG, 0.20, win)
    print("\ntheta=20% 主要转折点（研究样本）：leg=本段涨跌幅/交易日, fib_ratio=本段/上一段点数比, confirm_lag=过滤规则确认滞后,"
          " macd_cross_lag/dist=转折后第一次金叉(底)或死叉(顶)的滞后日数和点位距离")
    print(T20)
    T20.write_csv("results/sse_turning_points_theta20_1997_2024.csv")
    FB = fib_tests(s, win)
    FB.write_csv("results/sse_fib_levels_1997_2024.csv")
    fib_time(s, win)
    # ---------------- recent window on the full series
    s2 = sse.load()
    win2 = s2.date >= np.datetime64(sse.OOS_START)
    I2 = indicators(s2)
    SIG2 = signals(s2, I2)
    G2, tp2 = grade(s2, I2, SIG2, win2)
    print(f"\n\n========== 近期样本外 {sse.OOS_START} .. {s2.date[-1]}（{win2.sum()} 个交易日；数量很少，只作核对）==========")
    for th in (0.05, 0.10):
        pts = [p for p in sse.zigzag(s2.close, th) if win2[p[0]]]
        print(f"  theta={th:.0%} 转折点: " + ", ".join(f"{s2.date[i]} {'顶' if k == 'T' else '底'} {s2.close[i]:.0f}" for i, k, _ in pts))
    print_grades(G2.filter(pl.col("n") > 0), "信号评分（近期样本外；60/120 日远期收益在数据末端缺失）")
    G2.write_csv("results/sse_turning_signals_2025_2026.csv")
    print("\n  近期各信号触发日期:")
    for (fam, name), (e, side) in SIG2.items():
        ev = np.where(e & win2)[0]
        if len(ev):
            f20 = sse.fwd_log(s2.close, 20)
            print(f"    {name[:26]:26s} " + ", ".join(f"{str(s2.date[i])[2:]}({'' if np.isnan(f20[i]) else f'{100 * f20[i]:+.0f}%'})" for i in ev))
    latest(s2, I2)
