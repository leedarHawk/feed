"""Figures for run_sse_turning.py (research sample 1997-2024 unless marked). Writes results/sse_*.png."""
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import sse
from run_sse_turning import indicators, signals

for f in font_manager.findSystemFonts():
    if "wqy-zenhei" in f:
        font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "WenQuanYi Zen Hei", "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "axes.titleweight": "bold",
                     "axes.titlesize": 11, "lines.linewidth": 2})
BLUE, ORANGE, RED, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#e34948", "#8a8984", "#0b0b0b", "#52514e"

s_full = sse.load()
s = sse.load(end=sse.IS_END)
win = s.date >= np.datetime64(sse.IS_START)
I = indicators(s)

# 1. SSE with major turning points ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4.6))
m = s_full.date >= np.datetime64(sse.IS_START)
ax.plot(s_full.date[m], s_full.close[m], color=INK2, lw=1.2)
ax.set_yscale("log")
ax.set_yticks([1000, 1500, 2000, 3000, 4000, 5000, 6000])
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
pts = [p for p in sse.zigzag(s_full.close, 0.20) if m[p[0]]]
last = -10 ** 9
for i, k, _ in pts:
    col = ORANGE if k == "T" else BLUE
    ax.scatter(s_full.date[i], s_full.close[i], s=34, color=col, zorder=3, edgecolor="#fcfcfb", linewidth=1.5)
    if i - last > 100:      # label only points that are not crowded by the previous one
        ax.annotate(str(s_full.date[i])[:7], (s_full.date[i], s_full.close[i]), xytext=(0, 8 if k == "T" else -14),
                    textcoords="offset points", ha="center", fontsize=7, color=INK2)
    last = i
ax.axvspan(np.datetime64(sse.OOS_START), s_full.date[-1], color="#f0efec", zorder=0)
ax.text(np.datetime64("2025-10-01"), 1050, "样本外\n2025-26", ha="center", fontsize=8, color=INK2)
ax.scatter([], [], s=34, color=ORANGE, label="顶（回落≥20%）")
ax.scatter([], [], s=34, color=BLUE, label="底（反弹≥20%）")
ax.legend(loc="upper left", frameon=False)
ax.set_title("上证指数 1997–2026（对数坐标）与 20% 级别转折点（相邻过近的点不标日期，完整列表见 csv）")
fig.tight_layout()
fig.savefig("results/sse_turning_points.png", dpi=150)

# 2. indicator paths around turning points ------------------------------------------------------------
offs = np.arange(-40, 41)
feats = [("量比 MA5量/MA250量", I["vr"]), ("RSI14", I["rsi14"]), ("MACD柱 / 收盘 (%)", 100 * I["hist"] / s.close)]
fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharex=True)
for r, (kind, lab) in enumerate((("T", "顶部"), ("B", "底部"))):
    for col, (name, x) in enumerate(feats):
        ax = axes[r, col]
        for th, color in ((0.20, BLUE), (0.10, ORANGE)):
            idx = np.array([p[0] for p in sse.zigzag(s.close, th) if win[p[0]] and p[1] == kind])
            med = [np.nanmedian(x[np.clip(idx + o, 0, len(x) - 1)]) for o in offs]
            ax.plot(offs, med, color=color, label=f"{th:.0%} 级别 (n={len(idx)})")
        ax.axhline(np.nanmedian(x[win]), color=GRAY, lw=1, ls="--", label="全样本中位数")
        ax.axvline(0, color=GRAY, lw=1)
        if name == "RSI14":
            for lvl in (30, 70):
                ax.axhline(lvl, color="#d6d5d0", lw=1, ls=":")
        ax.set_title(f"{lab}：{name}", loc="left")
        if r == 1:
            ax.set_xlabel("相对转折点的交易日")
axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("转折点前后的指标中位数路径（1997–2024）", fontweight="bold", x=0.01, ha="left")
fig.tight_layout()
fig.savefig("results/sse_turning_profiles.png", dpi=150)

# 3. signal scoreboard ------------------------------------------------------------------------------
G = pl.read_csv("results/sse_turning_signals_1997_2024.csv").filter(~pl.col("family").is_in(["事后组合"]))
G = G.with_columns(edge=pl.when(pl.col("side") == "bull").then(pl.col("fwd60") - pl.col("fwd60_base"))
                   .otherwise(pl.col("fwd60_base") - pl.col("fwd60")))
labels = [f"{f} · {n}  (n={k})" for f, n, k in zip(G["family"], G["signal"], G["n"])]
y = np.arange(len(G))[::-1]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 8.4), sharey=True)
ax = axes[0]
ax.barh(y, G["catch_lift"], color=BLUE, height=0.62)
ax.axvline(1, color=INK2, lw=1)
ax.set_yticks(y, labels, fontsize=8)
ax.set_title("抓转折能力：±20 日内有 10% 级别同向转折的比例 / 随机", loc="left", fontsize=10)
ax.set_xlabel("倍数（1 = 与随机日期无异）")
for yy, v in zip(y, G["catch_lift"]):
    ax.text(v + 0.03, yy, f"{v:.2f}", va="center", fontsize=7, color=INK2)
ax = axes[1]
e = G["edge"].to_numpy() * 100
ax.barh(y, e, color=[BLUE if v > 0 else RED for v in e], height=0.62)
ax.axvline(0, color=INK2, lw=1)
ax.set_title("按信号方向的 60 日超额（点位对数收益，相对全样本平均）", loc="left", fontsize=10)
ax.set_xlabel("百分点（正 = 方向判断对了，负 = 方向反了）")
for yy, v in zip(y, e):
    ax.text(v + (0.3 if v >= 0 else -0.3), yy, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=7, color=INK2)
fig.suptitle("各信号在上证指数上的表现（1997–2024，看多信号看是否先涨，看空信号看是否先跌）", fontweight="bold", x=0.01, ha="left")
fig.tight_layout()
fig.savefig("results/sse_turning_signals.png", dpi=150)

# 4. Fibonacci retracement histogram -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, th in zip(axes, (0.05, 0.10)):
    pts = sse.zigzag(s.close, th)
    c = s.close
    rr = np.array([abs(c[pts[k][0]] - c[pts[k - 1][0]]) / abs(c[pts[k - 1][0]] - c[pts[k - 2][0]])
                   for k in range(2, len(pts)) if win[pts[k][0]]])
    bins = np.arange(0, 2.0001, 0.05)
    ax.hist(rr[rr <= 2], bins=bins, color=BLUE, edgecolor="#fcfcfb", linewidth=2)
    xs = np.linspace(0, 2, 400)
    x = rr[rr <= 3]
    kde = np.array([np.mean(np.exp(-0.5 * ((v - x) / 0.10) ** 2)) / (0.10 * np.sqrt(2 * np.pi)) for v in xs])
    ax.plot(xs, kde * len(x) * 0.05, color=INK2, lw=1.5, label="平滑后的期望分布")
    for f in sse.FIB:
        ax.axvline(f, color=ORANGE, lw=1.2, ls="--")
        ax.text(f, ax.get_ylim()[1] * 0.97, f"{f:g}", rotation=90, va="top", ha="right", fontsize=7, color=INK2)
    ax.set_title(f"{th:.0%} 级别：回撤/反弹幅度 ÷ 前一段幅度 (n={len(rr)})", loc="left")
    ax.set_xlabel("比例（虚线 = 斐波那契水平）")
    ax.set_ylabel("段数")
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.suptitle("斐波那契回撤：5 个水平合计没有扎堆；0.618 附近略多但不显著、与非斐波那契的 0.56 相当，且 2011 年后消失（1997–2024）",
             fontweight="bold", x=0.01, ha="left", fontsize=10)
fig.tight_layout()
fig.savefig("results/sse_fib_retracement.png", dpi=150)
print("figures written")
