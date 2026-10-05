"""Daily account report (Markdown) and the market-temperature context line."""
import datetime as dt
import os

import numpy as np
import polars as pl

from .bridge import research

BANDS = ("冰点", "偏冷", "中性", "偏热", "过热")


def industry_map(day, level="sw_l1"):
    """code -> industry name effective on `day` (month-end sampled; may lag by up to a month)."""
    path = os.path.join(research().lib.DATA, "industry_intervals.parquet")
    d = dt.date.fromisoformat(str(day)[:10])
    df = pl.read_parquet(path).filter(
        (pl.col("classification_system") == level) & (pl.col("effective_from") <= d) &
        (pl.col("effective_to_exclusive").is_null() | (pl.col("effective_to_exclusive") > d)))
    return dict(zip(df["code"].to_list(), [n.rstrip("I") for n in df["industry_name"].to_list()]))


def temperature():
    """Latest 5-day sentiment composite of research/sentiment.py and its band versus the trailing 500 days
    (the thermometer's own rule). Context only: it is not an input of the strategy."""
    import sentiment                      # research/sentiment.py (on sys.path via bridge)
    df = sentiment.build()
    x = df["sentiment_5d"].to_numpy()
    t = len(x) - 1
    hist = x[max(0, t - 500):t]
    hist = hist[~np.isnan(hist)]
    if np.isnan(x[t]) or len(hist) < 250:
        return None
    q = np.quantile(hist, [0.1, 0.3, 0.7, 0.9])
    band = BANDS[int(np.searchsorted(q, x[t], side="right"))]
    drivers = {k: float(df[k][t]) for k in ("z_up", "z_dn", "z_brk", "z_stk", "z_prem", "z_pu", "z_to")}
    return dict(date=str(df["date"][t])[:10], value=float(x[t]), band=band, pct=float((hist < x[t]).mean()),
                drivers=drivers)


LABEL = {"z_up": "涨停家数", "z_dn": "跌停家数", "z_brk": "炸板率", "z_stk": "最高连板", "z_prem": "昨日涨停今日",
         "z_pu": "上涨家数占比", "z_to": "成交额"}
KIND = {"rebalance": "定期调仓", "reduce": "降仓（卖出最弱）", "add": "加仓（买入最强）", "none": "不交易", "skip": "跳过"}


def _money(x):
    return f"{x:,.0f}"


def render(acct, ledger, plan, checks, names, industries, next_day, temp=None, extra=None):
    s = acct.strategy
    days = ledger.days()
    last = days[-1] if days else None
    L = [f"# 交易日报 · {acct.name} · {last['date'] if last else '尚未开始'}", ""]
    L.append(f"策略 {s.version}（{s.score}，{s.n_hold} 只，一月空仓{'是' if s.january else '否'}，"
             f"每 {s.rebal} 个交易日调仓，锚点 {s.anchor} 偏移 {s.offset}）· 券商 {acct.broker} · 下一交易日 {next_day}")
    L.append("")
    L.append("## 账户")
    if last:
        nav0 = acct.capital or days[0]["nav"]
        cum = last["nav"] / nav0 - 1 if nav0 else 0.0
        bench = float(np.prod([1 + (d["bench_ret"] or 0) for d in days]) - 1)
        L += ["| 净值 | 当日 | 累计 | 同期全市场等权 | 回撤 | 现金 | 持仓数 |", "| --- | --- | --- | --- | --- | --- | --- |",
              f"| {_money(last['nav'])} | {last['ret']:+.2%} | {cum:+.2%} | {bench:+.2%} | {last['drawdown']:.2%} | "
              f"{_money(last['cash'])} | {last['n_pos']} |"]
    else:
        L.append(f"尚无交易日记录。初始资金 {_money(acct.capital)}。")
    L.append("")
    dec = plan.decision
    L.append(f"## {next_day} 计划：{KIND.get(dec.kind, dec.kind)}")
    L.append(f"目标仓位 {plan.exposure:.0%}（{dec.n_sel} 只），{'调仓日' if plan.scheduled else '非调仓日'}；"
             f"可选股票池 {plan.universe} 只。{dec.note}")
    if extra:
        L += [f"- {x}" for x in extra]
    if plan.orders:
        sells = [o for o in plan.orders if o.side == "sell"]
        buys = [o for o in plan.orders if o.side == "buy"]
        L.append(f"卖出 {len(sells)} 笔 {_money(sum(o.value for o in sells))} 元，买入 {len(buys)} 笔 "
                 f"{_money(sum(o.value for o in buys))} 元（按 {plan.day} 前一收盘价估算，开盘后按集合竞价价重算）。")
        L += ["", "| 方向 | 代码 | 名称 | 数量 | 参考价 | 金额 | 说明 |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for o in sorted(plan.orders, key=lambda o: (o.side != "sell", -o.value)):
            L.append(f"| {'卖' if o.side == 'sell' else '买'} | {o.code[:6]} | {names.get(o.code, '')} | {o.qty:,.0f} | "
                     f"{o.price:.2f} | {_money(o.value)} | {o.reason} |")
    else:
        L.append("无订单。")
    L.append("")
    L.append("## 风控")
    bad = [c for c in checks if c.level != "ok"]
    L += [f"- **{c.level}** {c.name}：{c.message}" for c in bad] or ["- 全部通过"]
    for c in checks:
        if c.level == "ok":
            L.append(f"- ok {c.name}：{c.message}")
    L.append("")
    pos = ledger.db.execute("SELECT code, shares, price, value, cost FROM positions WHERE date=? ORDER BY value DESC",
                            (last["date"],)).fetchall() if last else []
    if pos:
        L.append(f"## 持仓（{last['date']} 收盘）")
        w_ind = {}
        for c, _, _, v, _ in pos:
            w_ind[industries.get(c, "未知")] = w_ind.get(industries.get(c, "未知"), 0.0) + v / last["nav"]
        top = sorted(w_ind.items(), key=lambda kv: -kv[1])[:5]
        L.append("行业（申万一级）前五：" + "，".join(f"{k} {v:.0%}" for k, v in top))
        L += ["", "| 代码 | 名称 | 行业 | 股数 | 收盘 | 市值 | 权重 | 浮盈 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for c, sh, px, v, cost in pos:
            pnl = px / cost - 1 if cost else float("nan")
            L.append(f"| {c[:6]} | {names.get(c, '')} | {industries.get(c, '')} | {sh:,.0f} | {px:.2f} | {_money(v)} | "
                     f"{v / last['nav']:.1%} | {pnl:+.1%} |")
        L.append("")
    if temp:
        drv = sorted(temp["drivers"].items(), key=lambda kv: kv[1])
        L.append("## 市场温度计（参考，不参与交易）")
        L.append(f"{temp['date']} 情绪综合分 {temp['value']:+.2f}，{temp['band']}，高于近两年 {temp['pct']:.0%} 的交易日。"
                 f"最冷的两项：{LABEL[drv[0][0]]} {drv[0][1]:+.2f}、{LABEL[drv[1][0]]} {drv[1][1]:+.2f}。")
        L.append("")
    ev = [e for e in ledger.events() if e[2] != "info"][-10:]
    if ev:
        L.append("## 最近事件")
        L += [f"- {e[1]} {e[2]} {e[3]}：{e[4]}" for e in ev]
    return "\n".join(L) + "\n"
