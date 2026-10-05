"""miniQMT (xtquant) adapter.

NOT TESTED against a live terminal: written to the xtquant API (XtQuantTrader, StockAccount, xtdata) but this
repository's environment has no QMT. Orders are only sent when the account config sets
execution.allow_live = true AND the command is run with --live; otherwise every call that would trade only
logs what it would send. Try it first in a simulated QMT account.

Timing used by `python -m trading open` on a QMT account (Beijing time):
  09:25:30  read the call-auction result for every name (open price, pre_close)
            -> form the score with the observed buyability, decide, size (identical to the paper path)
  09:26     send sells, then buys, as limit orders priced 1.5% through the auction price (inside the
            continuous-trading price cage); they match at 09:30
  09:31+    `python -m trading qmt-chase` re-prices unfilled orders (bounded number of rounds)
After the close, `python -m trading eod` reads positions and cash back from QMT.
"""
import math
import time

import numpy as np

from .base import Book, is_a_share, to_jq, to_qmt


class QmtUnavailable(RuntimeError):
    pass


class QmtBroker:
    kind = "qmt"

    def __init__(self, execution, live=False, log=print):
        try:
            from xtquant import xtconstant, xtdata
            from xtquant.xttrader import XtQuantTrader
            from xtquant.xttype import StockAccount
        except ImportError as e:
            raise QmtUnavailable("xtquant is not installed; run this on the miniQMT host") from e
        if not execution.qmt_path or not execution.qmt_account:
            raise QmtUnavailable("set execution.qmt_path and execution.qmt_account in the account config")
        self.xc, self.xd, self.log = xtconstant, xtdata, log
        self.live = bool(live and execution.allow_live)
        self.trader = XtQuantTrader(execution.qmt_path, int(time.time()) % 1_000_000)
        self.trader.start()
        rc = self.trader.connect()
        if rc != 0:
            raise QmtUnavailable(f"cannot connect to miniQMT at {execution.qmt_path} (code {rc})")
        self.acc = StockAccount(execution.qmt_account, "STOCK")
        self.trader.subscribe(self.acc)
        if not self.live:
            log("QMT DRY RUN: orders are logged, not sent (needs allow_live = true and --live)")

    # ---------------------------------------------------------------- account
    def book(self):
        asset = self.trader.query_stock_asset(self.acc)
        pos, external = {}, []
        for p in self.trader.query_stock_positions(self.acc) or []:
            if p.volume <= 0:
                continue
            if not is_a_share(p.stock_code):
                external.append((p.stock_code, float(p.volume)))
                continue
            cost = getattr(p, "avg_price", None) or getattr(p, "open_price", 0.0)
            pos[to_jq(p.stock_code)] = [float(p.volume), float(cost)]
        return Book(cash=float(asset.cash), pos=pos, external=external)

    # ---------------------------------------------------------------- market
    def auction_info(self, codes):
        """Open-time arrays aligned with `codes` (the panel's codes), from the call-auction ticks."""
        q = [to_qmt(c) for c in codes]
        ticks = {}
        for i in range(0, len(q), 500):
            ticks.update(self.xd.get_full_tick(q[i:i + 500]) or {})
        n = len(codes)
        op, pc = np.full(n, np.nan), np.full(n, np.nan)
        for j, c in enumerate(q):
            t = ticks.get(c)
            if not t:
                continue
            o = t.get("open") or t.get("lastPrice") or 0.0
            if o > 0 and (t.get("volume") or 0) > 0:
                op[j] = o
            if t.get("lastClose"):
                pc[j] = t["lastClose"]
        hl = np.round(pc * 1.10 + 1e-9, 2)        # main board, non-ST (the only names the strategy trades)
        ll = np.round(pc * 0.90 + 1e-9, 2)
        at_hl = np.isclose(op, hl, atol=0.005) | (op > hl)
        at_ll = np.isclose(op, ll, atol=0.005) | (op < ll)
        return dict(open=op, pre_close=pc, high_limit=hl, low_limit=ll,
                    can_buy=np.isfinite(op) & ~at_hl, can_sell=np.isfinite(op) & ~at_ll)

    # ---------------------------------------------------------------- orders
    @staticmethod
    def order_price(side, px, hl, ll, through=0.015):
        if side == "buy":
            return round(min(hl, math.floor(px * (1 + through) * 100 + 1e-6) / 100), 2)
        return round(max(ll, math.ceil(px * (1 - through) * 100 - 1e-6) / 100), 2)

    def submit(self, orders, info, cidx, remark="feed"):
        ids = []
        for o in sorted(orders, key=lambda o: o.side != "sell"):
            j = cidx[o.code]
            price = self.order_price(o.side, o.price, info["high_limit"][j], info["low_limit"][j])
            o.limit = price
            qty = int(round(o.qty))
            side = self.xc.STOCK_BUY if o.side == "buy" else self.xc.STOCK_SELL
            if not self.live:
                self.log(f"[dry-run] {o.side} {o.code} {qty} @ {price}")
                ids.append(None)
                continue
            oid = self.trader.order_stock(self.acc, to_qmt(o.code), side, qty, self.xc.FIX_PRICE, price, remark,
                                          o.reason)
            self.log(f"sent {o.side} {o.code} {qty} @ {price} -> {oid}")
            ids.append(str(oid) if oid is not None and oid >= 0 else None)
        return ids

    def open_orders(self):
        return self.trader.query_stock_orders(self.acc, True) or []

    def chase(self, rounds=3, wait=20, through=0.015, remark="feed"):
        """Cancel and re-price unfilled orders at the latest price, at most `rounds` times."""
        for r in range(rounds):
            pending = self.open_orders()
            if not pending:
                return 0
            ticks = self.xd.get_full_tick([o.stock_code for o in pending]) or {}
            for o in pending:
                t = ticks.get(o.stock_code) or {}
                last, pc = t.get("lastPrice"), t.get("lastClose")
                if not last or not pc:
                    continue
                left = int(o.order_volume - o.traded_volume)
                side = "buy" if o.order_type == self.xc.STOCK_BUY else "sell"
                price = self.order_price(side, last, round(pc * 1.1 + 1e-9, 2), round(pc * 0.9 + 1e-9, 2), through)
                if not self.live:
                    self.log(f"[dry-run] re-price {o.stock_code} {side} {left} @ {price}")
                    continue
                self.trader.cancel_order_stock(self.acc, o.order_id)
                self.trader.order_stock(self.acc, o.stock_code, o.order_type, left, self.xc.FIX_PRICE, price,
                                        remark, "chase")
            time.sleep(wait)
        return len(self.open_orders())

    def trades(self):
        out = []
        for t in self.trader.query_stock_trades(self.acc) or []:
            out.append(dict(code=to_jq(t.stock_code), side="buy" if t.order_type == self.xc.STOCK_BUY else "sell",
                            qty=float(t.traded_volume), price=float(t.traded_price), broker_id=str(t.order_id)))
        return out
