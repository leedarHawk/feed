"""Simulated broker: every order fills at the opening price of its day, as in the research backtest.

* buys fail at a limit-up open or when the name does not trade; sells fail at a limit-down open or suspension
* fill price = open x (1 +/- slippage); commission (minimum fee per order) on the value at the open; stamp
  duty on sells
* ex-rights days: shares are scaled by last close / pre_close (pre_close is the exchange's ex-rights
  reference price), i.e. dividends and bonus shares are reinvested in the name, as with qfq prices
"""
import numpy as np


class PaperBroker:
    kind = "paper"

    def __init__(self, costs):
        self.costs = costs

    def corporate_actions(self, book, cidx, pre_close):
        notes = []
        for c, v in book.pos.items():
            last, pc = book.last_close.get(c), pre_close[cidx[c]]
            if last and np.isfinite(pc) and pc > 0 and abs(last / pc - 1) > 1e-9:
                k = last / pc
                v[0] *= k
                v[1] /= k
                book.last_close[c] = float(pc)
                if abs(k - 1) > 1e-3:
                    notes.append(f"{c} ex-rights x{k:.4f}")
        return notes

    def execute(self, book, orders, day, cidx, info):
        """Fill orders (sells first) at the open of `day`. Returns (fills, statuses) aligned with orders."""
        c = self.costs
        stamp_rate = c.stamp(day)
        fills, status = [], [None] * len(orders)
        for i in sorted(range(len(orders)), key=lambda i: orders[i].side != "sell"):
            o = orders[i]
            j = cidx[o.code]
            px = info["open"][j]
            if o.side == "sell":
                held = book.pos.get(o.code, [0.0, 0.0])[0]
                qty = min(o.qty, held)
                if not (info["can_sell"][j] and np.isfinite(px)) or qty <= 1e-9:
                    status[i] = "blocked"
                    continue
                v = qty * px
                fee, stamp = c.fee(v), v * stamp_rate
                delta = v * (1 - c.slippage) - fee - stamp
                book.cash += delta
                left = held - qty
                if left <= 1e-6:
                    book.pos.pop(o.code, None)
                else:
                    book.pos[o.code][0] = left
            else:
                if not (info["can_buy"][j] and np.isfinite(px)):
                    status[i] = "blocked"
                    continue
                qty = o.qty
                while qty > 1e-9 and qty * px * (1 + c.slippage) + c.fee(qty * px) > book.cash + 1e-6:
                    qty = qty - c.lot if qty >= c.lot else 0.0          # never overdraw cash
                if qty <= 1e-9:
                    status[i] = "no_cash"
                    continue
                v = qty * px
                fee, stamp = c.fee(v), 0.0
                delta = -(v * (1 + c.slippage) + fee)
                book.cash += delta
                s0, k0 = book.pos.get(o.code, [0.0, 0.0])
                book.pos[o.code] = [s0 + qty, (s0 * k0 - delta) / (s0 + qty)]
                book.last_close.setdefault(o.code, float(px))
            status[i] = "filled" if abs(qty - o.qty) < 1e-9 else "partial"
            fills.append(dict(order=i, date=str(day), code=o.code, side=o.side, qty=float(qty),
                              price=float(px * (1 + c.slippage if o.side == "buy" else 1 - c.slippage)),
                              fee=float(fee), stamp=float(stamp), cash_delta=float(delta)))
        return fills, status
