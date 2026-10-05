"""QMT adapter against a fake xtquant module: checks our own code paths, not miniQMT itself."""
import sys
import types
from types import SimpleNamespace as NS

import numpy as np
import pytest

from trading.config import Execution
from trading.engine import Order


@pytest.fixture
def fake_xtquant(monkeypatch):
    sent, cancelled = [], []

    class Trader:
        def __init__(self, path, session):
            self.path = path

        def start(self):
            pass

        def connect(self):
            return 0

        def subscribe(self, acc):
            pass

        def query_stock_asset(self, acc):
            return NS(cash=12345.0)

        def query_stock_positions(self, acc):
            return [NS(stock_code="600000.SH", volume=1000, avg_price=9.5), NS(stock_code="204001.SH", volume=10),
                    NS(stock_code="000001.SZ", volume=0, avg_price=11.0)]

        def order_stock(self, acc, code, side, qty, ptype, price, strat, remark):
            sent.append((code, side, qty, price))
            return len(sent)

        def query_stock_orders(self, acc, cancelable_only):
            return []

        def cancel_order_stock(self, acc, oid):
            cancelled.append(oid)

        def query_stock_trades(self, acc):
            return [NS(stock_code="600000.SH", order_type=24, traded_volume=1000, traded_price=10.0, order_id=1)]

    ticks = {"600000.SH": dict(open=10.0, lastPrice=10.0, lastClose=10.0, volume=100),
             "000001.SZ": dict(open=12.1, lastPrice=12.1, lastClose=11.0, volume=50),      # opens at limit-up
             "600004.SH": dict(open=0.0, lastPrice=0.0, lastClose=8.0, volume=0)}           # suspended
    xt = types.ModuleType("xtquant")
    xt.xtconstant = NS(STOCK_BUY=23, STOCK_SELL=24, FIX_PRICE=11)
    xt.xtdata = NS(get_full_tick=lambda codes: {c: ticks[c] for c in codes if c in ticks})
    xt.xttrader = NS(XtQuantTrader=Trader)
    xt.xttype = NS(StockAccount=lambda acc, kind="STOCK": NS(acc=acc))
    for name, mod in {"xtquant": xt, "xtquant.xtconstant": xt.xtconstant, "xtquant.xtdata": xt.xtdata,
                      "xtquant.xttrader": xt.xttrader, "xtquant.xttype": xt.xttype}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    return sent, cancelled


def test_qmt_book_auction_and_orders(fake_xtquant):
    from trading.brokers.qmt import QmtBroker
    sent, _ = fake_xtquant
    ex = Execution(qmt_path="x", qmt_account="1", allow_live=True)
    b = QmtBroker(ex, live=True, log=lambda *a: None)
    book = b.book()
    assert book.cash == 12345.0 and book.pos == {"600000.XSHG": [1000.0, 9.5]}
    assert book.external == [("204001.SH", 10.0)]
    codes = np.array(["600000.XSHG", "000001.XSHE", "600004.XSHG"])
    info = b.auction_info(codes)
    assert list(info["can_buy"]) == [True, False, False] and list(info["can_sell"]) == [True, True, False]
    assert info["high_limit"][1] == 12.10
    cidx = {c: i for i, c in enumerate(codes)}
    ids = b.submit([Order("600000.XSHG", "buy", 200, 10.0, "new"), Order("600000.XSHG", "sell", 100, 10.0, "trim")],
                   info, cidx)
    assert sent[0] == ("600000.SH", 24, 100, 9.85)                 # sells go first, priced 1.5% through
    assert sent[1] == ("600000.SH", 23, 200, 10.15)
    assert ids == ["1", "2"]
    assert b.trades()[0]["side"] == "sell"


def test_qmt_dry_run_sends_nothing(fake_xtquant):
    from trading.brokers.qmt import QmtBroker
    sent, _ = fake_xtquant
    b = QmtBroker(Execution(qmt_path="x", qmt_account="1", allow_live=False), live=True, log=lambda *a: None)
    codes = np.array(["600000.XSHG"])
    b.submit([Order("600000.XSHG", "buy", 100, 10.0, "new")], b.auction_info(codes), {"600000.XSHG": 0})
    assert sent == []
