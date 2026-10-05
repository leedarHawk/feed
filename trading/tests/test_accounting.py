import datetime as dt

import numpy as np
import pytest

from trading.brokers.base import Book, to_jq, to_qmt
from trading.brokers.manual import limit_price, read_holdings, write_order_sheet
from trading.brokers.paper import PaperBroker
from trading.config import CostModel
from trading.engine import Order
from trading.ledger import Ledger
from trading.tradecal import Calendar, CalendarError

codes = ["600000.XSHG", "000001.XSHE"]
cidx = {c: i for i, c in enumerate(codes)}


def test_paper_fills_costs_and_limits():
    c = CostModel()
    b = Book(cash=100000.0)
    info = dict(open=np.array([10.0, 20.0]), can_buy=np.array([True, False]), can_sell=np.array([True, True]))
    orders = [Order(codes[0], "buy", 1000, 10.0, "new"), Order(codes[1], "buy", 100, 20.0, "new")]
    fills, status = PaperBroker(c).execute(b, orders, "2026-10-08", cidx, info)
    assert status == ["filled", "blocked"]
    cost = 10000 * (1 + c.slippage) + 5.0                       # commission 2.5 < minimum 5
    assert b.cash == pytest.approx(100000 - cost)
    assert b.pos[codes[0]][0] == 1000
    fills, status = PaperBroker(c).execute(b, [Order(codes[0], "sell", 1000, 10.0, "exit")], "2026-10-09", cidx,
                                           dict(open=np.array([11.0, 20.0]), can_buy=np.ones(2, bool),
                                                can_sell=np.array([True, True])))
    assert codes[0] not in b.pos
    assert fills[0]["stamp"] == pytest.approx(11000 * 0.0005)


def test_ex_rights_scales_shares_like_qfq():
    b = Book(cash=0.0, pos={codes[0]: [1000.0, 10.0]}, last_close={codes[0]: 10.0})
    PaperBroker(CostModel()).corporate_actions(b, cidx, np.array([9.5, np.nan]))   # 0.5 cash dividend
    assert b.pos[codes[0]][0] == pytest.approx(1000 * 10 / 9.5)
    assert b.mark({codes[0]: 9.5})[1] == pytest.approx(10000.0)                      # value unchanged ex-date


def test_ledger_roundtrip(tmp_path):
    L = Ledger(tmp_path / "l.sqlite")
    b = Book(cash=123.0, pos={codes[0]: [100.0, 9.9]}, last_close={codes[0]: 10.0})
    b.save(L)
    L.commit()
    b2 = Book.from_ledger(Ledger(tmp_path / "l.sqlite"))
    assert b2.cash == 123.0 and b2.pos == {codes[0]: [100.0, 9.9]} and b2.last_close == {codes[0]: 10.0}


def test_code_mapping():
    assert to_jq("600000") == "600000.XSHG" and to_jq("000001.SZ") == "000001.XSHE" and to_jq("SH603000") == "603000.XSHG"
    assert to_qmt("600000.XSHG") == "600000.SH"


def test_holdings_import_gbk_with_cash_row(tmp_path):
    p = tmp_path / "h.csv"
    p.write_bytes("证券代码,证券名称,股票余额,成本价\n600000,浦发银行,1200,9.80\n000001,平安银行,0,11\nCASH,,52000.5,\n".encode("gbk"))
    b = read_holdings(p)
    assert b.cash == 52000.5 and b.pos == {"600000.XSHG": [1200.0, 9.8]}


def test_limit_prices_and_order_sheet(tmp_path):
    assert limit_price("buy", 10.0, 0.03) == 10.30 and limit_price("sell", 10.0, 0.03) == 9.70
    assert limit_price("buy", 10.0, 0.20) == 11.00                       # capped at the +10% limit
    o = [Order("600000.XSHG", "buy", 100, 10.0, "new", name="浦发银行"), Order("000001.XSHE", "sell", 200, 11.0, "exit")]
    text = write_order_sheet(tmp_path / "o.csv", "2026-10-08", o, 0.03).read_text(encoding="utf-8-sig")
    assert text.splitlines()[1].split(",")[4] == "卖出"                   # sells first


def test_calendar_projection():
    cal = Calendar(np.array(["2026-09-29", "2026-09-30"], dtype="datetime64[D]"))
    assert cal.next_day("2026-09-30") == dt.date(2026, 10, 8)            # National Day closure
    days = cal.january_exit_days(2027)
    assert days[0] == dt.date(2027, 1, 18) and days[-1] == dt.date(2027, 1, 29) and len(days) == 10
    assert any("provisional" in w for w in cal.warnings)
    with pytest.raises(CalendarError):
        cal.next_day("2027-12-31")


def test_non_stock_holdings_are_kept_out_of_the_book(tmp_path):
    from trading.brokers.base import is_a_share
    assert is_a_share("600000") and is_a_share("000001.SZ") and not is_a_share("204001") and not is_a_share("510300")
    p = tmp_path / "h.csv"
    p.write_text("证券代码,股票余额\n600000,1000\n204001,10\n510300,500\nCASH,1000\n", encoding="utf-8")
    b = read_holdings(p)
    assert list(b.pos) == ["600000.XSHG"] and dict(b.external) == {"204001": 10.0, "510300": 500.0}


def test_first_run_keeps_imported_holdings(tmp_path):
    from trading.config import Account, strategy_from_spec
    from trading.runner import ensure_ledger
    acct = Account(name="m", broker="manual", capital=5e5, start="2026-10-08", strategy=strategy_from_spec("MAIN J1-50"))
    L = Ledger(tmp_path / "l.sqlite")
    ensure_ledger(acct, L)
    assert L.cash() == 0.0 and L.never_invested()
    Book(cash=1000.0, pos={codes[0]: [100.0, 10.0]}).save(L)
    ensure_ledger(acct, L)                                      # the next `run` must not wipe the import
    assert L.cash() == 1000.0 and codes[0] in L.holdings() and not L.never_invested()
    p = Account(name="p", broker="paper", capital=5e5, start="2026-10-08", strategy=strategy_from_spec("MAIN J1-50"))
    L2 = Ledger(tmp_path / "p.sqlite")
    ensure_ledger(p, L2)
    assert L2.cash() == 5e5
