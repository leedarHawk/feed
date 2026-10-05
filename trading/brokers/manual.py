"""Manual broker: the system writes an order sheet, a person enters it in the broker client, and the account
is brought back in sync by importing the client's holdings export after the close."""
import csv
import io
import math
from pathlib import Path

from .base import Book, is_a_share, to_jq

CODE_COLS = ("code", "证券代码", "代码", "股票代码")
SHARE_COLS = ("shares", "股票余额", "当前持仓", "持仓数量", "实际数量", "证券数量", "持股数量", "参考持股")
COST_COLS = ("cost", "成本价", "参考成本价", "买入成本", "持仓成本")
CASH_ROW = ("CASH", "现金", "资金")


def _round_down(x, tick=0.01):
    return math.floor(x / tick + 1e-9) * tick


def _round_up(x, tick=0.01):
    return math.ceil(x / tick - 1e-9) * tick


def limit_price(side, ref, band, prev_close=None):
    """A limit inside the main-board +/-10% band of the previous close and `band` around the reference."""
    pc = prev_close or ref
    hi, lo = round(pc * 1.10 + 1e-9, 2), round(pc * 0.90 + 1e-9, 2)
    if side == "buy":
        return round(min(hi, _round_down(ref * (1 + band))), 2)
    return round(max(lo, _round_up(ref * (1 - band))), 2)


def write_order_sheet(path, day, orders, band):
    """CSV (UTF-8 with BOM so Excel opens it) in the order to enter: all sells, then buys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(orders, key=lambda o: (o.side != "sell", -o.value))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["交易日", "序号", "证券代码", "证券名称", "方向", "数量", "参考价", "委托限价", "预计金额", "说明"])
        for i, o in enumerate(rows, 1):
            lim = o.limit if o.limit else limit_price(o.side, o.price, band)
            w.writerow([day, i, o.code[:6], o.name, "卖出" if o.side == "sell" else "买入", int(round(o.qty)),
                        f"{o.price:.2f}", f"{lim:.2f}", f"{o.value:.0f}", o.reason])
    return path


def _read_table(path):
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "gbk"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    dialect = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",\t;")
    return list(csv.DictReader(io.StringIO(text), dialect=dialect))


def _pick(row, names):
    for n in names:
        for k, v in row.items():
            if k and k.strip() == n and v not in (None, ""):
                return v.strip().strip("=\"'")
    return None


def read_holdings(path, cash=None):
    """Holdings export -> Book. Cash comes from `cash` or a row whose code is CASH / 现金 / 资金."""
    pos, found_cash, external = {}, None, []
    for row in _read_table(path):
        code = _pick(row, CODE_COLS)
        if code is None:
            continue
        if code.upper() in CASH_ROW:
            found_cash = float(_pick(row, SHARE_COLS) or _pick(row, ("amount", "金额", "可用资金")))
            continue
        shares = float(_pick(row, SHARE_COLS) or 0)
        if shares <= 0:
            continue
        if not is_a_share(code):
            external.append((code, shares))
            continue
        cost = float(_pick(row, COST_COLS) or 0)
        pos[to_jq(code)] = [shares, cost]
    cash = cash if cash is not None else found_cash
    if cash is None:
        raise ValueError(f"{path}: no cash given (pass --cash or add a CASH row)")
    return Book(cash=float(cash), pos=pos, external=external)
