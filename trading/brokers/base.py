"""Account book (holdings + cash) and helpers shared by the brokers."""
from dataclasses import dataclass, field

import numpy as np


A_SHARE_PREFIXES = ("600", "601", "603", "605", "000", "001", "002", "003", "300", "301", "688", "689")


def is_a_share(code):
    """True for SSE/SZSE A-share stock codes; ETFs, bonds, repos (e.g. 204001) and BSE names are not."""
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    return len(digits) == 6 and digits[:3] in A_SHARE_PREFIXES


def to_jq(code):
    """'600000' / '600000.SH' / 'SH600000' -> '600000.XSHG' (the panel's codes)."""
    c = str(code).strip().upper()
    if c.endswith((".XSHG", ".XSHE")):
        return c
    digits = "".join(ch for ch in c if ch.isdigit()).zfill(6)[-6:]
    if c.endswith(".SH") or c.startswith("SH") or digits[0] in "69":
        return digits + ".XSHG"
    if c.endswith(".SZ") or c.startswith("SZ") or digits[0] in "023":
        return digits + ".XSHE"
    raise ValueError(f"cannot map {code!r} to an SSE/SZSE code")


def to_qmt(code):
    return code.replace(".XSHG", ".SH").replace(".XSHE", ".SZ")


@dataclass
class Book:
    cash: float
    pos: dict = field(default_factory=dict)          # code -> [shares, cost per share]
    last_close: dict = field(default_factory=dict)   # code -> last traded close (values suspended names)
    external: list = field(default_factory=list)     # (code, quantity) held but outside the strategy (ETF, repo...)

    @classmethod
    def from_ledger(cls, ledger):
        return cls(cash=ledger.cash(), pos={c: [s, k] for c, (s, k) in ledger.holdings().items()},
                   last_close=dict(ledger.get("last_close", {})))

    def save(self, ledger):
        ledger.replace_holdings({c: tuple(v) for c, v in self.pos.items()}, self.cash)
        ledger.set("last_close", {c: self.last_close[c] for c in self.pos if c in self.last_close})

    def shares_array(self, cidx, n):
        a = np.zeros(n)
        for c, (s, _) in self.pos.items():
            if c not in cidx:
                raise KeyError(f"held code {c} is not in the panel")
            a[cidx[c]] = s
        return a

    def close_array(self, cidx, n):
        a = np.full(n, np.nan)
        for c in self.pos:
            if c in self.last_close:
                a[cidx[c]] = self.last_close[c]
        return a

    def mark(self, prices):
        """prices: code -> close (NaN / missing = not traded, keep the last close)."""
        for c in self.pos:
            p = prices.get(c)
            if p is not None and np.isfinite(p):
                self.last_close[c] = float(p)
        mv = sum(s * self.last_close.get(c, k) for c, (s, k) in self.pos.items())
        return mv, mv + self.cash
