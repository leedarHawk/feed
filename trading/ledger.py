"""SQLite ledger of one account: holdings, cash, plans/orders, fills, daily NAV and risk events."""
import datetime as dt
import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta     (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS state    (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS holdings (code TEXT PRIMARY KEY, shares REAL NOT NULL, cost REAL NOT NULL);
CREATE TABLE IF NOT EXISTS days     (date TEXT PRIMARY KEY, nav REAL, cash REAL, mv REAL, ret REAL, bench_ret REAL,
                                     exposure REAL, n_pos INTEGER, peak REAL, drawdown REAL, turnover REAL, note TEXT);
CREATE TABLE IF NOT EXISTS positions(date TEXT, code TEXT, shares REAL, price REAL, value REAL, cost REAL,
                                     PRIMARY KEY (date, code));
CREATE TABLE IF NOT EXISTS orders   (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, stage TEXT, code TEXT,
                                     name TEXT, side TEXT, qty REAL, price REAL, limit_price REAL, reason TEXT,
                                     status TEXT, broker_id TEXT);
CREATE TABLE IF NOT EXISTS fills    (id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER, date TEXT, code TEXT,
                                     side TEXT, qty REAL, price REAL, fee REAL, stamp REAL, cash_delta REAL);
CREATE TABLE IF NOT EXISTS events   (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, date TEXT, level TEXT,
                                     kind TEXT, message TEXT);
CREATE INDEX IF NOT EXISTS orders_date ON orders(date);
CREATE INDEX IF NOT EXISTS fills_date ON fills(date);
"""


class Ledger:
    def __init__(self, path, fast=False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)
        if fast:                                   # replays: durability is irrelevant, speed is not
            self.db.execute("PRAGMA synchronous=OFF")
            self.db.execute("PRAGMA journal_mode=MEMORY")

    def close(self):
        self.db.commit()
        self.db.close()

    def commit(self):
        self.db.commit()

    # ---------------------------------------------------------------- key/value
    def _get(self, table, key, default=None):
        r = self.db.execute(f"SELECT value FROM {table} WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    def _set(self, table, key, value):
        self.db.execute(f"INSERT OR REPLACE INTO {table}(key, value) VALUES (?, ?)", (key, json.dumps(value)))

    def meta(self, key, default=None):
        return self._get("meta", key, default)

    def set_meta(self, key, value):
        self._set("meta", key, value)

    def get(self, key, default=None):
        return self._get("state", key, default)

    def set(self, key, value):
        self._set("state", key, value)

    # ---------------------------------------------------------------- holdings
    def holdings(self):
        return {c: (s, k) for c, s, k in self.db.execute("SELECT code, shares, cost FROM holdings")}

    def never_invested(self):
        """True until the account has held something: its first trading day is a full portfolio build."""
        return self.get("first_trade") is None and not self.holdings()

    def cash(self):
        return float(self.get("cash", 0.0))

    def replace_holdings(self, holdings, cash):
        """holdings: code -> (shares, cost per share)."""
        self.db.execute("DELETE FROM holdings")
        self.db.executemany("INSERT INTO holdings(code, shares, cost) VALUES (?, ?, ?)",
                            [(c, float(s), float(k)) for c, (s, k) in holdings.items() if s > 1e-9])
        self.set("cash", float(cash))

    # ---------------------------------------------------------------- history
    def last_day(self):
        r = self.db.execute("SELECT max(date) FROM days").fetchone()
        return r[0]

    def last_row(self):
        cur = self.db.execute("SELECT * FROM days ORDER BY date DESC LIMIT 1")
        r = cur.fetchone()
        return dict(zip([c[0] for c in cur.description], r)) if r else None

    def days(self):
        cur = self.db.execute("SELECT * FROM days ORDER BY date")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur]

    def record_day(self, row, positions):
        keys = list(row)
        self.db.execute(f"INSERT OR REPLACE INTO days({','.join(keys)}) VALUES ({','.join('?' * len(keys))})",
                        [row[k] for k in keys])
        self.db.execute("DELETE FROM positions WHERE date=?", (row["date"],))
        self.db.executemany("INSERT INTO positions(date, code, shares, price, value, cost) VALUES (?, ?, ?, ?, ?, ?)",
                            [(row["date"], *p) for p in positions])

    def add_orders(self, day, stage, orders, status):
        ids = []
        for o in orders:
            cur = self.db.execute(
                "INSERT INTO orders(date, stage, code, name, side, qty, price, limit_price, reason, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (day, stage, o.code, o.name, o.side, o.qty, o.price, o.limit, o.reason, status))
            ids.append(cur.lastrowid)
        return ids

    def set_order_status(self, order_id, status, broker_id=None):
        self.db.execute("UPDATE orders SET status=?, broker_id=coalesce(?, broker_id) WHERE id=?",
                        (status, broker_id, order_id))

    def orders(self, day, stage=None):
        q, a = "SELECT * FROM orders WHERE date=?", [day]
        if stage:
            q += " AND stage=?"
            a.append(stage)
        cur = self.db.execute(q + " ORDER BY id", a)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur]

    def add_fill(self, f):
        self.db.execute("INSERT INTO fills(order_id, date, code, side, qty, price, fee, stamp, cash_delta) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (f["order_id"], f["date"], f["code"], f["side"], f["qty"], f["price"], f["fee"], f["stamp"],
                         f["cash_delta"]))

    def fills(self, day):
        cur = self.db.execute("SELECT * FROM fills WHERE date=? ORDER BY id", (day,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur]

    def event(self, day, level, kind, message):
        self.db.execute("INSERT INTO events(ts, date, level, kind, message) VALUES (?, ?, ?, ?, ?)",
                        (dt.datetime.now().isoformat(timespec="seconds"), day, level, kind, message))

    def events(self, day=None):
        q = "SELECT ts, date, level, kind, message FROM events" + (" WHERE date=?" if day else "") + " ORDER BY id"
        return self.db.execute(q, (day,) if day else ()).fetchall()
