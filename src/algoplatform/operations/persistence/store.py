"""SQLite store: an append-only event journal plus query-friendly projections.

The journal (``events``) is the source of truth; ``orders`` and ``fills`` are upserted projections for fast
recovery. WAL mode and a commit per write mean an acknowledged write survives a process crash. Schema changes go
through numbered migrations so an old database file upgrades in place.

Tables: ``events`` (journal), ``orders`` / ``fills`` (projections), ``kill_switch``, ``kv`` (small JSON documents:
platform settings and services), ``bar_series`` / ``bars`` (downloaded market history, one row per bar).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ...foundation.core.exceptions import PlatformError
from ...foundation.core.logging import get_logger
from ...foundation.core.models import Fill, Order
from .codec import dumps, fill_from_dict, fill_to_dict, order_from_dict, order_to_dict

log = get_logger(__name__)


class StoreError(PlatformError):
    pass


MIGRATIONS: Dict[int, List[str]] = {
    1: [
        "CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, topic TEXT NOT NULL, payload TEXT NOT NULL)",
        "CREATE TABLE orders (client_order_id TEXT PRIMARY KEY, data TEXT NOT NULL, updated TEXT NOT NULL)",
        "CREATE TABLE fills (rowid_seq INTEGER PRIMARY KEY AUTOINCREMENT, fill_id TEXT UNIQUE NOT NULL, "
        "client_order_id TEXT NOT NULL, data TEXT NOT NULL)",
    ],
    2: [
        "CREATE TABLE kill_switch (id INTEGER PRIMARY KEY CHECK (id = 1), state TEXT NOT NULL, reason TEXT, source TEXT, ts TEXT)",
        "CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        "CREATE INDEX idx_events_topic ON events(topic)",
    ],
    3: [
        "CREATE TABLE bar_series (symbol TEXT NOT NULL, timeframe TEXT NOT NULL, provider TEXT NOT NULL, fetched_at TEXT NOT NULL, "
        "cleaning TEXT NOT NULL DEFAULT '{}', PRIMARY KEY (symbol, timeframe))",
        "CREATE TABLE bars (symbol TEXT NOT NULL, timeframe TEXT NOT NULL, ts TEXT NOT NULL, open REAL NOT NULL, high REAL NOT NULL, "
        "low REAL NOT NULL, close REAL NOT NULL, volume REAL NOT NULL, PRIMARY KEY (symbol, timeframe, ts)) WITHOUT ROWID",
    ],
}
SCHEMA_VERSION = max(MIGRATIONS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteStore:
    def __init__(self, path: str = ":memory:", _max_version: int = SCHEMA_VERSION) -> None:
        self.path = path
        self._lock = threading.RLock()
        try:
            self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)   # autocommit; explicit BEGIN
            if path != ":memory:":
                self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.Error as exc:
            raise StoreError(f"cannot open store {path}: {exc}") from exc
        self._migrate(_max_version)

    # ---- schema -------------------------------------------------------------------------------------
    def _migrate(self, max_version: int) -> None:
        with self._lock:
            self._db.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = self._db.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = row[0] or 0
            if current > SCHEMA_VERSION:
                raise StoreError(f"database schema v{current} is newer than this code (v{SCHEMA_VERSION})")
            for v in range(current + 1, max_version + 1):
                self._db.execute("BEGIN")
                try:
                    for stmt in MIGRATIONS[v]:
                        self._db.execute(stmt)
                    self._db.execute("INSERT INTO schema_version(version) VALUES (?)", (v,))
                    self._db.execute("COMMIT")
                except sqlite3.Error as exc:
                    self._db.execute("ROLLBACK")
                    raise StoreError(f"migration to v{v} failed: {exc}") from exc
                log.info("store migrated to schema v%d", v)

    @property
    def version(self) -> int:
        with self._lock:
            return self._db.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]

    def _tx(self, statements: Iterable[Tuple[str, tuple]]) -> int:
        with self._lock:
            try:
                self._db.execute("BEGIN")
                last = 0
                for sql, args in statements:
                    cur = self._db.execute(sql, args)
                    last = cur.lastrowid or last
                self._db.execute("COMMIT")
                return last
            except sqlite3.Error as exc:
                try:
                    self._db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise StoreError(f"store write failed: {exc}") from exc

    # ---- journal ------------------------------------------------------------------------------------
    def append_event(self, topic: str, payload: Dict[str, Any]) -> int:
        return self._tx([("INSERT INTO events(ts, topic, payload) VALUES (?,?,?)", (_now(), topic, dumps(payload)))])

    def events(self, after_seq: int = 0, topics: Optional[Iterable[str]] = None) -> Tuple[List[Tuple[int, str, str, Dict[str, Any]]], int]:
        """Returns (rows, corrupt_count). Rows with unreadable payloads are skipped and counted, never fatal."""
        with self._lock:
            rows = self._db.execute("SELECT seq, ts, topic, payload FROM events WHERE seq > ? ORDER BY seq", (after_seq,)).fetchall()
        wanted = set(topics) if topics else None
        out, corrupt = [], 0
        for seq, ts, topic, payload in rows:
            if wanted and topic not in wanted:
                continue
            try:
                out.append((seq, ts, topic, json.loads(payload)))
            except ValueError:
                corrupt += 1
        return out, corrupt

    def last_seq(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COALESCE(MAX(seq), 0) FROM events").fetchone()[0]

    def sequence_gaps(self) -> List[int]:
        """Missing journal sequence numbers (should be none: AUTOINCREMENT never reuses or skips on commit)."""
        with self._lock:
            seqs = [r[0] for r in self._db.execute("SELECT seq FROM events ORDER BY seq")]
        return [s for a, b in zip(seqs, seqs[1:]) for s in range(a + 1, b)]

    # ---- projections ----------------------------------------------------------------------------------
    def save_order(self, order: Order) -> None:
        self._tx([("INSERT INTO orders(client_order_id, data, updated) VALUES (?,?,?) "
                   "ON CONFLICT(client_order_id) DO UPDATE SET data=excluded.data, updated=excluded.updated",
                   (order.client_order_id, dumps(order_to_dict(order)), _now()))])

    def save_fill(self, fill: Fill) -> bool:
        """Idempotent: returns False if this fill id was already stored."""
        with self._lock:
            if self._db.execute("SELECT 1 FROM fills WHERE fill_id = ?", (fill.fill_id,)).fetchone():
                return False
        self._tx([("INSERT INTO fills(fill_id, client_order_id, data) VALUES (?,?,?)",
                   (fill.fill_id, fill.client_order_id, dumps(fill_to_dict(fill))))])
        return True

    def load_orders(self) -> List[Order]:
        with self._lock:
            rows = self._db.execute("SELECT data FROM orders ORDER BY rowid").fetchall()
        return [order_from_dict(json.loads(r[0])) for r in rows]

    def load_fills(self) -> List[Fill]:
        with self._lock:
            rows = self._db.execute("SELECT data FROM fills ORDER BY rowid_seq").fetchall()
        return [fill_from_dict(json.loads(r[0])) for r in rows]

    # ---- kill switch + key/value ----------------------------------------------------------------------
    def set_kill_switch(self, state: str, reason: str = "", source: str = "") -> None:
        self._tx([("INSERT INTO kill_switch(id, state, reason, source, ts) VALUES (1,?,?,?,?) "
                   "ON CONFLICT(id) DO UPDATE SET state=excluded.state, reason=excluded.reason, source=excluded.source, ts=excluded.ts",
                   (state, reason, source, _now()))])

    def get_kill_switch(self) -> Optional[Dict[str, str]]:
        with self._lock:
            r = self._db.execute("SELECT state, reason, source, ts FROM kill_switch WHERE id = 1").fetchone()
        return None if r is None else {"state": r[0], "reason": r[1] or "", "source": r[2] or "", "ts": r[3]}

    def put(self, key: str, value: Any) -> None:
        self._tx([("INSERT INTO kv(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, dumps(value)))])

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            r = self._db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return default if r is None else json.loads(r[0])

    def delete(self, key: str) -> bool:
        with self._lock:
            found = self._db.execute("SELECT 1 FROM kv WHERE key = ?", (key,)).fetchone() is not None
        if found:
            self._tx([("DELETE FROM kv WHERE key = ?", (key,))])
        return found

    # ---- market history ---------------------------------------------------------------------------------
    def replace_bars(self, symbol: str, timeframe: str, provider: str, fetched_at: str, cleaning: Dict[str, Any],
                     rows: Iterable[Tuple[str, float, float, float, float, float]]) -> int:
        """Replace one series atomically: its description and every bar. Returns the number of bars written."""
        rows = list(rows)
        with self._lock:
            try:
                self._db.execute("BEGIN")
                self._db.execute("DELETE FROM bars WHERE symbol = ? AND timeframe = ?", (symbol, timeframe))
                self._db.executemany("INSERT INTO bars(symbol, timeframe, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?)",
                                     [(symbol, timeframe, *r) for r in rows])
                self._db.execute("INSERT INTO bar_series(symbol, timeframe, provider, fetched_at, cleaning) VALUES (?,?,?,?,?) "
                                 "ON CONFLICT(symbol, timeframe) DO UPDATE SET provider=excluded.provider, fetched_at=excluded.fetched_at, "
                                 "cleaning=excluded.cleaning", (symbol, timeframe, provider, fetched_at, dumps(cleaning)))
                self._db.execute("COMMIT")
            except sqlite3.Error as exc:
                try:
                    self._db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise StoreError(f"could not store bars for {symbol} {timeframe}: {exc}") from exc
        return len(rows)

    def delete_bars(self, symbol: str, timeframe: str) -> bool:
        with self._lock:
            found = self._db.execute("SELECT 1 FROM bar_series WHERE symbol = ? AND timeframe = ?", (symbol, timeframe)).fetchone() is not None
        self._tx([("DELETE FROM bars WHERE symbol = ? AND timeframe = ?", (symbol, timeframe)),
                  ("DELETE FROM bar_series WHERE symbol = ? AND timeframe = ?", (symbol, timeframe))])
        return found

    def bar_series(self) -> List[Dict[str, Any]]:
        """Every stored series with its description and bar count, ordered by symbol and timeframe."""
        with self._lock:
            rows = self._db.execute("SELECT s.symbol, s.timeframe, s.provider, s.fetched_at, s.cleaning, COUNT(b.ts) FROM bar_series s "
                                    "LEFT JOIN bars b ON b.symbol = s.symbol AND b.timeframe = s.timeframe "
                                    "GROUP BY s.symbol, s.timeframe ORDER BY s.symbol, s.timeframe").fetchall()
        return [{"symbol": r[0], "timeframe": r[1], "provider": r[2], "fetched_at": r[3], "cleaning": json.loads(r[4] or "{}"), "rows": r[5]}
                for r in rows]

    def load_bars(self, symbol: str, timeframe: str) -> List[Tuple[str, float, float, float, float, float]]:
        with self._lock:
            return [tuple(r) for r in self._db.execute("SELECT ts, open, high, low, close, volume FROM bars WHERE symbol = ? AND timeframe = ? "
                                                       "ORDER BY ts", (symbol, timeframe)).fetchall()]

    def close(self) -> None:
        with self._lock:
            self._db.close()
