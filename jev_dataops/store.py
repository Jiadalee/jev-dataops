"""Small durable metadata store. Dataset contents remain on disk."""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "metadata.sqlite3"
        self.lock = threading.RLock()
        with closing(self.connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("CREATE TABLE IF NOT EXISTS items (kind TEXT, id TEXT, body TEXT, PRIMARY KEY(kind,id))")

    def connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def put(self, kind, item):
        with self.lock, closing(self.connect()) as db, db:
            db.execute("INSERT OR REPLACE INTO items VALUES (?,?,?)", (kind, item["id"], json.dumps(item, ensure_ascii=False, allow_nan=False)))
        return item

    def get(self, kind, item_id):
        with closing(self.connect()) as db:
            row = db.execute("SELECT body FROM items WHERE kind=? AND id=?", (kind, item_id)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, kind, limit=100, offset=0):
        with closing(self.connect()) as db:
            rows = db.execute("SELECT body FROM items WHERE kind=? ORDER BY rowid DESC LIMIT ? OFFSET ?", (kind, limit, offset)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def update(self, item_id, **values):
        with self.lock:
            item = self.get("run", item_id)
            item.update(values, updated_at=now())
            return self.put("run", item)

    def log(self, item_id, message):
        with self.lock:
            item = self.get("run", item_id)
            item["logs"] = (item.get("logs", []) + [{"time": now(), "message": message}])[-200:]
            return self.put("run", item)

    def recover(self):
        with self.lock, closing(self.connect()) as db, db:
            for item_id, body in db.execute("SELECT id,body FROM items WHERE kind='run'").fetchall():
                item = json.loads(body)
                if item["status"] in {"running", "queued"}:
                    item.update(status="failed", error="Service interrupted. Retry to reuse the screening cache.", updated_at=now())
                    db.execute("UPDATE items SET body=? WHERE kind='run' AND id=?", (json.dumps(item), item_id))
