"""
EventLogger — one place that fans an event out to (a) stdout, (b) the storage `events`
table, and (c) a daily-rotated JSON file (the same pattern the repo's other bots use, which
project memory flags as the trustworthy log source over stdout .log files).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .storage.base import Storage


class EventLogger:
    def __init__(self, storage: Storage, log_dir: Path, *, echo: bool = True) -> None:
        self.storage = storage
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.echo = echo

    def _json_path(self) -> Path:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"events-{day}.json"

    def emit(self, event_type: str, *, severity: str = "info", message: str = "",
             cycle_id: str | None = None, **payload: Any) -> None:
        ts = datetime.now(timezone.utc)
        rec = {"ts": ts.isoformat(), "event_type": event_type, "severity": severity,
               "cycle_id": cycle_id, "message": message, "payload": payload}
        # storage
        try:
            self.storage.record_event({
                "ts": ts, "event_type": event_type, "severity": severity,
                "cycle_id": cycle_id, "message": message[:512], "payload": payload,
            })
        except Exception as exc:  # never let logging kill the loop
            print(f"[eventlog] storage write failed: {exc}")
        # json file (append one object per line)
        try:
            with self._json_path().open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass
        # stdout
        if self.echo:
            tag = severity.upper()
            cid = f" [{cycle_id}]" if cycle_id else ""
            print(f"{ts.strftime('%H:%M:%S')} {tag:8s}{cid} {event_type}: {message}")
