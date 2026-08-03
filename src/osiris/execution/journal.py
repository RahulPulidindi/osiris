"""Append-only journal. The audit trail and the tax substantiation.

Two properties are load-bearing:

  1. **Append-only.** Records are never mutated or deleted. A correction is a new
     record that supersedes an earlier one. A journal that can be rewritten is
     not evidence of anything.

  2. **Vetoes are recorded, not just fills.** A kernel that silently blocks every
     order looks identical to a quiet market from the outside. Without veto
     records you cannot tell "no opportunities" from "the system is broken."

Stored as JSONL: durable, greppable, survives a schema change, and needs no
migration. Each line is a self-describing event.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from osiris.logging import get_logger

log = get_logger(__name__)


class EventType(str, Enum):
    CYCLE_START = "cycle_start"
    CYCLE_END = "cycle_end"
    UNIVERSE_BUILT = "universe_built"
    REGIME_CLASSIFIED = "regime_classified"
    FUNNEL_TRACE = "funnel_trace"
    PLAN_PRODUCED = "plan_produced"
    INTENT_EMITTED = "intent_emitted"
    KERNEL_APPROVED = "kernel_approved"
    KERNEL_VETO = "kernel_veto"
    REVIEW_PASSED = "review_passed"
    REVIEW_REJECTED = "review_rejected"
    ORDER_PLACED = "order_placed"
    ORDER_FAILED = "order_failed"
    FILL = "fill"
    RECONCILIATION = "reconciliation"
    RECONCILIATION_BREAK = "reconciliation_break"
    BREAKER_TRIPPED = "breaker_tripped"
    BREAKER_RESET = "breaker_reset"
    KILL_SWITCH = "kill_switch"
    POSTMORTEM = "postmortem"
    ERROR = "error"


def _default(obj: Any) -> Any:
    """Serialize the domain types that show up in payloads."""
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, frozenset | set):
        return sorted(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


@dataclass(frozen=True)
class JournalEvent:
    ts: datetime
    event: EventType
    payload: dict[str, Any]
    correlation_id: str = ""
    seq: int = 0

    def to_json(self) -> str:
        return json.dumps(
            {
                "seq": self.seq,
                "ts": self.ts.isoformat(),
                "event": self.event.value,
                "correlation_id": self.correlation_id,
                "payload": self.payload,
            },
            default=_default,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, line: str) -> JournalEvent:
        raw = json.loads(line)
        return cls(
            ts=datetime.fromisoformat(raw["ts"]),
            event=EventType(raw["event"]),
            payload=raw.get("payload", {}),
            correlation_id=raw.get("correlation_id", ""),
            seq=int(raw.get("seq", 0)),
        )


class Journal:
    """Append-only JSONL writer with an in-process sequence counter.

    Each write is flushed and fsynced. Slower, deliberately: an order that was
    placed but not journaled is an unreconcilable position.
    """

    def __init__(self, path: Path, *, fsync: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fsync = fsync
        self._lock = threading.Lock()
        self._seq = self._last_seq()

    def _last_seq(self) -> int:
        if not self.path.exists():
            return 0
        last = 0
        with self.path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    last = max(last, int(json.loads(line).get("seq", 0)))
                except (json.JSONDecodeError, ValueError, TypeError):
                    continue
        return last

    def append(
        self,
        event: EventType,
        payload: dict[str, Any] | None = None,
        *,
        correlation_id: str = "",
    ) -> JournalEvent:
        with self._lock:
            self._seq += 1
            rec = JournalEvent(
                ts=datetime.now(UTC),
                event=event,
                payload=payload or {},
                correlation_id=correlation_id,
                seq=self._seq,
            )
            with self.path.open("a") as fh:
                fh.write(rec.to_json() + "\n")
                fh.flush()
                if self.fsync:
                    os.fsync(fh.fileno())
        return rec

    def read(
        self,
        *,
        event: EventType | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[JournalEvent]:
        out = list(self.iter_events(event=event, since=since))
        return out[-limit:] if limit else out

    def iter_events(
        self, *, event: EventType | None = None, since: datetime | None = None
    ) -> Iterator[JournalEvent]:
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = JournalEvent.from_json(line)
                except (json.JSONDecodeError, KeyError, ValueError):
                    log.warning("journal.corrupt_line", line=line[:120])
                    continue
                if event is not None and rec.event is not event:
                    continue
                if since is not None and rec.ts < since:
                    continue
                yield rec

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for rec in self.iter_events():
            out[rec.event.value] = out.get(rec.event.value, 0) + 1
        return out

    def veto_summary(self) -> dict[str, int]:
        """Veto codes by frequency. The first place to look when nothing trades."""
        out: dict[str, int] = {}
        for rec in self.iter_events(event=EventType.KERNEL_VETO):
            for code in rec.payload.get("vetoes", []):
                out[code] = out.get(code, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))
