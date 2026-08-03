"""In-process event bus feeding one multiplexed SSE stream.

**One EventSource, not one per channel.** Browsers cap concurrent connections per
origin (6 on HTTP/1.1), so a stream-per-channel design starves the rest of the app
of connections. Everything is multiplexed over a single connection and demuxed on
the `event` field client-side.

Subscribers are **bounded and lossy by design**. A slow or stalled client must
never apply backpressure to the trading loop: dropping dashboard frames is
acceptable, delaying an order is not. When a queue fills, the oldest frames are
discarded and a `dropped` counter is surfaced so the UI can show it is behind
rather than silently displaying stale numbers.
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from osiris.execution.journal import _default
from osiris.logging import get_logger

log = get_logger(__name__)


class Channel(str, Enum):
    """SSE event names. The client switches on these."""

    HEARTBEAT = "heartbeat"
    CYCLE = "cycle"
    AGENT = "agent"            # streamed reasoning tokens
    FILL = "fill"
    INTENT = "intent"
    VETO = "veto"
    PNL = "pnl"
    BREAKER = "breaker"
    RANKING = "ranking"
    RECONCILIATION = "reconciliation"
    ERROR = "error"


@dataclass(frozen=True)
class Event:
    channel: Channel
    data: dict[str, Any]
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    seq: int = 0

    def to_sse_data(self) -> str:
        return json.dumps(
            {"seq": self.seq, "ts": self.ts.isoformat(), **self.data},
            default=_default,
        )


class Subscriber:
    """A bounded, lossy queue for one connected client."""

    def __init__(self, maxsize: int = 500) -> None:
        self.queue: deque[Event] = deque(maxlen=maxsize)
        self.dropped = 0
        self._wakeup = asyncio.Event()

    def push(self, event: Event) -> None:
        if len(self.queue) == self.queue.maxlen:
            self.dropped += 1
        self.queue.append(event)
        self._wakeup.set()

    async def get(self, timeout: float = 15.0) -> Event | None:
        """Next event, or None on timeout so the caller can send a keepalive."""
        if not self.queue:
            self._wakeup.clear()
            try:
                await asyncio.wait_for(self._wakeup.wait(), timeout=timeout)
            except TimeoutError:
                return None
        return self.queue.popleft() if self.queue else None


class EventBus:
    """Fan-out to subscribers, plus a replay buffer for late joiners.

    The replay buffer matters: a dashboard opened mid-session should show the
    day's context immediately rather than an empty screen until the next event.
    """

    def __init__(self, *, replay_size: int = 200) -> None:
        self._subscribers: set[Subscriber] = set()
        self._replay: deque[Event] = deque(maxlen=replay_size)
        self._seq = 0

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, channel: Channel, data: dict[str, Any]) -> Event:
        self._seq += 1
        event = Event(channel=channel, data=data, seq=self._seq)
        if channel is not Channel.HEARTBEAT:
            self._replay.append(event)
        for sub in list(self._subscribers):
            sub.push(event)
        return event

    def subscribe(self, *, replay: bool = True) -> Subscriber:
        sub = Subscriber()
        if replay:
            for event in self._replay:
                sub.push(event)
        self._subscribers.add(sub)
        log.debug("eventbus.subscribed", subscribers=len(self._subscribers))
        return sub

    def unsubscribe(self, sub: Subscriber) -> None:
        self._subscribers.discard(sub)
        log.debug("eventbus.unsubscribed", subscribers=len(self._subscribers))

    def replay_buffer(self) -> list[Event]:
        return list(self._replay)


# Module-level bus. The API and the trading loop share one instance.
BUS = EventBus()
