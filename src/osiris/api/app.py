"""FastAPI application: read-mostly API plus one multiplexed SSE stream.

Deliberate constraint: **this API cannot trade.** It exposes no order endpoint, no
limit mutation, and no way to arm the live path. The only writes are the kill
switch and a manual breaker reset, which are safety controls that fail toward
stopping. A dashboard that can place orders is an attack surface aimed at the
account, and it would put an HTTP handler inside the risk path.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.sse import EventSourceResponse, ServerSentEvent

from osiris.api.events import Channel
from osiris.api.schemas import (
    Health,
    KillSwitchReleaseRequest,
    KillSwitchRequest,
    PreflightOut,
)
from osiris.api.state import RuntimeState, build_runtime_state
from osiris.logging import configure_logging, get_logger

log = get_logger(__name__)

_STATE: RuntimeState | None = None


def get_state() -> RuntimeState:
    if _STATE is None:  # pragma: no cover - set during lifespan
        raise HTTPException(503, "runtime state not initialized")
    return _STATE


def set_state(state: RuntimeState) -> None:
    """Injection point for tests and for the paper/live runners."""
    global _STATE
    _STATE = state


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    if _STATE is None:
        set_state(build_runtime_state())
    log.info("api.started", mode=get_state().settings.mode.value)
    yield
    log.info("api.stopped")


app = FastAPI(
    title="Osiris",
    description="Autonomous equity ranking agent with a deterministic risk kernel.",
    version="0.1.0",
    lifespan=lifespan,
)

# The dashboard is served from a separate origin in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=Health)
async def health(state: RuntimeState = Depends(get_state)) -> Health:
    ks = state.killswitch.check()
    return Health(
        status="ok",
        mode=state.settings.mode.value,
        armed=state.armed,
        account_type=state.settings.account_type.value,
        broker=state.broker.name,
        kill_switch_engaged=ks.engaged,
        breakers_tripped=[c.value for c in state.breakers.tripped],
        subscribers=state.bus.subscriber_count,
    )


@app.get("/api/preflight", response_model=PreflightOut)
async def preflight(state: RuntimeState = Depends(get_state)) -> PreflightOut:
    """Go-live readiness. READ-ONLY: reporting readiness cannot cause arming.

    Exposed so an operator can see exactly which checks block a live run without
    shelling into the box. `armed: true` means *cleared to arm* -- actually
    arming still requires the two independent affirmations in `Settings`, neither
    of which is reachable over HTTP.
    """
    from osiris.runner.preflight import run_preflight

    return PreflightOut(**run_preflight(state).to_dict())


@app.get("/api/stream", response_class=EventSourceResponse)
async def stream(state: RuntimeState = Depends(get_state)):
    """Single multiplexed SSE stream. The client demuxes on the event name.

    One connection rather than one per channel: browsers cap concurrent
    connections per origin, so a stream-per-channel design would starve the rest
    of the app of connections.

    Note the shape here: this is an async GENERATOR path operation declaring
    `response_class=EventSourceResponse`, not a function returning
    `EventSourceResponse(...)`. In FastAPI 0.141 the SSE encoding lives in the
    routing layer, and that class is only a marker that sets the media type.
    Returning it directly bypasses the encoder, which yields a 200 with the right
    `text/event-stream` header and then an immediately closed body -- a confusing
    failure that looks like a client bug.
    """
    subscriber = state.bus.subscribe(replay=True)

    async def generator():
        # Disconnect is detected via CancelledError, NOT request.is_disconnected().
        # On a streaming response that helper polls the ASGI receive channel, and
        # for a client that has sent its request and is only reading, it reports a
        # disconnect immediately -- closing the stream on the first iteration. The
        # server correctly returns text/event-stream and then hangs up, which is a
        # confusing failure to diagnose from the client side.
        # `raw_data` rather than `data`, because the payloads here are ALREADY
        # serialized JSON. The `data` field is always JSON-encoded by FastAPI, so
        # passing a JSON string through it double-encodes: the wire carries
        # `data: "{\"seq\": 1}"` and the client's JSON.parse returns a string
        # instead of an object. That failure is silent on the server side.
        try:
            yield ServerSentEvent(
                event=Channel.HEARTBEAT.value,
                raw_data='{"status":"connected"}',
                retry=3_000,
            )
            while True:
                event = await subscriber.get(timeout=15.0)
                if event is None:
                    # Keepalive comment: idle connections get dropped by proxies.
                    yield ServerSentEvent(comment="keepalive")
                    continue
                yield ServerSentEvent(
                    event=event.channel.value,
                    raw_data=event.to_sse_data(),
                    id=str(event.seq),
                )
        except asyncio.CancelledError:
            raise
        finally:
            state.bus.unsubscribe(subscriber)

    async for event in generator():
        yield event


@app.post("/api/control/kill-switch", response_model=Health)
async def engage_kill_switch(
    body: KillSwitchRequest, state: RuntimeState = Depends(get_state)
) -> Health:
    """Engage the kill switch. Stops new risk; exits still run."""
    state.killswitch.engage(body.reason)
    state.publish(Channel.BREAKER, {"kind": "kill_switch", "engaged": True, "reason": body.reason})
    return await health(state)


@app.post("/api/control/kill-switch/release", response_model=Health)
async def release_kill_switch(
    body: KillSwitchReleaseRequest, state: RuntimeState = Depends(get_state)
) -> Health:
    """Release the kill switch. Requires a named acknowledgement."""
    try:
        state.killswitch.release(body.acknowledged_by)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    state.publish(
        Channel.BREAKER,
        {"kind": "kill_switch", "engaged": False, "acknowledged_by": body.acknowledged_by},
    )
    return await health(state)


@app.post("/api/control/breakers/reset", response_model=Health)
async def reset_breakers(
    body: KillSwitchReleaseRequest, state: RuntimeState = Depends(get_state)
) -> Health:
    """Manual breaker reset. Human-initiated only.

    A fuse that resets itself is not a fuse, so there is no automatic path here.
    """
    state.breakers = state.breakers.reset()
    from osiris.execution.journal import EventType

    state.journal.append(
        EventType.BREAKER_RESET, {"acknowledged_by": body.acknowledged_by}
    )
    state.publish(Channel.BREAKER, {"kind": "breakers", "tripped": []})
    return await health(state)
