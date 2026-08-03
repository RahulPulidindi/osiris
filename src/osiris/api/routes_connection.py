"""Connection status and the risk acknowledgement.

Two facts the dashboard needs that nothing else exposed:

1. Whether Robinhood credentials exist on this machine (`~/.osiris/`). The
   `/api/health` broker name says which broker the *running process* got, but a
   fresh install needs to know whether `python -m osiris.connect` has ever run.

2. A way to record "I understand the risk" from the UI. Arming still requires
   the process to be restarted -- the dry-run flag is welded to the MCP adapter
   at connection time, deliberately, so an HTTP handler can never arm a live
   session in place. This endpoint only writes the two affirmations to `.env`;
   the restart is the second factor.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from osiris.api.state import RuntimeState
from osiris.config import REPO_ROOT
from osiris.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["connection"])

ENV_PATH = REPO_ROOT / ".env"

# The exact phrase the operator must type. A checkbox is a reflex; typing a
# sentence is a decision.
REQUIRED_PHRASE = "I understand the risk"


def _get_state() -> RuntimeState:
    from osiris.api.app import get_state

    return get_state()


class ConnectionOut(BaseModel):
    robinhood_linked: bool
    broker: str
    mode: str
    armed: bool
    risk_acknowledged: bool
    restart_required: bool
    connect_command: str = "python -m osiris.connect"


class ArmRequest(BaseModel):
    acknowledgement: str = Field(min_length=1, max_length=200)
    acknowledged_by: str = Field(min_length=1, max_length=120)


def _credentials_cached() -> bool:
    try:
        from osiris.mcp.session import FileTokenStorage

        return FileTokenStorage().has_tokens
    except Exception:  # pragma: no cover - import guard
        return False


def _env_says_armed() -> bool:
    """What `.env` will produce on the NEXT start, independent of this process."""
    if not ENV_PATH.exists():
        return False
    text = ENV_PATH.read_text()
    mode = re.search(r"^OSIRIS_MODE=(\S+)", text, re.M)
    risk = re.search(r"^OSIRIS_I_UNDERSTAND_THE_RISK=(\S+)", text, re.M)
    return bool(
        mode and mode.group(1).strip().lower() == "live"
        and risk and risk.group(1).strip().lower() == "yes"
    )


@router.get("/connection", response_model=ConnectionOut)
async def connection(state: RuntimeState = Depends(_get_state)) -> ConnectionOut:
    env_armed = _env_says_armed()
    return ConnectionOut(
        robinhood_linked=_credentials_cached(),
        broker=state.broker.name,
        mode=state.settings.mode.value,
        armed=state.armed,
        risk_acknowledged=env_armed or state.armed,
        # Acknowledged on disk but this process started before it was set.
        restart_required=env_armed and not state.armed,
    )


@router.post("/control/arm", response_model=ConnectionOut)
async def arm(
    body: ArmRequest, state: RuntimeState = Depends(_get_state)
) -> ConnectionOut:
    """Record both live-trading affirmations in `.env`.

    Refuses unless the acknowledgement phrase matches exactly and Robinhood
    credentials already exist -- acknowledging risk for an account the agent
    cannot even read is a sequencing error worth stopping.
    """
    if body.acknowledgement.strip().lower() != REQUIRED_PHRASE.lower():
        raise HTTPException(
            400, f'Acknowledgement must read exactly: "{REQUIRED_PHRASE}"'
        )
    if not _credentials_cached():
        raise HTTPException(
            409,
            "No Robinhood credentials cached. Run `python -m osiris.connect` first.",
        )

    # Live mode refuses to boot with zero recorded equity (a hard validator in
    # Settings), so arming must record the balance we actually observed or the
    # restart it asks for would crash the service. Refusing to arm before the
    # account has been read is the same guard, earlier and politer.
    equity = state.ledger.equity(state.prices)
    if equity <= 0:
        raise HTTPException(
            409,
            "The account balance has not been read yet. Wait for the service "
            "to finish connecting to Robinhood, then try again.",
        )

    lines = ENV_PATH.read_text().splitlines() if ENV_PATH.exists() else []
    wanted = {
        "OSIRIS_MODE": "live",
        "OSIRIS_I_UNDERSTAND_THE_RISK": "yes",
        "OSIRIS_ACCOUNT_EQUITY_USD": f"{equity:.2f}",
    }
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key in wanted:
            out.append(f"{key}={wanted[key]}")
            seen.add(key)
        else:
            out.append(line)
    for key, value in wanted.items():
        if key not in seen:
            out.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(out) + "\n")

    log.info(
        "connection.risk_acknowledged",
        by=body.acknowledged_by,
        at=datetime.now(UTC).isoformat(),
    )
    return await connection(state)
