"""Kill switch. File-based, checked before every action.

File-based rather than a config flag or an API call, for three reasons:

  1. It works when the process is wedged, the API is down, or the dashboard
     cannot load. `touch KILL_SWITCH` needs no working software.
  2. It survives a restart. A flag in memory does not, and a supervisor that
     restarts a halted agent into an un-halted state is a real failure mode.
  3. It is checkable without the agent's cooperation. A switch the agent must
     agree to honor is not a switch.

Engaging the switch stops NEW risk. It does not stop exits: halt means "take no
new risk," never "stop managing existing risk."
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from osiris.config import KILL_SWITCH_PATH
from osiris.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class KillSwitchState:
    engaged: bool
    reason: str = ""
    engaged_at: datetime | None = None

    @property
    def allows_new_risk(self) -> bool:
        return not self.engaged

    @property
    def allows_exits(self) -> bool:
        """Always true. Exits are risk management, not risk taking."""
        return True


class KillSwitch:
    def __init__(self, path: Path = KILL_SWITCH_PATH) -> None:
        self.path = Path(path)

    def check(self) -> KillSwitchState:
        if not self.path.exists():
            return KillSwitchState(engaged=False)
        try:
            reason = self.path.read_text().strip() or "engaged (no reason recorded)"
            mtime = datetime.fromtimestamp(self.path.stat().st_mtime, tz=UTC)
        except OSError as exc:
            # Cannot read it but it exists: fail closed.
            return KillSwitchState(True, f"unreadable kill switch: {exc}")
        return KillSwitchState(True, reason, mtime)

    @property
    def engaged(self) -> bool:
        return self.check().engaged

    def engage(self, reason: str) -> KillSwitchState:
        stamp = datetime.now(UTC).isoformat()
        self.path.write_text(f"{reason}\nengaged_at={stamp}\n")
        log.error("killswitch.engaged", reason=reason)
        return self.check()

    def release(self, acknowledged_by: str) -> KillSwitchState:
        """Human-initiated only. A switch that clears itself is not a switch."""
        if not acknowledged_by.strip():
            raise ValueError(
                "Releasing the kill switch requires an acknowledgement. "
                "Automated release defeats the purpose."
            )
        if self.path.exists():
            self.path.unlink()
        log.warning("killswitch.released", acknowledged_by=acknowledged_by)
        return self.check()
