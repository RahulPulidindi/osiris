"""Robinhood Trading MCP client.

Transport is streamable HTTP with OAuth 2.1 + PKCE. The official `mcp` SDK
handles the protocol; this module owns pagination, snapshotting, retry, and the
isError trap.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from osiris.logging import get_logger
from osiris.mcp.capabilities import (
    CapabilityRegistry,
    ToolCallFailed,
    ToolSpec,
    is_write,
)

log = get_logger(__name__)

# src/osiris/mcp/client.py -> src/osiris -> src -> repo root
SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "docs" / "mcp" / "tools-snapshot.json"
)


async def enumerate_tools(session: Any) -> list[ToolSpec]:
    """Drain the paginated tools/list cursor.

    A single listTools() call can return a partial surface with a nextCursor.
    Code that reads one page may conclude a tool does not exist.

    SDK v2 takes the cursor inside a `params` object rather than as a keyword.
    The old `cursor=` form raised TypeError, so pagination silently never worked.
    """
    from mcp.types import PaginatedRequestParams

    tools: list[ToolSpec] = []
    cursor: str | None = None
    pages = 0
    while True:
        page = await session.list_tools(
            params=PaginatedRequestParams(cursor=cursor) if cursor else None
        )
        for t in page.tools:
            # SDK v2 exposes this as `input_schema`; the wire alias is
            # `inputSchema`. Reading only the camelCase name silently produced an
            # EMPTY schema for every tool, which disabled client-side argument
            # validation entirely -- the server then rejected calls for missing
            # required properties that the snapshot claimed did not exist.
            schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", None) or {}
            tools.append(
                ToolSpec(
                    name=t.name,
                    description=getattr(t, "description", "") or "",
                    input_schema=schema if isinstance(schema, dict) else {},
                )
            )
        pages += 1
        cursor = getattr(page, "nextCursor", None)
        if not cursor or pages > 50:
            break
    log.info("mcp.enumerated", tool_count=len(tools), pages=pages)
    return sorted(tools, key=lambda t: t.name)


def write_snapshot(tools: list[ToolSpec], path: Path = SNAPSHOT_PATH) -> dict[str, Any]:
    """Write a diffable snapshot. Contains no tokens, only names and schemas."""
    from datetime import UTC, datetime

    snapshot = {
        "captured_at": datetime.now(UTC).isoformat(),
        "tool_count": len(tools),
        "write_tools": sorted(t.name for t in tools if is_write(t)),
        "tools": [
            {
                "name": t.name,
                "description": t.description,
                "required": t.required,
                "properties": sorted(t.properties),
                "inputSchema": t.input_schema,
            }
            for t in tools
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=2) + "\n")
    return snapshot


def load_snapshot(path: Path = SNAPSHOT_PATH) -> list[ToolSpec]:
    """Load a snapshot into ToolSpecs, for offline work and drift checks."""
    if not path.exists():
        raise FileNotFoundError(
            f"No MCP snapshot at {path}. Run `python -m osiris.mcp.enumerate` first "
            "(see docs/PHASE0.md)."
        )
    raw = json.loads(path.read_text())
    return [
        ToolSpec(
            name=t["name"],
            description=t.get("description", ""),
            input_schema=t.get("inputSchema", {}),
        )
        for t in raw["tools"]
    ]


def check_drift(live: list[ToolSpec], snapshot: list[ToolSpec]) -> list[str]:
    """Compare live surface to snapshot. Non-empty result should fail CI.

    A server-side rename must break the build, not a trade.
    """
    live_map = {t.name: t for t in live}
    snap_map = {t.name: t for t in snapshot}
    problems: list[str] = []

    for name in sorted(set(snap_map) - set(live_map)):
        problems.append(f"REMOVED: {name}")
    for name in sorted(set(live_map) - set(snap_map)):
        problems.append(f"ADDED: {name}")
    for name in sorted(set(live_map) & set(snap_map)):
        lr, sr = set(live_map[name].required), set(snap_map[name].required)
        if lr != sr:
            problems.append(f"REQUIRED CHANGED: {name}: {sorted(sr)} -> {sorted(lr)}")
    return problems


def text_of(result: Any) -> str:
    """Extract text from an MCP tool result content array."""
    parts: list[str] = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


class MCPAdapter:
    """Capability-oriented wrapper. Deliberately exposes no tool names.

    Every write passes through an optional guard before leaving the process.
    """

    def __init__(
        self,
        session: Any,
        tools: list[ToolSpec],
        guard: Any = None,
        *,
        dry_run: bool = True,
    ) -> None:
        self.session = session
        self.registry = CapabilityRegistry(tools)
        self.guard = guard
        self.dry_run = dry_run
        self._write_tools = self.registry.write_tools

    @property
    def tool_names(self) -> list[str]:
        return self.registry.tool_names

    def has(self, capability: str) -> bool:
        return self.registry.has(capability)

    async def call(
        self,
        capability: str,
        args: dict[str, Any] | None = None,
        *,
        retries: int = 3,
    ) -> Any:
        args = dict(args or {})
        tool = self.registry.resolve(capability)
        self.registry.validate_args(tool, args)

        is_write_call = tool.name in self._write_tools
        if is_write_call and self.dry_run:
            raise PermissionError(
                f"Refusing write {tool.name!r} in dry_run mode. "
                "Arm the live path explicitly (see osiris.config.Settings.live_armed)."
            )

        delay = 0.5
        last_exc: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                result = await self.session.call_tool(tool.name, args)
                # An MCP tool that fails logically returns isError with a 200.
                if getattr(result, "isError", False):
                    raise ToolCallFailed(tool.name, text_of(result) or "isError", result)
                log.debug("mcp.call.ok", tool=tool.name, capability=capability)
                return result
            except ToolCallFailed:
                raise  # logical failure; retrying will not help
            except Exception as exc:  # transport-level
                last_exc = exc
                if attempt == retries:
                    break
                log.warning(
                    "mcp.call.retry",
                    tool=tool.name,
                    attempt=attempt,
                    error=str(exc),
                )
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError(f"{tool.name} failed after {retries} attempts") from last_exc

    async def review_then_place(self, order_args: dict[str, Any]) -> Any:
        """Simulation is mandatory. No review capability means no order."""
        if not self.has("reviewOrder"):
            from osiris.mcp.capabilities import ToolUnavailable

            raise ToolUnavailable("reviewOrder", self.tool_names)
        review = await self.call("reviewOrder", order_args)
        return review
