"""CLI: enumerate the live MCP tool surface and snapshot it.

    python -m osiris.mcp.enumerate            # snapshot + capability report
    python -m osiris.mcp.enumerate --check    # drift check against snapshot (CI)
    python -m osiris.mcp.enumerate --offline  # report from existing snapshot

OAuth 2.1 + PKCE is interactive on first run; the SDK caches the token.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from osiris.config import load_settings
from osiris.logging import configure_logging, get_logger
from osiris.mcp.capabilities import CapabilityRegistry, ToolSpec
from osiris.mcp.client import (
    check_drift,
    enumerate_tools,
    load_snapshot,
    write_snapshot,
)

log = get_logger(__name__)


async def _connect_and_enumerate(endpoint: str) -> list[ToolSpec]:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The `mcp` SDK is required for live enumeration: pip install 'mcp>=1.29.0'"
        ) from exc

    async with streamablehttp_client(endpoint) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await enumerate_tools(session)


def _print_report(tools: list[ToolSpec]) -> None:
    registry = CapabilityRegistry(tools)
    print(f"\n{len(tools)} tools advertised\n")

    writes = sorted(registry.write_tools)
    print(f"Classified as WRITE ({len(writes)}) — these pass through the risk kernel:")
    for name in writes:
        print(f"  {name}")

    print("\nCapability resolution:")
    report = registry.capability_report()
    for cap, tool in sorted(report.items()):
        mark = "ok  " if tool else "MISS"
        print(f"  {mark} {cap:<24} -> {tool or '(unavailable)'}")

    missing = [c for c, t in report.items() if t is None]
    critical = {"placeOrder", "reviewOrder", "listPositions", "getPortfolio", "getQuotes"}
    blocked = critical & set(missing)
    if blocked:
        print(f"\nCRITICAL capabilities unavailable: {sorted(blocked)}")
        print("The agent cannot operate until these resolve.")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enumerate Robinhood MCP tool surface")
    parser.add_argument("--check", action="store_true", help="drift check (exit 1 on drift)")
    parser.add_argument("--offline", action="store_true", help="report from snapshot only")
    args = parser.parse_args(argv)

    configure_logging()
    settings = load_settings()

    if args.offline:
        _print_report(load_snapshot())
        return 0

    live = asyncio.run(_connect_and_enumerate(settings.mcp_endpoint))

    if args.check:
        problems = check_drift(live, load_snapshot())
        if problems:
            print("MCP SCHEMA DRIFT DETECTED:", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            print(
                "\nReview each change, then re-snapshot deliberately.",
                file=sys.stderr,
            )
            return 1
        print("No drift. Live surface matches snapshot.")
        return 0

    snapshot = write_snapshot(live)
    print(f"Wrote docs/mcp/tools-snapshot.json ({snapshot['tool_count']} tools)")
    _print_report(live)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
