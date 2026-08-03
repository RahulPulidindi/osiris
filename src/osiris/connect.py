"""Connect Osiris to your Robinhood account.

    python -m osiris.connect              # authorize + verify + snapshot
    python -m osiris.connect --status     # are we connected?
    python -m osiris.connect --logout     # forget cached credentials

Run this once. It opens a browser for Robinhood's OAuth consent, caches the
tokens under `~/.osiris/`, and then proves the connection actually works by
reading your account rather than merely reporting "connected".

That last part is the point. A successful token exchange means authentication
worked, not that the agent can trade: the tool surface is account-specific, so a
connection can succeed while `placeOrder` is absent. This command tells you which
one you have.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from osiris.config import load_settings
from osiris.logging import configure_logging, get_logger
from osiris.mcp.client import write_snapshot
from osiris.mcp.session import FileTokenStorage, LiveConnection

log = get_logger(__name__)

# Without these the agent cannot operate, and it is better to say so at connect
# time than to discover it mid-session.
REQUIRED = ("getPortfolio", "listPositions", "getQuotes", "reviewOrder", "placeOrder")


def _fmt(label: str, value: str) -> str:
    return f"  {label:.<26} {value}"


async def _dump_accounts(adapter) -> None:
    """Print the raw `get_accounts` response, with obvious secrets withheld.

    Account numbers are redacted to their last four. The point is to reveal the
    STRUCTURE, and a full account number in terminal scrollback is a needless
    liability.
    """
    import json as _json

    from osiris.execution.mcp_broker import _extract_payload

    print("\n  --- raw get_accounts response ---")
    try:
        result = await adapter.call("listAccounts", {})
    except Exception as exc:
        print(f"  call failed: {exc}\n")
        return

    payload = _extract_payload(result)

    def redact(obj):
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in {"account_number", "accountNumber", "number"} and isinstance(
                    v, str | int
                ):
                    out[k] = f"...{str(v)[-4:]}"
                else:
                    out[k] = redact(v)
            return out
        if isinstance(obj, list):
            return [redact(v) for v in obj]
        return obj

    text = _json.dumps(redact(payload), indent=2)
    print("\n".join(f"  {line}" for line in text.splitlines()[:60]))
    print("  --- end ---\n")


async def _dump_portfolio(adapter, account: str) -> None:
    """Print every numeric field in the portfolio response, with its path.

    Deliberately shows VALUES here, unlike the account dump: the whole question is
    which number represents total account value, and that is unanswerable from
    field names alone when several plausible candidates coexist.
    """
    from osiris.execution.mcp_broker import _coerce_number, _extract_payload

    print("\n  --- numeric fields in get_portfolio ---")
    try:
        result = await adapter.call("getPortfolio", {"account_number": account})
    except Exception as exc:
        print(f"  call failed: {exc}\n")
        return

    payload = _extract_payload(result)
    found: list[tuple[str, float]] = []

    def walk(node, path: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else key)
        elif isinstance(node, list):
            for i, item in enumerate(node[:3]):
                walk(item, f"{path}[{i}]")
        else:
            number = _coerce_number(node)
            if number is not None:
                found.append((path, number))

    walk(payload)
    if not found:
        print("  no numeric fields found at all")
    for path, value in found[:40]:
        print(f"  {path:<44} {value:>16,.2f}")
    print("  --- end ---\n")


async def run_connect(*, snapshot: bool, debug_accounts: bool = False) -> int:
    settings = load_settings()
    storage = FileTokenStorage()

    print("\nOsiris → Robinhood MCP")
    print(f"  endpoint: {settings.mcp_endpoint}")
    if storage.has_tokens:
        print("  using cached credentials (delete with --logout)\n")
    else:
        print("  no cached credentials; browser authorization required\n")

    conn = LiveConnection(endpoint=settings.mcp_endpoint, dry_run=True)
    try:
        await conn.open(interactive=True)
    except Exception as exc:
        print(f"\nConnection FAILED: {exc}\n")
        print("Common causes:")
        print("  - Robinhood Agentic access not enabled on the account")
        print("  - authorization window timed out (re-run this command)")
        print("  - the callback port is in use (close other Osiris processes)")
        return 1

    try:
        print(f"Connected. {len(conn.tools)} tools advertised.\n")

        missing = conn.require_capabilities(*REQUIRED)
        print("Capabilities the agent needs:")
        for name in REQUIRED:
            tool = None
            if conn.adapter.has(name):
                tool = conn.adapter.registry.resolve(name).name
            print(_fmt(name, tool or "UNAVAILABLE"))
        print()

        # --- Prove it by reading the account, not by trusting the handshake. ---
        print("Reading your account:")
        from osiris.execution.mcp_broker import MCPBroker

        broker = MCPBroker(conn.adapter)
        read_failures: list[str] = []

        account = await broker.resolve_account()
        if account:
            print(_fmt("account", f"...{account[-4:]}"))
        else:
            print(_fmt("account", "COULD NOT RESOLVE"))
            read_failures.append(
                "account number could not be determined; portfolio and position "
                "reads require it"
            )
            # Dump the raw payload so the envelope can be identified. The tool's
            # schema documents only its INPUT, so the response shape has to be
            # observed -- and guessing envelope names one release at a time is
            # slower than just looking.
            if debug_accounts:
                await _dump_accounts(conn.adapter)
            else:
                print("      re-run with --debug-accounts to dump the raw response")

        try:
            equity = await broker.get_account_equity()
            print(_fmt("equity", f"${equity:,.2f}"))
        except Exception as exc:
            print(_fmt("equity", f"FAILED: {exc}"))
            read_failures.append(f"equity read failed: {exc}")
            # Dump the numeric fields that DO exist so the right one can be
            # named, instead of expanding a guessed key list another round.
            if account:
                await _dump_portfolio(conn.adapter, account)

        try:
            positions = await broker.get_positions()
            print(_fmt("open positions", str(len(positions))))
            for sym, qty in sorted(positions.items())[:8]:
                print(f"      {sym:<8} {qty:>12,.4f}")
        except Exception as exc:
            print(_fmt("positions", f"FAILED: {exc}"))
            read_failures.append(f"position read failed: {exc}")
        print()

        if snapshot:
            written = write_snapshot(conn.tools)
            print(f"Snapshot written: docs/mcp/tools-snapshot.json "
                  f"({written['tool_count']} tools)")
            print("This is the baseline that makes schema drift detectable.\n")

        if missing:
            print(f"NOT READY: missing {', '.join(missing)}.")
            print("The agent cannot trade without these.\n")
            return 1

        # A resolved capability is not a working one. Reporting "Ready" while
        # equity and positions both failed to read is the worst possible outcome
        # of this command: it invites you to arm an account the agent cannot see.
        if read_failures:
            print("NOT READY: the tools resolved but the account could not be read.\n")
            for problem in read_failures:
                print(f"  - {problem}")
            print(
                "\nWithout equity, every risk limit is a fraction of an unknown "
                "number.\nWithout positions, reconciliation cannot verify what you "
                "own.\n"
            )
            return 1

        print("Ready. Osiris can read your account and place orders through the")
        print("review → kernel → place pipeline.\n")
        print("Next:")
        print("  python -m osiris.run --once      # one supervised cycle")
        print("  python -m osiris.run --serve     # dashboard + scheduled cycles\n")
        return 0
    finally:
        await conn.close()


async def run_status() -> int:
    storage = FileTokenStorage()
    settings = load_settings()
    print(f"\n  endpoint .......... {settings.mcp_endpoint}")
    print(f"  credentials ....... {'cached' if storage.has_tokens else 'none'}")
    print(f"  token file ........ {storage.path}")
    print(f"  mode .............. {settings.mode.value}")
    print(f"  live armed ........ {'yes' if settings.live_armed else 'no'}\n")
    return 0 if storage.has_tokens else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Connect Osiris to Robinhood.")
    parser.add_argument("--status", action="store_true", help="show connection state")
    parser.add_argument("--logout", action="store_true", help="forget credentials")
    parser.add_argument(
        "--no-snapshot",
        action="store_true",
        help="skip writing the MCP tool snapshot",
    )
    parser.add_argument(
        "--debug-accounts",
        action="store_true",
        help="dump the raw get_accounts response when resolution fails",
    )
    args = parser.parse_args(argv)

    configure_logging()

    if args.logout:
        FileTokenStorage().clear()
        print("Cached Robinhood credentials removed.")
        return 0
    if args.status:
        return asyncio.run(run_status())
    return asyncio.run(
        run_connect(
            snapshot=not args.no_snapshot, debug_accounts=args.debug_accounts
        )
    )


if __name__ == "__main__":
    sys.exit(main())
