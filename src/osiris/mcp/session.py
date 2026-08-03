"""Live connection to the Robinhood Trading MCP, with OAuth 2.1 + PKCE.

This is the module that actually reaches your account. Three things it owns:

  1. **The OAuth dance.** Interactive on first run: the SDK opens a browser, you
     approve, and the callback lands on a loopback server. Tokens are then cached
     so subsequent runs are non-interactive.

  2. **Token storage on disk, outside the repo.** Under `~/.osiris/` rather than
     the working tree, because a token file inside a git repo eventually gets
     committed. Written with 0600 permissions.

  3. **A single long-lived session.** Connecting is expensive and rate limits are
     unpublished, so the session is opened once and held for the process
     lifetime rather than per call.

Note on SDK version: the transport entry point is `streamable_http_client` in
`mcp>=2.0`. It was `streamablehttp_client` in 1.x. Importing the old name raises
ImportError at connect time -- which is exactly the kind of break that only shows
up when you finally try to trade.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from osiris.logging import get_logger

log = get_logger(__name__)

# Deliberately outside the repository. A cached token inside the working tree is
# a credential one `git add -A` away from being published.
OSIRIS_HOME = Path(os.environ.get("OSIRIS_HOME", Path.home() / ".osiris"))
TOKEN_PATH = OSIRIS_HOME / "mcp-auth.json"

CLIENT_NAME = "Osiris"
# Loopback redirect: the SDK spins up a local server to catch the callback.
CALLBACK_PORT = 33477
REDIRECT_URI = f"http://127.0.0.1:{CALLBACK_PORT}/callback"


class NotConnected(RuntimeError):
    """Raised when live operations are attempted without a session."""


@dataclass
class FileTokenStorage:
    """Persist OAuth tokens and client registration to disk.

    Implements the SDK's `TokenStorage` protocol. Kept as plain JSON so a human
    can inspect or delete it; deleting the file is how you force re-auth.
    """

    path: Path = TOKEN_PATH

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("mcp.token_store_unreadable", error=str(exc))
            return {}

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))
        # 0600: tokens are account credentials. A world-readable token on a
        # shared machine is equivalent to a leaked password.
        os.chmod(self.path, 0o600)

    async def get_tokens(self):
        raw = self._read().get("tokens")
        if not raw:
            return None
        from mcp.shared.auth import OAuthToken

        try:
            return OAuthToken.model_validate(raw)
        except Exception as exc:
            log.warning("mcp.stored_token_invalid", error=str(exc))
            return None

    async def set_tokens(self, tokens) -> None:
        data = self._read()
        data["tokens"] = tokens.model_dump(mode="json")
        self._write(data)

    async def get_client_info(self):
        raw = self._read().get("client_info")
        if not raw:
            return None
        from mcp.shared.auth import OAuthClientInformationFull

        try:
            return OAuthClientInformationFull.model_validate(raw)
        except Exception as exc:
            log.warning("mcp.stored_client_info_invalid", error=str(exc))
            return None

    async def set_client_info(self, client_info) -> None:
        data = self._read()
        data["client_info"] = client_info.model_dump(mode="json")
        self._write(data)

    @property
    def has_tokens(self) -> bool:
        return bool(self._read().get("tokens"))

    def clear(self) -> None:
        if self.path.exists():
            self.path.unlink()


async def _open_browser(url: str) -> None:
    """Send the user to the authorization page."""
    import webbrowser

    print("\nOpening your browser to authorize Osiris with Robinhood.")
    print("If it does not open, paste this URL:\n")
    print(f"  {url}\n")
    webbrowser.open(url)


async def _await_callback():
    """Serve the loopback redirect once and return the authorization code.

    A tiny one-shot HTTP server rather than asking the user to paste a code back
    into the terminal: the code arrives in a query string, and copy-paste of a
    long opaque token is where this flow usually goes wrong for people.
    """
    import asyncio
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse

    from mcp.shared.auth import AuthorizationCodeResult

    captured: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        # Name is fixed by BaseHTTPRequestHandler's dispatch.
        def do_GET(self) -> None:
            query = parse_qs(urlparse(self.path).query)
            captured.update({k: v[0] for k, v in query.items() if v})
            body = (
                b"<html><body style='font-family:system-ui;background:#0a0a0b;"
                b"color:#e8e6e1;padding:3rem'>"
                b"<h2>Osiris is connected.</h2>"
                b"<p>You can close this tab and return to the terminal.</p>"
                b"</body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            """Silence the default stderr access log."""

    def serve_once() -> None:
        server = HTTPServer(("127.0.0.1", CALLBACK_PORT), Handler)
        server.timeout = 300
        server.handle_request()
        server.server_close()

    # Off the event loop: `handle_request` blocks, and blocking here would stall
    # the very transport that needs to complete the token exchange.
    await asyncio.to_thread(serve_once)

    if "error" in captured:
        raise RuntimeError(
            f"Authorization failed: {captured.get('error')} "
            f"{captured.get('error_description', '')}".strip()
        )
    if "code" not in captured:
        raise RuntimeError("Authorization callback carried no code.")
    return AuthorizationCodeResult(
        code=captured["code"],
        state=captured.get("state"),
    )


def _client_metadata():
    from mcp.shared.auth import OAuthClientMetadata

    return OAuthClientMetadata(
        client_name=CLIENT_NAME,
        redirect_uris=[REDIRECT_URI],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="none",  # public client; PKCE is the proof
    )


@dataclass
class LiveConnection:
    """An open MCP session plus the resolved capability adapter.

    Held open for the process lifetime. `AsyncExitStack` rather than nested `async
    with` blocks so the connection can outlive the function that created it --
    the alternative is threading the whole application through a context manager.
    """

    endpoint: str
    dry_run: bool = True
    storage: FileTokenStorage = field(default_factory=FileTokenStorage)

    session: Any = None
    adapter: Any = None
    tools: list = field(default_factory=list)
    _stack: Any = None

    @property
    def connected(self) -> bool:
        return self.session is not None

    async def open(self, *, interactive: bool = True):
        """Connect, authenticate, enumerate the surface, build the adapter."""
        from contextlib import AsyncExitStack

        from mcp import ClientSession
        from mcp.client.auth import OAuthClientProvider

        # SDK v2 renamed this from `streamablehttp_client`. Importing here rather
        # than at module scope keeps `osiris.mcp.session` importable (and the rest
        # of the app testable) on a machine with a mismatched SDK.
        try:
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise RuntimeError(
                "Incompatible `mcp` SDK: no `streamable_http_client` in "
                "mcp.client.streamable_http. Install mcp>=2.0.0."
            ) from exc

        if not interactive and not self.storage.has_tokens:
            raise NotConnected(
                "No cached Robinhood credentials. Run `python -m osiris.connect` "
                "once to authorize; it needs a browser."
            )

        auth = OAuthClientProvider(
            server_url=self.endpoint,
            client_metadata=_client_metadata(),
            storage=self.storage,
            redirect_handler=_open_browser,
            callback_handler=_await_callback,
        )

        stack = AsyncExitStack()
        try:
            from mcp.client.streamable_http import httpx2

            http_client = await stack.enter_async_context(
                httpx2.AsyncClient(auth=auth, timeout=60.0)
            )
            streams = await stack.enter_async_context(
                streamable_http_client(self.endpoint, http_client=http_client)
            )
            read, write = streams[0], streams[1]
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise

        self._stack = stack
        self.session = session

        from osiris.mcp.client import MCPAdapter, enumerate_tools

        self.tools = await enumerate_tools(session)
        self.adapter = MCPAdapter(session, self.tools, dry_run=self.dry_run)

        log.info(
            "mcp.connected",
            endpoint=self.endpoint,
            tools=len(self.tools),
            dry_run=self.dry_run,
        )
        return self

    async def close(self) -> None:
        if self._stack is not None:
            # Robinhood rejects the MCP session-termination DELETE with a 400.
            # It happens after all work is done and the SDK prints it to stdout,
            # where it reads like a failure. Suppressed here rather than left to
            # alarm an operator over a teardown call that changes nothing.
            import contextlib
            import io

            with contextlib.suppress(Exception), contextlib.redirect_stdout(
                io.StringIO()
            ):
                await self._stack.aclose()
        self._stack = None
        self.session = None
        self.adapter = None
        log.info("mcp.disconnected")

    def require_capabilities(self, *names: str) -> list[str]:
        """Return the requested capabilities that did NOT resolve."""
        if self.adapter is None:
            raise NotConnected("not connected")
        return [n for n in names if not self.adapter.has(n)]


async def connect(
    endpoint: str, *, dry_run: bool = True, interactive: bool = True
) -> LiveConnection:
    """Open a live connection. Caller owns closing it."""
    return await LiveConnection(endpoint=endpoint, dry_run=dry_run).open(
        interactive=interactive
    )
