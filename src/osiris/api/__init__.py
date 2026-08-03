"""HTTP API and SSE stream for the dashboard."""

from osiris.api.events import BUS, Channel, Event, EventBus
from osiris.api.state import RuntimeState, build_runtime_state


def create_app():
    """Assemble the app with all routers attached.

    A factory rather than import-time wiring so tests can build an isolated app
    with injected state.
    """
    from osiris.api.app import app
    from osiris.api.routes_connection import router as connection_router
    from osiris.api.routes_portfolio import router as portfolio_router
    from osiris.api.routes_research import router as research_router

    if not getattr(app, "_osiris_routers_attached", False):
        app.include_router(portfolio_router)
        app.include_router(research_router)
        app.include_router(connection_router)
        _mount_dashboard(app)
        app._osiris_routers_attached = True
    return app


def _mount_dashboard(app) -> None:
    """Serve the built dashboard from the same process, if it exists.

    One process IS the product in production: `python -m osiris.run --serve`
    holds the broker connection, runs the schedule, answers the API, and serves
    the UI. The Vite dev server remains for development only.

    Mounted after the API routers so `/api/*` always wins, and skipped silently
    when `web/dist` is absent -- an API-only deployment is still valid.
    """
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles

    # src/osiris/api/__init__.py -> src/osiris -> src -> repo root
    dist = (
        Path(__file__).resolve().parent.parent.parent.parent / "web" / "dist"
    )
    if (dist / "index.html").exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="dashboard")


__all__ = [
    "BUS",
    "Channel",
    "Event",
    "EventBus",
    "RuntimeState",
    "build_runtime_state",
    "create_app",
]
