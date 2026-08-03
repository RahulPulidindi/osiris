"""Build a MarketSnapshot from the live Robinhood MCP.

The replacement for the synthetic market. Same output type, so the loop, kernel,
executor, and ledger are unchanged -- which is the point of having had a snapshot
abstraction at all.

Two rules govern everything here:

**Fail loudly, never substitute.** If quotes are missing the symbol is dropped
from the universe rather than defaulting a price. A fabricated price flows
straight into position sizing, and the resulting order is wrong in a way nothing
downstream can detect.

**Tolerate schema variation.** Field names differ across MCP tool versions
(`last_trade_price` vs `price` vs `last`). Readers try several keys and return
None when none match, so a rename degrades to "no data for this symbol" instead of
a crash mid-cycle.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np

from osiris.data.macro import session_date
from osiris.execution.loop import MarketSnapshot
from osiris.execution.mcp_broker import _extract_payload, _first_number
from osiris.logging import get_logger
from osiris.types import Quote

log = get_logger(__name__)

# Sector labels vary by data source; normalize toward the benchmark's vocabulary
# so the sector-deviation gate compares like with like.
SECTOR_ALIASES = {
    "information technology": "Technology",
    "tech": "Technology",
    "consumer discretionary": "Discretionary",
    "consumer cyclical": "Discretionary",
    "consumer staples": "Staples",
    "consumer defensive": "Staples",
    "health care": "Healthcare",
    "financial services": "Financials",
    "financial": "Financials",
    "basic materials": "Materials",
    "communication services": "Technology",
    "real estate": "Real Estate",
    "utilities": "Utilities",
    "energy": "Energy",
    "industrials": "Industrials",
}

# S&P 500 sector weights, used as the neutrality reference for the deviation gate.
BENCHMARK_SECTOR_WEIGHTS = {
    "Technology": 0.31,
    "Financials": 0.13,
    "Healthcare": 0.12,
    "Discretionary": 0.11,
    "Industrials": 0.09,
    "Staples": 0.06,
    "Energy": 0.04,
    "Utilities": 0.03,
    "Materials": 0.02,
    "Real Estate": 0.02,
}


def normalize_sector(raw: str | None) -> str:
    if not raw:
        return "Unknown"
    return SECTOR_ALIASES.get(raw.strip().lower(), raw.strip().title())


def _rows(payload: Any, *keys: str) -> list[dict]:
    """Pull a list of records out of a payload whose envelope key varies."""
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if not isinstance(payload, dict):
        return []
    for key in (*keys, "results", "data", "items"):
        val = payload.get(key)
        if isinstance(val, list):
            return [r for r in val if isinstance(r, dict)]
    # A single record returned bare.
    return [payload] if payload else []


def _batched(items: list[str], size: int) -> list[list[str]]:
    """Split into chunks the venue will accept.

    Batch ceilings are per-tool and enforced server-side: historicals and
    fundamentals cap at 10 symbols, and quotes above 20 silently drop the `closes`
    field. Exceeding them does not error usefully -- it returns partial data.
    """
    return [items[i : i + size] for i in range(0, len(items), size)]


def _symbol_of(row: dict) -> str:
    for key in ("symbol", "ticker", "instrument_symbol", "name"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().upper()
    return ""


MIN_HISTORY_BARS = 30

# Any of these on a record means it is a quote rather than an envelope.
_QUOTE_FIELDS = (
    "last_trade_price",
    "last_price",
    "bid_price",
    "ask_price",
    "mark_price",
    "price",
    "bid",
    "ask",
    "last",
    "previous_close",
    "close_price",
)


def _quote_rows(payload: Any, *, depth: int = 0) -> list[dict]:
    """Collect quote records from a payload of unknown shape.

    `_rows` only checked a fixed set of top-level envelope keys, which silently
    returned nothing when the venue nested quotes under `data` -- and the caller
    could not distinguish that from a market with no quotes.

    A record counts as a quote when it names a symbol AND carries at least one
    price field. Requiring both avoids collecting envelope objects that happen to
    mention a symbol.
    """
    if depth > 5:
        return []

    if isinstance(payload, dict):
        if _symbol_of(payload) and any(k in payload for k in _QUOTE_FIELDS):
            return [payload]

        out: list[dict] = []
        for key in ("quotes", "data", "results", *payload):
            child = payload.get(key)
            if not isinstance(child, dict | list):
                continue
            # A symbol-keyed mapping, e.g. {"AAPL": {...prices...}}. The record
            # carries no symbol of its own, so the KEY supplies it -- otherwise the
            # quote is parsed and then discarded for lacking a symbol.
            if (
                isinstance(child, dict)
                and not _symbol_of(child)
                and any(k in child for k in _QUOTE_FIELDS)
            ):
                out.append({**child, "symbol": key.upper()})
                continue
            out.extend(_quote_rows(child, depth=depth + 1))
            if out:
                return out
        return out

    if isinstance(payload, list):
        out = []
        for item in payload:
            out.extend(_quote_rows(item, depth=depth + 1))
        return out
    return []


def _shape_hint(payload: Any, *, limit: int = 12) -> str:
    """Compact structural summary for a diagnostic log line."""
    if isinstance(payload, dict):
        parts = []
        for key, value in list(payload.items())[:limit]:
            if isinstance(value, dict):
                parts.append(f"{key}{{{','.join(list(value)[:6])}}}")
            elif isinstance(value, list):
                inner = (
                    ",".join(list(value[0])[:6])
                    if value and isinstance(value[0], dict)
                    else ""
                )
                parts.append(f"{key}[{len(value)}]{{{inner}}}")
            else:
                parts.append(key)
        return " ".join(parts)
    if isinstance(payload, list):
        return f"list[{len(payload)}]"
    return type(payload).__name__


def _parse_history(payload: Any, requested: list[str]) -> dict[str, np.ndarray]:
    """Extract per-symbol close series from a multi-symbol historicals response.

    The grouping shape is not documented by the input schema, so both plausible
    forms are handled: records keyed by symbol, and a flat list of bars each
    carrying its own symbol. Anything with too few bars is dropped rather than
    padded -- a short series produces a momentum number that is pure noise, and
    the ranker cannot tell the difference.
    """
    out: dict[str, list[float]] = {}

    def close_of(row: dict) -> float | None:
        value = _first_number(
            row, "close_price", "close", "adjusted_close", "price", "last_trade_price"
        )
        return value if value and value > 0 else None

    def absorb(symbol: str, rows: list[dict]) -> None:
        closes = [c for c in (close_of(r) for r in rows) if c is not None]
        if closes:
            out.setdefault(symbol.upper(), []).extend(closes)

    def walk(node: Any, symbol_hint: str = "", depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(node, dict):
            # A record that names its symbol and carries its own bars.
            symbol = _symbol_of(node) or symbol_hint
            for key in ("historicals", "bars", "candles", "data", "results"):
                child = node.get(key)
                if isinstance(child, list) and child and isinstance(child[0], dict):
                    if close_of(child[0]) is not None and symbol:
                        absorb(symbol, child)
                    else:
                        walk(child, symbol, depth + 1)
                    return
            # A bare bar.
            if symbol and close_of(node) is not None:
                absorb(symbol, [node])
                return
            for key, child in node.items():
                # Keys may BE symbols, e.g. {"AAPL": [...]}.
                hint = key.upper() if key.upper() in {s.upper() for s in requested} else symbol
                walk(child, hint, depth + 1)
        elif isinstance(node, list):
            for item in node:
                walk(item, symbol_hint, depth + 1)

    walk(payload)
    return {
        symbol: np.asarray(closes, dtype=float)
        for symbol, closes in out.items()
        if len(closes) >= MIN_HISTORY_BARS
    }


def _parse_dollar_volume(payload: Any, requested: list[str]) -> dict[str, float]:
    """Average daily dollar volume per symbol, from REAL reported volume only.

    Symbols whose bars carry no volume are omitted entirely rather than estimated.
    The ADV participation gate then abstains for them, which is safe; feeding it a
    fabricated denominator would make it approve exactly the oversized orders it
    exists to block.
    """
    per_symbol: dict[str, list[float]] = {}

    def walk(node: Any, symbol_hint: str = "", depth: int = 0) -> None:
        if depth > 5:
            return
        if isinstance(node, dict):
            symbol = _symbol_of(node) or symbol_hint
            volume = _first_number(node, "volume", "total_volume")
            close = _first_number(node, "close_price", "close", "price")
            if symbol and volume and close and volume > 0 and close > 0:
                per_symbol.setdefault(symbol.upper(), []).append(volume * close)
                return
            for key, child in node.items():
                hint = (
                    key.upper()
                    if key.upper() in {s.upper() for s in requested}
                    else symbol
                )
                walk(child, hint, depth + 1)
        elif isinstance(node, list):
            for item in node:
                walk(item, symbol_hint, depth + 1)

    walk(payload)
    return {
        symbol: float(np.mean(values))
        for symbol, values in per_symbol.items()
        if len(values) >= 5
    }


class LiveMarket:
    """Reads the live market through the MCP adapter.

    Every method degrades rather than raising: a partial snapshot still lets the
    cycle run on the symbols that did resolve, whereas an exception aborts the
    session and leaves existing positions unmanaged.
    """

    def __init__(self, adapter: Any, *, max_concurrency: int = 8) -> None:
        self.adapter = adapter
        self._sem = asyncio.Semaphore(max_concurrency)
        # Sectors and betas change slowly; caching avoids re-fetching daily.
        self._sector_cache: dict[str, str] = {}

    async def _call(self, capability: str, args: dict | None = None) -> Any:
        async with self._sem:
            return await self.adapter.call(capability, args or {})

    # ------------------------------------------------------------------ quotes
    async def quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Live bid/ask/last. Symbols without a usable quote are omitted.

        Omission is deliberate: the kernel's spread and staleness gates need a
        real quote, and a synthesized one would silently pass gates designed to
        catch exactly that.
        """
        if not symbols:
            return {}
        out: dict[str, Quote] = {}

        # 20 per call: above that the venue omits `closes` rather than failing,
        # which would look like missing data instead of a batch-size problem.
        async def batch(chunk: list[str]) -> list[dict]:
            try:
                result = await self._call("getQuotes", {"symbols": chunk})
            except Exception as exc:
                log.warning(
                    "live.quotes_batch_failed", count=len(chunk), error=str(exc)
                )
                return []
            payload = _extract_payload(result)
            rows = _quote_rows(payload)
            if not rows:
                # The call succeeded but nothing parsed, which means the envelope
                # is not what this code expects. Report the structure rather than
                # counting it as "missing quotes" -- a parse failure and a genuine
                # absence of market data look identical from the caller's side.
                log.error(
                    "live.quotes_unparseable",
                    count=len(chunk),
                    shape=_shape_hint(payload),
                )
            return rows

        results = await asyncio.gather(*(batch(c) for c in _batched(symbols, 20)))
        rows = [row for group in results for row in group]

        now = datetime.now(UTC)
        for row in rows:
            symbol = _symbol_of(row)
            if not symbol:
                continue
            bid = _first_number(row, "bid_price", "bid", "bid_inclusive_of_sell_spread")
            ask = _first_number(row, "ask_price", "ask", "ask_inclusive_of_buy_spread")
            last = _first_number(
                row, "last_trade_price", "last_price", "price", "last", "mark_price"
            )
            # Reconstruct what is missing ONLY from other real observations.
            if last is None and bid and ask:
                last = (bid + ask) / 2.0
            if last is None or last <= 0:
                continue
            if not bid or bid <= 0:
                bid = last
            if not ask or ask <= 0:
                ask = last
            if ask < bid:
                bid, ask = ask, bid
            out[symbol] = Quote(symbol=symbol, bid=bid, ask=ask, last=last, ts=now)

        missing = set(symbols) - set(out)
        if missing:
            log.info("live.quotes_missing", count=len(missing),
                     sample=sorted(missing)[:6])
        return out

    # -------------------------------------------------------------- price history
    async def history(
        self, symbols: list[str], *, days: int = 400, interval: str = "day"
    ) -> dict[str, np.ndarray]:
        """Daily closes per symbol, oldest first.

        Needed for momentum, volatility, and regime. The tool takes a `symbols`
        ARRAY (max 10) and REQUIRES an RFC3339 `start_time` -- there is no `span`
        parameter. `adjustment_type: split` is requested explicitly because
        unadjusted prices contain synthetic gaps at every split, which a momentum
        or volatility calculation reads as real moves.
        """
        if not symbols:
            return {}

        start = (datetime.now(UTC) - timedelta(days=days)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        async def batch(chunk: list[str]) -> dict[str, np.ndarray]:
            try:
                result = await self._call(
                    "getHistoricals",
                    {
                        "symbols": chunk,
                        "start_time": start,
                        "interval": interval,
                        "adjustment_type": "split",
                    },
                )
            except Exception as exc:
                log.warning(
                    "live.history_batch_failed", count=len(chunk), error=str(exc)
                )
                return {}
            return _parse_history(_extract_payload(result), chunk)

        groups = await asyncio.gather(*(batch(c) for c in _batched(symbols, 10)))
        out: dict[str, np.ndarray] = {}
        for group in groups:
            out.update(group)
        log.info("live.history_loaded", requested=len(symbols), resolved=len(out))
        return out

    # ------------------------------------------------------------------ sectors
    async def sectors(self, symbols: list[str]) -> dict[str, str]:
        """Sector per symbol, from fundamentals. Cached across cycles."""
        unknown = [s for s in symbols if s not in self._sector_cache]

        async def batch(chunk: list[str]) -> None:
            try:
                # `symbols` array, max 10 per call.
                result = await self._call("getFundamentals", {"symbols": chunk})
            except Exception as exc:
                log.debug("live.fundamentals_failed", count=len(chunk), error=str(exc))
                return
            for row in _rows(_extract_payload(result), "fundamentals"):
                symbol = _symbol_of(row)
                if not symbol:
                    continue
                raw = row.get("sector") or row.get("industry") or row.get("category")
                self._sector_cache[symbol] = normalize_sector(
                    raw if isinstance(raw, str) else None
                )

        if unknown:
            await asyncio.gather(*(batch(c) for c in _batched(unknown, 10)))
        return {s: self._sector_cache.get(s, "Unknown") for s in symbols}

    # ---------------------------------------------------------------------- ADV
    async def _advs(self, symbols: list[str]) -> dict[str, float]:
        """Average daily DOLLAR volume, from real reported volume only.

        Returns only symbols where volume was actually observed. The ADV gate
        skips symbols it has no data for, which is correct: abstaining is safe,
        while approving against a fabricated denominator is not.
        """
        if not symbols:
            return {}

        start = (datetime.now(UTC) - timedelta(days=45)).strftime("%Y-%m-%dT%H:%M:%SZ")

        async def batch(chunk: list[str]) -> dict[str, float]:
            try:
                result = await self._call(
                    "getHistoricals",
                    {"symbols": chunk, "start_time": start, "interval": "day"},
                )
            except Exception as exc:
                log.debug("live.adv_batch_failed", count=len(chunk), error=str(exc))
                return {}
            return _parse_dollar_volume(_extract_payload(result), chunk)

        groups = await asyncio.gather(*(batch(c) for c in _batched(symbols, 10)))
        out: dict[str, float] = {}
        for group in groups:
            out.update(group)
        log.info("live.adv_resolved", requested=len(symbols), resolved=len(out))
        return out

    # ----------------------------------------------------------------- universe
    async def universe(self, *, fallback: list[str], limit: int = 120) -> list[str]:
        """Candidate symbols, preferring the broker's own scanner.

        Falls back to a static liquid list when no scanner capability resolves.
        The fallback is explicit rather than silent: trading a hardcoded list
        while believing you are screening the market is a meaningful difference.
        """
        # `run_scan` REQUIRES a scan_id, which comes from `get_scans`. Calling it
        # bare fails validation, so the id has to be resolved first.
        if self.adapter.has("runScan") and self.adapter.has("listScans"):
            try:
                scans = _rows(
                    _extract_payload(await self._call("listScans", {})), "scans"
                )
                scan_id = next(
                    (
                        str(row[key])
                        for row in scans
                        for key in ("scan_id", "id", "scanId")
                        if row.get(key)
                    ),
                    None,
                )
                if scan_id:
                    result = await self._call("runScan", {"scan_id": scan_id})
                    rows = _rows(_extract_payload(result), "results", "symbols")
                    found = [s for s in (_symbol_of(r) for r in rows) if s]
                    if len(found) >= 20:
                        log.info("live.universe_from_scanner", count=len(found))
                        return found[:limit]
                    log.warning("live.scanner_too_few", count=len(found))
                else:
                    log.info("live.no_saved_scans", detail="using fallback universe")
            except Exception as exc:
                log.warning("live.scanner_failed", error=str(exc))

        log.info("live.universe_fallback", count=min(len(fallback), limit))
        return fallback[:limit]

    # ----------------------------------------------------------------- snapshot
    async def snapshot(
        self,
        *,
        universe: list[str],
        as_of: date | None = None,
        benchmark: str = "SPY",
        held: list[str] | None = None,
    ) -> MarketSnapshot:
        """Assemble everything one cycle needs from live data.

        Held names are force-included in the universe. Dropping a held symbol
        because it failed a screen would make it unrankable and therefore
        un-exitable -- the position would be stranded.
        """
        symbols = list(dict.fromkeys([*(held or []), *universe]))

        quotes, closes, bench_hist = await asyncio.gather(
            self.quotes([*symbols, benchmark]),
            self.history(symbols),
            self.history([benchmark]),
        )

        # Tradable = we have BOTH a quote and enough history to rank it.
        tradable_symbols = [s for s in symbols if s in quotes and s in closes]
        if not tradable_symbols:
            log.error("live.snapshot_empty", requested=len(symbols))

        sectors = await self.sectors(tradable_symbols)
        benchmark_closes = bench_hist.get(benchmark)
        betas = _estimate_betas(closes, benchmark_closes)

        # ADV is deliberately left EMPTY unless the venue reported real volume.
        # A price-derived proxy is not average dollar volume, and feeding the ADV
        # participation gate an invented denominator would make it approve orders
        # it exists to block. Absent data means that gate abstains, which is the
        # honest failure mode.
        adv = await self._advs(tradable_symbols)

        metrics = {}
        for symbol in tradable_symbols:
            series = closes[symbol]
            if series.size > 126:
                metrics[symbol] = {
                    "momentum": round(float(series[-1] / series[-126] - 1.0), 4),
                    "price": round(float(series[-1]), 2),
                }

        return MarketSnapshot(
            as_of=as_of or session_date(),
            universe=tradable_symbols,
            closes={s: closes[s] for s in tradable_symbols},
            quotes={s: quotes[s] for s in tradable_symbols},
            benchmark_closes=benchmark_closes,
            adv=adv,
            sectors=sectors,
            betas=betas,
            benchmark_sector_weights=dict(BENCHMARK_SECTOR_WEIGHTS),
            tradable=dict.fromkeys(tradable_symbols, True),
            metrics=metrics,
        )


def _estimate_betas(
    closes: dict[str, np.ndarray], benchmark_closes: np.ndarray | None
) -> dict[str, float]:
    """OLS beta against the benchmark over the overlapping window.

    Defaults to 1.0 when it cannot be computed. That is the neutral assumption:
    guessing a low beta would let the beta budget be breached silently, which is
    the direction that costs money.
    """
    if benchmark_closes is None or benchmark_closes.size < 40:
        return dict.fromkeys(closes, 1.0)

    bench_returns = np.diff(benchmark_closes) / benchmark_closes[:-1]
    out: dict[str, float] = {}
    for symbol, series in closes.items():
        if series.size < 40:
            out[symbol] = 1.0
            continue
        returns = np.diff(series) / series[:-1]
        n = min(returns.size, bench_returns.size)
        if n < 30:
            out[symbol] = 1.0
            continue
        x, y = bench_returns[-n:], returns[-n:]
        var = float(np.var(x))
        if var <= 1e-12:
            out[symbol] = 1.0
            continue
        beta = float(np.cov(y, x)[0, 1] / var)
        # Clamp: an estimate outside this range on daily data is noise or a
        # corporate action, not a real exposure.
        out[symbol] = float(np.clip(beta, 0.0, 3.0))
    return out
