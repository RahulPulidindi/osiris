"""Exa research client. Research and filings, not a news feed.

The decisive capability is `startPublishedDate` / `endPublishedDate`: it lets a
query be bounded to what was published before a given morning. That is what
makes an honest backtest possible. A backtest run against today's web index
silently leaks the future and flatters itself.

Cost discipline: /search is $7/1k requests, /contents is $1/1k pages. So search
once, then pull contents for known URLs -- 7x cheaper and 100 QPS vs 10 QPS.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

import httpx

from osiris.data.sanitize import make_provenanced
from osiris.eval.pit import assert_text_not_future
from osiris.logging import get_logger
from osiris.types import Provenanced, Trust

log = get_logger(__name__)

EXA_BASE = "https://api.exa.ai"

# Domains trusted enough to weight more heavily. Not a security boundary --
# sanitization still applies -- but a quality signal.
PREFERRED_DOMAINS = (
    "sec.gov", "reuters.com", "bloomberg.com", "wsj.com", "ft.com",
    "cnbc.com", "barrons.com", "marketwatch.com", "investors.com",
    "seekingalpha.com", "fool.com",
)


@dataclass
class ExaCostTracker:
    """Tracks spend so the funnel cannot silently run away.

    Naively searching 1000 names daily is ~$210/month in search alone; the
    funnel exists partly to keep this bounded.
    """

    search_requests: int = 0
    extra_results: int = 0
    content_pages: int = 0
    summaries: int = 0

    SEARCH_PER_1K: float = 7.0
    RESULT_PER_1K: float = 1.0
    CONTENT_PER_1K: float = 1.0
    SUMMARY_PER_1K: float = 1.0

    @property
    def usd(self) -> float:
        return (
            self.search_requests * self.SEARCH_PER_1K / 1000.0
            + self.extra_results * self.RESULT_PER_1K / 1000.0
            + self.content_pages * self.CONTENT_PER_1K / 1000.0
            + self.summaries * self.SUMMARY_PER_1K / 1000.0
        )

    def record_search(self, n_results: int, summaries: bool = False) -> None:
        self.search_requests += 1
        self.extra_results += max(0, n_results - 10)
        if summaries:
            self.summaries += n_results

    def record_contents(self, n_pages: int) -> None:
        self.content_pages += n_pages


class _RateLimited(Exception):
    """HTTP 429. Transient by definition, so it is retried rather than dropped."""

    def __init__(self, retry_after: str | None = None) -> None:
        self.retry_after: float | None = None
        if retry_after:
            try:
                self.retry_after = float(retry_after)
            except ValueError:
                self.retry_after = None
        super().__init__("rate limited")


class ResearchClient:
    """Exa wrapper with point-in-time bounding and cost tracking."""

    def __init__(
        self,
        api_key: str | None,
        *,
        cost_tracker: ExaCostTracker | None = None,
        daily_usd_ceiling: float = 5.0,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.costs = cost_tracker or ExaCostTracker()
        self.daily_usd_ceiling = daily_usd_ceiling
        self.timeout = timeout
        # 4, not 8. The documented ceiling is 10 QPS, but 8 in flight produced
        # sustained 429s in practice -- concurrency is not the same as request
        # rate, and each request can retry. Headroom is cheaper than lost research.
        self._semaphore = asyncio.Semaphore(4)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key or "", "Content-Type": "application/json"}

    def _check_budget(self) -> None:
        if self.costs.usd >= self.daily_usd_ceiling:
            raise RuntimeError(
                f"Exa daily budget exhausted: ${self.costs.usd:.2f} >= "
                f"${self.daily_usd_ceiling:.2f}"
            )

    async def search(
        self,
        query: str,
        *,
        as_of: date | None = None,
        lookback_days: int = 14,
        num_results: int = 6,
        category: str | None = "news",
        search_type: str = "fast",
        include_domains: list[str] | None = None,
    ) -> list[Provenanced]:
        """Search with a hard publication-date ceiling.

        `as_of` is the simulation date. Nothing published after it can be
        returned, which is what makes a backtest honest.
        """
        if not self.enabled:
            return []
        self._check_budget()

        payload: dict = {
            "query": query,
            "numResults": num_results,
            "type": search_type,
            "contents": {"text": {"maxCharacters": 2_000}, "highlights": True},
        }
        if category:
            payload["category"] = category
        if include_domains:
            payload["includeDomains"] = include_domains

        if as_of is not None:
            end = datetime.combine(as_of, datetime.min.time(), tzinfo=UTC)
            start = end - timedelta(days=lookback_days)
            payload["startPublishedDate"] = start.isoformat()
            payload["endPublishedDate"] = end.isoformat()

        # 429 is retried with backoff rather than dropped. Losing a search means
        # the strategist scores that name on price history alone, silently -- and a
        # rate limit is the most transient failure there is.
        data: dict | None = None
        delay = 1.0
        for attempt in range(1, 4):
            try:
                async with self._semaphore:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(
                            f"{EXA_BASE}/search", json=payload, headers=self._headers()
                        )
                        if resp.status_code == 429:
                            raise _RateLimited(
                                resp.headers.get("retry-after")
                            )
                        resp.raise_for_status()
                        data = resp.json()
                break
            except _RateLimited as exc:
                if attempt == 3:
                    log.warning("exa.rate_limited", query=query[:60])
                    return []
                wait = exc.retry_after or delay
                log.info("exa.backoff", attempt=attempt, wait_s=round(wait, 1))
                await asyncio.sleep(wait)
                delay *= 2
            except Exception as exc:
                log.warning("exa.search.failed", query=query[:60], error=str(exc))
                return []

        if data is None:
            return []

        self.costs.record_search(num_results)
        return self._to_documents(data.get("results", []), as_of=as_of)

    def _to_documents(
        self, results: list[dict], *, as_of: date | None
    ) -> list[Provenanced]:
        docs: list[Provenanced] = []
        for r in results:
            text = r.get("text") or " ".join(r.get("highlights") or [])
            if not text:
                continue
            published = _parse_date(r.get("publishedDate"))
            if as_of is not None and published is not None:
                try:
                    assert_text_not_future(published, as_of)
                except Exception:
                    # Exa's date filter should prevent this; if it slips through
                    # we drop the document rather than leak the future.
                    log.warning(
                        "exa.lookahead.dropped",
                        url=r.get("url"),
                        published=str(published),
                        as_of=as_of.isoformat(),
                    )
                    continue
            docs.append(
                make_provenanced(
                    content=f"{r.get('title', '')}\n{text}",
                    source=f"exa:{_domain(r.get('url', ''))}",
                    trust=Trust.UNTRUSTED_EXTERNAL,
                    url=r.get("url"),
                    published_at=published,
                )
            )
        return docs

    async def research_symbol(
        self, symbol: str, company: str = "", *, as_of: date | None = None,
        deep: bool = False,
    ) -> list[Provenanced]:
        """Stage 2 (cheap) or Stage 3 (deep) research for one name."""
        # Only include the company name when it ADDS information. With `company`
        # empty this previously produced "PEP PEP earnings results", where the
        # duplicated token skews relevance scoring against the terms that matter.
        subject = f"{company} ({symbol})" if company and company != symbol else symbol
        if deep:
            queries = [
                (f"{subject} earnings results guidance outlook", "financial report", "deep"),
                (f"{subject} analyst rating price target catalyst", "news", "fast"),
            ]
        else:
            queries = [(f"{subject} stock news", "news", "fast")]

        results = await asyncio.gather(
            *(
                self.search(
                    q,
                    as_of=as_of,
                    category=cat,
                    search_type=st,
                    num_results=6 if deep else 4,
                )
                for q, cat, st in queries
            ),
            return_exceptions=True,
        )
        docs: list[Provenanced] = []
        for r in results:
            if isinstance(r, list):
                docs.extend(r)
        return docs


def _domain(url: str) -> str:
    if not url:
        return "unknown"
    stripped = url.split("//")[-1]
    return stripped.split("/")[0].removeprefix("www.")


def _parse_date(raw) -> datetime | None:
    if not raw:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    try:
        text = str(raw).replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    except ValueError:
        return None
