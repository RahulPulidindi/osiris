# Osiris — Autonomous Trading Agent: Build Plan

## Reality check first

I researched the Robinhood Agentic docs, the MCP tool surface, and TradingView's
automation model before planning. Four findings contradict the brief and change
the architecture. Read this section before the design.

### 1. Half the requested strategies are not executable on this venue

The Robinhood MCP supports **long equities and long options orders only**. From
the official tool list: `place_equity_order`, `place_option_order`,
`review_*`, `cancel_*`. There is no short-sell tool, no futures, no crypto,
no event contracts. Robinhood's newsroom states agentic launched "with support
for equities only out of the gate. Support for options, crypto, event contracts,
futures, and more are coming soon."

Consequences for the strategy list in the brief:

| Requested | Status on Robinhood MCP |
|---|---|
| Long equities | Supported |
| Long options (single-leg) | Supported |
| Trend following | Supported (long-only) |
| Scalping | **Not viable** — see #2 |
| Short selling | **Not available** — no tool; also RH bans boxed positions |
| Multi-leg spreads | Unverified; Level 3 needs a margin account |
| Futures / crypto / event contracts | **Not available** via agentic MCP |

Bearish exposure is only expressible via long puts or inverse ETFs. That is a
real constraint on "any strategy, any asset class," not a detail.

### 2. Scalping is structurally impossible through this stack

Three independent reasons:

- **No streaming data.** The MCP is request/response JSON-RPC. `get_equity_quotes`
  and `get_equity_price_book` are polled. There is no WebSocket tick feed.
- **LLM latency.** An OpenRouter round trip is ~1–20s. Scalping edges live in
  milliseconds. By the time a model reasons about a quote, the quote is stale.
- **TradingView adds nondeterministic delay.** Webhooks fire once, no retry, no
  delivery guarantee, 3-second server timeout.

Osiris's viable horizon is **minutes to days**, not sub-second. I've designed for
that. If you genuinely want scalping, it needs a different venue (co-located,
direct market data) and no LLM in the hot path — a separate project.

### 3. The PDT rule you're probably designing around no longer exists

Effective **June 4, 2026**, FINRA amended Rule 4210: the pattern-day-trader
designation, the 4-trades-in-5-days trigger, and the $25,000 minimum are
**eliminated**, replaced by real-time intraday margin requirements. Robinhood
adopted this on June 4, 2026.

This is good news, but the replacement constraint is sharper: **intraday margin
deficits (IMD)** are monitored in real time and repeated failures cause account
restrictions. So the guardrail Osiris needs is not a day-trade counter — it's an
intraday exposure monitor.

Also: the Agentic account is a *self-directed individual* account. Whether it is
cash or margin decides everything about turnover. **In a cash account, T+1
settlement applies and reusing unsettled proceeds triggers good-faith
violations** — which throttle you to roughly one round trip per position per day.
Verify your account type before writing sizing logic. This is task 0.

### 4. TradingView is the wrong primary tool here

TradingView has no official public data API. Automation is Pine Script alerts →
webhooks, which are fire-and-forget with no retries. And you don't need it: the
MCP already ships `get_equity_technical_indicators` (RSI, MACD, Bollinger,
moving averages), `get_equity_historicals` (OHLCV), and a full scanner suite
(`create_scan`, `run_scan`, `get_scanner_filter_specs`).

Recommendation: **skip TradingView for v1.** Compute indicators locally from
OHLCV so they're testable and backtestable. Revisit only if you have specific
Pine strategies to port, in which case treat webhooks as low-priority hints
into a queue, never as direct order triggers.

## Where the real risk sits

The brief treats "full autonomy" as the goal and profit as the objective
function. The failure mode isn't a bad trade — it's an unbounded loss loop with
nobody watching. Three specific dangers:

**Prompt injection is a live attack path.** Osiris reads news, filings, and
possibly social sentiment. That is untrusted input flowing into a model that can
place orders. A crafted headline is a code path to your capital. Every piece of
retrieved text must be treated as adversarial data, never as instructions.

**"Maximize profit" with no risk term converges on ruin.** An unconstrained
optimizer facing a profit objective learns leverage and concentration, which
maximizes *expected* return while making bankruptcy near-certain over enough
draws. The objective must be risk-adjusted (Sharpe/Sortino, or return subject to
a drawdown ceiling). This is the single most important design decision.

**Robinhood explicitly disclaims all of it.** "Robinhood does not control,
supervise, monitor, recommend, or audit these AI agents... You assume all risk."
There is no counterparty to appeal to. Guardrails are yours to build.

Autonomy is preserved throughout what follows — Osiris places trades without
per-trade approval. What's bounded is *blast radius*, via deterministic limits
the model cannot talk its way past.
