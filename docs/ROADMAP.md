# Osiris — Build Roadmap

Ordered so that nothing can lose money until the machinery that prevents runaway
losses exists and has been proven. Each phase has an explicit exit gate.

## Phase 0 — Ground truth (before any code)

Facts to establish, because they invalidate design choices downstream:

1. **Is the Agentic account cash or margin?** Decides turnover, settlement
   rules, shorting, and options level. Everything in sizing depends on it.
2. **Enumerate the live MCP tool surface.** `initialize` → `tools/list`, drain
   pagination, snapshot to `docs/mcp/tools-snapshot.json`. Don't trust the docs
   table or this plan — read your own account's surface.
3. **Options level granted?** Determines single-leg vs. spreads.
4. **Confirm PDT removal is live on your account** (adopted June 4, 2026).
5. Note rate limits empirically; none are published.

**Gate:** snapshot committed, account type known in writing.

## Phase 1 — Skeleton and safety (no orders)

- Repo, typed config, secrets via env (never committed)
- MCP client: OAuth 2.1 + PKCE, token refresh, reconnect with backoff
- **Capability resolver** — match on schema shape, not tool-name strings
- Schema-drift check wired into CI
- Structured logging + append-only journal from day one
- **Risk Kernel with full unit tests** — written *before* any order path
- Kill switch: file-based + CLI, checked before every action

**Gate:** kernel test suite green, including every breaker. All reads work; no
write path exists yet.

## Phase 2 — Perception

- OHLCV ingest + local indicator library (own it; don't depend on TradingView)
- News/sentiment adapter (Finnhub free tier first)
- Macro calendar + blackout windows
- Earnings calendar integration
- Regime classifier (trend / chop / high-vol)
- Sanitization layer: all external text wrapped, provenance-tagged, never
  concatenated into instruction slots

**Gate:** one command prints a full, timestamped market picture with sources.

## Phase 3 — Cognition (paper only)

- OpenRouter client: per-role model routing, retries, cost ceiling per day
- Analyst / Strategist / Red Team / PM roles
- Strict `OrderIntent` schema; reject malformed output rather than repairing it
- Mandatory invalidation condition per proposal
- Prompt-injection test suite: hostile headlines in a fixture corpus, asserting
  the agent never emits an intent traceable to injected instructions

**Gate:** injection suite passes 100%. Intents produced but nothing sent.

## Phase 4 — Backtest and paper trade

- Event-driven backtester with realistic costs: slippage, spread, partial fills
- **Point-in-time data only** — no lookahead, no survivorship bias. This is where
  most retail systems fool themselves.
- Walk-forward validation, not a single in-sample curve
- Paper broker implementing the same interface as the MCP adapter
- Run paper for a **minimum of 4 weeks** across at least one earnings cycle

**Gate:** positive walk-forward expectancy *after* costs, and paper results that
don't diverge wildly from backtest. If backtest is great and paper is bad, the
backtest is wrong — fix it before proceeding.

**Status: met against the synthetic market.** 186 sessions over 259 calendar days
(> 4 weeks, spanning multiple earnings cycles): +19.8% vs +11.6% benchmark,
Sharpe 1.63, max DD 8.2%, zero reconciliation breaks, and a realistic veto
distribution. Paper-vs-backtest Sharpe gap is 0.35, well inside the 1.0 band, and
that comparison runs the *same* momentum ranking through both paths so any
divergence is attributable to execution mechanics rather than to a different
strategy. Re-run against live data before this counts for a funded account —
synthetic prices cannot falsify a strategy, only the machinery around it.

## Phase 5 — Live, minimum viable capital

- Fund the Agentic account with an amount you would shrug off losing entirely
- Enable one strategy, one symbol, smallest tradeable size
- **Every order still goes review → kernel → place**
- Daily reconciliation: MCP `get_realized_pnl` vs. internal ledger; any
  divergence halts trading
- Mobile alerts on fills and breaker trips

**Gate:** two weeks live with zero reconciliation breaks and zero kernel
bypasses. Not "profitable" — *correct*. Profitability at this size is noise.

**Status: machinery built, gate not passed.** `python -m osiris.runner.gate` runs
16 arming checks and exits non-zero unless every blocking one passes; alerting
ships in `osiris/runner/alerts.py`; the procedure is in `docs/RUNBOOK.md`.

Two blocking checks remain open **by construction**, and both require a real
account rather than more code:

- `account_type_known` — cash vs margin is Phase 0 output (`docs/PHASE0.md`).
- `mcp_snapshot_present` — the tool surface must be enumerated from your own
  account. Without that baseline, schema drift is undetectable.

That is the correct state for an unfunded repo. A gate that cleared here would be
lying.

## Phase 6 — Scale deliberately

Widen only after each step holds: universe → strategies → size → options
(defined-risk only, after equities are stable). One variable at a time, so
attribution stays interpretable.

## Phase 7 — Continuous operation

- Nightly post-mortem → lessons memory
- Weekly attribution review; auto-bench decaying strategies
- Monthly: re-verify MCP surface, rotate credentials, restore-from-backup drill
- Watchdog on the agent itself (dead-man's switch → flatten or halt)

## Cost estimate (monthly)

| Item | Cost |
|---|---|
| OpenRouter | $50–300 (dominated by loop frequency; cache aggressively) |
| News API | $0 (Finnhub free) → $50 |
| Market data extras | $0–30 |
| VPS | $10–20 |
| TradingView | $0 (recommend skipping) |

Biggest lever: **loop frequency**. A 5-minute cycle costs ~12x a 1-hour cycle
and, at this stack's latency, is unlikely to be 12x better.

## Tech stack

Python 3.12 · official `mcp` SDK · Pydantic (strict schemas) · Postgres or
SQLite+WAL · APScheduler · structlog · pytest · Docker.

Python for the ecosystem (pandas, vectorbt, empyrical). Pydantic everywhere on
LLM boundaries — parse, don't validate.

## Legal and tax

Not legal advice; confirm with a professional:
- Trading your own money in your own account: no registration needed. Managing
  anyone else's or taking a fee changes that immediately (adviser/broker rules).
- High turnover generates **wash sales** and heavy short-term gains. `get_equity_tax_lots`
  exists — use it for tax-aware exits.
- Keep the journal; it is your audit trail and your tax substantiation.

## Honest expectations

Most retail day trading loses money — a *Journal of Finance* study of the most-
bought Robinhood stocks (2018–2020) found average 20-day returns of **-4.7%**.
An LLM does not repeal that. The edge here, if any, is in breadth, discipline,
and never deviating from risk rules — not in the model's cleverness.

Judge Osiris on risk-adjusted return vs. buy-and-hold SPY over 6+ months. That
is the honest benchmark, and the only comparison that separates skill from beta.
