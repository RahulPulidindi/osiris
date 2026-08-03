# Osiris — Architecture

## Core principle: the model proposes, the kernel disposes

The single most important structural decision is that **the LLM never calls the
broker directly.** Every order passes through a deterministic Risk Kernel that
can veto. The model's output is a *proposal*; the kernel is the only thing
holding a broker connection.

This is what makes autonomy safe rather than reckless. The agent still trades
with zero human approval — but it cannot exceed limits it doesn't control,
cannot be argued out of them by injected text, and cannot bypass them through a
clever tool call. A prompt injection that fully captures the model still cannot
place an order that violates policy.

```
                    ┌──────────────────────────────────┐
                    │        Data Plane (untrusted)     │
                    │  MCP market data · news · filings │
                    │  macro calendar · OHLCV · L2 book │
                    └────────────────┬─────────────────┘
                                     │ sanitized, provenance-tagged
                                     ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │                     Cognition Plane (LLM)                        │
   │  Analyst → Strategist → Red Team → Portfolio Manager             │
   │  (OpenRouter; model tiered per role)                             │
   │  Output: OrderIntent JSON + thesis + invalidation condition       │
   └────────────────────────────┬────────────────────────────────────┘
                                │ OrderIntent (never a broker call)
                                ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │           RISK KERNEL — deterministic code, no LLM                │
   │  position sizing · exposure caps · drawdown halt · kill switch     │
   │  correlation limits · settlement/IMD guard · idempotency           │
   │  VETO or PASS. Not persuadable. Not promptable.                    │
   └────────────────────────────┬────────────────────────────────────┘
                                │ approved orders only
                                ▼
   ┌─────────────────────────────────────────────────────────────────┐
   │      Execution Plane · review_* → place_* → reconcile             │
   │      Robinhood MCP · capability-resolved, schema-validated         │
   └────────────────────────────┬────────────────────────────────────┘
                                ▼
        Journal (append-only) → Attribution → Nightly review → memory
```

## Components

### 1. Data plane

Osiris is only as good as what it sees. The MCP covers more than expected —
use it as the spine and fill gaps deliberately.

**From the MCP (free, already authenticated):**
- `get_equity_historicals` — OHLCV bars
- `get_equity_technical_indicators` — RSI, MACD, Bollinger, MAs
- `get_equity_price_book` — real-time L2, up to 4 symbols
- `get_equity_quotes` — real-time, up to 20 symbols
- `get_equity_fundamentals`, `get_financials`
- `get_earnings_calendar` / `get_earnings_results`
- `run_scan` / `create_scan` — server-side universe screening
- `get_realized_pnl`, `get_pnl_trade_history` — ground-truth P&L
- `get_equity_tax_lots` — lot-level basis, needed for tax-aware exits

**Gaps the MCP does not fill — these are the "what am I missing" answer:**

| Gap | Why it matters | Fill |
|---|---|---|
| News + sentiment | Biggest driver of short-horizon moves | Finnhub free (60/min) to prototype; Alpha Vantage or Polygon/Massive for scale |
| Macro calendar | CPI/FOMC/NFP dominate everything else that day | FRED (free) + econ calendar; hard-block trading in blackout windows |
| Options analytics | MCP gives quotes, not IV rank/surface/greeks context | Compute locally from chains; ORATS if you need term structure |
| Options flow | Unusual activity as a signal | Unusual Whales / Polygon options (optional, paid) |
| Corporate actions | Splits silently corrupt OHLCV and cost basis | Polygon or Nasdaq Data Link |
| Halt / LULD status | Trading into a halt is a real failure mode | Nasdaq trader feed; also check `get_equity_tradability` |
| Borrow / short interest | Squeeze risk on small caps | Finnhub / paid provider |
| Backtest engine | Cannot validate a strategy without it | vectorbt or a custom event-driven loop |
| Regime classifier | Same strategy that wins in trend loses in chop | Local model: VIX + realized vol + breadth |

Everything entering the cognition plane is wrapped:

```json
{
  "source": "finnhub_news",
  "fetched_at": "2026-07-31T20:15:00Z",
  "trust": "untrusted_external",
  "content": "<verbatim text, never interpolated into an instruction slot>"
}
```

### 2. Cognition plane

Multi-role rather than one monolithic prompt, because the roles have genuinely
different failure modes and deserve different models.

| Role | Job | Model tier |
|---|---|---|
| **Analyst** | Summarize evidence per symbol. No opinions, no orders. | Cheap/fast |
| **Strategist** | Propose thesis + entry/exit/invalidation | Strong reasoning |
| **Red Team** | Argue the bear case; find the flaw. Can veto. | Strong, *different family* |
| **Portfolio Manager** | Fit proposals to existing book; dedupe correlation | Strong reasoning |
| **Post-mortem** | Nightly: why did trades win/lose? Write to memory | Strong, offline |

The Red Team role is not decoration. A single model asked to find trades will
find trades — confirmation bias is the default. Using a *different model family*
for the bear case (e.g. Claude proposes, GPT critiques) decorrelates errors. If
Red Team can't be satisfied, the trade doesn't happen.

Every proposal must include a **falsifiable invalidation condition** ("thesis is
wrong if price closes below 200MA" / "if IV rank drops under 30"). No
invalidation condition → automatic kernel veto. This is what converts vibes into
a testable position with a defined exit.

### 3. Risk Kernel — the heart of the system

Plain deterministic code. No LLM. Every rule fails closed.

**Pre-trade gates** (all must pass):
1. Kill switch not engaged
2. Within trading window; not in a macro blackout
3. `review_equity_order` / `review_option_order` ran clean — **mandatory**
4. Notional ≤ per-trade cap (start ~2% of equity)
5. Symbol exposure ≤ cap (~10%); sector ≤ cap (~25%)
6. Correlation-adjusted gross exposure within limit
7. Daily order count under budget (blocks runaway loops)
8. Settlement check if cash account (GFV prevention)
9. Intraday margin headroom if margin account (IMD prevention)
10. Idempotency key unused — the #1 duplicate-order defense
11. Symbol passes `get_equity_tradability`; liquidity/spread floor met
12. Options: defined-risk only; no naked short premium

**Circuit breakers** (halt everything, require human reset):
- Daily loss > 3% of equity
- Peak-to-trough drawdown > 10%
- 5 consecutive losses
- Realized vol or slippage far outside model
- Any MCP schema drift detected
- P&L reported by MCP diverges from internal ledger

Numbers above are starting points to tune, not gospel. What matters is that they
exist in code, outside the model's reach, before any capital is live.

**Position sizing:** volatility-targeted (ATR-scaled), with a fractional-Kelly
ceiling. Never fixed share counts — that silently sizes up as price rises.

### 4. Execution plane

- **Resolve capabilities from the live `tools/list`, don't hardcode tool names.**
  The surface is account-specific, paginated (drain `nextCursor`), and Robinhood
  is actively adding asset classes. Snapshot it and diff in CI so a server-side
  rename fails loudly at build time, not at trade time.
- **`isError: true` is a failure, not a success.** An MCP tool that logically
  fails returns `isError` with a 200. Code that only catches exceptions will
  book a rejected order as filled. This is the most likely silent-corruption bug
  in the whole system.
- Always `review_*` before `place_*`.
- **Limit orders by default.** Market orders don't execute in extended/overnight
  sessions, and stop orders queue to next open — so any stop-based exit logic
  must know it is not protection outside regular hours.
- Reconcile after every fill: broker state is truth, local ledger is a cache.

### 5. Memory and learning

- **Journal**: append-only record of every intent, veto, order, fill, thesis.
- **Attribution**: per-strategy P&L, hit rate, expectancy, slippage.
- **Lessons**: nightly post-mortem writes compact retrieval-ready notes.
- **Strategy registry**: each strategy carries live stats and gets auto-benched
  when expectancy decays.

Without attribution, "maximize profit" has no gradient — you can't tell skill
from a bull market. This is how Osiris improves rather than just churning.
