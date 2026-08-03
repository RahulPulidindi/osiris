# Phase 0 — Ground truth checklist

Facts that must be established before sizing logic is trusted. Each has a
verification command and a place to record the answer. **Do not skip this and
do not infer these from documentation** — the MCP tool surface is
account-specific, and the account type changes the entire turnover model.

Run: `python -m osiris.mcp.enumerate` (requires interactive OAuth once).

## 1. Account type: cash or margin

**Why it matters.** In a **cash account**, T+1 settlement applies and reusing
unsettled proceeds triggers good-faith violations, which throttles the strategy
to roughly one round trip per position per day. In a **margin account**,
unsettled funds are immediately reusable, but real-time intraday margin
requirements apply.

This single fact decides whether daily rebalancing is even legal for the book.

- [ ] Verified account type: `________` (cash | margin)
- [ ] Recorded in `.env` as `OSIRIS_ACCOUNT_TYPE`

Check via `get_accounts` output, and confirm in the Robinhood UI. The Agentic
account is a self-directed individual account; the type is not guaranteed to
match your primary account.

## 2. Live MCP tool surface

The published docs list ~45 tools, but the surface advertised to **your** account
may differ, and `tools/list` is paginated.

- [ ] `python -m osiris.mcp.enumerate` run successfully
- [ ] `docs/mcp/tools-snapshot.json` committed
- [ ] Tool count recorded: `______`
- [ ] Capability table shows `placeOrder`, `reviewOrder`, `listPositions`,
      `listOrders`, `cancelOrder`, `runScan` all resolving

If any capability fails to resolve, that is a genuine gap in what the account
can do today, not a bug in the resolver.

## 3. Options level

Not needed for v1 (equities only), but record it so the constraint is known.

- [ ] Options level granted: `______` (none | L2 | L3)

Note L3 (spreads) requires a margin account.

## 4. PDT status

FINRA eliminated the pattern-day-trader designation effective **June 4, 2026**,
replaced by real-time intraday margin requirements. Robinhood adopted this on
that date. Confirm no legacy PDT flag persists on the account.

- [ ] No PDT flag / restriction present
- [ ] Understood that the replacement constraint is **intraday margin deficit
      (IMD)**, monitored in real time, with repeated failures causing restrictions

## 5. Rate limits (empirical)

None are published. Discover them and record, since the funnel's Stage 0/1
polling volume depends on this.

- [ ] `get_equity_quotes` batch limit confirmed (docs say 20 symbols)
- [ ] `get_equity_price_book` batch limit confirmed (docs say 4 symbols)
- [ ] Observed throttling threshold, if any: `______`

## 6. Scanner filter vocabulary

- [ ] `get_scanner_filter_specs` output saved to `docs/mcp/scanner-filters.json`
- [ ] Confirmed filters exist for: dollar volume / average volume, market cap, price

These are the inputs to the Stage 0 liquidity floor. If the scanner cannot
filter on liquidity, the universe must be built from a local screen instead.

## Gate

Phase 0 is complete when the tool snapshot is committed and the account type is
known **in writing**. Until then, `OSIRIS_MODE` stays `paper` and the live order
path stays disarmed.
