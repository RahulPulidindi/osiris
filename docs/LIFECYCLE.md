# Osiris — Autonomous Trade Lifecycle

Confirming exactly what runs without human involvement. Answer: the entire loop
below. No human approves any individual trade.

## The daily cycle (fully autonomous)

```
07:00  Wake. Health check, kill-switch check, reconcile broker vs. ledger.
07:15  Universe filter — run_scan → ~500-1000 liquid names.
07:20  Ingest: OHLCV, fundamentals, news, earnings calendar, macro regime.
08:00  Rank. Analyst → Strategist → Red Team → PM produce a scored ranking.
09:15  Diff target book vs. current book. Emit OrderIntents for the delta.
09:25  Risk Kernel evaluates every intent. VETO or PASS.
09:30  Execute at the open: review_* → place_* for each approved order.
09:45  Verify fills. Reconcile. Journal everything.
16:00  Close-of-day snapshot. Attribution.
20:00  Post-mortem. Write lessons to memory. Sleep.
```

Every step is machine-executed. Osiris opens positions, closes positions, and
sizes them without asking.

## Opening positions — autonomous

A stock entering the top-N ranking generates a buy intent. Kernel checks it
(sizing, exposure, correlation, liquidity, tradability, idempotency), then
`review_equity_order` → `place_equity_order`. No confirmation prompt.

## Closing positions — autonomous, and *better* than the original design

This is worth dwelling on, because it's the part the reframe genuinely improves.

Under a ranking engine, **exits are mechanical**: a position closes when the
stock falls out of the top-N at rebalance. It doesn't need an opinion, a thesis
review, or a judgment call. It's a set-difference operation.

```
held = {AAPL, MSFT, NVDA, ...}      # current book
target = top_n(ranking)             # today's ranking

to_buy  = target - held             # open these
to_sell = held - target             # close these  ← automatic exit
```

Compare that to discretionary exits, which is where LLM trading agents most
reliably fail: the model forms a thesis, the position moves against it, and the
model rationalizes holding ("the thesis is still intact, this is just noise").
That's the classic path to one position eating the portfolio. Mechanical
ranking-based exits delete that failure mode — the model never gets a vote on
whether to hold a loser.

Three exit paths, all autonomous:

1. **Rank exit** — drops out of top-N → sold at next rebalance
2. **Risk exit** — kernel forces a flatten on a stop or exposure breach,
   independent of what the ranking says, checked continuously not just at 09:30
3. **Invalidation exit** — the position's pre-declared falsifiable condition
   fires (e.g. closes below 200MA)

The kernel can force an exit that the model disagrees with. It cannot be talked
out of one.

## The one deliberate exception

**Circuit breakers halt trading and require a human reset.** If daily loss
exceeds ~3%, drawdown exceeds ~10%, or the ledger diverges from the broker,
Osiris stops opening positions and pages you.

This is the only human-in-the-loop moment in the system, and it is intentional.
It's not a per-trade approval — it's a fuse. A fuse that reset itself isn't a
fuse. Note the asymmetry: a halted Osiris will still *close* positions and
honor risk exits. Halt means "stop taking new risk," never "stop managing
existing risk."

## Autonomy vs. constraint

To be exact about the distinction, since it's easy to blur:

| | Who decides |
|---|---|
| Which stocks to buy/sell | **Osiris** (no human) |
| When to enter/exit | **Osiris** (no human) |
| Position sizes | Osiris proposes, kernel bounds |
| Whether an order is *permitted* | **Kernel** (deterministic code) |
| Resuming after a breaker trip | Human |

The kernel isn't a human in the loop — it's code you wrote once, running
unattended. It's the difference between a car with no driver and a car with no
brakes. You asked for the first one.
