# Osiris

An autonomous trading agent for your Robinhood Agentic account. It researches
stocks, decides what to buy and sell, places the orders itself, and shows you
every action with the reasoning behind it.

## Getting started

```bash
python3 -m venv .venv
source .venv/bin/activate        # do this in every new terminal
pip install -e ".[dev]"
cp .env.example .env
```

The Python package lives in `src/osiris/`; the dashboard in `web/`; the
container build at the root. Tests: `pytest tests -q`.

Every `python` below assumes that `activate` has run. Without it you get
`ModuleNotFoundError: No module named 'pydantic'`, because the dependencies live
in `.venv` rather than in the system interpreter. If you would rather not
activate, call the interpreter directly: `.venv/bin/python -m osiris.connect`.

**1. Connect your account.** Opens a browser once for Robinhood's consent screen,
then caches credentials in `~/.osiris/` (outside this repo, mode 0600).

```bash
python -m osiris.connect
```

This does more than authenticate — it reads your balance and positions back to
you. A token exchange succeeding means *login* worked; it does not mean the agent
can trade, because the available tools are account-specific. This command tells
you which one you have.

**2. Watch it think, without letting it trade.**

```bash
python -m osiris.run --once --dry-run
```

Full research and decision-making. Every intended order is explained. Nothing is
sent.

**3. Let it trade.** Set both of these in `.env` — not with `export`, which dies
with the shell:

```
OSIRIS_MODE=live
OSIRIS_I_UNDERSTAND_THE_RISK=yes
```

Two separate affirmations, so no single stray variable can arm a real account.
Then:

```bash
cd web && npm install && npm run build && cd ..   # build the dashboard once
python -m osiris.run --serve                      # agent + dashboard on :8030
```

One process is the whole product: it holds the broker connection, runs the
schedule, answers the API, and serves the dashboard at http://127.0.0.1:8030.
(`cd web && npm run dev` still gives a hot-reloading UI at :5173 for
development.)

## Running it unattended

`--serve` is the always-on process. It holds the broker connection, serves the
dashboard, and **wakes on the market's schedule** — not on a timer.

You issue no commands. There is no "trade" button; the dashboard cannot place an
order at all. The agent decides on its own each session.

It prints its schedule at startup so a sleeping process is distinguishable from a
hung one:

```
  SCHEDULE
    market time ...... 2026-08-02 11:10:47 ET
    trading today .... no
    next cycle ....... in 22.5h  (market closed; next session 2026-08-03)
    cadence .......... once per session, shortly after the open
```

**Cadence.** Two loops at two speeds:

- **Research** runs once per session, ~10 minutes after the open. That offset
  is deliberate: the first minutes carry the widest spreads of the day, and
  this book rebalances daily so a second research pass adds cost and turnover
  without adding signal. `--interval 60` runs it hourly if you want that.
- **Monitoring** runs continuously while the market is open: every
  `OSIRIS_WATCH_INTERVAL_SECONDS` (default 60) the agent pulls quotes for the
  names it holds, marks equity so the dashboard and circuit breakers see
  intraday moves, and fires any stop-loss immediately rather than at the next
  open. The monitor can only sell — it opens no positions, so it cannot churn.
  Signal is daily; risk is continuous.

**Skipped automatically:** weekends, the ten US market holidays, pre-market,
and the last 20 minutes before the close (an order placed at 15:59 has no time to
fill and reconcile).

**Surviving disconnects.** An MCP session does not live forever — tokens refresh,
sockets drop. After three consecutive failed cycles the agent reconnects and
keeps its journal and ledger intact. Failures alert; three in a row escalate to
critical, because an unmanaged position has no stop.

**Keeping it alive across reboots** is your OS's job, not the agent's. On macOS:

The machine must be awake every market minute — research at the open, then
minute-by-minute position monitoring until the close. A sleeping laptop runs
nothing, so the real answer is a $5/month Linux VM running the provided
container:

```bash
docker compose up -d --build      # on the server; survives crashes and reboots
```

**See `docs/DEPLOY.md`** for the full recipe: copying credentials (OAuth needs
a browser exactly once, on your Mac), reaching the dashboard privately over
SSH or Tailscale, and day-2 operations. This process places real orders
through your cached credentials, so it must run somewhere you control — a
serverless host like Vercel cannot run an always-on scheduler and would mean
uploading your brokerage tokens to a third party.

For a Mac that stays awake anyway, `nohup python -m osiris.run --serve` or the
launchd unit in `deploy/com.osiris.agent.plist` both work.

Stop it any time with `touch KILL_SWITCH` — that works even if the process is
wedged and the dashboard will not load.

## What you see

One page, in order of importance:

**Portfolio** — the account value, today's change, cash and exposure.

**Positions** — each holding with its size, gain or loss, and (expanded on
click) the thesis for holding it and the condition that triggers a sale.

**Agent activity** — what the agent did, newest first, each row carrying its
reason:

```
14:47  BOUGHT  NVDA  $2,140
       entered the top ranks
       earnings beat with guidance raised; still below sector P/E

14:47  SKIPPED AAPL
       would exceed the sector limit  sector Technology would reach 26.6%
```

Skipped orders appear beside fills deliberately. An agent whose every order is
being rejected looks exactly like a quiet market if you only show fills.

Before the account is linked and the risk acknowledged, the page shows a
two-step setup instead: connect Robinhood (`python -m osiris.connect`), then
type the acknowledgement. The acknowledgement writes both affirmations to
`.env`; restarting the service is what actually arms it, so the page alone can
never arm live trading. There is **no sample data** anywhere — a UI that
invents numbers to look populated teaches you to trust figures that were never
real.

## How it decides

1. **Universe** — your account's scanner, or a liquid large-cap list if no
   scanner tool is available (it says which).
2. **Cheap pre-rank** — free, deterministic, no LLM. Not a momentum screen:
   that would discard the "good news not yet in the price" names worth finding.
3. **Triage** → **deep research** → **portfolio construction** — progressively
   more expensive models over progressively fewer names, so the token bill scales
   with conviction rather than universe size.
4. **Red team** — a *different model family* argues the bear case and holds veto
   power. One model asked to both propose and critique produces correlated
   errors; a model asked to find trades will find trades.
5. **Risk kernel** — deterministic, no LLM, cannot be influenced by the model.
6. **Simulate, then place** — every order is dry-run against the venue first.

Every proposal must carry a falsifiable invalidation condition. One without a
stated exit is rejected automatically.

## Safety

Configure any limit in `.env`, including effectively unlimited:

| Setting | Default | Meaning |
|---|---|---|
| `OSIRIS_MAX_TRADE_NOTIONAL_PCT` | `0.02` | Max per order, as a fraction of equity |
| `OSIRIS_MAX_SYMBOL_WEIGHT` | `0.10` | Max in any one stock |
| `OSIRIS_MAX_SECTOR_WEIGHT` | `0.25` | Max in any one sector |
| `OSIRIS_DAILY_LOSS_HALT_PCT` | `0.03` | Stop trading after this daily loss |
| `OSIRIS_MAX_DRAWDOWN_HALT_PCT` | `0.10` | Stop trading at this drawdown |

`python -m osiris.run --unrestricted` disables the caps entirely.

**Small accounts get scaled limits automatically.** The defaults assume five
figures. Below $10,000 they are not conservative but *infeasible* — at $366 a 2%
order is $7.32 against $18 target positions, needing ~50 orders to build a book
the kernel would then veto. So below that threshold the book concentrates: fewer
names, larger orders.

That is a real increase in risk, not a free optimization. A few hundred dollars
cannot be both spread across 20 names and meaningfully invested in any of them.
Concentration is the honest choice; the startup banner prints the actual dollar
figures so you can see what it decided.

Note also that Robinhood allows fractional shares on **market orders only**. A
limit order below one share is therefore converted to a dollar-denominated market
order and logged — trading limit-price protection for the ability to trade at all,
which is the right call at $7 notional.

**Two things are not configurable**, and they are not caps:

- **Simulate before placing.** This is what makes a rejected order
  distinguishable from a filled one. MCP tools return `isError: true` with an
  HTTP 200, so code that only catches exceptions books a rejection as a fill —
  and then every later sizing decision is computed against a position you do not
  own.
- **The kill switch.** `touch KILL_SWITCH` stops new positions immediately. No
  process restart, no API, no working dashboard required — which is precisely
  what is unavailable during an incident.

Halting stops *new* positions. Exits and stop-losses keep running: an agent that
freezes while holding losers is worse than one that never traded.

## Honest expectations

Most retail trading loses money. A *Journal of Finance* study of the most-bought
Robinhood stocks (2018–2020) found average 20-day returns of **−4.7%**, and an
LLM does not repeal that. The edge here, if any, is breadth and discipline — not
the model's cleverness.

Judge it on risk-adjusted return versus buy-and-hold SPY over 6+ months. That is
the only comparison that separates skill from a rising market.

Validate the strategy offline before funding anything:

```bash
python -m osiris.runner.gate      # exits non-zero unless it clears
```

See `docs/RUNBOOK.md` for what to do when something breaks.

Not investment advice. Trading your own money in your own account requires no
registration; managing anyone else's changes that immediately.
