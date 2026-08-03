# Is there real upside? What the research actually shows

## Short answer

Yes — but the upside lives in a specific, narrow shape, and it is almost the
opposite of what "autonomous trading bot" usually means.

**What works (peer-reviewed, out-of-sample):** ranking a *large* universe of
liquid stocks by LLM-synthesized analysis, holding a *diversified basket* of the
top-ranked names, rebalancing *daily-at-open or monthly*.

**What doesn't:** concentrated discretionary bets, intraday timing, scalping,
and predicting losers.

## The evidence

### 1. Agentic AI nowcasting, Russell 1000 (Chen & Pu, Jan 2026)

An LLM autonomously searched the live web daily and scored all ~1000 Russell
stocks. 155,000 predictions, April 2025 – Jan 2026. Structurally free of
look-ahead bias (predictions made at the edge of time).

- Top-20 long portfolio: **+50%** vs. **+26%** for the benchmark over 9 months
- Daily 6-factor alpha **18.4 bps** (t = 2.46), annualized ~46%
- Sharpe **2.43**; transaction costs <10% of gross alpha
- Market beta only **0.298** — it built a *defensive* book, not a leveraged one

The load-bearing caveat: **"this predictability is highly concentrated:
expanding beyond the top tier rapidly dilutes alpha, and bottom-ranked stocks
exhibit returns statistically indistinguishable from the market."**

The AI could find winners. It could **not** find losers. The authors hypothesize
negative news is contaminated by corporate obfuscation and social noise.

### 2. MarketSenseAI multi-agent system (Fatouros & Metaxas, 2024–2026)

Four specialist agents (News, Fundamentals, Dynamics, Macro) → synthesis agent →
monthly recommendation. Signals generated live.

- S&P 500 cohort, 19 months: **+2.18%/mo** vs. **+1.15%** equal-weight benchmark
- **+25.2pp** compound excess, 99.7th percentile of 10,000 Monte Carlo random
  portfolios (**p = 0.003**) — beats random selection, not just the index
- Portfolio beta **0.865** (below 1), and it preserved *more* alpha in down
  months (+1.31%) than up months (+0.82%) — inconsistent with a hidden
  high-beta bet
- S&P 100 cohort: directionally consistent but **p = 0.17, not significant**

Note the architecture that won here is close to the multi-role design already in
`ARCHITECTURE.md`. That is encouraging, and not a coincidence — specialist
decomposition plus synthesis is the pattern that keeps showing up.

### 3. ChatGPT headline scoring (Lopez-Lira & Tang, UCLA)

- GPT-4 headline sentiment predicts next-day returns; Sharpe **3.8**
- Return forecasting is an **emergent capability of larger models** — GPT-1,
  GPT-2, and BERT show *no* predictive ability
- Strongest in small caps and after negative news

## The other side of the ledger

Reasons to size expectations down, all from the same literature:

1. **Short samples, mostly bull markets.** 9–19 months. None of these systems
   has been through a 2008 or a 2022. A Sharpe of 2.43 over 9 months is not a
   Sharpe of 2.43.
2. **Alpha decays after publication.** McLean & Pontiff document substantial
   post-publication decay; Chordia et al. show most daily-frequency anomalies
   have already been arbitraged away. These papers are now public.
3. **Marginal significance.** t = 2.46 and p = 0.17 are not laws of physics.
4. **79% of professional active large-cap managers underperformed the S&P 500 in
   2025** (SPIVA). Over 15 years it's **~90%**, and on a *risk-adjusted* basis
   over 10 years, **95.7%**. These are full-time funded teams.
5. **Your venue is long-only.** All three papers benefit from at least a
   ranking spread; you can only express the long leg. Given that finding losers
   didn't work anyway, this costs less than it sounds — but it caps you.

Point 4 is the one to internalize. It's not that alpha is impossible; it's that
alpha is *scarce* and mostly gets competed away. The realistic ambition is a
modest, durable edge over buy-and-hold — not a money printer.

## What this implies for Osiris

The evidence says build a **cross-sectional ranking engine**, not a trade picker.

| Design choice | Wrong version | Evidence-backed version |
|---|---|---|
| Universe | A few stocks the agent "likes" | Rank ~500–1000 liquid names |
| Concentration | 3–5 high-conviction bets | Top 20+ names, diversified |
| Frequency | Intraday / scalping | Daily-at-open or monthly rebalance |
| Direction | Long and short | **Long-only** (shorts didn't work, and RH can't) |
| Universe quality | Small caps, high spread | Large caps, ~1.6 bps spreads |
| Objective | Maximize profit | Risk-adjusted, low beta |
| Benchmark | "Did I make money?" | vs. equal-weight index **and** vs. random |

Two things worth stressing because they're counterintuitive:

**Breadth is the edge, not conviction.** The papers win by making ~1000 small
judgments and averaging them, not 5 brilliant ones. This is where an LLM has a
genuine structural advantage over you: it can read about 1000 companies every
day and you cannot. That is the actual reason to build this.

**The random-portfolio benchmark is non-negotiable.** MarketSenseAI's key result
isn't beating the index — it's beating 10,000 random same-sized portfolios from
the same universe on the same dates. Without that test you cannot distinguish
skill from "I happened to hold 20 stocks in a rising market." Most retail bot
builders never run it, which is why most of them believe they have an edge.

## Recommended revision to the plan

Reframe Osiris from "autonomous trader" to **"autonomous equity research analyst
that rebalances a diversified long book."** Same autonomy, same MCP, same risk
kernel — different, evidence-backed strategy shape.

Concretely, this *simplifies* the build:
- Drop scalping (impossible anyway — see PLAN.md)
- Drop options for v1 (none of the evidence involves options)
- Drop TradingView (no intraday timing to chart)
- Drop L2 order book (irrelevant at daily rebalance)
- **Add** the Monte Carlo random-portfolio test as a hard gate
- **Add** the MCP scanner (`run_scan`) as the universe filter — it's built for this

Lower frequency also collapses the cost problem: one ranking pass at market open
instead of a 5-minute loop cuts OpenRouter spend by ~100x, and the evidence says
the slower version performs *better*.
