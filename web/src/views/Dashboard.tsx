import { useMemo } from "react";
import { Empty, Label, tone } from "../components/Primitives";
import { money, moneySigned, pct, pctSigned, timeOfDay } from "../lib/format";
import { humanize } from "../lib/humanize";
import type { Activity, Position } from "../lib/types";
import { useStore } from "../store/useStore";

/** The whole product on one page, in order of importance:
 *  a band of figures across the top, then positions (left) and the agent's
 *  actions (right) side by side, so "what I own" and "what it's doing" are
 *  visible at once instead of stacked a scroll apart. */
export function Dashboard() {
  return (
    <div className="flex flex-col">
      <Hero />
      <div className="grid grid-cols-1 gap-x-16 gap-y-12 lg:grid-cols-[5fr_7fr]">
        <Positions />
        <ActivityFeed />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ hero */

function Hero() {
  const portfolio = useStore((s) => s.portfolio);

  const equity = portfolio?.equity ?? 0;
  const dailyPnl = portfolio?.daily_pnl ?? 0;
  const dailyPct = portfolio?.daily_pnl_pct ?? 0;
  const t = tone(dailyPnl);

  return (
    <section className="rise flex flex-wrap items-end justify-between gap-x-12 gap-y-8 py-12">
      <div>
        <Label>Portfolio</Label>
        <div className="hero-num mt-3">{money(equity, true)}</div>
        <div className={`num mt-3 text-sm ${t}`}>
          {moneySigned(dailyPnl, true)} ({pctSigned(dailyPct)}) today
        </div>
      </div>

      {portfolio && (
        <div className="flex divide-x divide-[color:var(--color-line-soft)] pb-1.5">
          <Stat label="Cash" value={money(portfolio.cash, true)} first />
          <Stat label="Invested" value={pct(portfolio.net_exposure_pct, 0)} />
          <Stat label="Positions" value={String(portfolio.position_count)} />
          <Stat
            label="All-time high"
            value={money(portfolio.peak_equity, true)}
            className="hidden sm:block"
          />
        </div>
      )}
    </section>
  );
}

function Stat({
  label,
  value,
  first = false,
  className = "",
}: {
  label: string;
  value: string;
  first?: boolean;
  className?: string;
}) {
  return (
    <div className={`${first ? "pr-8 sm:pr-12" : "px-8 sm:px-12"} ${className}`}>
      <div className="num text-xl font-light">{value}</div>
      <div className="label mt-1.5">{label}</div>
    </div>
  );
}

/* -------------------------------------------------------------- positions */

function Positions() {
  const portfolio = useStore((s) => s.portfolio);
  const positions = portfolio?.positions ?? [];

  return (
    <section className="rise" style={{ animationDelay: "60ms" }}>
      <div className="flex items-baseline justify-between border-b border-[color:var(--color-line)] pb-2.5">
        <Label>What you own</Label>
        {positions.length > 0 && (
          <span className="num text-xs text-[color:var(--color-fg-3)]">
            {positions.length}
          </span>
        )}
      </div>
      {positions.length === 0 ? (
        <Empty title="No open positions.">
          When the agent buys a stock it appears here, with how it&rsquo;s
          doing and the reason it&rsquo;s held.
        </Empty>
      ) : (
        <div>
          {positions.map((p) => (
            <PositionRow key={p.symbol} position={p} />
          ))}
        </div>
      )}
    </section>
  );
}

function PositionRow({ position: p }: { position: Position }) {
  const t = tone(p.unrealized_pnl);
  return (
    <details className="row group">
      <summary className="flex cursor-pointer list-none items-baseline gap-4 py-4 [&::-webkit-details-marker]:hidden">
        <span className="num w-14 shrink-0 text-base font-medium">
          {p.symbol}
        </span>
        <span className="min-w-0 flex-1 truncate text-xs text-[color:var(--color-fg-3)]">
          {money(p.market_value, true)} · {pct(p.weight, 0)} of portfolio
        </span>
        <span className={`num shrink-0 text-right text-sm ${t}`}>
          {moneySigned(p.unrealized_pnl, true)}
          <span className="ml-2 text-xs opacity-80">
            {pctSigned(p.unrealized_pnl_pct)}
          </span>
        </span>
      </summary>
      <div className="pb-5 pl-18 text-sm leading-relaxed">
        {p.thesis ? (
          <p className="max-w-prose text-[color:var(--color-fg-2)]">
            <span className="text-[color:var(--color-fg-3)]">Why it&rsquo;s held: </span>
            {p.thesis}
          </p>
        ) : (
          <p className="text-[color:var(--color-fg-3)] italic">
            Bought before Osiris took over — no reasoning on record.
          </p>
        )}
        {p.invalidation && (
          <p className="mt-2 max-w-prose text-[color:var(--color-fg-3)]">
            <span className="text-[color:var(--color-down)]">
              It will sell if:{" "}
            </span>
            {p.invalidation}
          </p>
        )}
        <p className="num mt-3 text-xs text-[color:var(--color-fg-4)]">
          {p.quantity.toFixed(4)} shares · paid {money(p.avg_cost, true)} avg ·
          now {money(p.last_price, true)}
        </p>
      </div>
    </details>
  );
}

/* --------------------------------------------------------------- activity */

function ActivityFeed() {
  const activity = useStore((s) => s.activity);

  // Keep only the most recent zero-trade check-in; a week of "did nothing"
  // rows buries the entries that matter.
  const rows = useMemo(() => {
    let keptNote = false;
    return activity.filter((r) => {
      if (r.kind !== "note") return true;
      if (keptNote) return false;
      keptNote = true;
      return true;
    });
  }, [activity]);

  return (
    <section className="rise" style={{ animationDelay: "120ms" }}>
      <div className="border-b border-[color:var(--color-line)] pb-2.5">
        <Label>What the agent is doing</Label>
      </div>
      {rows.length === 0 ? (
        <Empty title="Nothing yet.">
          The agent researches and trades shortly after each market open, then
          watches every position minute-by-minute for the rest of the day —
          selling automatically if one falls past its stop. Everything it does,
          and everything it decides against, shows up here in plain terms.
        </Empty>
      ) : (
        <div>
          {rows.map((r) => (
            <ActivityRow key={r.seq} row={r} />
          ))}
        </div>
      )}
    </section>
  );
}

const KIND_STYLE: Record<
  Activity["kind"],
  { word: string; mark: string; color: string }
> = {
  bought: { word: "Buy", mark: "mark-buy", color: "var(--color-up)" },
  sold: { word: "Sell", mark: "mark-sell", color: "var(--color-fg-2)" },
  blocked: { word: "Passed", mark: "mark-block", color: "var(--color-amber)" },
  halted: { word: "Stopped", mark: "mark-block", color: "var(--color-down)" },
  note: { word: "Check-in", mark: "", color: "var(--color-fg-3)" },
};

function ActivityRow({ row }: { row: Activity }) {
  const h = humanize(row);
  const style = KIND_STYLE[row.kind];

  // The daily check-in is a heartbeat, not an event. It renders as one quiet
  // line rather than a full entry, so the trades and passes stand out.
  if (row.kind === "note") {
    return (
      <div className="row flex items-baseline gap-3 py-3">
        <span className="dot mt-px shrink-0 self-center opacity-60" />
        <span className="min-w-0 flex-1 truncate text-xs text-[color:var(--color-fg-3)]">
          <span className="text-[color:var(--color-fg-2)]">{h.headline}.</span>{" "}
          {h.explanation}
        </span>
        <time className="num shrink-0 text-xs text-[color:var(--color-fg-4)]">
          {timeOfDay(row.ts)}
        </time>
      </div>
    );
  }

  return (
    <article className={`row ${style.mark} py-4 ${style.mark ? "pl-4" : ""}`}>
      <div className="flex items-baseline gap-4">
        <span className="label w-14 shrink-0" style={{ color: style.color }}>
          {style.word}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-4">
            <span className="text-sm font-medium text-[color:var(--color-fg)]">
              {h.headline}
              {row.notional_usd != null && (
                <span className="num ml-2.5 font-normal text-[color:var(--color-fg-3)]">
                  ${Math.round(row.notional_usd).toLocaleString("en-US")}
                </span>
              )}
            </span>
            <time className="num shrink-0 text-xs text-[color:var(--color-fg-4)]">
              {timeOfDay(row.ts)}
            </time>
          </div>

          {(h.explanation || h.fact) && (
            <p className="mt-1.5 max-w-prose text-sm leading-relaxed text-[color:var(--color-fg-2)]">
              {h.explanation}
              {h.fact && (
                <span className="text-[color:var(--color-fg-3)]"> {h.fact}</span>
              )}
            </p>
          )}
          {h.thesis && (
            <details className="group mt-2">
              <summary className="cursor-pointer list-none text-xs text-[color:var(--color-fg-4)] transition-colors hover:text-[color:var(--color-fg-3)] [&::-webkit-details-marker]:hidden">
                <span className="group-open:hidden">
                  Why it was interested&thinsp;→
                </span>
                <span className="hidden group-open:inline">
                  Why it was interested&thinsp;↓
                </span>
              </summary>
              <p className="mt-1.5 max-w-prose text-sm leading-relaxed text-[color:var(--color-fg-3)]">
                {h.thesis}
              </p>
            </details>
          )}
        </div>
      </div>
    </article>
  );
}
