import type { Activity } from "./types";

/** Translate the agent's shorthand into sentences a first-time trader can
 *  read. The backend speaks in risk-desk terms ("spread 218.5bps exceeds
 *  25.0bps"); this file is the interpreter. Nothing is invented — every
 *  sentence is a restatement of a reason the agent actually recorded. */

/** Plain-English versions of the kernel's veto reasons, keyed on fragments of
 *  the API's `reason` text (see VETO_EXPLANATIONS on the backend). */
const VETO_TRANSLATIONS: [match: string, plain: string][] = [
  [
    "spread too wide",
    "The gap between the buying and selling price was too large right now — buying would mean overpaying just to get in.",
  ],
  ["kill switch", "Trading is stopped until you resume it."],
  ["circuit breaker", "An automatic safety stop is active."],
  ["macro event", "A major economic announcement is close — prices whip around and it sat out."],
  ["too close to earnings", "The company reports earnings soon, and prices swing hard on those days."],
  ["no exit condition", "The research never said when to sell it, so it refused to buy. Every position must have a planned exit."],
  ["per-order cap", "The order was bigger than the per-trade safety limit. Positions are built up over several days instead."],
  ["single-name limit", "Buying more would put too much of the portfolio in one stock."],
  ["sector limit", "Buying more would concentrate too much money in one industry."],
  ["benchmark's sector mix", "It would tilt the portfolio too far toward one industry."],
  ["beta over budget", "It would make the whole portfolio swing harder than the market — more risk than allowed."],
  ["diversification floor", "Selling would leave too few stocks in the portfolio."],
  ["daily volume", "The order was too big relative to how much this stock trades — it would move the price."],
  ["not currently tradable", "This stock can't be traded right now."],
  ["too old to trust", "The price quote was stale, and it won't trade on old numbers."],
  ["order budget", "It already used up its allowance of orders for the day."],
  ["duplicate", "It had already sent this exact order."],
  ["not enough buying power", "Not enough cash available for this order."],
  ["not settled", "Cash from a recent sale hasn't cleared yet."],
  ["rejected the simulation", "Robinhood's test run of the order came back rejected, so nothing was sent."],
];

/** Reasons a trade existed, from the planner. Keyed the same way. */
const INTENT_TRANSLATIONS: [match: string, plain: string][] = [
  ["entered the top ranks", "its research ranked this stock among the best buys today"],
  ["dropped out of the top ranks", "its research no longer ranks this stock worth holding"],
  ["hit its stop", "the price fell to the pre-set level where it cuts losses"],
  ["thesis was invalidated", "the reason for owning it stopped being true"],
  ["rebalancing", "it was adjusting the position back to its planned size"],
  ["trimming", "the position had grown larger than planned, so it sold some"],
  ["adding", "the position was smaller than planned, so it bought more"],
];

function translate(
  table: [string, string][],
  text: string,
): string | null {
  const lower = text.toLowerCase();
  for (const [match, plain] of table) {
    if (lower.includes(match)) return plain;
  }
  return null;
}

/** "spread 218.5bps exceeds 25.0bps" → "the gap was 2.19%, nearly 9× the 0.25% limit". */
function humanizeSpreadFact(fact: string): string | null {
  const m = fact.match(/spread\s+([\d.]+)bps\s+exceeds\s+([\d.]+)bps/i);
  if (!m?.[1] || !m[2]) return null;
  const actual = parseFloat(m[1]) / 100;
  const limit = parseFloat(m[2]) / 100;
  const times = limit > 0 ? actual / limit : 0;
  const ratio =
    times >= 2 ? `, about ${Math.round(times)}× its limit of ${limit.toFixed(2)}%` : ` against a limit of ${limit.toFixed(2)}%`;
  return `The gap was ${actual.toFixed(2)}%${ratio}.`;
}

export interface HumanActivity {
  /** One-line summary: "Decided against buying V". */
  headline: string;
  /** The plain-English reason. */
  explanation: string;
  /** The measurement, restated ("The gap was 2.19%…"), when one exists. */
  fact: string;
  /** The research thesis — why the agent wanted the trade at all. */
  thesis: string;
}

const DETAIL_MARKER = " · wanted it because: ";

export function humanize(row: Activity): HumanActivity {
  const detail = row.detail ?? "";
  const at = detail.indexOf(DETAIL_MARKER);
  const rawFact = at >= 0 ? detail.slice(0, at).trim() : "";
  const thesis =
    at >= 0 ? detail.slice(at + DETAIL_MARKER.length).trim() : "";

  switch (row.kind) {
    case "bought":
    case "sold": {
      const verb = row.kind === "bought" ? "Bought" : "Sold";
      const because = translate(INTENT_TRANSLATIONS, row.reason);
      return {
        headline: `${verb} ${row.symbol}`,
        explanation: because ? `Because ${because}.` : row.reason,
        fact: "",
        // On a buy the detail IS the thesis; on a sell it is the exit trigger.
        thesis: detail && at < 0 ? detail : thesis,
      };
    }
    case "blocked": {
      const why = translate(VETO_TRANSLATIONS, row.reason) ?? row.reason;
      const fact = humanizeSpreadFact(rawFact || detail) ?? "";
      // The API's headline is "Blocked a buy of V" / "Blocked a sell of V".
      const selling = /a sell of/i.test(row.headline);
      return {
        headline: `Decided against ${selling ? "selling" : "buying"} ${row.symbol}`.trim(),
        explanation: why,
        fact,
        thesis,
      };
    }
    case "halted":
      return {
        headline: row.headline,
        explanation:
          translate(VETO_TRANSLATIONS, row.reason) ?? row.reason,
        fact: "",
        thesis: detail && at < 0 ? detail : thesis,
      };
    default: {
      // Session note: "Session complete — 0 trade(s), 5 blocked".
      const m = row.headline.match(/(\d+)\s*trade\(s\),\s*(\d+)\s*blocked/);
      if (m) {
        const trades = Number(m[1]);
        const blocked = Number(m[2]);
        const parts: string[] = [];
        parts.push(
          trades === 0
            ? "made no trades"
            : `made ${trades} trade${trades === 1 ? "" : "s"}`,
        );
        if (blocked > 0)
          parts.push(
            `held back ${blocked} it judged too risky or too expensive right now`,
          );
        return {
          headline: "Daily check-in",
          explanation: `Reviewed the market and the portfolio, ${parts.join(" and ")}.`,
          fact: "",
          thesis: "",
        };
      }
      return {
        headline: row.headline,
        explanation: row.reason,
        fact: "",
        thesis: detail,
      };
    }
  }
}
