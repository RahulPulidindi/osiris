/** Number formatting.
 *
 *  All formatters return plain strings with no sign-colouring: colour is a
 *  presentation decision made by the component, because the two-tier emphasis
 *  rule applies different treatment to a hero total and a repeated table cell.
 */

const usd0 = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const usd2 = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

const compact = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
});

export function money(value: number, decimals = false): string {
  if (!Number.isFinite(value)) return "—";
  return decimals ? usd2.format(value) : usd0.format(value);
}

/** Signed money. Uses a true minus sign (U+2212) so digits stay grid-aligned. */
export function moneySigned(value: number, decimals = false): string {
  if (!Number.isFinite(value)) return "—";
  const formatted = money(Math.abs(value), decimals);
  if (value > 0) return `+${formatted}`;
  if (value < 0) return `\u2212${formatted}`;
  return formatted;
}

export function compactMoney(value: number): string {
  if (!Number.isFinite(value)) return "—";
  return `$${compact.format(value)}`;
}

export function pct(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function pctSigned(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "—";
  const formatted = `${(Math.abs(value) * 100).toFixed(digits)}%`;
  if (value > 0) return `+${formatted}`;
  if (value < 0) return `\u2212${formatted}`;
  return formatted;
}

export function bps(value: number, digits = 1): string {
  if (!Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : value < 0 ? "\u2212" : "";
  return `${sign}${Math.abs(value).toFixed(digits)} bps`;
}

export function num(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

export function shares(value: number): string {
  if (!Number.isFinite(value)) return "—";
  // Fractional shares are real on this venue, so trailing precision matters
  // when it is present and is noise when it is not.
  return Number.isInteger(value) ? String(value) : value.toFixed(4);
}

/** Sign class for the two-tier P&L emphasis rule. */
export function signClass(value: number): "is-gain" | "is-loss" | "is-flat" {
  if (!Number.isFinite(value) || Math.abs(value) < 1e-9) return "is-flat";
  return value > 0 ? "is-gain" : "is-loss";
}

/** Hour and minute only.
 *
 *  Seconds were dropped from the ledger: on a book that rebalances once a
 *  session, second-level precision is noise that widens the column and pulls the
 *  eye away from the reasoning beside it. */
export function timeOfDay(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return d.toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function shortDate(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

/** Human-readable event label from a journal event key. */
export function eventLabel(event: string): string {
  return event.replace(/_/g, " ");
}

export function titleCase(value: string): string {
  return value
    .replace(/[_-]/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
