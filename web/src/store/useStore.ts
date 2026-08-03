import { create } from "zustand";
import { api } from "../lib/api";
import type {
  Activity,
  AgentLine,
  Breaker,
  Channel,
  Connection,
  Health,
  Portfolio,
  Preflight,
} from "../lib/types";

/** Cap the in-memory stream. An agent left running for a week would otherwise
 *  grow this array without bound; the journal is the durable record. */
const MAX_STREAM_LINES = 300;

export type ConnectionState = "connecting" | "open" | "closed";

interface State {
  health: Health | null;
  preflight: Preflight | null;
  portfolio: Portfolio | null;
  breakers: Breaker[];
  activity: Activity[];
  connectionInfo: Connection | null;

  stream: AgentLine[];
  connection: ConnectionState;
  // Whether the BACKEND answered at all, as opposed to answering with no data.
  // Starts true so the first render is not a false alarm during the initial fetch.
  apiReachable: boolean;
  lastError: string | null;
  selectedSymbol: string | null;

  refreshAll: () => Promise<void>;
  refreshPortfolio: () => Promise<void>;
  connect: () => () => void;
  select: (symbol: string | null) => void;
}

function push(lines: AgentLine[], line: AgentLine): AgentLine[] {
  const next = [...lines, line];
  return next.length > MAX_STREAM_LINES ? next.slice(-MAX_STREAM_LINES) : next;
}

/** Turn an SSE payload into one readable log line.
 *
 *  Vetoes are rendered as prominently as fills. A kernel silently blocking every
 *  order looks identical to a quiet market, so a stream that only showed fills
 *  would hide the most important failure mode. */
function describe(channel: Channel, data: Record<string, unknown>): AgentLine | null {
  const seq = Number(data.seq ?? 0);
  const ts = String(data.ts ?? new Date().toISOString());
  const base = { seq, ts, channel };
  const sym = typeof data.symbol === "string" ? data.symbol : undefined;

  switch (channel) {
    case "agent":
      return {
        ...base,
        role: String(data.role ?? "agent"),
        text: String(data.text ?? ""),
        symbol: sym,
      };
    case "fill":
      return {
        ...base,
        role: "fill",
        text: `${String(data.side ?? "").toUpperCase()} ${data.symbol} ${Number(
          data.quantity ?? 0,
        ).toFixed(4)} @ ${Number(data.price ?? 0).toFixed(2)}`,
        symbol: sym,
      };
    case "veto": {
      const codes = Array.isArray(data.vetoes) ? data.vetoes.join(", ") : "";
      return {
        ...base,
        role: "veto",
        text: `${data.symbol ?? ""} blocked: ${codes}`,
        symbol: sym,
      };
    }
    case "breaker":
      return {
        ...base,
        role: "breaker",
        text: String(data.reason ?? data.detail ?? JSON.stringify(data)),
      };
    case "cycle":
      return { ...base, role: "cycle", text: String(data.summary ?? "cycle event") };
    case "reconciliation":
      return {
        ...base,
        role: "reconcile",
        text: data.clean ? "ledger reconciled" : `DIVERGENCE: ${data.divergences}`,
      };
    case "error":
      return { ...base, role: "error", text: String(data.error ?? "unknown error") };
    default:
      return null; // pnl and heartbeat update state, not the log
  }
}

export const useStore = create<State>((set, get) => ({
  health: null,
  preflight: null,
  portfolio: null,
  breakers: [],
  activity: [],
  connectionInfo: null,

  stream: [],
  connection: "connecting",
  apiReachable: true,
  lastError: null,
  selectedSymbol: null,

  select: (symbol) => set({ selectedSymbol: symbol }),

  refreshPortfolio: async () => {
    try {
      const [portfolio, activity] = await Promise.all([
        api.portfolio(),
        api.activity(200),
      ]);
      set({ portfolio, activity });
    } catch (err) {
      set({ lastError: err instanceof Error ? err.message : String(err) });
    }
  },

  refreshAll: async () => {
    // Settled rather than all: one failing panel must not blank the dashboard.
    const results = await Promise.allSettled([
      api.health(),
      api.preflight(),
      api.portfolio(),
      api.breakers(),
      api.activity(200),
      api.connection(),
    ]);
    const [health, preflight, portfolio, breakers, activity, connectionInfo] =
      results;

    const value = <T,>(r: PromiseSettledResult<T>, fallback: T): T =>
      r.status === "fulfilled" ? r.value : fallback;

    const failed = results.filter((r) => r.status === "rejected").length;
    // EVERY request failing means the backend is down, not that one panel broke.
    // Distinguished because the two need different messages, and a proxy error
    // shows up as a rejected fetch identical to a real API error.
    const allFailed = failed === results.length;

    set({
      health: value(health, get().health),
      preflight: value(preflight, get().preflight),
      portfolio: value(portfolio, get().portfolio),
      breakers: value(breakers, []),
      activity: value(activity, []),
      connectionInfo: value(connectionInfo, get().connectionInfo),
      apiReachable: !allFailed,
      lastError: allFailed
        ? "backend unreachable — is `python -m osiris.run --serve` running?"
        : failed > 0
          ? `${failed} request(s) failed`
          : null,
    });
  },

  /** ONE EventSource for every channel.
   *
   *  Browsers cap concurrent connections per origin (6 on HTTP/1.1), so a
   *  stream-per-channel design would starve the rest of the app. Everything is
   *  multiplexed server-side and demuxed here on the event name. */
  connect: () => {
    const source = new EventSource("/api/stream");
    const channels: Channel[] = [
      "cycle",
      "agent",
      "fill",
      "intent",
      "veto",
      "pnl",
      "breaker",
      "reconciliation",
      "error",
    ];

    source.onopen = () => set({ connection: "open", lastError: null });
    source.onerror = () => {
      // EventSource reconnects on its own; surface the state rather than
      // tearing down and racing a manual retry against the built-in backoff.
      set({ connection: "closed" });
    };

    const handlers = new Map<string, (e: MessageEvent) => void>();

    for (const channel of channels) {
      const handler = (event: MessageEvent) => {
        let data: Record<string, unknown>;
        try {
          data = JSON.parse(event.data as string) as Record<string, unknown>;
        } catch {
          return;
        }

        if (channel === "pnl") {
          const p = get().portfolio;
          if (p) {
            set({
              portfolio: {
                ...p,
                equity: Number(data.equity ?? p.equity),
                daily_pnl: Number(data.daily_pnl ?? p.daily_pnl),
                daily_pnl_pct: Number(data.daily_pnl_pct ?? p.daily_pnl_pct),
                peak_equity: Number(data.peak_equity ?? p.peak_equity),
                drawdown_pct: Number(data.drawdown_pct ?? p.drawdown_pct),
              },
            });
          }
          return;
        }

        const line = describe(channel, data);
        if (line) set({ stream: push(get().stream, line) });

        // A fill or a breaker changes the book: re-pull rather than patching,
        // so the ledger stays authoritative over the UI's guess.
        if (channel === "fill" || channel === "breaker" || channel === "reconciliation") {
          void get().refreshPortfolio();
        }
        if (channel === "cycle") {
          void get().refreshAll();
        }
      };
      handlers.set(channel, handler);
      source.addEventListener(channel, handler as EventListener);
    }

    set({ connection: "connecting" });

    return () => {
      for (const [channel, handler] of handlers) {
        source.removeEventListener(channel, handler as EventListener);
      }
      source.close();
      set({ connection: "closed" });
    };
  },
}));
