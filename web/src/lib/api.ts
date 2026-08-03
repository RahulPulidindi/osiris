import type {
  Activity,
  Breaker,
  Connection,
  Health,
  Portfolio,
  Preflight,
} from "./types";

const BASE = "/api";

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    signal: signal ?? null,
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new ApiError(`GET ${path} failed: ${res.statusText}`, res.status);
  }
  return (await res.json()) as T;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const payload = (await res.json()) as { detail?: unknown };
      if (payload.detail) detail = JSON.stringify(payload.detail);
    } catch {
      /* keep statusText */
    }
    throw new ApiError(`POST ${path} failed: ${detail}`, res.status);
  }
  return (await res.json()) as T;
}

export const api = {
  health: (signal?: AbortSignal) => get<Health>("/health", signal),
  preflight: (signal?: AbortSignal) => get<Preflight>("/preflight", signal),
  portfolio: (signal?: AbortSignal) => get<Portfolio>("/portfolio", signal),
  breakers: (signal?: AbortSignal) => get<Breaker[]>("/breakers", signal),
  activity: (limit = 200, kind?: string, signal?: AbortSignal) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (kind) params.set("kind", kind);
    return get<Activity[]>(`/activity?${params}`, signal);
  },

  connection: (signal?: AbortSignal) => get<Connection>("/connection", signal),

  /** Records both live-trading affirmations in .env. A restart still applies
   *  them — the running process cannot arm itself over HTTP. */
  acknowledgeRisk: (acknowledgement: string, acknowledgedBy: string) =>
    post<Connection>("/control/arm", {
      acknowledgement,
      acknowledged_by: acknowledgedBy,
    }),

  // The only other writes the API exposes. Both are human-safety controls that
  // fail toward stopping; neither can place an order or change a risk limit.
  engageKillSwitch: (reason: string) => post<Health>("/control/kill-switch", { reason }),
  releaseKillSwitch: (acknowledgedBy: string) =>
    post<Health>("/control/kill-switch/release", { acknowledged_by: acknowledgedBy }),
  resetBreakers: (acknowledgedBy: string) =>
    post<Health>("/control/breakers/reset", { acknowledged_by: acknowledgedBy }),
};

export { ApiError };
