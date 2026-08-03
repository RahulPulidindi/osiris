/** Mirrors osiris/api/schemas.py. Kept hand-written and narrow on purpose:
 *  a generated client would drag in the whole OpenAPI surface for six views. */

export interface Health {
  status: string;
  mode: "paper" | "live";
  armed: boolean;
  account_type: string;
  broker: string;
  kill_switch_engaged: boolean;
  breakers_tripped: string[];
  subscribers: number;
  version: string;
}

export type ActivityKind = "bought" | "sold" | "blocked" | "halted" | "note";

/** One thing the agent did, with its reason attached. */
export interface Activity {
  seq: number;
  ts: string;
  kind: ActivityKind;
  symbol: string;
  headline: string;
  reason: string;
  detail: string;
  notional_usd: number | null;
  quantity: number | null;
  price: number | null;
  correlation_id: string;
}

/** Robinhood link + arming state, from /api/connection. */
export interface Connection {
  robinhood_linked: boolean;
  broker: string;
  mode: string;
  armed: boolean;
  risk_acknowledged: boolean;
  restart_required: boolean;
  connect_command: string;
}

export interface PreflightCheck {
  name: string;
  passed: boolean;
  severity: "blocking" | "advisory";
  detail: string;
}

/** Go-live readiness. `armed` means *cleared to arm*, not armed. */
export interface Preflight {
  armed: boolean;
  evaluated_at: string;
  checks: PreflightCheck[];
  blocking_failures: string[];
  advisories: string[];
}

export interface Position {
  symbol: string;
  quantity: number;
  avg_cost: number;
  last_price: number;
  market_value: number;
  weight: number;
  unrealized_pnl: number;
  unrealized_pnl_pct: number;
  realized_pnl: number;
  sector: string;
  beta: number;
  thesis: string;
  invalidation: string;
  opened_at: string | null;
}

export interface Portfolio {
  equity: number;
  cash: number;
  buying_power: number;
  gross_exposure: number;
  net_exposure_pct: number;
  position_count: number;
  portfolio_beta: number;
  realized_pnl: number;
  unrealized_pnl: number;
  daily_pnl: number;
  daily_pnl_pct: number;
  drawdown_pct: number;
  peak_equity: number;
  positions: Position[];
  sector_weights: Record<string, number>;
  as_of: string;
}

export interface RankingRow {
  symbol: string;
  rank: number | null;
  score: number;
  conviction: number;
  stage: number;
  sector: string;
  last_price: number;
  change_pct: number;
  target_weight: number;
  held_weight: number;
  thesis: string;
  invalidation: string;
  red_team_verdict: string;
  sparkline: number[];
  sources: string[];
}

export interface EquityPoint {
  date: string;
  equity: number;
  benchmark: number | null;
  drawdown: number;
}

export interface Breaker {
  name: string;
  tripped: boolean;
  value: number;
  threshold: number;
  detail: string;
}

export interface FactorExposure {
  alpha_annualized: number;
  alpha_t_stat: number;
  market_beta: number;
  loadings: Record<string, number>;
  r_squared: number;
  n_periods: number;
  verdict: string;
  is_significant: boolean;
}

export interface SectorDeviation {
  sector: string;
  portfolio_weight: number;
  benchmark_weight: number;
  deviation: number;
  within_band: boolean;
}

export interface Attribution {
  selection: number;
  allocation: number;
  interaction: number;
  total_excess: number;
  selection_share: number;
  verdict: string;
  by_sector: Record<string, Record<string, number>>;
  realized_slippage_bps: number;
  modeled_slippage_bps: number;
  slippage_excess_bps: number;
  slippage_degrading: boolean;
}

export interface Gate {
  name: string;
  passed: boolean;
  statistic: number;
  detail: string;
}

export interface Evaluation {
  gates: Gate[];
  all_passed: boolean;
  verdict: string;
  sharpe: number;
  deflated_sharpe: number;
  sortino: number;
  max_drawdown: number;
  total_return: number;
  after_tax_return: number;
  cagr: number;
  win_rate: number;
  monte_carlo_percentile: number;
  monte_carlo_p_value: number;
  null_distribution: number[];
  observed_return: number;
  walk_forward: Array<Record<string, number>>;
  equity_curve: EquityPoint[];
  funnel_fidelity: number;
  cost_sensitivity: Record<string, number>;
}

export interface JournalEntry {
  seq: number;
  ts: string;
  event: string;
  correlation_id: string;
  payload: Record<string, unknown>;
}

export interface FunnelStage {
  stage: number;
  name: string;
  count: number;
  cost_usd: number;
  description: string;
}

/** SSE event names. Must stay in sync with osiris/api/events.py Channel. */
export type Channel =
  | "heartbeat"
  | "cycle"
  | "agent"
  | "fill"
  | "intent"
  | "veto"
  | "pnl"
  | "breaker"
  | "ranking"
  | "reconciliation"
  | "error";

export interface StreamEvent {
  seq: number;
  ts: string;
  [key: string]: unknown;
}

/** One line in the live agent reasoning log. */
export interface AgentLine {
  seq: number;
  ts: string;
  channel: Channel;
  role: string;
  text: string;
  symbol?: string;
}
