"""Go-live arming checks. The last gate before real money.

Every check here answers one question: *if this is wrong, does the system lose
money silently?* Anything that fails loudly at trade time is not in this list --
a crash is safe, because it does not trade. What this module hunts for is the
class of defect where the machinery keeps running while being wrong.

Design decisions worth stating:

**Checks are pure and offline.** They read config, the journal, the ledger, and a
snapshot. None of them place an order or need a live session, so `preflight` is
safe to run on a schedule and in CI.

**A check can be BLOCKING or ADVISORY.** Only blocking failures prevent arming.
Advisory ones (paper duration, sample size) are judgement calls a human should
see but should not be silently overridden by a passing aggregate.

**The default answer is no.** `PreflightReport.armed` requires every blocking
check to pass explicitly. A check that errors counts as a failure, because an
inconclusive safety check is not a passing one.

This mirrors the Phase 5 gate in `docs/ROADMAP.md`: the bar is *correctness*, not
profitability. Profit at minimum size is noise; a reconciliation break is not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

from osiris.config import AccountType, Mode, RiskLimits, Settings
from osiris.execution.journal import EventType, Journal
from osiris.logging import get_logger
from osiris.mcp.client import SNAPSHOT_PATH

log = get_logger(__name__)

# Phase 4 gate: "minimum of 4 weeks across at least one earnings cycle".
MIN_PAPER_SESSIONS = 20          # ~4 weeks of trading days
MIN_PAPER_CALENDAR_DAYS = 28
# A quarter is the earnings cycle; 60 sessions spans one comfortably.
EARNINGS_CYCLE_SESSIONS = 60
# Below this the return series cannot support the evaluation gates.
MIN_RETURN_SAMPLES = 30


class Severity(str, Enum):
    BLOCKING = "blocking"
    ADVISORY = "advisory"


@dataclass(frozen=True)
class Check:
    """One preflight question and its verdict."""

    name: str
    passed: bool
    severity: Severity
    detail: str

    @property
    def blocks(self) -> bool:
        return self.severity is Severity.BLOCKING and not self.passed

    def line(self) -> str:
        if self.passed:
            mark = "PASS"
        else:
            mark = "FAIL" if self.severity is Severity.BLOCKING else "WARN"
        return f"[{mark}] {self.name}: {self.detail}"


@dataclass
class PreflightReport:
    checks: list[Check] = field(default_factory=list)
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def blocking_failures(self) -> list[Check]:
        return [c for c in self.checks if c.blocks]

    @property
    def advisories(self) -> list[Check]:
        return [
            c for c in self.checks if not c.passed and c.severity is Severity.ADVISORY
        ]

    @property
    def armed(self) -> bool:
        """True only when every blocking check passed.

        Note this is an assertion about *readiness*, not an instruction to trade.
        Arming still requires the two independent affirmations in `Settings`.
        """
        return not self.blocking_failures and bool(self.checks)

    def describe(self) -> str:
        lines = [c.line() for c in self.checks]
        lines.append("")
        if self.armed:
            lines.append(
                f"CLEARED: {len(self.checks)} checks, "
                f"{len(self.advisories)} advisory warning(s)."
            )
            if self.advisories:
                lines.append(
                    "Advisories are not blocking, but each one is a known gap. "
                    "Read them before committing capital."
                )
        else:
            lines.append(
                f"NOT CLEARED: {len(self.blocking_failures)} blocking failure(s). "
                "Do not commit capital."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "armed": self.armed,
            "evaluated_at": self.evaluated_at.isoformat(),
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "severity": c.severity.value,
                    "detail": c.detail,
                }
                for c in self.checks
            ],
            "blocking_failures": [c.name for c in self.blocking_failures],
            "advisories": [c.name for c in self.advisories],
        }


# --------------------------------------------------------------- config checks
def check_account_type(settings: Settings) -> Check:
    """Cash vs margin governs settlement, turnover, and sizing.

    Trading a cash account as though it were margin produces good-faith
    violations rather than an error: the orders fill, and the restriction lands
    weeks later. That is exactly the silent-wrongness class this gate exists for.
    """
    known = settings.account_type is not AccountType.UNKNOWN
    return Check(
        name="account_type_known",
        passed=known,
        severity=Severity.BLOCKING,
        detail=(
            f"account type {settings.account_type.value}"
            if known
            else "account type UNKNOWN -- complete Phase 0 (docs/PHASE0.md); "
            "cash vs margin governs settlement and sizing"
        ),
    )


def check_explicit_arming(settings: Settings) -> Check:
    """Two independent affirmations, so no single stray env var arms real money."""
    if settings.mode is not Mode.LIVE:
        return Check(
            name="explicit_arming",
            passed=True,
            severity=Severity.BLOCKING,
            detail=f"mode {settings.mode.value}: no live path to arm",
        )
    return Check(
        name="explicit_arming",
        passed=settings.live_armed,
        severity=Severity.BLOCKING,
        detail=(
            "live mode with explicit risk acknowledgement"
            if settings.live_armed
            else "live mode WITHOUT acknowledgement: set OSIRIS_I_UNDERSTAND_THE_RISK=yes"
        ),
    )


def check_minimum_capital(settings: Settings, *, max_usd: float = 5_000.0) -> Check:
    """Phase 5 says *minimum viable capital*, so a large balance is the failure.

    Inverted on purpose. The risk at go-live is not too little size, it is
    starting at a size where a systematic bug is expensive.
    """
    equity = settings.account_equity_usd
    if settings.mode is not Mode.LIVE:
        return Check(
            name="minimum_viable_capital",
            passed=True,
            severity=Severity.ADVISORY,
            detail=f"paper mode, notional equity ${equity:,.0f}",
        )
    ok = 0 < equity <= max_usd
    return Check(
        name="minimum_viable_capital",
        passed=ok,
        severity=Severity.ADVISORY,
        detail=(
            f"${equity:,.0f} is within the ${max_usd:,.0f} first-live ceiling"
            if ok
            else f"${equity:,.0f} exceeds the ${max_usd:,.0f} first-live ceiling: "
            "Phase 5 calls for an amount you would shrug off losing entirely"
        ),
    )


def check_risk_limits_coherent(limits: RiskLimits) -> Check:
    """The limits must not deadlock against each other.

    `RiskLimits` validates this at construction, so this check is a tripwire for
    a future edit that loosens the validator: a book that can never reach full
    investment would look like a broken strategy, not a bad config.
    """
    problems: list[str] = []
    if limits.max_symbol_weight * limits.target_position_count < 1.0:
        problems.append("symbol cap x target count cannot reach 100% invested")
    if limits.max_trade_notional_pct > limits.max_symbol_weight:
        problems.append("per-order cap exceeds the per-symbol cap")
    if limits.daily_loss_halt_pct > limits.max_drawdown_halt_pct:
        problems.append("daily loss halt exceeds drawdown halt")
    if limits.min_position_count > limits.target_position_count:
        problems.append("position floor exceeds target count")
    return Check(
        name="risk_limits_coherent",
        passed=not problems,
        severity=Severity.BLOCKING,
        detail=(
            "; ".join(problems)
            if problems
            else (
                f"per-order {limits.max_trade_notional_pct:.1%}, symbol "
                f"{limits.max_symbol_weight:.0%}, sector {limits.max_sector_weight:.0%}, "
                f"halts {limits.daily_loss_halt_pct:.0%}/{limits.max_drawdown_halt_pct:.0%}"
            )
        ),
    )


# --------------------------------------------------------------- safety checks
def check_kill_switch_clear(killswitch) -> Check:
    """An engaged switch means a human halted this system. Respect it."""
    state = killswitch.check()
    return Check(
        name="kill_switch_clear",
        passed=not state.engaged,
        severity=Severity.BLOCKING,
        detail=(
            f"clear ({killswitch.path})"
            if not state.engaged
            else f"ENGAGED: {state.reason}"
        ),
    )


def check_kill_switch_writable(killswitch) -> Check:
    """A switch you cannot engage under load is not a switch.

    Verified by actually writing and removing the file. Discovering that the
    directory is read-only during an incident is too late, and the failure mode
    is that the operator believes they have halted the system when they have not.
    """
    if killswitch.check().engaged:
        return Check(
            name="kill_switch_writable",
            passed=True,
            severity=Severity.BLOCKING,
            detail="already engaged, so the path is demonstrably writable",
        )
    probe = killswitch.path.parent / f".{killswitch.path.name}.probe"
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text("preflight probe\n")
        probe.unlink()
    except OSError as exc:
        return Check(
            name="kill_switch_writable",
            passed=False,
            severity=Severity.BLOCKING,
            detail=f"cannot write to {probe.parent}: {exc}",
        )
    return Check(
        name="kill_switch_writable",
        passed=True,
        severity=Severity.BLOCKING,
        detail=f"verified writable at {killswitch.path}",
    )


def check_breakers_clear(breakers) -> Check:
    """A tripped breaker requires a human reset, never an automatic one."""
    tripped = list(getattr(breakers, "tripped", ()))
    reasons = list(getattr(breakers, "reasons", ()))
    return Check(
        name="breakers_clear",
        passed=not tripped,
        severity=Severity.BLOCKING,
        detail=(
            "no breakers tripped"
            if not tripped
            else f"TRIPPED: {'; '.join(reasons) or [c.value for c in tripped]}"
        ),
    )


def check_mcp_snapshot(path=SNAPSHOT_PATH) -> Check:
    """A committed tool snapshot is what makes drift detectable.

    Without it the schema-drift breaker has no baseline, so a server-side rename
    surfaces as a failed trade instead of a failed build.
    """
    exists = path.exists()
    if not exists:
        return Check(
            name="mcp_snapshot_present",
            passed=False,
            severity=Severity.BLOCKING,
            detail=(
                f"no snapshot at {path}: run Phase 0 enumeration so schema drift "
                "is detectable before it reaches a trade"
            ),
        )
    import json

    try:
        raw = json.loads(path.read_text())
        count = int(raw.get("tool_count", 0))
        captured = raw.get("captured_at", "unknown")
    except (OSError, ValueError) as exc:
        return Check(
            name="mcp_snapshot_present",
            passed=False,
            severity=Severity.BLOCKING,
            detail=f"snapshot unreadable: {exc}",
        )
    return Check(
        name="mcp_snapshot_present",
        passed=count > 0,
        severity=Severity.BLOCKING,
        detail=(
            f"{count} tools captured {captured}"
            if count > 0
            else "snapshot contains zero tools"
        ),
    )


# -------------------------------------------------------------- journal checks
def check_no_reconciliation_breaks(journal: Journal) -> Check:
    """The Phase 5 gate. Any break means the ledger and the venue disagreed.

    Blocking without exception. A single unexplained break means every
    subsequent sizing decision was computed against a book that may not exist.
    """
    counts = journal.counts()
    breaks = counts.get(EventType.RECONCILIATION_BREAK.value, 0)
    clean = counts.get(EventType.RECONCILIATION.value, 0)
    if breaks:
        return Check(
            name="zero_reconciliation_breaks",
            passed=False,
            severity=Severity.BLOCKING,
            detail=(
                f"{breaks} reconciliation break(s) recorded. Each one means the "
                "ledger and the broker disagreed; resolve every one before arming."
            ),
        )
    return Check(
        name="zero_reconciliation_breaks",
        passed=clean > 0,
        severity=Severity.BLOCKING,
        detail=(
            f"{clean} clean reconciliations, zero breaks"
            if clean
            else "no reconciliations recorded at all: the loop has never verified "
            "its book against a broker"
        ),
    )


def check_no_kernel_bypasses(journal: Journal) -> Check:
    """Every placement must be preceded by a review. Placements > reviews is a bypass.

    This is the structural invariant of the whole design. If it can be violated,
    the kernel is advisory rather than mandatory, and nothing else in this report
    means anything.
    """
    counts = journal.counts()
    placed = counts.get(EventType.ORDER_PLACED.value, 0)
    reviewed = counts.get(EventType.REVIEW_PASSED.value, 0)
    ok = placed <= reviewed
    return Check(
        name="zero_kernel_bypasses",
        passed=ok,
        severity=Severity.BLOCKING,
        detail=(
            f"{placed} placements against {reviewed} passed reviews"
            if ok
            else f"BYPASS: {placed} placements but only {reviewed} reviews -- "
            f"{placed - reviewed} order(s) reached the venue unsimulated"
        ),
    )


def check_paper_duration(journal: Journal) -> Check:
    """Phase 4: minimum 4 weeks of paper trading.

    Measured in both sessions and calendar span, because 20 cycles replayed in
    one afternoon is not four weeks of market conditions.
    """
    cycles = journal.read(event=EventType.CYCLE_END)
    ran = [
        c for c in cycles if not c.payload.get("skipped") and c.payload.get("as_of")
    ]
    if not ran:
        return Check(
            name="paper_duration",
            passed=False,
            severity=Severity.ADVISORY,
            detail="no completed sessions recorded in the journal",
        )

    dates = sorted({c.payload["as_of"][:10] for c in ran})
    sessions = len(dates)
    span = (
        datetime.fromisoformat(dates[-1]) - datetime.fromisoformat(dates[0])
    ).days + 1

    enough = sessions >= MIN_PAPER_SESSIONS and span >= MIN_PAPER_CALENDAR_DAYS
    earnings = sessions >= EARNINGS_CYCLE_SESSIONS
    detail = f"{sessions} sessions over {span} calendar days"
    if not enough:
        detail += (
            f" -- need >={MIN_PAPER_SESSIONS} sessions and "
            f">={MIN_PAPER_CALENDAR_DAYS} days"
        )
    elif not earnings:
        detail += (
            f" -- spans <{EARNINGS_CYCLE_SESSIONS} sessions, so it may not cover "
            "a full earnings cycle"
        )
    return Check(
        name="paper_duration",
        passed=enough and earnings,
        severity=Severity.ADVISORY,
        detail=detail,
    )


def check_vetoes_are_visible(journal: Journal) -> Check:
    """A kernel blocking everything looks identical to a quiet market.

    Zero vetoes over a long run is suspicious rather than reassuring: it usually
    means the gates are not wired to the intents at all. All-vetoes is the other
    failure, and it is the one that produces a system that appears to run.
    """
    counts = journal.counts()
    vetoes = counts.get(EventType.KERNEL_VETO.value, 0)
    intents = counts.get(EventType.INTENT_EMITTED.value, 0)
    fills = counts.get(EventType.FILL.value, 0)

    if intents == 0:
        return Check(
            name="veto_visibility",
            passed=False,
            severity=Severity.BLOCKING,
            detail="no intents were ever emitted: the planner produced nothing",
        )
    if fills == 0:
        return Check(
            name="veto_visibility",
            passed=False,
            severity=Severity.BLOCKING,
            detail=(
                f"{intents} intents and {vetoes} vetoes but ZERO fills: the planner "
                "and the kernel disagree on every order"
            ),
        )
    rate = vetoes / intents
    return Check(
        name="veto_visibility",
        passed=True,
        severity=Severity.ADVISORY if rate < 0.9 else Severity.BLOCKING,
        detail=(
            f"{vetoes}/{intents} intents vetoed ({rate:.0%}), {fills} fills; "
            f"top causes: {list(journal.veto_summary())[:3]}"
        ),
    )


# ----------------------------------------------------------- evaluation checks
def check_evaluation_gates(evaluation: dict | None) -> Check:
    """The four gates from `osiris.eval.gates`. All must pass before capital.

    Blocking, because this is the only evidence that the strategy has an edge at
    all. If it cannot be produced, the honest outcome is an index fund.
    """
    if not evaluation:
        return Check(
            name="evaluation_gates",
            passed=False,
            severity=Severity.BLOCKING,
            detail="no evaluation has been run: the edge is unfalsified",
        )
    gates = evaluation.get("gates") or []
    if not gates:
        return Check(
            name="evaluation_gates",
            passed=False,
            severity=Severity.BLOCKING,
            detail="evaluation present but contains no gate results",
        )

    # A MISSING gate is as disqualifying as a failing one. Checking only that
    # the present gates passed means a harness that silently stopped emitting
    # the Monte Carlo test would clear this check -- the bar would drop without
    # anything reporting a failure. `funnel_fidelity` is excluded because it is
    # only meaningful when the LLM funnel produced the ranking; there is no
    # pre-rank to be faithful to when a deterministic ranker is in use.
    required = {"monte_carlo", "factor_attribution", "cost_sensitivity"}
    present = {g["name"] for g in gates}
    missing = sorted(required - present)
    failed = [g["name"] for g in gates if not g.get("passed")]

    problems: list[str] = []
    if missing:
        problems.append(f"never ran: {', '.join(missing)}")
    if failed:
        problems.append(f"failed: {', '.join(failed)}")

    return Check(
        name="evaluation_gates",
        passed=not problems,
        severity=Severity.BLOCKING,
        detail=(
            f"all {len(gates)} required gates passed "
            f"(MC percentile {evaluation.get('monte_carlo_percentile', 0):.1%}, "
            f"sharpe {evaluation.get('sharpe', 0):.2f})"
            if not problems
            else "; ".join(problems)
        ),
    )


def check_return_sample_size(daily_returns: list[float]) -> Check:
    """Gates computed on a handful of points are theatre.

    A Sharpe from 10 observations has a confidence interval wide enough to
    include zero, so a passing gate on that sample is not evidence.
    """
    n = len(daily_returns)
    return Check(
        name="return_sample_size",
        passed=n >= MIN_RETURN_SAMPLES,
        severity=Severity.ADVISORY,
        detail=(
            f"{n} daily observations"
            if n >= MIN_RETURN_SAMPLES
            else f"only {n} observations; gates need >={MIN_RETURN_SAMPLES} to mean anything"
        ),
    )


def check_no_fabricated_returns(daily_returns: list[float]) -> Check:
    """A -100% day on a long-only book is an accounting artifact, not a loss.

    It arises from a zero equity mark on a skipped session, and it silently
    destroys every downstream metric. Blocking because the corruption is
    invisible in aggregate statistics -- they simply become wrong.
    """
    bad = [r for r in daily_returns if r <= -0.9 or r != r]
    return Check(
        name="no_fabricated_returns",
        passed=not bad,
        severity=Severity.BLOCKING,
        detail=(
            f"{len(daily_returns)} returns, none implausible"
            if not bad
            else f"{len(bad)} implausible return(s) (e.g. {bad[0]:.4f}): a zero "
            "equity mark has corrupted the series"
        ),
    )


def check_paper_matches_backtest(
    paper_returns: list[float],
    backtest_returns: list[float] | None,
    *,
    max_sharpe_gap: float = 1.0,
) -> Check:
    """Phase 4 gate: paper must not diverge wildly from backtest.

    If backtest is great and paper is bad, the backtest is wrong -- it is almost
    always lookahead or unmodeled cost. Advisory rather than blocking only
    because a missing backtest is a known gap rather than evidence of a defect.
    """
    from osiris.eval.metrics import sharpe, to_array

    if not backtest_returns:
        return Check(
            name="paper_matches_backtest",
            passed=False,
            severity=Severity.ADVISORY,
            detail="no backtest returns supplied for comparison",
        )
    if len(paper_returns) < 3:
        return Check(
            name="paper_matches_backtest",
            passed=False,
            severity=Severity.ADVISORY,
            detail=f"only {len(paper_returns)} paper observations to compare",
        )

    ps = sharpe(to_array(paper_returns))
    bs = sharpe(to_array(backtest_returns))
    gap = abs(ps - bs)
    ok = gap <= max_sharpe_gap
    return Check(
        name="paper_matches_backtest",
        passed=ok,
        severity=Severity.ADVISORY,
        detail=(
            f"paper sharpe {ps:.2f} vs backtest {bs:.2f} (gap {gap:.2f})"
            + (
                ""
                if ok
                else f" exceeds {max_sharpe_gap:.2f}: if backtest is much better, "
                "suspect lookahead or unmodeled cost in the backtest"
            )
        ),
    )


# ------------------------------------------------------------------- aggregate
def run_preflight(
    state,
    *,
    backtest_returns: list[float] | None = None,
    snapshot_path=SNAPSHOT_PATH,
) -> PreflightReport:
    """Run every arming check against live runtime state.

    A check that raises is recorded as a BLOCKING failure rather than being
    allowed to propagate. An inconclusive safety check is not a passing one, and
    a preflight that crashes should not be indistinguishable from one that was
    never run.
    """
    checks: list[Check] = []

    def guarded(name: str, fn: Callable[[], Check]) -> None:
        try:
            checks.append(fn())
        # Any failure is a failed check, deliberately including unexpected ones.
        except Exception as exc:
            log.error("preflight.check_errored", check=name, error=str(exc))
            checks.append(
                Check(
                    name=name,
                    passed=False,
                    severity=Severity.BLOCKING,
                    detail=f"check raised {type(exc).__name__}: {exc}",
                )
            )

    guarded("account_type_known", lambda: check_account_type(state.settings))
    guarded("explicit_arming", lambda: check_explicit_arming(state.settings))
    guarded("minimum_viable_capital", lambda: check_minimum_capital(state.settings))
    guarded("risk_limits_coherent", lambda: check_risk_limits_coherent(state.limits))
    guarded("kill_switch_clear", lambda: check_kill_switch_clear(state.killswitch))
    guarded("kill_switch_writable", lambda: check_kill_switch_writable(state.killswitch))
    guarded("breakers_clear", lambda: check_breakers_clear(state.breakers))
    guarded("mcp_snapshot_present", lambda: check_mcp_snapshot(snapshot_path))
    guarded(
        "zero_reconciliation_breaks",
        lambda: check_no_reconciliation_breaks(state.journal),
    )
    guarded("zero_kernel_bypasses", lambda: check_no_kernel_bypasses(state.journal))
    guarded("paper_duration", lambda: check_paper_duration(state.journal))
    guarded("veto_visibility", lambda: check_vetoes_are_visible(state.journal))
    guarded("evaluation_gates", lambda: check_evaluation_gates(state.evaluation))
    guarded("return_sample_size", lambda: check_return_sample_size(state.daily_returns))
    guarded(
        "no_fabricated_returns",
        lambda: check_no_fabricated_returns(state.daily_returns),
    )
    guarded(
        "paper_matches_backtest",
        lambda: check_paper_matches_backtest(state.daily_returns, backtest_returns),
    )

    report = PreflightReport(checks=checks)
    log.info(
        "preflight.complete",
        armed=report.armed,
        blocking=len(report.blocking_failures),
        advisory=len(report.advisories),
    )
    return report
