"""Regressions from the first full live cognition run.

Every bug here produced a *plausible-looking* outcome rather than an error, which
is what made them expensive:

  - Token budgets too small: 17 of 20 analyses truncated mid-JSON and discarded,
    so the strategist scored on a third of the evidence and nobody could tell.
  - `final_count` hardcoded to 20 while the account supported 5.
  - `StrategistScore.stage` / `.sources` do not exist: the run crashed AFTER a
    complete successful cycle, throwing away the work at the display layer.
  - Exa 429s dropped rather than retried, silently degrading theses to price-only.
  - Duplicated ticker in search queries ("PEP PEP earnings").
"""

from __future__ import annotations

import pytest

from osiris.cognition.roles import _looks_truncated
from osiris.cognition.schemas import StrategistScore


class TestTruncationDetection:
    """Truncation and malformed JSON need opposite fixes.

    Both raise JSONDecodeError, so without distinguishing them the logs point at
    neither cause -- more token budget versus a better prompt.
    """

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('{"a": 1}', False),
            ('[{"a": 1}]', False),
            ('{"a": 1, "b": [2, 3', True),
            ('{"symbol": "AAPL", "thesis": "some long text', True),
            ("", False),
            ("not json at all", False),
        ],
    )
    def test_identifies_unbalanced_structures(self, text, expected):
        assert _looks_truncated(text) is expected

    def test_a_real_truncated_analyst_response_is_detected(self):
        """Shaped like the responses that failed at char ~4000."""
        text = '{"symbol": "PEP", "summary": "' + "x" * 3900 + '", "evidence": [{"k"'

        assert _looks_truncated(text)


class TestTruncatedResponseSalvage:
    """A response cut off at the token limit is usually 19 good records and a
    partial 20th. Raising discarded all 19, which is how a complete research
    cycle produced zero targets while every step reported success.

    Salvage only ever drops a trailing incomplete element. It never repairs or
    invents a field, because a half-read thesis attached to a real ticker is more
    dangerous than a missing one -- it would flow into scoring looking complete.
    """

    def salvage(self, body: str):
        from osiris.cognition.llm import _salvage_json

        return _salvage_json(body)

    def test_recovers_complete_records_from_a_bare_array(self):
        body = '[{"symbol":"A","score":1},{"symbol":"B","score":2},{"symbol":"C","sco'
        out = self.salvage(body)

        assert out == [{"symbol": "A", "score": 1}, {"symbol": "B", "score": 2}]

    def test_recovers_from_a_wrapped_array(self):
        body = '{"scores":[{"symbol":"A","score":1},{"symbol":"B","score":2},{"sym'
        out = self.salvage(body)

        assert out == {"scores": [{"symbol": "A", "score": 1}, {"symbol": "B", "score": 2}]}

    def test_brackets_inside_strings_do_not_confuse_depth_tracking(self):
        body = '[{"symbol":"A","thesis":"up [strong] move"},{"symbol":"B","the'
        out = self.salvage(body)

        assert out == [{"symbol": "A", "thesis": "up [strong] move"}]

    def test_escaped_quotes_are_handled(self):
        body = '[{"symbol":"A","thesis":"said \\"buy\\" loudly"},{"sym'
        out = self.salvage(body)

        assert out[0]["thesis"] == 'said "buy" loudly'

    def test_returns_none_when_no_record_completed(self):
        """Nothing salvageable must not become an empty success."""
        assert self.salvage('[{"symbol":"A","sc') is None

    def test_never_invents_a_missing_field(self):
        """The partial record is dropped entirely, not completed with defaults."""
        body = '[{"symbol":"A","score":1,"thesis":"real"},{"symbol":"B","score":2,"thes'
        out = self.salvage(body)

        assert len(out) == 1
        assert out[0]["symbol"] == "A"

    def test_parse_json_recovers_instead_of_raising(self):
        from osiris.cognition.llm import LLMResponse, Role

        resp = LLMResponse(
            role=Role.STRATEGIST,
            model="test",
            content='{"scores":[{"symbol":"A","score":1},{"symbol":"B","sc',
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )

        assert resp.parse_json() == {"scores": [{"symbol": "A", "score": 1}]}

    def test_valid_json_is_unaffected(self):
        from osiris.cognition.llm import LLMResponse, Role

        resp = LLMResponse(
            role=Role.STRATEGIST,
            model="test",
            content='{"scores":[{"symbol":"A","score":1}]}',
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )

        assert resp.parse_json() == {"scores": [{"symbol": "A", "score": 1}]}

    def test_genuinely_malformed_json_still_raises(self):
        """Salvage must not mask a real prompt or schema problem."""
        import json

        from osiris.cognition.llm import LLMResponse, Role

        resp = LLMResponse(
            role=Role.STRATEGIST,
            model="test",
            content="{this is not json at all}",
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
        )
        with pytest.raises((json.JSONDecodeError, ValueError)):
            resp.parse_json()


class TestRankingFailureVisibility:
    """A failed ranking and a quiet market must not produce the same summary.

    The cycle correctly continued (exits must run) but reported
    "0 fills, 0 vetoed, 0 rejected", which reads as a market with no
    opportunities rather than as broken research.
    """

    def test_the_result_carries_the_error(self):
        from datetime import date

        from osiris.execution.loop import CycleResult

        result = CycleResult(
            correlation_id="x",
            as_of=date(2026, 8, 3),
            ran=True,
            equity=1000.0,
            ranking_error="Expecting ',' delimiter",
        )

        assert "RANKING FAILED" in result.summary()

    def test_a_clean_cycle_says_nothing_about_ranking(self):
        from datetime import date

        from osiris.execution.loop import CycleResult

        result = CycleResult(
            correlation_id="x", as_of=date(2026, 8, 3), ran=True, equity=1000.0
        )

        assert "RANKING FAILED" not in result.summary()


class TestPromptFieldTrimming:
    def test_long_fields_are_bounded(self):
        from osiris.cognition.roles import _trim

        out = _trim("x" * 2000)

        assert len(out) < 700
        assert "trimmed" in out

    def test_short_fields_pass_through_unchanged(self):
        from osiris.cognition.roles import _trim

        assert _trim("short text") == "short text"

    def test_trimming_is_marked_not_silent(self):
        """The model must see that evidence was elided.

        Otherwise a cut-off sentence reads as the complete picture.
        """
        from osiris.cognition.roles import _trim

        assert _trim("y" * 1000).endswith("…[trimmed]")


class TestWeightNormalization:
    """Scaling down is safe; scaling up would override a judgement.

    I initially read the first live plan's 0.82 sum as a normalization bug and
    made it scale up. Reading the journal disproved that: the PM's own notes said
    "thin evidence base... red team reduction flags on top 4 names... prudent risk
    management." Holding 18% cash was the decision, not an accident.

    So under-allocation is surfaced, never corrected. Over-allocation is still
    scaled down, because a plan summing above 1.0 cannot be executed.
    """

    def plan(self, weights: list[float]):
        from osiris.cognition.schemas import PortfolioPlan, TargetHolding

        return PortfolioPlan(
            holdings=[
                TargetHolding(symbol=f"S{i}", target_weight=w, rationale="r")
                for i, w in enumerate(weights)
            ]
        )

    def invested(
        self, weights: list[float], target: float = 0.97, scale_up: bool = False
    ) -> float:
        return sum(
            h.target_weight
            for h in self.plan(weights).normalized(
                target_invested=target, scale_up=scale_up
            )
        )

    def test_a_deliberate_cash_holding_is_preserved(self):
        """The exact live case: 0.82 was a conviction judgement, not an error.

        Forcing it to 0.97 would discard the reasoning the funnel exists to
        produce.
        """
        assert self.invested([0.18, 0.16, 0.18, 0.16, 0.14]) == pytest.approx(0.82)

    def test_scaling_up_is_available_but_opt_in(self):
        assert self.invested(
            [0.18, 0.16, 0.18, 0.16, 0.14], scale_up=True
        ) == pytest.approx(0.97)

    def test_the_loop_does_not_scale_up(self):
        import inspect

        from osiris.execution.loop import DailyLoop

        source = inspect.getsource(DailyLoop.run_cycle)
        assert "scale_up=False" in source

    def test_an_under_allocated_plan_is_surfaced(self):
        """Silence would make a cash drag invisible in every metric."""
        import inspect

        from osiris.execution.loop import DailyLoop

        source = inspect.getsource(DailyLoop.run_cycle)
        assert "loop.plan_holds_cash" in source

    def test_an_over_allocated_plan_is_still_scaled_down(self):
        assert self.invested([0.30, 0.30, 0.30, 0.20, 0.20]) == pytest.approx(0.97)

    def test_relative_proportions_are_preserved(self):
        """Scaling must not change the PM's relative conviction."""
        from osiris.cognition.schemas import PortfolioPlan, TargetHolding

        plan = PortfolioPlan(
            holdings=[
                TargetHolding(symbol="A", target_weight=0.40, rationale="r"),
                TargetHolding(symbol="B", target_weight=0.20, rationale="r"),
            ]
        )
        out = {h.symbol: h.target_weight for h in plan.normalized(target_invested=0.9)}

        assert out["A"] / out["B"] == pytest.approx(2.0)

    def test_a_nearly_correct_plan_is_left_alone(self):
        """Rescaling by a couple of percent is churn for no benefit."""
        weights = [0.20, 0.20, 0.19, 0.19, 0.19]
        out = self.plan(weights).normalized(target_invested=0.97)

        assert [h.target_weight for h in out] == weights

    def test_an_empty_plan_is_safe(self):
        assert self.plan([]).normalized(target_invested=0.97) == []

    def test_zero_weights_do_not_divide_by_zero(self):
        assert self.plan([0.0, 0.0]).normalized(target_invested=0.97) is not None

    def test_the_loop_reserves_cash_for_slippage(self):
        """A 100% target leaves nothing for spread and drift between sizing and
        fill, so the last order in the batch fails on buying power."""
        import inspect

        from osiris.execution.loop import DailyLoop

        source = inspect.getsource(DailyLoop.run_cycle)
        assert "target_invested=0.97" in source


class TestUnreviewedNamesFailClosed:
    """The most consequential regression in this file.

    On the first full live run the red team's response truncated after 4 of 9
    reviews. The 5 names it never assessed -- GE, JPM, LIN, MRK, V -- became the
    ENTIRE book, because `construct` excluded only explicitly vetoed symbols.

    Every position had bypassed adversarial review while the log cheerfully
    reported `red_team.vetoed`, so the safety check appeared to be working. The
    distinction that matters: "no objection was raised" is not "no objection
    exists."
    """

    def scores(self, symbols_and_scores: list[tuple[str, float]]):
        return [
            StrategistScore(
                symbol=s, score=v, conviction=0.6, thesis="t", invalidation="i"
            )
            for s, v in symbols_and_scores
        ]

    def reviews(self, verdicts: list[tuple[str, str]]):
        from osiris.cognition.schemas import RedTeamReview

        return [
            RedTeamReview(symbol=s, verdict=v, bear_case="b", key_risk="r")
            for s, v in verdicts
        ]

    def eligible_symbols(self, scores, reviews) -> set[str]:
        """Mirror of `construct`'s eligibility rule, without an LLM call."""
        from osiris.cognition.roles import RED_TEAM_THRESHOLD

        vetoed = {r.symbol for r in reviews if r.is_veto}
        reviewed = {r.symbol for r in reviews}
        return {
            s.symbol
            for s in scores
            if s.symbol not in vetoed
            and s.score > 0
            and not (s.score > RED_TEAM_THRESHOLD and s.symbol not in reviewed)
        }

    def test_an_unreviewed_high_scorer_is_excluded(self):
        scores = self.scores([("AAPL", 3.0), ("MSFT", 3.0)])
        reviews = self.reviews([("AAPL", "pass")])

        assert self.eligible_symbols(scores, reviews) == {"AAPL"}

    def test_the_exact_live_failure_produces_an_empty_book(self):
        """4 of 9 reviewed; the 5 unreviewed must NOT become the book."""
        reviewed_names = ["AMD", "CSCO", "AAPL", "MSFT"]
        unreviewed_names = ["GE", "JPM", "LIN", "MRK", "V"]
        scores = self.scores([(s, 3.0) for s in reviewed_names + unreviewed_names])
        reviews = self.reviews(
            [("AMD", "veto"), ("CSCO", "veto"), ("AAPL", "pass"), ("MSFT", "pass")]
        )

        eligible = self.eligible_symbols(scores, reviews)
        assert eligible == {"AAPL", "MSFT"}
        for name in unreviewed_names:
            assert name not in eligible

    def test_explicit_vetoes_still_exclude(self):
        scores = self.scores([("AAPL", 3.0)])
        reviews = self.reviews([("AAPL", "veto")])

        assert self.eligible_symbols(scores, reviews) == set()

    def test_low_scorers_do_not_require_a_review(self):
        """Below the review threshold, absence of a review is expected.

        Requiring one would empty the book for a reason unrelated to safety.
        """
        from osiris.cognition.roles import RED_TEAM_THRESHOLD

        scores = self.scores([("AAPL", RED_TEAM_THRESHOLD - 0.5)])

        assert self.eligible_symbols(scores, []) == {"AAPL"}

    def test_the_threshold_is_shared_between_the_two_stages(self):
        """If red_team and construct disagree, names slip through or vanish."""
        import inspect

        from osiris.cognition.roles import CognitionPipeline

        red_team_src = inspect.getsource(CognitionPipeline.red_team)
        construct_src = inspect.getsource(CognitionPipeline.construct)
        assert "RED_TEAM_THRESHOLD" in red_team_src
        assert "RED_TEAM_THRESHOLD" in construct_src

    def test_exclusions_are_logged(self):
        import inspect

        from osiris.cognition.roles import CognitionPipeline

        source = inspect.getsource(CognitionPipeline.construct)
        assert "pm.excluded_unreviewed" in source

    def test_the_red_team_batch_is_small_enough_to_survive(self):
        """A reasoning model spends tokens thinking before emitting anything."""
        from osiris.cognition.roles import RED_TEAM_BATCH, STRATEGIST_BATCH

        assert RED_TEAM_BATCH <= 5
        assert RED_TEAM_BATCH < STRATEGIST_BATCH


class TestStageBudgetScaling:
    """Funnel width must follow the book the kernel allows.

    Researching 40 names for a 5-name book is not merely wasteful: the strategist
    and red team emit one record per candidate, so an oversized funnel truncates
    them. That produced failures at successive stages, each looking like a new bug
    while sharing one cause.
    """

    def test_a_small_book_narrows_the_funnel(self):
        from osiris.cognition.funnel import StageBudget

        budget = StageBudget.for_book(5)

        assert budget.final_count == 5
        assert budget.deep_width == 20

    def test_selection_breadth_is_preserved(self):
        """A narrow funnel must still choose from several times the target."""
        from osiris.cognition.funnel import StageBudget

        for target in (5, 8, 10):
            budget = StageBudget.for_book(target)
            assert budget.deep_width >= target * 3

    def test_a_full_size_book_keeps_the_original_width(self):
        from osiris.cognition.funnel import StageBudget

        assert StageBudget.for_book(20).deep_width == 40

    def test_a_tiny_book_still_researches_a_floor(self):
        """One name should not mean researching only four."""
        from osiris.cognition.funnel import StageBudget

        assert StageBudget.for_book(1).deep_width >= 10

    def test_deep_width_never_exceeds_the_cap(self):
        from osiris.cognition.funnel import StageBudget

        assert StageBudget.for_book(100).deep_width == 40


class TestOutputBoundedByBatching:
    """Roles whose OUTPUT scales with input must be batched.

    Otherwise widening the funnel silently truncates them, and a red team whose
    reviews are discarded is worse than none: the vetoes vanish while the theses
    survive, so the book looks reviewed and is not.
    """

    def test_batch_sizes_are_defined(self):
        from osiris.cognition.roles import RED_TEAM_BATCH, STRATEGIST_BATCH

        assert 0 < STRATEGIST_BATCH <= 20
        assert 0 < RED_TEAM_BATCH <= 20

    def test_strategist_batches_above_the_threshold(self):
        import inspect

        from osiris.cognition.roles import CognitionPipeline

        source = inspect.getsource(CognitionPipeline.score)
        assert "STRATEGIST_BATCH" in source
        assert "asyncio.gather" in source

    def test_red_team_batches(self):
        import inspect

        from osiris.cognition.roles import CognitionPipeline

        source = inspect.getsource(CognitionPipeline.red_team)
        assert "RED_TEAM_BATCH" in source

    def test_unreviewed_names_are_surfaced(self):
        """A lost red-team batch must not read as approval."""
        import inspect

        from osiris.cognition.roles import CognitionPipeline

        source = inspect.getsource(CognitionPipeline.red_team)
        assert "red_team.unreviewed" in source
        assert "red_team.batch_failed" in source


class TestEmptyCompletionHandling:
    """An empty response is a budget problem, not a schema problem.

    `no JSON found in response: ` with nothing after the colon pointed at the
    parser rather than the token ceiling.
    """

    def test_empty_content_raises_a_named_error(self):
        from osiris.cognition.llm import EmptyCompletion, LLMResponse, Role

        resp = LLMResponse(
            role=Role.RED_TEAM,
            model="test",
            content="   ",
            input_tokens=100,
            output_tokens=0,
            cost_usd=0.0,
        )
        with pytest.raises(EmptyCompletion, match="empty completion"):
            resp.parse_json()

    def test_the_error_names_the_role_and_token_count(self):
        from osiris.cognition.llm import EmptyCompletion, LLMResponse, Role

        resp = LLMResponse(
            role=Role.RED_TEAM,
            model="test",
            content="",
            input_tokens=100,
            output_tokens=0,
            cost_usd=0.0,
        )
        try:
            resp.parse_json()
        except EmptyCompletion as exc:
            assert "red_team" in str(exc)
            assert "budget" in str(exc)

    def test_it_remains_catchable_as_a_value_error(self):
        """Existing handlers catch ValueError; this must not escape them."""
        from osiris.cognition.llm import EmptyCompletion

        assert issubclass(EmptyCompletion, ValueError)

    def test_non_json_text_names_the_role_and_length(self):
        from osiris.cognition.llm import LLMResponse, Role

        resp = LLMResponse(
            role=Role.STRATEGIST,
            model="test",
            content="I cannot complete this request.",
            input_tokens=10,
            output_tokens=8,
            cost_usd=0.0,
        )
        with pytest.raises(ValueError, match="strategist"):
            resp.parse_json()

    def test_empty_completions_are_retried_with_more_headroom(self):
        import inspect

        from osiris.cognition.llm import LLMClient

        source = inspect.getsource(LLMClient.complete)
        assert "llm.empty_completion_retry" in source
        assert "* 2" in source


class TestTokenBudgets:
    """Budgets must fit the output the schema asks for."""

    def test_analyst_budget_exceeds_observed_response_size(self):
        import inspect

        from osiris.cognition.roles import CognitionPipeline

        source = inspect.getsource(CognitionPipeline.analyze)
        # Observed truncation at ~1,363 tokens against a 1,500 limit.
        assert "max_tokens=4_000" in source

    def test_triage_budget_scales_to_a_full_batch(self):
        import inspect

        from osiris.cognition.roles import CognitionPipeline

        source = inspect.getsource(CognitionPipeline._triage_batch)
        # 40 symbols x ~100 tokens each cannot fit in 1,500.
        assert "max_tokens=6_000" in source


class TestStrategistScoreContract:
    """The display layer must read fields that exist."""

    def test_has_no_stage_attribute(self):
        """Reaching the strategist IS stage 3; there is no per-score stage."""
        assert "stage" not in StrategistScore.model_fields

    def test_citations_not_sources(self):
        assert "citations" in StrategistScore.model_fields
        assert "sources" not in StrategistScore.model_fields

    def test_the_publish_path_only_reads_real_fields(self):
        """Guards the exact crash: a complete cycle discarded at the last step."""
        import inspect

        from osiris.run import LiveAgent

        source = inspect.getsource(LiveAgent._publish)
        assert "s.stage" not in source
        assert "s.sources" not in source
        assert "s.citations" in source

    def test_a_score_can_be_projected_without_error(self):
        score = StrategistScore(
            symbol="AAPL",
            score=2.0,
            conviction=0.7,
            thesis="t",
            invalidation="i",
        )
        row = {
            "symbol": score.symbol,
            "score": score.score,
            "conviction": score.conviction,
            "stage": 3,
            "thesis": score.thesis,
            "invalidation": score.invalidation,
            "sources": list(score.citations),
        }

        assert row["stage"] == 3
        assert row["sources"] == []


class TestFunnelTargetCount:
    def test_target_count_follows_the_kernels_limit(self):
        """A 20-name plan on a 5-name account yields untradeably small positions."""
        from osiris.config import load_settings
        from osiris.run import build_funnel

        funnel = build_funnel(load_settings(), target_count=5)
        if funnel is None:
            pytest.skip("no OPENROUTER_API_KEY configured")

        assert funnel.budget.final_count == 5

    def test_default_stays_at_twenty(self):
        from osiris.config import load_settings
        from osiris.run import build_funnel

        funnel = build_funnel(load_settings())
        if funnel is None:
            pytest.skip("no OPENROUTER_API_KEY configured")

        assert funnel.budget.final_count == 20


class TestResearchQueries:
    def test_a_bare_symbol_is_not_duplicated(self):
        """"PEP PEP earnings results" skews relevance against the real terms."""
        import inspect

        from osiris.data.research import ResearchClient

        source = inspect.getsource(ResearchClient.research_symbol)
        assert "{name} {symbol}" not in source
        assert "subject" in source

    def test_rate_limits_are_retried_not_dropped(self):
        import inspect

        from osiris.data.research import ResearchClient

        source = inspect.getsource(ResearchClient.search)
        assert "_RateLimited" in source
        assert "exa.backoff" in source

    def test_retry_after_header_is_honored(self):
        from osiris.data.research import _RateLimited

        assert _RateLimited("2.5").retry_after == 2.5
        assert _RateLimited(None).retry_after is None
        assert _RateLimited("garbage").retry_after is None

    def test_concurrency_leaves_headroom_under_the_documented_limit(self):
        from osiris.data.research import ResearchClient

        client = ResearchClient("key")
        assert client._semaphore._value <= 4
