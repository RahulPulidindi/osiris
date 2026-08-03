"""OpenRouter client with per-role model routing and a hard cost ceiling.

Two design decisions worth stating:

  1. Roles get different models. RedTeam deliberately uses a different model
     FAMILY than Strategist, because asking one model to both propose and
     critique produces correlated errors -- a single model asked to find trades
     will find trades.

  2. The daily cost ceiling is enforced in code, not by intention. An agent loop
     with a bug can otherwise spend unbounded money overnight.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from enum import Enum

import httpx

from osiris.logging import get_logger

log = get_logger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class Role(str, Enum):
    ANALYST = "analyst"        # summarize evidence, no opinions
    STRATEGIST = "strategist"  # propose thesis
    RED_TEAM = "red_team"      # argue the bear case, holds veto
    PM = "pm"                  # portfolio construction
    POSTMORTEM = "postmortem"  # nightly review
    TRIAGE = "triage"          # cheap stage-2 funnel pass
    VISION = "vision"          # chart reading


# Deliberate family separation between STRATEGIST and RED_TEAM.
#
# COST TIER: the default set runs every role on budget models. The four-role
# architecture, the red team's veto, and family separation are all preserved --
# only the price per token changes. On a small account the strategy's edge (if
# any) is breadth and discipline, not marginal model IQ, and a $0.70 research
# cycle against a $100 book is 0.7% of equity EVERY DAY, which is a larger drag
# than the spread. Scale the models up with the account via OSIRIS_MODEL_*.
DEFAULT_MODELS: dict[Role, str] = {
    Role.TRIAGE: "google/gemini-2.5-flash-lite",
    Role.ANALYST: "google/gemini-2.5-flash",
    Role.STRATEGIST: "google/gemini-2.5-flash",
    Role.RED_TEAM: "openai/gpt-5-mini",  # different family, holds the veto
    Role.PM: "google/gemini-2.5-flash",
    Role.POSTMORTEM: "google/gemini-2.5-flash-lite",
    Role.VISION: "google/gemini-2.5-flash-lite",
}

# The premium set the defaults replaced. Selectable per-role from the
# environment; documented here so the upgrade path is one env var, not a diff.
PREMIUM_MODELS: dict[Role, str] = {
    Role.TRIAGE: "google/gemini-2.5-flash",
    Role.ANALYST: "google/gemini-2.5-flash",
    Role.STRATEGIST: "anthropic/claude-sonnet-4.5",
    Role.RED_TEAM: "openai/gpt-5",
    Role.PM: "anthropic/claude-sonnet-4.5",
    Role.POSTMORTEM: "anthropic/claude-sonnet-4.5",
    Role.VISION: "google/gemini-2.5-flash",
}

# Approximate USD per 1M tokens (input, output). Used for the ceiling, so
# over-estimating is the safe direction.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "google/gemini-2.5-flash-lite": (0.10, 0.40),
    "google/gemini-2.5-flash": (0.30, 2.50),
    "openai/gpt-5-mini": (0.25, 2.00),
    "anthropic/claude-sonnet-4.5": (3.00, 15.00),
    "openai/gpt-5": (1.25, 10.00),
}
DEFAULT_PRICING = (3.00, 15.00)


def models_from_env() -> dict[Role, str]:
    """Per-role model overrides: OSIRIS_MODEL_STRATEGIST=... etc.

    Env-driven so upgrading models with account size is a config change on the
    server, not a code deploy. Unset roles keep the budget defaults.
    """
    import os

    out: dict[Role, str] = {}
    for role in Role:
        value = os.environ.get(f"OSIRIS_MODEL_{role.name}", "").strip()
        if value:
            out[role] = value
    return out


class BudgetExhausted(RuntimeError):
    """Raised when the daily LLM ceiling is hit. Halts cognition, not exits."""


class EmptyCompletion(ValueError):
    """The model returned no content at all.

    Distinct from malformed JSON: the fix is a larger budget or a smaller prompt,
    not a better schema. Subclasses ValueError so existing handlers still catch it.
    """


@dataclass
class CostLedger:
    spent_usd: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_role: dict[str, float] = field(default_factory=dict)

    def record(self, role: Role, model: str, in_tok: int, out_tok: int) -> float:
        pin, pout = MODEL_PRICING.get(model, DEFAULT_PRICING)
        cost = (in_tok / 1_000_000) * pin + (out_tok / 1_000_000) * pout
        self.spent_usd += cost
        self.calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.by_role[role.value] = self.by_role.get(role.value, 0.0) + cost
        return cost


def _count_records(payload: dict | list) -> int:
    if isinstance(payload, list):
        return len(payload)
    for value in payload.values():
        if isinstance(value, list):
            return len(value)
    return 1


def _salvage_json(body: str) -> dict | list | None:
    """Recover the complete leading records from a truncated JSON response.

    Walks the text tracking bracket depth and string state, remembering the last
    position where a top-level array element closed cleanly. Everything after that
    is an incomplete record and is dropped.

    Only whole elements survive. No field is defaulted, inferred, or repaired: a
    half-read thesis attached to a real ticker is more dangerous than no thesis,
    because it would flow into a scoring decision looking complete.
    """
    depth = 0
    in_string = False
    escaped = False
    array_depth: int | None = None
    last_clean: int | None = None

    for i, ch in enumerate(body):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "[{":
            depth += 1
            if ch == "[" and array_depth is None:
                array_depth = depth
        elif ch in "]}":
            # An element of the outermost array just closed.
            if array_depth is not None and depth == array_depth + 1 and ch == "}":
                last_clean = i
            depth -= 1

    if array_depth is None or last_clean is None:
        return None

    candidate = body[: last_clean + 1] + "]"
    # Close any object levels that wrapped the array, e.g. {"scores": [...]}.
    for _ in range(max(0, array_depth - 1)):
        candidate += "}"
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


@dataclass(frozen=True)
class LLMResponse:
    role: Role
    model: str
    content: str
    input_tokens: int
    output_tokens: int
    cost_usd: float

    def parse_json(self) -> dict | list:
        """Extract JSON, tolerating fenced code blocks and truncation.

        Truncation is treated as recoverable rather than fatal. A response cut off
        at the token limit is usually 19 complete records followed by a partial
        20th, and raising discards all 19 -- which is how a full research cycle
        produced zero targets while every individual step reported success.

        Salvage is deliberately conservative: it only ever drops a trailing
        incomplete element. It never repairs or invents field values, because a
        fabricated thesis is worse than a missing one.
        """
        text = self.content.strip()
        if text.startswith("```"):
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        if not text:
            # An EMPTY completion, not malformed output. Usually the model spent
            # its entire budget on reasoning tokens and emitted nothing, or the
            # provider returned a blank choice. "no JSON found in response: "
            # with nothing after the colon is a genuinely confusing way to say
            # that, and it sent me looking at the parser rather than the budget.
            raise EmptyCompletion(
                f"{self.role.value} returned an empty completion "
                f"({self.output_tokens} output tokens). The token budget may be "
                "exhausted by reasoning before any content is produced."
            )

        start = min(
            (i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1
        )
        if start < 0:
            raise ValueError(
                f"no JSON found in {self.role.value} response "
                f"({len(text)} chars): {text[:200]!r}"
            )

        body = text[start:]
        end = max(body.rfind("}"), body.rfind("]"))
        try:
            return json.loads(body[: end + 1])
        except json.JSONDecodeError:
            salvaged = _salvage_json(body)
            if salvaged is None:
                raise
            log.warning(
                "llm.response_truncated_salvaged",
                role=self.role.value,
                model=self.model,
                chars=len(self.content),
                recovered=_count_records(salvaged),
            )
            return salvaged


class LLMClient:
    """OpenRouter wrapper. Enforces the budget before every call."""

    def __init__(
        self,
        api_key: str | None,
        *,
        daily_usd_ceiling: float = 15.0,
        models: dict[Role, str] | None = None,
        ledger: CostLedger | None = None,
        timeout: float = 120.0,
        max_concurrency: int = 6,
    ) -> None:
        self.api_key = api_key
        self.daily_usd_ceiling = daily_usd_ceiling
        # Budget defaults <- env overrides <- explicit constructor argument.
        self.models = {**DEFAULT_MODELS, **models_from_env(), **(models or {})}
        self.ledger = ledger or CostLedger()
        self.timeout = timeout
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    @property
    def budget_remaining(self) -> float:
        return max(0.0, self.daily_usd_ceiling - self.ledger.spent_usd)

    def _check_budget(self) -> None:
        if self.ledger.spent_usd >= self.daily_usd_ceiling:
            raise BudgetExhausted(
                f"LLM daily ceiling reached: ${self.ledger.spent_usd:.2f} >= "
                f"${self.daily_usd_ceiling:.2f}"
            )

    async def complete(
        self,
        role: Role,
        system: str,
        user: str,
        *,
        images: list[str] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2_000,
        retries: int = 3,
    ) -> LLMResponse:
        """Single completion. `images` are base64 data URLs for vision calls."""
        if not self.enabled:
            raise RuntimeError("OPENROUTER_API_KEY not configured")
        self._check_budget()

        model = self.models[role]
        content: list[dict] | str
        if images:
            content = [{"type": "text", "text": user}]
            content += [
                {"type": "image_url", "image_url": {"url": u}} for u in images
            ]
        else:
            content = user

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        delay = 1.0
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                async with self._semaphore:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(
                            OPENROUTER_URL,
                            json=payload,
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                                "HTTP-Referer": "https://github.com/osiris",
                                "X-Title": "Osiris",
                            },
                        )
                        resp.raise_for_status()
                        data = resp.json()

                text = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage", {})
                in_tok = int(usage.get("prompt_tokens", 0))
                out_tok = int(usage.get("completion_tokens", 0))
                cost = self.ledger.record(role, model, in_tok, out_tok)

                # An empty completion after a successful HTTP call means the model
                # produced nothing usable. Retry ONCE with more headroom before
                # giving up: losing this response costs the whole stage, and the
                # usual cause (budget consumed before content began) is fixed by
                # exactly this.
                if not text.strip() and attempt < retries:
                    log.warning(
                        "llm.empty_completion_retry",
                        role=role.value,
                        model=model,
                        out_tokens=out_tok,
                        next_max_tokens=payload["max_tokens"] * 2,
                    )
                    payload["max_tokens"] = int(payload["max_tokens"]) * 2
                    await asyncio.sleep(delay)
                    delay *= 2
                    continue
                log.debug(
                    "llm.complete",
                    role=role.value,
                    model=model,
                    in_tokens=in_tok,
                    out_tokens=out_tok,
                    cost_usd=round(cost, 5),
                )
                return LLMResponse(role, model, text, in_tok, out_tok, cost)
            except Exception as exc:
                last = exc
                if attempt == retries:
                    break
                log.warning(
                    "llm.retry", role=role.value, attempt=attempt, error=str(exc)
                )
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError(f"LLM call failed for {role.value} after {retries} tries") from last
