"""Per-run tool-call budget enforcement for the orchestrator.

Two budgets, one middleware:

- a **global** per-run cap (`max_tool_calls`, counted under the ``__all__`` key) that,
  once reached, forces the next planner turn to answer from context with no further
  tool calls (the "answer-now" soft-landing), and
- optional **per-tool** per-run caps (`per_tool_max_calls`, counted under each tool
  name) that block *only* the exhausted tool with a recoverable error ``ToolMessage``,
  leaving the planner free to use other tools or answer.

Both budgets are accounted in a single ``after_model`` pass that writes the
``run_tool_call_count`` / ``thread_tool_call_count`` channels exactly once. This is a
deliberate override of the stock per-instance accounting: those count channels are
``UntrackedValue`` (last-writer-wins), so stacking several stock
``ToolCallLimitMiddleware`` instances would let one instance's dict write clobber
another's in the same super-step. Owning every count key in one writer is the
race-free way to combine a global cap, multiple per-tool caps, and the global-only
answer-now behavior in one middleware.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Annotated, Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, ToolCallLimitMiddleware
from langchain.agents.middleware.tool_call_limit import ToolCallLimitState
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


ORCHESTRATOR_TOOL_BUDGET_GUARD_ENV_PREFIX = "ORCHESTRATOR_TOOL_BUDGET_GUARD_"
_GLOBAL_TOOL_COUNT_KEY = "__all__"

_ANSWER_NOW_INSTRUCTION = """\
<tool_budget>
## TOOL BUDGET REACHED FOR THIS REQUEST

You have used every tool call allotted to answer THIS message. This is a
per-request limit only — it resets automatically on the user's next message and
is not a permanent, account-level, or service restriction.

Respond honestly from the current context. Do not fabricate missing tool results, citations, or data.
Do not call tools.
Start with a short bold TLDR telling the user you reached the tool-exploration limit for this
request and so could not complete every step — and that they can continue by sending another
message, which resets the limit.
Then state what you found and what you did not find before proposing any follow-ups or hints.
</tool_budget>
"""


class ToolBudgetGuardSettings(BaseSettings):
    """Server-owned settings for :class:`ToolBudgetEnforcementMiddleware`."""

    model_config = SettingsConfigDict(
        env_prefix="",
        case_sensitive=False,
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    max_tool_calls: int | None = Field(
        default=None,
        ge=0,
        description="Maximum planner tool calls per graph invocation. None disables the global guard.",
    )

    per_tool_max_calls: dict[str, Annotated[int, Field(ge=0)]] | None = Field(
        default=None,
        description=(
            "Optional per-run cap for specific tool names, e.g. {'task': 5}. Provided via env as JSON "
            "(ORCHESTRATOR_TOOL_BUDGET_GUARD_PER_TOOL_MAX_CALLS='{\"task\": 5}'). None/empty disables per-tool caps. "
            "Each cap must be >= 0 (validated at the settings boundary, not only when the middleware is built). "
            "An exhausted per-tool cap blocks only that tool; it does not trigger the global answer-now soft-landing."
        ),
    )

    def __init__(self, *, _env_prefix: str | None = None, **values: Any) -> None:
        """Initialize settings with an optional caller-owned environment prefix."""
        super().__init__(_env_prefix=_env_prefix or "", **values)

    @field_validator("max_tool_calls", mode="before")
    @classmethod
    def _blank_max_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return int(stripped) if stripped else None
        return value

    @field_validator("per_tool_max_calls", mode="before")
    @classmethod
    def _blank_per_tool_to_none(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class ToolBudgetEnforcementMiddleware(ToolCallLimitMiddleware):
    """Enforce the orchestrator tool budget — a global per-run cap plus optional per-tool caps.

    Extends LangChain's ``ToolCallLimitMiddleware`` for its state channels
    (``run_tool_call_count`` / ``thread_tool_call_count``) and middleware identity, but
    owns counting in a single ``after_model`` pass (see module docstring for why a single
    writer is required). Two planner-facing behaviors:

    - once the **global** run count reaches ``max_tool_calls``, the next model call is made
      with no usable tools and an instruction to answer from already-available context, and
    - any tool whose **per-tool** cap is reached is blocked with a recoverable error
      ``ToolMessage`` while the rest of the run continues unaffected.
    """

    def __init__(
        self,
        *,
        max_tool_calls: int | None = None,
        per_tool_max_calls: Mapping[str, int] | None = None,
        settings: ToolBudgetGuardSettings | None = None,
    ) -> None:
        if settings is not None:
            resolved_max = settings.max_tool_calls
            resolved_per_tool: Mapping[str, int] | None = settings.per_tool_max_calls
        else:
            resolved_max = max_tool_calls
            resolved_per_tool = per_tool_max_calls

        limits: dict[str, int] = {}
        if resolved_max is not None:
            if resolved_max < 0:
                raise ValueError("max_tool_calls must be >= 0")
            limits[_GLOBAL_TOOL_COUNT_KEY] = resolved_max
        for tool_name, cap in (resolved_per_tool or {}).items():
            if cap < 0:
                raise ValueError(f"per_tool_max_calls[{tool_name!r}] must be >= 0")
            limits[tool_name] = cap
        if not limits:
            raise ValueError("max_tool_calls must be set (or per_tool_max_calls provided) when installing ToolBudgetEnforcementMiddleware")

        # Bypass the stock __init__ (it rejects a per-tool-only config because it requires a
        # thread/run limit) and set the attributes other middleware/tests inspect directly.
        AgentMiddleware.__init__(self)  # type: ignore[arg-type]  # invariant StateT false-positive; base __init__ is a no-op
        self.max_tool_calls = resolved_max
        self.per_tool_max_calls: dict[str, int] = {name: cap for name, cap in limits.items() if name != _GLOBAL_TOOL_COUNT_KEY}
        self._limits = limits
        self.run_limit = resolved_max
        self.thread_limit: int | None = None
        self.tool_name: str | None = None
        self.exit_behavior = "continue"

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Force an answer-only request once the global run tool budget is reached."""
        if not self._run_budget_reached(request):
            return handler(request)
        return self._answer_now_response(handler(self._answer_now_request(request)))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Force an answer-only request once the global run tool budget is reached."""
        if not self._run_budget_reached(request):
            return await handler(request)
        return self._answer_now_response(await handler(self._answer_now_request(request)))

    def after_model(self, state: ToolCallLimitState[Any], runtime: Runtime[Any] | None = None) -> dict[str, Any] | None:
        """Count this turn's tool calls against every configured limit in one pass.

        A tool call is blocked when the global cap **or** its own per-tool cap would be
        exceeded; allowed calls increment both run and thread counts, blocked calls
        increment the run count only (mirroring the stock limiter), and blocked calls
        receive a recoverable error ``ToolMessage``. The two count channels are written
        once so concurrent dict writes cannot clobber each other.
        """
        last_ai_message = next((message for message in reversed(state.get("messages", [])) if isinstance(message, AIMessage)), None)
        if last_ai_message is None or not last_ai_message.tool_calls:
            return None

        run_counts = dict(state.get("run_tool_call_count", {}) or {})
        thread_counts = dict(state.get("thread_tool_call_count", {}) or {})
        blocked_messages: list[ToolMessage] = []
        changed = False

        for tool_call in last_ai_message.tool_calls:
            count_keys = [key for key in (_GLOBAL_TOOL_COUNT_KEY, tool_call["name"]) if key in self._limits]
            if not count_keys:
                continue
            exceeded_key = next((key for key in count_keys if run_counts.get(key, 0) + 1 > self._limits[key]), None)
            for key in count_keys:
                run_counts[key] = run_counts.get(key, 0) + 1
            if exceeded_key is None:
                for key in count_keys:
                    thread_counts[key] = thread_counts.get(key, 0) + 1
            else:
                limited_tool = None if exceeded_key == _GLOBAL_TOOL_COUNT_KEY else tool_call["name"]
                blocked_messages.append(
                    ToolMessage(
                        content=_blocked_tool_message(limited_tool),
                        tool_call_id=tool_call["id"],
                        name=tool_call.get("name"),
                        status="error",
                    )
                )
            changed = True

        if not changed:
            return None
        update: dict[str, Any] = {
            "run_tool_call_count": run_counts,
            "thread_tool_call_count": thread_counts,
        }
        if blocked_messages:
            update["messages"] = blocked_messages
        return update

    async def aafter_model(self, state: ToolCallLimitState[Any], runtime: Runtime[Any] | None = None) -> dict[str, Any] | None:
        """Async wrapper around the single-pass :meth:`after_model` accounting."""
        return self.after_model(state, runtime)

    @staticmethod
    def _answer_now_request(request: ModelRequest) -> ModelRequest:
        """Force an answer-only turn while keeping the vLLM prefix cache warm.

        The validated ``tools → system`` prefix is left byte-identical (so it stays
        cached); the budget instruction is appended as a trailing ``HumanMessage`` and
        ``tool_choice='none'`` forbids further tool selection at decode time.
        """
        answer_now = HumanMessage(content=_ANSWER_NOW_INSTRUCTION)
        return request.override(
            messages=[*request.messages, answer_now],
            tool_choice="none",
            response_format=None,
        )

    @staticmethod
    def _answer_now_response(response: ModelResponse) -> ModelResponse:
        return ModelResponse(
            result=[_remove_tool_calls(message) for message in response.result],
            structured_response=None,
        )

    def _run_budget_reached(self, request: ModelRequest) -> bool:
        if self.max_tool_calls is None:
            return False
        state = request.state or {}
        return _global_run_tool_count(state) >= self.max_tool_calls


def _blocked_tool_message(tool_name: str | None) -> str:
    """Build the recoverable error sent to the planner when a cap is reached."""
    if tool_name:
        return (
            f"Tool call limit for '{tool_name}' reached for this request. Do not call '{tool_name}' "
            "again before the user's next message; the limit resets then. Continue with other tools "
            "or answer from what you already have."
        )
    return (
        "Tool call limit reached for this request. Do not make additional tool calls before the "
        "user's next message; the limit resets then. Answer from the information you already have."
    )


def _global_run_tool_count(state: Mapping[str, Any]) -> int:
    counts = state.get("run_tool_call_count", {})
    if not isinstance(counts, Mapping):
        return 0
    value = counts.get(_GLOBAL_TOOL_COUNT_KEY, 0)
    return value if isinstance(value, int) and value > 0 else 0


def _remove_tool_calls(message: BaseMessage) -> BaseMessage:
    if not isinstance(message, AIMessage):
        return message

    additional_kwargs = dict(message.additional_kwargs)
    additional_kwargs.pop("tool_calls", None)
    return message.model_copy(
        update={
            "additional_kwargs": additional_kwargs,
            "tool_calls": [],
            "invalid_tool_calls": [],
        }
    )
