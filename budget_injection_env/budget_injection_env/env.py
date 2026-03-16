"""
BudgetInjectionEnv: A MultiTurnEnv wrapper that periodically injects budget messages.

Wraps any verifiers SingleTurnEnv (or MultiTurnEnv) and adds:
1. Total budget info appended to the system prompt
2. After every `inject_budget_every` tokens of model output, injects a user message
   with the remaining token budget
3. Delegates scoring/rubric to the inner env

Each "turn" in the multi-turn loop generates up to `inject_budget_every` tokens.
After each turn, a budget message is injected as the env_response.
Stops when total effective tokens >= max_total_tokens or model emits EOS.
"""

import logging
import random

import verifiers as vf
from verifiers.types import Messages, SamplingArgs, State

logger = logging.getLogger(__name__)

# Budget message format templates
BUDGET_FORMATS = {
    "absolute": "Budget: {used}/{total} tokens generated. ~{remaining} tokens remaining.",
    "ratio": "[{percent}% used] ~{remaining_percent}% budget remaining.",
    "minimal": "<budget>{remaining}</budget>",
}

# Urgency format uses phase-dependent messages
URGENCY_MESSAGES = {
    "early": "Continue reasoning. {remaining} tokens remaining.",
    "mid": "Begin concluding. ~{remaining} tokens remaining.",
    "late": "Wrap up now. Only {remaining} tokens left.",
}


def _format_budget_message(
    fmt: str,
    used: int,
    total: int,
    remaining: int,
    template: str | None = None,
) -> str:
    """Format a budget message based on the format type."""
    if fmt == "urgency":
        fraction_used = used / max(total, 1)
        if fraction_used < 0.5:
            tmpl = URGENCY_MESSAGES["early"]
        elif fraction_used < 0.75:
            tmpl = URGENCY_MESSAGES["mid"]
        else:
            tmpl = URGENCY_MESSAGES["late"]
        return tmpl.format(used=used, total=total, remaining=remaining)

    if template:
        tmpl = template
    else:
        tmpl = BUDGET_FORMATS.get(fmt, BUDGET_FORMATS["absolute"])

    percent = round(100 * used / max(total, 1))
    remaining_percent = 100 - percent
    return tmpl.format(
        used=used,
        total=total,
        remaining=remaining,
        percent=percent,
        remaining_percent=remaining_percent,
    )


class BudgetInjectionEnv(vf.MultiTurnEnv):
    """Wraps an inner env to inject periodic budget messages during generation.

    Each turn generates up to `inject_budget_every` tokens. Between turns,
    a budget message is injected as a user message. The inner env's rubric
    scores the final concatenated response.
    """

    def __init__(
        self,
        inner_env: vf.Environment,
        inject_budget_every: int = 2048,
        max_total_tokens: int = 8192,
        min_total_tokens: int | None = None,
        budget_format: str = "absolute",
        budget_message_template: str | None = None,
        budget_system_prompt: str = "You have a budget of {total} tokens for your response.",
        **kwargs,
    ):
        self.inner_env = inner_env
        self.inject_budget_every = inject_budget_every
        self.max_total_tokens = max_total_tokens
        self.min_total_tokens = min_total_tokens
        self.budget_format = budget_format
        self.budget_message_template = budget_message_template
        self.budget_system_prompt = budget_system_prompt

        max_injections = max_total_tokens // inject_budget_every
        max_turns = max_injections + 1

        # Build augmented system prompt
        inner_system = inner_env.system_prompt or ""
        budget_line = budget_system_prompt.format(total=max_total_tokens)
        augmented_system = f"{inner_system}\n{budget_line}".strip() if inner_system else budget_line

        super().__init__(
            max_turns=max_turns,
            dataset=inner_env.dataset,
            eval_dataset=getattr(inner_env, "eval_dataset", None),
            system_prompt=augmented_system,
            parser=inner_env.parser,
            rubric=inner_env.rubric,
            sampling_args={"max_tokens": inject_budget_every},
            **kwargs,
        )

    async def setup_state(self, state: State) -> State:
        """Initialize budget tracking in state."""
        # For variable budget: sample a random budget per problem
        if self.min_total_tokens is not None:
            budget = random.randint(self.min_total_tokens, self.max_total_tokens)
        else:
            budget = self.max_total_tokens
        state["budget_total"] = budget
        state["budget_used"] = 0
        return state

    async def env_response(self, messages: Messages, state: State, **kwargs) -> str:
        """Inject a budget message between model turns."""
        total = state["budget_total"]
        used = state["budget_used"]
        remaining = max(0, total - used)

        if remaining <= 0:
            state["final_env_response"] = "Budget exhausted."
            return "Budget exhausted."

        return _format_budget_message(
            fmt=self.budget_format,
            used=used,
            total=total,
            remaining=remaining,
            template=self.budget_message_template,
        )

    @vf.stop
    async def budget_exhausted(self, state: State) -> bool:
        """Stop when the total budget is used up."""
        return state.get("budget_used", 0) >= state.get("budget_total", self.max_total_tokens)

    async def add_model_response(self, state, prompt_messages, response):
        """Track token usage after each model response."""
        await super().add_model_response(state, prompt_messages, response)
        # Count completion tokens from the latest trajectory step
        last_step = state["trajectory"][-1]
        if last_step.get("tokens") and last_step["tokens"].get("completion_ids"):
            n_tokens = len(last_step["tokens"]["completion_ids"])
        elif response.usage:
            n_tokens = response.usage.completion_tokens
        else:
            n_tokens = 0
        state["budget_used"] = state.get("budget_used", 0) + n_tokens

    async def get_model_response(self, state, prompt, client=None, model=None,
                                  tool_defs=None, sampling_args=None):
        """Override to set max_tokens for this turn based on remaining budget."""
        total = state.get("budget_total", self.max_total_tokens)
        used = state.get("budget_used", 0)
        remaining = max(0, total - used)
        tokens_this_turn = min(self.inject_budget_every, remaining)

        sa = dict(sampling_args or state.get("sampling_args") or {})
        sa["max_tokens"] = tokens_this_turn

        return await super().get_model_response(
            state, prompt, client=client, model=model,
            tool_defs=tool_defs, sampling_args=sa,
        )


def load_environment(
    gym: str = "countdown",
    inject_budget_every: int = 2048,
    max_total_tokens: int = 8192,
    min_total_tokens: int | None = None,
    budget_format: str = "absolute",
    budget_message_template: str | None = None,
    budget_system_prompt: str = "You have a budget of {total} tokens for your response.",
    **inner_env_kwargs,
) -> BudgetInjectionEnv:
    """Load a BudgetInjectionEnv wrapping the specified gym environment.

    Called by verifiers' load_environment() when env_id="budget_injection_env".
    """
    inner_env = vf.load_environment(gym, **inner_env_kwargs)
    return BudgetInjectionEnv(
        inner_env=inner_env,
        inject_budget_every=inject_budget_every,
        max_total_tokens=max_total_tokens,
        min_total_tokens=min_total_tokens,
        budget_format=budget_format,
        budget_message_template=budget_message_template,
        budget_system_prompt=budget_system_prompt,
    )
