# Budget Injection Research Project

Research project on periodic natural language budget injection during RL training.
See `RESEARCH_PLAN.md` for hypotheses and experiments, `TODO.md` for status,
`INSTRUCTIONS.md` for how to run.

## Structure

```
configs/                   # TOML configs (10 experiments)
scripts/                   # Launch, monitor, eval scripts
budget_injection_env/      # BudgetInjectionEnv verifiers package
  env.py                   # Core env: MultiTurnEnv wrapper
  tests/                   # Unit tests
prime-rl/                  # Stock prime-rl clone (git submodule or clone)
analysis/                  # Post-training analysis scripts
outputs/                   # Training outputs (gitignored)
logs/                      # SLURM logs (gitignored)
```

## Quick Reference

| Property | Value |
|----------|-------|
| Primary model | Qwen/Qwen3-4B (Full FT) |
| Secondary model | Qwen/Qwen3-0.6B (Full FT, lower priority) |
| Env | rg-mix (zebra_puzzles, arc_1d, sokoban, cryptarithm, countdown) |
| Training | GRPO, 400 steps, batch=256, rollouts=4 |
| Hardware | 1 node x 4 A100-80GB (2 inf + 2 train) |
| Jobs | 48h premium, ckpt interval=30, auto-resume |
| Seeds | 2 per 4B run (42, 123), 1 per 0.6B run |

## Architecture

`BudgetInjectionEnv` is a `MultiTurnEnv` that wraps any verifiers env:
- Turn 1: model generates up to `inject_budget_every` tokens
- Turn 2+: env injects budget message as user turn, model continues
- Stops when total effective tokens >= max_total_tokens or EOS
- Inner env's rubric scores the final concatenated response
- Works with stock prime-rl (no custom inference endpoints)

## Key Design Decisions

1. **MultiTurnEnv, not custom inference server**: Each turn generates
   `inject_budget_every` tokens via standard vLLM. Budget messages are
   injected as user turns between model responses. This works with
   completely stock prime-rl.

2. **System prompt includes total budget**: First turn sees
   "You have a budget of 8192 tokens for your response." in system prompt.

3. **Sampling args per turn**: `get_model_response()` overrides max_tokens
   per turn to min(inject_budget_every, remaining_budget).

4. **Variable budget**: When `min_total_tokens` is set, each problem gets
   a random budget in [min, max]. Tests budget generalization.

## Dependencies

- prime-rl (stock clone)
- verifiers (comes with prime-rl)
- reasoning_gym (for rg-mix tasks)

## Running

```bash
# Setup
cd prime-rl && uv sync --all-extras
uv pip install -e ../budget_injection_env

# Launch Phase 1
bash scripts/launch_phase1.sh

# Monitor
bash scripts/check_runs.sh

# Auto-resume crashed jobs
bash scripts/monitor_and_resume.sh
```
