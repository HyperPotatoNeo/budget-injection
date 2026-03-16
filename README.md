# Budget Injection: Teaching Reasoning Models to Manage Their Token Budget

We study whether periodically injecting natural language budget messages ("You have 6144 tokens remaining") during RL training teaches reasoning models to allocate compute adaptively — spending more tokens on hard problems and wrapping up efficiently on easy ones.

## Key Idea

During RL training (GRPO), we interrupt model generation every N tokens and inject a user-turn message reporting the remaining token budget. The model learns to condition its reasoning strategy on this signal through reward alone — no SFT warmup, no special tokens, no vocabulary changes.

```
System: You are a helpful assistant. You have a budget of 8192 tokens.
User: Solve this zebra puzzle: ...
Assistant: Let me work through the constraints step by step...
  [generates 2048 tokens]
User: Budget: 2048/8192 tokens generated. ~6144 tokens remaining.
Assistant: Good progress. Now combining constraints 3 and 5...
  [generates 2048 tokens]
User: Budget: 4096/8192 tokens generated. ~4096 tokens remaining.
Assistant: I have enough information to determine the answer...
  [generates final answer within budget]
```

## Why This Matters

Reasoning models (Qwen3, DeepSeek-R1, etc.) have no awareness of their generation budget. They routinely overshoot on easy problems and get truncated on hard ones. Prior work uses special control tokens ([BudgetThinker](https://arxiv.org/abs/2508.17196)) or inference-time forcing ([s1](https://arxiv.org/abs/2501.19393)), but neither teaches the model a generalizable concept of budget through natural language.

## Hypotheses

1. **H1**: RL training with budget injection outperforms inference-only injection
2. **H2**: Optimal injection frequency exists (testing every 1024, 2048, 4096 tokens)
3. **H3**: Budget-trained models internalize pacing even WITHOUT injection at eval
4. **H4**: Ratio-based messages ("75% remaining") generalize across budget sizes better than absolute counts
5. **H5**: Budget-aware models improve test-time scaling efficiency (fewer truncated responses in best-of-N)

## Experiment Matrix

### Training (10 configs, 18 runs total)

| Phase | Config | Injection | Format | Seeds |
|-------|--------|-----------|--------|-------|
| 1 | `baseline` | None | — | 2 |
| 1 | `inject-2048` | Every 2048 tok | Absolute | 2 |
| 1 | `inject-1024` | Every 1024 tok | Absolute | 2 |
| 1 | `inject-4096` | Every 4096 tok | Absolute | 2 |
| 2 | `inject-ratio` | Every 2048 tok | Ratio (%) | 2 |
| 2 | `inject-urgency` | Every 2048 tok | Urgency-graded | 2 |
| 2 | `inject-minimal` | Every 2048 tok | `<budget>N</budget>` | 2 |
| 3 | `inject-variable` | Every 2048 tok | Absolute, random budget 4K-8K | 2 |
| 4 | `baseline-06b` | None | — | 1 |
| 4 | `inject-06b` | Every 2048 tok | Absolute | 1 |

### Evaluation (4 conditions per model)

| Condition | Injection | Budget |
|-----------|-----------|--------|
| E1 | None | 8192 |
| E2 | Matching train | 8192 |
| E3 | Matching train | 4096 |
| E4 | Matching train | 16384 |

## Architecture

### BudgetInjectionEnv

A `MultiTurnEnv` wrapper that adds budget injection to any [verifiers](https://github.com/PrimeIntellect-ai/verifiers) environment:

- **Turn 1**: Model generates up to `inject_budget_every` tokens
- **Turn 2+**: Env injects budget message as user turn → model continues
- **Stop**: When total effective tokens ≥ `max_total_tokens` or EOS
- **Scoring**: Delegated to inner env's rubric

Works with completely stock [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) — no custom inference endpoints needed.

### Message Formats

| Format | Example | Tokens |
|--------|---------|--------|
| Absolute | `Budget: 2048/8192 tokens generated. ~6144 tokens remaining.` | ~20 |
| Ratio | `[25% used] ~75% budget remaining.` | ~10 |
| Urgency | `Begin concluding. ~3000 tokens remaining.` (escalates) | ~12 |
| Minimal | `<budget>6144</budget>` | ~6 |

## Setup

```bash
# Clone
git clone https://github.com/HyperPotatoNeo/budget-injection.git
cd budget-injection

# Clone prime-rl
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl && uv sync --all-extras

# Install BudgetInjectionEnv
uv pip install -e ../budget_injection_env
```

## Running Experiments

```bash
# Launch a single run
sbatch scripts/launch.sh configs/inject_2048.toml --seed 42

# Launch all Phase 1 (4 configs x 2 seeds = 8 jobs)
bash scripts/launch_phase1.sh

# Check status of all runs
bash scripts/check_runs.sh

# Auto-resume crashed jobs
bash scripts/monitor_and_resume.sh
```

### Infrastructure

- **Model**: Qwen/Qwen3-4B (Full FT), Qwen/Qwen3-0.6B (scale comparison)
- **Env**: rg-mix (zebra puzzles, arc_1d, sokoban, cryptarithm, countdown)
- **Training**: GRPO, 400 steps, batch=256, rollouts=4
- **Hardware**: 1 node × 4 A100-80GB (2 inference + 2 trainer)
- **Jobs**: 48h, checkpoint every 30 steps, auto-resume on crash

## Novelty vs Prior Work

| | This work | BudgetThinker | SelfBudgeter | L1/LCPO |
|---|-----------|--------------|--------------|---------|
| Signal type | NL chat turns | Special tokens | Self-predicted tag | Prompt suffix |
| Training | RL-only | SFT + RL | SFT + RL | RL |
| Tasks | 5 diverse reasoning | Math only | Math + code | Math |
| Budget generalization | Tested (4K→16K) | Not tested | Not tested | Fixed targets |
| Works zero-shot | Yes (NL semantics) | No (needs training) | No | No |

## Related Work

See [RELATED_WORK.md](RELATED_WORK.md) for a survey of 30+ papers including BudgetThinker, SelfBudgeter, L1/LCPO, s1, BRPO, Kimi K2.5 Toggle, and production model budget mechanisms.

## Project Structure

```
configs/                   # TOML training configs (10 experiments)
scripts/                   # Launch, monitor, eval scripts
budget_injection_env/      # BudgetInjectionEnv verifiers package
  env.py                   # MultiTurnEnv wrapper
  tests/                   # Unit tests
prime-rl/                  # Stock prime-rl clone (not checked in)
analysis/                  # Post-training analysis
RESEARCH_PLAN.md           # Full experimental design
RELATED_WORK.md            # Literature survey
INSTRUCTIONS.md            # How to run experiments
TODO.md                    # Progress tracking
```
