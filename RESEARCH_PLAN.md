# Budget Injection: Teaching Reasoning Models to Manage Their Token Budget

## One-Sentence Contribution

We show that periodically injecting natural language budget messages during RL training
teaches reasoning models a generalizable concept of compute budgeting that transfers to
unseen budget levels, without requiring special tokens or SFT supervision.

## Motivation

Reasoning models (Qwen3, DeepSeek-R1, etc.) use extended chain-of-thought but have no
awareness of their remaining generation budget. They routinely overshoot on easy problems
and get truncated on hard ones. Prior work addresses this with:
- Special control tokens (BudgetThinker, 2508.17196) requiring SFT + vocabulary extension
- Prompt-only signals (TALE, L1) that are ignored mid-generation
- Inference-time forcing (s1) that truncates/extends without model awareness

**Our approach**: Insert natural language user-turn messages ("Budget: 25% used. 75%
remaining.") at regular intervals during RL training. The model learns to condition its
reasoning strategy on remaining budget through reward signal alone — no SFT warmup, no
special tokens, no vocabulary changes.

## Hypotheses

### H1: Training with injection > Inference-only injection
RL training with budget injection produces models that outperform both (a) standard RL
evaluated with injection and (b) standard RL evaluated without injection.

### H2: Optimal injection frequency exists (non-monotonic)
Too few injections starve the model of budget signal. Too many interrupt multi-step
reasoning. Test N=1024, 2048, 4096.

### H3: Internalization — budget-trained models pace themselves WITHOUT injection
If a model trained with budget messages paces itself even without them at inference,
it has internalized reasoning-time allocation.

### H4: Ratio-based messages generalize across budget sizes better than absolute counts
"75% remaining" is scale-invariant; "6144/8192 remaining" embeds the specific budget.
Train on 8K, eval on 4K and 16K. Ratio format should transfer better.

### H5: Budget injection improves test-time scaling efficiency
Budget-aware models + best-of-N should produce fewer truncated responses and more diverse
valid candidates, improving accuracy-per-token at any compute budget.

## Experiment Plan

**Model**: Qwen/Qwen3-4B (primary, Full FT). All 4B runs use 2 seeds.
**Training**: 400 steps, GRPO, batch_size=256, rollouts_per_example=4
**Hardware**: 1 node x 4 A100-80GB (2 inference TP=1 + 2 trainer FSDP2)
**Jobs**: 48h premium, ckpt.interval=30 for crash recovery, resume_step=-1

### Phase 1: Core Training Runs (4 configs x 2 seeds = 8 runs)

| Config | Name | Injection | Frequency | Template | Tests |
|--------|------|-----------|-----------|----------|-------|
| 1 | `baseline` | None | N/A | N/A | Control |
| 2 | `inject-2048` | Yes | Every 2048 | Absolute | H1, H3 |
| 3 | `inject-1024` | Yes | Every 1024 | Absolute | H2 |
| 4 | `inject-4096` | Yes | Every 4096 | Absolute | H2 |

### Phase 2: Format Variants (3 configs x 2 seeds = 6 runs)

| Config | Name | Injection | Frequency | Template | Tests |
|--------|------|-----------|-----------|----------|-------|
| 5 | `inject-ratio` | Yes | Every 2048 | Ratio (%) | H4 |
| 6 | `inject-urgency` | Yes | Every 2048 | Urgency-graded | Format ablation |
| 7 | `inject-minimal` | Yes | Every 2048 | `<budget>N</budget>` | Format ablation |

### Phase 3: Generalization (1 config x 2 seeds = 2 runs)

| Config | Name | Injection | Frequency | Template | Notes |
|--------|------|-----------|-----------|----------|-------|
| 8 | `inject-variable` | Yes | Every 2048 | Absolute | Budget randomized 4K-8K |

### Phase 4: Lower priority — Qwen3-0.6B (2 configs x 1 seed = 2 runs)

| Config | Name | Model | Injection | Notes |
|--------|------|-------|-----------|-------|
| 9 | `baseline-0.6b` | Qwen3-0.6B | None | Scale comparison control |
| 10 | `inject-0.6b` | Qwen3-0.6B | Every 2048 | Scale comparison |

**Total: 8 configs, 18 training runs** (16 with 2 seeds for 4B, 2 with 1 seed for 0.6B)

### Eval Matrix (applied to all trained models)

| Condition | Injection at Eval | Budget at Eval |
|-----------|-------------------|----------------|
| E1 | None | 8192 |
| E2 | Yes (matching train freq) | 8192 |
| E3 | Yes (matching train freq) | 4096 |
| E4 | Yes (matching train freq) | 16384 |

### Test-Time Scaling Experiments (inference-only, best model from above)

| Experiment | Description |
|-----------|-------------|
| TTS-1: Best-of-N | N={1,4,8} at budgets {4K,8K}. Compare truncation rate, pass@N. |
| TTS-2: Variable budget | Train-8K model evaluated at {2K, 4K, 6K, 8K, 12K, 16K}. Plot accuracy vs budget. |
| TTS-3: Pareto frontier | Fixed total compute C: compare 1 sample at C vs N samples at C/N (budget-trained). |

## Message Format Templates

### A: Absolute (control)
```
Budget: {used}/{total} tokens generated. ~{remaining} tokens remaining.
```

### B: Ratio
```
[{percent}% used] ~{remaining_percent}% budget remaining.
```

### C: Urgency-graded
- >50% remaining: `Continue reasoning. {remaining} tokens remaining.`
- 25-50% remaining: `Begin concluding. ~{remaining} tokens remaining.`
- <25% remaining: `Wrap up now. Only {remaining} tokens left.`

### D: Minimal
```
<budget>{remaining}</budget>
```

## Metrics

### Primary
- **Accuracy** (pass@1)
- **Token efficiency**: accuracy / avg_tokens
- **Truncation rate**: % responses hitting max budget without EOS
- **Budget utilization**: avg_effective_tokens / budget

### Secondary
- Per-task accuracy delta (injection vs no-injection)
- Length CV (tighter = more consistent budget awareness)
- Budget adherence at different eval budgets

## Architecture

### BudgetInjectionEnv (new verifiers env)

A wrapper env that adds budget injection to ANY verifiers env:
1. Adds total budget info to the initial system prompt
2. Every N effective tokens, injects a user-turn budget message
3. Delegates scoring/rubric to the inner env
4. Tracks inject_ranges for completion_mask (injected tokens excluded from loss)

This is implemented as a standalone verifiers env package inside the budget-injection
repo, pluggable into stock prime-rl via `load_environment()`.

### Infrastructure
- Fresh clone of prime-rl (stock, no modifications)
- BudgetInjectionEnv as a separate package
- 1 node x 4 A100-80GB (2 inf + 2 train)
- SLURM 48h premium jobs with auto-resume on crash

## Execution Priority

1. Phase 1 (core): 8 runs, highest priority
2. Phase 2 (format): 6 runs, after Phase 1 shows positive signal
3. Phase 3 (generalization): 2 runs
4. Phase 4 (0.6B scale): 2 runs, lowest priority
5. Evals + TTS: after training completes
