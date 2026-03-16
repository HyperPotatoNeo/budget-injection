# Budget Injection: Teaching Reasoning Models to Manage Their Token Budget

## One-Sentence Contribution

We show that periodically injecting natural language budget messages during RL training
teaches reasoning models a generalizable concept of compute budgeting that transfers to
unseen budget levels and composes with test-time scaling methods, without requiring special
tokens or SFT supervision.

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

## Key Novelty vs Prior Work

| This work | BudgetThinker | SelfBudgeter | L1/LCPO | s1 |
|-----------|--------------|--------------|---------|-----|
| Natural language chat turns | Special control tokens | Self-predicted tag | Prompt suffix | Decode-time forcing |
| RL-only (no SFT) | SFT + RL | SFT + RL | RL | SFT |
| Multi-task (5 diverse tasks) | Math only | Math + code | Math | Math |
| Budget generalization tested | Single budget | Single budget | Fixed targets | N/A |
| TTS interaction studied | Not studied | Not studied | Not studied | N/A |
| Zero-shot via NL semantics | Requires training | Requires training | Requires training | Works zero-shot |

## Hypotheses

### H1: Training with injection > Inference-only injection
RL training with budget injection produces models that outperform both (a) standard RL
evaluated with injection and (b) standard RL evaluated without injection. The existing
eval shows RL-trained models are injection-neutral (eval-only injection: -0.3%). Training
WITH injection should teach the model to USE the signal.

### H2: Optimal injection frequency exists (non-monotonic)
Too few injections starve the model of budget signal. Too many interrupt multi-step
reasoning. Existing eval: injection helps zebra_puzzles (+34pp) but hurts cryptarithm
(-10.6pp) and sokoban (-4.9pp). Test N=1024, 2048, 4096.

### H3: Internalization — budget-trained models pace themselves WITHOUT injection
The strongest result. If a model trained with budget messages paces itself even without
them at inference, it has internalized reasoning-time allocation. Mechanism: budget
messages correlate with reward during training, and the pacing behavior persists.

### H4: Ratio-based messages generalize across budget sizes better than absolute counts
"75% remaining" is scale-invariant; "6144/8192 remaining" embeds the specific budget.
Train on 8K, eval on 4K and 16K. Ratio format should transfer better.

### H5: Budget injection improves test-time scaling efficiency
Budget-aware models + best-of-N should produce fewer truncated responses and more diverse
valid candidates, improving accuracy-per-token at any compute budget.

## Experiment Plan

### Phase 1: Core Training Runs (4 runs)

| Run | Name | Injection | Frequency | Template | Steps |
|-----|------|-----------|-----------|----------|-------|
| 1 | `baseline` | None | N/A | N/A | 600 (DONE) |
| 2 | `inject-2048` | Yes | Every 2048 | Absolute | 600 |
| 3 | `inject-1024` | Yes | Every 1024 | Absolute | 600 |
| 4 | `inject-4096` | Yes | Every 4096 | Absolute | 600 |

Tests: H1 (Run 2 vs 1), H2 (Runs 2 vs 3 vs 4)

### Phase 2: Format Variants (3 runs)

| Run | Name | Injection | Frequency | Template | Steps |
|-----|------|-----------|-----------|----------|-------|
| 5 | `inject-ratio` | Yes | Every 2048 | Ratio (%) | 600 |
| 6 | `inject-urgency` | Yes | Every 2048 | Urgency-graded | 600 |
| 7 | `inject-minimal` | Yes | Every 2048 | `<budget>N</budget>` | 600 |

Tests: H4 (Run 5 vs 2), format ablation (Runs 5 vs 6 vs 7 vs 2)

### Phase 3: Generalization & Scaling (3 runs)

| Run | Name | Injection | Frequency | Template | Steps | Notes |
|-----|------|-----------|-----------|----------|-------|-------|
| 8 | `inject-variable` | Yes | Every 2048 | Absolute | 600 | Budget randomized 4K-8K per problem |
| 9 | `inject-8b` | Yes | Every 2048 | Best from Phase 2 | 400 | Qwen3-8B LoRA r=32 |
| 10 | `baseline-8b` | None | N/A | N/A | 400 | Qwen3-8B LoRA r=32 control |

Tests: Budget generalization (Run 8), scaling (Run 9 vs 10)

### Eval Matrix (applied to all trained models)

| Condition | Injection at Eval | Budget at Eval |
|-----------|-------------------|----------------|
| E1 | None | 8192 |
| E2 | Yes (matching train freq) | 8192 |
| E3 | Yes (matching train freq) | 4096 |
| E4 | Yes (matching train freq) | 16384 |

10 models x 4 eval conditions = 40 eval runs (each ~15 min, trivial cost).

### Test-Time Scaling Experiments (inference-only, best model from above)

| Experiment | Description |
|-----------|-------------|
| TTS-1: Best-of-N | N={1,4,8,16} at budgets {4K,8K}. Compare truncation rate, pass@N. |
| TTS-2: Variable budget | Train-8K model evaluated at {2K, 4K, 6K, 8K, 12K, 16K}. Plot accuracy vs budget. |
| TTS-3: Budget forcing | Combine injection with s1-style "Wait" appending. Are they complementary? |
| TTS-4: Pareto frontier | Fixed total compute C: compare 1 sample at C vs N samples at C/N (budget-trained). |

## Message Format Templates

### A: Absolute (control)
```
Budget: {used}/{total} tokens generated. ~{remaining} tokens remaining.
```
~18-22 tokens per injection.

### B: Ratio
```
[{injection_num}/{total_injections}] {percent}% budget used
```
~8-12 tokens per injection. Scale-invariant.

### C: Urgency-graded
- >50% remaining: `Continue reasoning. {remaining} tokens remaining.`
- 25-50% remaining: `Begin concluding. ~{remaining} tokens remaining.`
- <25% remaining: `Wrap up now. Only {remaining} tokens left.`
~10-16 tokens. Phase-appropriate guidance.

### D: Minimal
```
<budget>{remaining}</budget>
```
~5-7 tokens. Tests whether NL scaffolding is load-bearing.

## Metrics

### Primary
- **Accuracy** (pass@1): The target
- **Token efficiency**: accuracy / avg_tokens
- **Truncation rate**: % responses hitting max budget without EOS
- **Budget utilization**: avg_effective_tokens / budget

### Secondary
- **Per-task accuracy delta**: injection vs no-injection per task type
- **Length CV**: std(lengths) / mean(lengths) — tighter = more consistent
- **Budget adherence**: |actual_tokens - budget| for models told a specific budget

### Diagnostic
- **Internalization test**: Run 2 model at E1 (no injection) vs Run 1 at E1
- **Reasoning quality near end**: Manual inspection of last 512 tokens when >75% used
- **Strategy switching**: Does the model change approach under tight vs loose budgets?

## Infrastructure

All runs use compaction-rl's `inject_only` mode (no KV compaction):
- `/inject_generate` endpoint for periodic budget message injection
- `completion_mask=0` for injected tokens (excluded from RL loss)
- `logprobs=0.0` for injected tokens
- Standard full forward pass in trainer (no segmented forward)

### Hardware
- NERSC Perlmutter, 2 nodes per run (4 inf TP=1 + 4 train FSDP2)
- 4x A100-SXM4-80GB per node
- Qwen3-4B Full FT: ~78 GiB peak with optim_cpu_offload
- Qwen3-8B LoRA r=32: ~50 GiB peak with optim_cpu_offload

### Estimated Cost
- Phase 1: ~1,120 GPU-hours (4 runs x ~280)
- Phase 2: ~840 GPU-hours (3 runs x ~280)
- Phase 3: ~752 GPU-hours (variable + 2x LoRA)
- Evals: ~40 GPU-hours
- TTS experiments: ~80 GPU-hours
- **Total: ~2,832 GPU-hours** (~354 node-hours on 8-GPU nodes)

## Extensions (if core results are positive)

1. **Curriculum training**: Start with generous budgets, gradually tighten
2. **Difficulty-adaptive budgets**: Easy problems get smaller budgets via oracle difficulty
3. **Behavioral phase transitions**: Automated analysis of strategy switching under budget pressure
4. **Cross-model transfer**: Does budget awareness survive distillation?
5. **Length penalty comparison**: Is budget injection strictly better than GRPO length penalties?

## Paper Framing

**Venue**: ICML 2026, NeurIPS 2026, or ICLR 2027

**Structure**:
1. Introduction: Efficiency gap in reasoning models
2. Related Work: BudgetThinker, SelfBudgeter, L1, s1, BRPO
3. Method: NL budget injection + GRPO (simple — this is a strength)
4. Core Experiments: Training effect, frequency, format, generalization
5. Test-Time Scaling: Budget-aware models + BoN, variable budget, Pareto frontier
6. Analysis: Internalization, strategy switching, per-task breakdown
7. Discussion and Limitations

## Anticipated Reviewer Questions

**Q: How is this different from BudgetThinker?**
A: (1) NL chat turns vs special tokens — enables zero-shot transfer and budget
generalization, (2) RL-only vs SFT+RL — simpler pipeline, (3) multi-task evaluation,
(4) test-time scaling interaction studied.

**Q: Isn't this just prompt engineering?**
A: No — the model is RL-TRAINED with these messages. Ablation: "be concise" in system
prompt vs periodic budget injection. The trained model should show precise calibration
that prompting alone cannot achieve.

**Q: Why not use special tokens? They're more efficient.**
A: Special tokens require vocabulary extension and SFT. NL signals work zero-shot,
generalize to unseen budgets, and compose with existing chat formats. We test a minimal
`<budget>N</budget>` format as an efficiency-matched control.

**Q: The model just learns shorter answers. Where's the metacognition?**
A: Behavioral analysis shows strategy switching, not just truncation. Under tight
budgets the model uses different (faster but valid) algorithms. Length-penalized RL
produces uniformly shorter outputs without this adaptation.
