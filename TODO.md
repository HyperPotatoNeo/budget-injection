# TODO — Budget Injection Project

## Implementation (before any runs)

- [ ] Clone fresh prime-rl into `prime-rl/` subdirectory
- [ ] Set up uv environment: `cd prime-rl && uv sync --all-extras`
- [ ] Implement `BudgetInjectionEnv` as a verifiers env wrapper package
  - [ ] Core: wrap any SingleTurnEnv, inject budget messages every N tokens
  - [ ] Add total budget to initial system prompt
  - [ ] Track inject_ranges for completion_mask (excluded from loss)
  - [ ] Support 4 message formats: absolute, ratio, urgency, minimal
  - [ ] Support variable budget (min_total_tokens → max_total_tokens random per problem)
  - [ ] `load_environment()` entry point for prime-rl discovery
- [ ] Unit tests for BudgetInjectionEnv (no GPU needed)
- [ ] Verify rg_mix dataset exists at `/pscratch/sd/s/siddart2/datasets/rg_mix_7500`
  - [ ] If not, generate it (check compaction-rl for how it was created)
- [ ] Smoke test: run 5 steps of `baseline.toml` on 1 interactive node

## Phase 1: Core Frequency Sweep (highest priority)

All 4B runs use 2 seeds (seed 42 and seed 123).

- [ ] Run 1a: `baseline` seed=42
- [ ] Run 1b: `baseline` seed=123
- [ ] Run 2a: `inject-2048` seed=42
- [ ] Run 2b: `inject-2048` seed=123
- [ ] Run 3a: `inject-1024` seed=42
- [ ] Run 3b: `inject-1024` seed=123
- [ ] Run 4a: `inject-4096` seed=42
- [ ] Run 4b: `inject-4096` seed=123
- [ ] Eval all 8 checkpoints at E1-E4

## Phase 2: Format Variants (after Phase 1 shows signal)

- [ ] Run 5a: `inject-ratio` seed=42
- [ ] Run 5b: `inject-ratio` seed=123
- [ ] Run 6a: `inject-urgency` seed=42
- [ ] Run 6b: `inject-urgency` seed=123
- [ ] Run 7a: `inject-minimal` seed=42
- [ ] Run 7b: `inject-minimal` seed=123
- [ ] Eval all 6 checkpoints at E1-E4

## Phase 3: Generalization

- [ ] Run 8a: `inject-variable` seed=42
- [ ] Run 8b: `inject-variable` seed=123
- [ ] Eval at E1-E4

## Phase 4: Scale (lower priority)

- [ ] Run 9: `baseline-06b` seed=42
- [ ] Run 10: `inject-06b` seed=42
- [ ] Eval at E1-E4

## Test-Time Scaling (after training)

- [ ] TTS-1: Best-of-N experiment
- [ ] TTS-2: Variable budget curve
- [ ] TTS-3: Pareto frontier

## Analysis & Writing

- [ ] Per-task accuracy breakdown tables
- [ ] Training curves (reward, accuracy, length over steps)
- [ ] Internalization test (inject-trained model eval'd without injection)
- [ ] Budget generalization curves
- [ ] Reasoning quality analysis (manual inspection)
- [ ] Draft blog post / paper
