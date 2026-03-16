# Budget Injection Research Project

Research project investigating periodic natural language budget injection during RL
training for reasoning models. See `RESEARCH_PLAN.md` for full experimental design.

## Structure

```
configs/          # TOML configs for each training run (10 total)
scripts/          # Launch scripts, eval harness
analysis/         # Post-training analysis scripts
RESEARCH_PLAN.md  # Full research plan with hypotheses, experiments, metrics
RELATED_WORK.md   # Literature survey (30+ papers)
```

## Quick Reference

- **Model**: Qwen3-4B (primary), Qwen3-8B (scaling)
- **Env**: rg-mix (zebra_puzzles, arc_1d, sokoban, cryptarithm, countdown)
- **Infrastructure**: compaction-rl's `inject_only` mode via CompactionEnv
- **Training**: prime-rl GRPO, 2-node layout (4 inf + 4 train), Full FT
- **Key configs**: `inject_2048.toml` (primary), `baseline.toml` (control)

## Dependencies

This project uses compaction-rl's infrastructure:
- `CompactionEnv` with `inject_only=true` for budget injection
- `/inject_generate` endpoint in compaction worker
- `completion_mask=0` handling for injected tokens

## Code Changes Needed (in compaction-rl)

1. **Ratio template vars**: Add `{percent}`, `{remaining_percent}` to worker.py template formatting
2. **Urgency format**: Add `budget_format_type` param with conditional template selection
3. **Variable budget**: Add `max_total_tokens_min` param to CompactionEnv for per-problem randomization
4. **Eval harness**: Extend `eval_rg_mix.py` for cross-budget evaluation
