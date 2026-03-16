# Instructions for Running Budget Injection Experiments

## Overview

This project runs RL training experiments on NERSC Perlmutter using prime-rl with a custom
BudgetInjectionEnv. Each run uses 1 node (4x A100-80GB): 2 GPUs for inference, 2 for training.

## Setup (one-time)

### 1. Clone prime-rl
```bash
cd /pscratch/sd/s/siddart2/budget-injection
git clone https://github.com/PrimeIntellect-ai/prime-rl.git
cd prime-rl
```

### 2. Set up environment
```bash
# On a compute node inside the skyrl container:
export UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache
uv sync --all-extras
```

### 3. Install BudgetInjectionEnv
```bash
# From budget-injection root:
cd /pscratch/sd/s/siddart2/budget-injection
uv pip install -e ./budget_injection_env
```

### 4. Verify dataset
```bash
ls /pscratch/sd/s/siddart2/datasets/rg_mix_7500/
# Should contain train and eval splits
```

## Launching a Run

### Single run with specific seed
```bash
sbatch scripts/launch.sh configs/inject_2048.toml --seed 42
```

### All Phase 1 runs
```bash
bash scripts/launch_phase1.sh
```

The launch script:
1. Allocates 1 node, 4 GPUs, 48h premium
2. Starts the skyrl container
3. Runs `uv run rl @ <config.toml>` with the appropriate seed
4. Output goes to `outputs/<run-name>/`
5. Checkpoints every 30 steps for crash recovery

### Seeded runs
Each 4B config runs with 2 seeds. The launch script appends `-seed42` or `-seed123`
to the output_dir and wandb name to distinguish them. Override seed via:
```bash
# In the TOML or via CLI override:
uv run rl @ configs/inject_2048.toml orchestrator.seed=42
```

## Monitoring

### Check running jobs
```bash
squeue --me
```

### Check training progress
```bash
# Look at latest log:
tail -50 outputs/<run-name>/run_default/logs/train/<name>.log

# Check W&B for reward curves:
# Project: budget-injection on wandb.ai
```

### Check for crashes
```bash
bash scripts/check_runs.sh
```

This script checks all expected output directories for:
- Whether the job is still running (squeue)
- Whether the latest checkpoint is recent
- Whether the log file has errors

## Resuming After Crash

Jobs WILL crash (node failures, OOM, NCCL timeouts). The `resume_step = -1` in all
configs means prime-rl automatically resumes from the latest checkpoint.

### Manual resume
```bash
sbatch scripts/launch.sh configs/inject_2048.toml --seed 42
# It will automatically find the latest checkpoint in outputs/inject-2048-seed42/
```

### Auto-monitor and resume
```bash
# Run this periodically (every ~2 hours) to check and resubmit crashed jobs:
bash scripts/monitor_and_resume.sh
```

This script:
1. Lists all expected runs from the experiment matrix
2. For each: checks if job is running, if not checks if training completed
3. If neither running nor completed: resubmits with the same config

## Evaluation

After training completes, run evals:
```bash
bash scripts/eval_all.sh <checkpoint_dir> <eval_budget>
```

Eval conditions:
- E1: no injection, budget=8192
- E2: injection matching training freq, budget=8192
- E3: injection, budget=4096
- E4: injection, budget=16384

## Key Files

| File | Purpose |
|------|---------|
| `configs/*.toml` | Training configs (1 per experiment) |
| `scripts/launch.sh` | Launch single training run |
| `scripts/launch_phase1.sh` | Launch all Phase 1 runs |
| `scripts/check_runs.sh` | Check status of all runs |
| `scripts/monitor_and_resume.sh` | Auto-resume crashed runs |
| `scripts/eval_all.sh` | Run evaluations |
| `budget_injection_env/` | BudgetInjectionEnv package |
| `prime-rl/` | Stock prime-rl clone |
| `outputs/` | Training outputs (gitignored) |

## Troubleshooting

### "Module not found: budget_injection_env"
The env package must be installed in the same venv as prime-rl:
```bash
cd prime-rl && uv pip install -e ../budget_injection_env
```

### NCCL timeout
Usually means a GPU died. The job will crash. Just resubmit — resume_step=-1 handles it.

### OOM
Qwen3-4B Full FT with 2 trainer GPUs + optim_cpu_offload should fit. If OOM:
- Check seq_len (should be 9216)
- Check batch_size (should be 256 with packing)
- Reduce rollouts_per_example from 4 to 2

### Job killed at 48h
Resubmit. The checkpoint at step 30*N will be picked up automatically.
At ~5 min/step, 400 steps = ~33h. Should complete in one 48h job.
