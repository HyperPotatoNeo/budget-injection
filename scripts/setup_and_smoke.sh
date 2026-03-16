#!/bin/bash
# Setup prime-rl environment and run smoke tests.
# Run this INSIDE the container on a compute node.
set -e

echo "=== Budget Injection Setup & Smoke Test ==="
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
echo "Time: $(date)"

cd /pscratch/sd/s/siddart2/budget-injection/prime-rl

# Step 1: Install prime-rl
echo ""
echo "=== Step 1: uv sync ==="
export UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache
uv sync --all-extras 2>&1 | tail -5

# Step 2: Install BudgetInjectionEnv
echo ""
echo "=== Step 2: Install BudgetInjectionEnv ==="
uv pip install -e ../budget_injection_env 2>&1 | tail -3

# Step 3: Verify imports
echo ""
echo "=== Step 3: Verify imports ==="
uv run python -c "
import verifiers as vf
print(f'verifiers: OK')
import budget_injection_env
print(f'budget_injection_env: OK')
from budget_injection_env.env import BudgetInjectionEnv, _format_budget_message
print(f'BudgetInjectionEnv class: OK')
msg = _format_budget_message('absolute', 2048, 8192, 6144)
print(f'Format test: {msg}')
print('All imports OK!')
"

# Step 4: Run unit tests
echo ""
echo "=== Step 4: Unit tests ==="
uv run python -m pytest ../budget_injection_env/tests/ -v 2>&1 | tail -15

# Step 5: Check dataset
echo ""
echo "=== Step 5: Check rg_mix dataset ==="
if [ -d "/pscratch/sd/s/siddart2/datasets/rg_mix_7500" ]; then
    echo "Dataset found at /pscratch/sd/s/siddart2/datasets/rg_mix_7500"
    ls /pscratch/sd/s/siddart2/datasets/rg_mix_7500/ | head -5
else
    echo "WARNING: Dataset not found! Need to generate rg_mix_7500"
fi

# Step 6: Smoke test - baseline (5 steps, no injection)
echo ""
echo "=== Step 6: Baseline smoke test (5 steps) ==="
uv run rl @ /pscratch/sd/s/siddart2/budget-injection/configs/baseline.toml \
    max_steps=5 \
    output_dir=outputs/smoke-baseline \
    wandb.name=smoke-baseline \
    wandb.project=budget-injection-smoke \
    ckpt.interval=5 \
    2>&1 | tail -30

echo ""
echo "=== Smoke test complete ==="
echo "Check outputs/smoke-baseline/ for results"
