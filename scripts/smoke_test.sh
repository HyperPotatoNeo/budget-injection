#!/bin/bash
# Smoke test: setup prime-rl env and run 5 steps of baseline.
#
#SBATCH -A m5017
#SBATCH -C "gpu&hbm80g"
#SBATCH --qos=premium
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --output=/pscratch/sd/s/siddart2/budget-injection/logs/smoke-%j.out
#SBATCH --error=/pscratch/sd/s/siddart2/budget-injection/logs/smoke-%j.err

set -ex

echo "=== Budget Injection Smoke Test ==="
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"

cd /global/homes/s/siddart2
export HOME=/pscratch/sd/s/siddart2
export SCRATCH=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd $SCRATCH

podman-hpc run --rm \
  --user "$(id -u):$(id -g)" --replace --name budget-inject-smoke \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH -e HOME -e WANDB_API_KEY=595199cad0de28f309ce22cb212dcbeeb21b06d8 \
  -v "$SCRATCH":"$SCRATCH" -v "/global/homes/s/siddart2":"/global/homes/s/siddart2" \
  -w "$SCRATCH/budget-injection/prime-rl" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash -exc '
unset NCCL_SOCKET_IFNAME
export UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache

echo "=== Step 1: uv sync ==="
uv sync --all-extras

echo ""
echo "=== Step 2: Install BudgetInjectionEnv ==="
uv pip install -e ../budget_injection_env

echo ""
echo "=== Step 3: Verify imports ==="
uv run python -c "
import verifiers as vf; print(\"verifiers OK\")
from budget_injection_env.env import BudgetInjectionEnv, _format_budget_message
msg = _format_budget_message(\"absolute\", 2048, 8192, 6144)
print(f\"Format: {msg}\")
print(\"All imports OK\")
"

echo ""
echo "=== Step 4: Unit tests ==="
uv run python -m pytest ../budget_injection_env/tests/ -v

echo ""
echo "=== Step 5: Check dataset ==="
ls /pscratch/sd/s/siddart2/datasets/rg_mix_7500/ 2>/dev/null && echo "Dataset OK" || echo "WARNING: Dataset not found"

echo ""
echo "=== Step 6: Baseline smoke test (5 steps) ==="
uv run rl @ /pscratch/sd/s/siddart2/budget-injection/configs/baseline.toml \
    max_steps=5 \
    output_dir=outputs/smoke-baseline \
    wandb.name=smoke-baseline \
    wandb.project=budget-injection-smoke \
    ckpt.interval=5

echo ""
echo "=== SMOKE TEST COMPLETE ==="
echo "Time: $(date)"
'
