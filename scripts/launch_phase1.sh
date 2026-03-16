#!/bin/bash
# Launch all Phase 1 experiments (4 configs x 2 seeds = 8 jobs).
# Each job is 1 node, 4 GPUs, 48h.
set -e

cd /pscratch/sd/s/siddart2/budget-injection
mkdir -p logs

CONFIGS=(
    "configs/baseline.toml"
    "configs/inject_2048.toml"
    "configs/inject_1024.toml"
    "configs/inject_4096.toml"
)

SEEDS=(42 123)

echo "=== Launching Phase 1: ${#CONFIGS[@]} configs x ${#SEEDS[@]} seeds = $(( ${#CONFIGS[@]} * ${#SEEDS[@]} )) jobs ==="

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        name=$(basename "$config" .toml)-seed${seed}
        echo "Submitting: $name"
        sbatch --job-name="bi-${name}" scripts/launch.sh "$config" --seed "$seed"
        sleep 1  # avoid SLURM rate limiting
    done
done

echo "=== All Phase 1 jobs submitted ==="
echo "Monitor with: squeue --me"
echo "Check status: bash scripts/check_runs.sh"
