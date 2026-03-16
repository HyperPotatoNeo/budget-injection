#!/bin/bash
# Launch all Phase 2 experiments (3 configs x 2 seeds = 6 jobs).
set -e

cd /pscratch/sd/s/siddart2/budget-injection
mkdir -p logs

CONFIGS=(
    "configs/inject_ratio.toml"
    "configs/inject_urgency.toml"
    "configs/inject_minimal.toml"
)

SEEDS=(42 123)

echo "=== Launching Phase 2: ${#CONFIGS[@]} configs x ${#SEEDS[@]} seeds = $(( ${#CONFIGS[@]} * ${#SEEDS[@]} )) jobs ==="

for config in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        name=$(basename "$config" .toml)-seed${seed}
        echo "Submitting: $name"
        sbatch --job-name="bi-${name}" scripts/launch.sh "$config" --seed "$seed"
        sleep 1
    done
done

echo "=== All Phase 2 jobs submitted ==="
