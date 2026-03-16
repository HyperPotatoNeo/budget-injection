#!/bin/bash
# Launch a single budget-injection training run on 1 node (4 GPUs).
# Usage: sbatch scripts/launch.sh configs/inject_2048.toml [--seed 42]
#
#SBATCH -A m4881
#SBATCH -C "gpu&hbm80g"
#SBATCH --qos=premium
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -e

CONFIG="${1:?Usage: sbatch scripts/launch.sh <config.toml> [--seed N]}"
SEED=42

# Parse --seed argument
shift
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Derive run name from config filename
CONFIG_NAME=$(basename "$CONFIG" .toml)
RUN_NAME="${CONFIG_NAME}-seed${SEED}"

echo "=== Budget Injection Training ==="
echo "Config: $CONFIG"
echo "Seed: $SEED"
echo "Run name: $RUN_NAME"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "================================="

# Setup
cd /global/homes/s/siddart2
export HOME=/pscratch/sd/s/siddart2
export SCRATCH=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd $SCRATCH

# Create log directory
mkdir -p $SCRATCH/budget-injection/logs

# Run inside container
podman-hpc run --rm \
  --user "$(id -u):$(id -g)" --replace --name budget-inject-${SLURM_JOB_ID} \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH -e HOME -e WANDB_API_KEY=595199cad0de28f309ce22cb212dcbeeb21b06d8 \
  -v "$SCRATCH":"$SCRATCH" -v "/global/homes/s/siddart2":"/global/homes/s/siddart2" \
  -w "$SCRATCH/budget-injection/prime-rl" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash -c "
    unset NCCL_SOCKET_IFNAME
    source .venv/bin/activate
    uv run rl @ $SCRATCH/budget-injection/${CONFIG} \
      output_dir=outputs/${RUN_NAME} \
      wandb.name=${RUN_NAME} \
      2>&1 | tee $SCRATCH/budget-injection/logs/${RUN_NAME}-\$(date +%Y%m%d-%H%M%S).log
  "
