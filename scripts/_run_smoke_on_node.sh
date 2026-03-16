#!/bin/bash
# Wrapper to run smoke test inside container on a compute node.
# Usage: ssh <node> "bash /pscratch/sd/s/siddart2/budget-injection/scripts/_run_smoke_on_node.sh"
set -e

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
  bash -c "unset NCCL_SOCKET_IFNAME && bash $SCRATCH/budget-injection/scripts/setup_and_smoke.sh"
