# Progress Log

**Read this file first** when picking up the project. It tracks what's been done, what's
running, and what to do next. Update it after every significant action.

## Current Status

**Phase**: Setup & Smoke Testing
**Last Updated**: 2026-03-16
**Last Instance**: #48

## What's Done

- [x] Created repo at https://github.com/HyperPotatoNeo/budget-injection.git
- [x] Research plan with 5 hypotheses, 10 configs, 18 training runs
- [x] Related work survey (30+ papers)
- [x] BudgetInjectionEnv skeleton (MultiTurnEnv wrapper, 4 message formats)
- [x] All 10 TOML configs (1 node, 4 GPU, 400 steps, rollouts=4)
- [x] Launch/monitor/resume scripts
- [x] INSTRUCTIONS.md for future instances
- [x] Cloned prime-rl into `prime-rl/` subdirectory
- [ ] uv sync inside container (needs compute node)
- [ ] Install budget_injection_env into prime-rl venv
- [ ] Smoke test (5 steps of baseline on interactive node)
- [ ] Verify BudgetInjectionEnv works with prime-rl training loop

## What's Running

| Job ID | Config | Node | Status | Notes |
|--------|--------|------|--------|-------|
| 50123152 | smoke_test.sh | nid008225 | Running | uv sync + 5-step baseline, m5017 premium |

<!--
When jobs are running, update this section:
| Job ID | Config | Seed | Node | Status | Step | Notes |
|--------|--------|------|------|--------|------|-------|
| 12345  | inject-2048 | 42 | nid00123 | Running | 120/400 | Started 2026-03-16 |
-->

## What To Do Next

1. **Get interactive node**: `salloc -A m4881 -C "gpu&hbm80g" --qos=interactive --time 4:00:00 --gpus-per-node 4`
2. **Setup env inside container**:
   ```bash
   export HOME=$SCRATCH
   export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
   cd $SCRATCH
   podman-hpc run --rm -it \
     --user "$(id -u):$(id -g)" --replace --name budget-inject \
     --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
     -e SCRATCH -e HOME \
     -v "$SCRATCH":"$SCRATCH" -v "/global/homes/s/siddart2":"/global/homes/s/siddart2" \
     -w "$SCRATCH/budget-injection/prime-rl" \
     docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 /bin/bash
   ```
3. **Inside container**:
   ```bash
   unset NCCL_SOCKET_IFNAME
   UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache uv sync --all-extras
   uv pip install -e ../budget_injection_env
   ```
4. **Smoke test baseline** (5 steps, no injection):
   ```bash
   uv run rl @ $SCRATCH/budget-injection/configs/baseline.toml \
     max_steps=5 output_dir=outputs/smoke-test wandb.name=smoke-test
   ```
5. **Smoke test injection** (5 steps, with BudgetInjectionEnv):
   ```bash
   uv run rl @ $SCRATCH/budget-injection/configs/inject_2048.toml \
     max_steps=5 output_dir=outputs/smoke-test-inject wandb.name=smoke-test-inject
   ```
6. If smoke tests pass → launch Phase 1

## Known Issues

- **BudgetInjectionEnv not yet tested with prime-rl training loop**: The MultiTurnEnv
  approach should work but needs verification that multi-turn trajectories flow correctly
  through the trainer (token concatenation, loss masking, etc.)
- **rg_mix dataset**: Verify it exists at `/pscratch/sd/s/siddart2/datasets/rg_mix_7500`.
  If not, need to generate it (check compaction-rl for how).
- **Qwen3-4B model**: May need to be `Qwen/Qwen3-4B` or `Qwen/Qwen3-4B-Instruct-2507`.
  Check which is available on the node's HF cache.
- **2 inf + 2 train on 1 node**: The `[deployment] num_train_gpus=2, num_infer_gpus=2`
  config needs verification. Prime-rl may handle GPU assignment differently than expected.

## Run Log

### 2026-03-16 — Instance #48: Project Setup
- Created entire project: research plan, configs, env, scripts, instructions
- Deep research: 8 web search agents found 30+ papers on budget-aware reasoning
- Key finding: BudgetThinker (2508.17196) is closest prior work; Seed-OSS-36B is only
  shipped model with inline budget reflection
- Cloned prime-rl. Need compute node for uv sync + smoke test.
