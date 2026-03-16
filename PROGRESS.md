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
- [x] uv sync inside container (works, 304 packages)
- [x] BudgetInjectionEnv imports + unit tests pass (9/9) inside container
- [ ] Smoke test — 5-step training with countdown env (job 50123943)
- [ ] Need rg_mix_env module (the rg-mix dataset loader from compaction-rl)
- [ ] Verify BudgetInjectionEnv works end-to-end with prime-rl training loop

## What's Running

| Job ID | Config | Node | Status | Notes |
|--------|--------|------|--------|-------|
| 50126477 | smoke_test.sh | pending | Running | rg-mix-env + Qwen3-0.6B, 5 steps, m5017 premium |

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

- **rg_mix_env is NOT a built-in verifiers env**: It's a custom env from compaction-rl
  that loads a pre-generated multi-task dataset. Need to either:
  (a) Copy the rg_mix_env module from compaction-rl, or
  (b) Use stock verifiers `ReasoningGymEnv` with multi-task gym config, or
  (c) Create a small rg_mix_env loader in this project
  Dataset exists at `/pscratch/sd/s/siddart2/datasets/rg_mix_7500/` (7500 problems,
  5 tasks: zebra_puzzles_7, arc_1d, sokoban_hard, cryptarithm, countdown_7).
- **PYTHONPATH for env discovery**: `uv run` doesn't see `uv pip install -e` packages.
  Must set `PYTHONPATH=$SCRATCH/budget-injection` so verifiers' `load_environment()`
  can find `budget_injection_env` (and eventually `rg_mix_env`).
- **pyproject.toml build-backend**: Fixed — use `setuptools.build_meta` not
  `setuptools.backends._legacy:_Backend`.
- **[inference] section required**: All configs must have `[inference]` section
  (even empty) for prime-rl to auto-start inference servers on the node.
- **Qwen3-4B model**: Verify `Qwen/Qwen3-4B` is available. May need HuggingFace token.

## Run Log

### 2026-03-16 — Instance #48: Project Setup + Smoke Test Debugging
- Created entire project: research plan, configs, env, scripts, instructions
- Deep research: 8 web search agents found 30+ papers on budget-aware reasoning
- Key finding: BudgetThinker (2508.17196) is closest prior work; Seed-OSS-36B is only
  shipped model with inline budget reflection
- Cloned prime-rl. uv sync works (304 packages from cache).
- BudgetInjectionEnv: imports work, 9/9 unit tests pass inside container.
- Fixed pyproject.toml build-backend (setuptools.build_meta).
- Fixed PYTHONPATH issue (uv run doesn't see uv pip install -e packages).
- Fixed missing [inference] section in configs.
- Discovered rg_mix_env is NOT a built-in verifiers env — need custom loader.
- Submitted countdown smoke test (job 50123943) as pipeline validation.
- TODO: create rg_mix_env loader, test BudgetInjectionEnv end-to-end.
