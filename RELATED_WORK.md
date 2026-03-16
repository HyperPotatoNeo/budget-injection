# Related Work: Budget-Aware Reasoning in LLMs

## Most Directly Related

### BudgetThinker (Aug 2025, arxiv 2508.17196)
Periodically inserts K=8 special control tokens at budget fraction positions (7/8, 6/8,
..., 1/8 remaining). Two-stage: SFT + curriculum GRPO. DeepSeek-R1-Distill-Qwen 1.5B/7B.
+4.9% accuracy. Key finding: continuous reminders beat upfront-only specification.
Ratio-based signals outperform fixed-interval.

**Difference from ours**: Special tokens vs natural language. SFT+RL vs RL-only. Math-only
vs multi-task. No budget generalization or TTS interaction studied.

### SelfBudgeter (May 2025, arxiv 2505.11274)
Model self-predicts `<budget>N</budget>` before reasoning, then adheres. Cold-start SFT +
budget-guided GRPO. 61% length compression on math. Under review ICLR 2026.

### L1 / LCPO (Mar 2025, arxiv 2503.04697, COLM 2025)
"Think for N tokens" appended to prompt. RL reward penalizes deviation from target length.
1.5B model surpasses GPT-4o at equal reasoning lengths. Training-based.

### s1: Budget Forcing (Jan 2025, arxiv 2501.19393, EMNLP 2025)
Inference-time: suppress end-of-thinking token, append "Wait" to extend reasoning. No
training. s1-32B exceeded o1-preview by 27% on math. Coined "budget forcing."

### Seed-OSS-36B (ByteDance, production)
Only shipped open-weight model with inline budget reflection. Generates
`<seed:cot_budget_reflect>I have used 129 tokens, 383 remaining</seed:cot_budget_reflect>`
tags. Trained on budget multiples of 512. Not documented in Seed1.5-Thinking paper.

## Budget Injection / Control (Training-Based)

| Paper | arxiv | Date | Mechanism |
|-------|-------|------|-----------|
| BARD | 2511.01470 | Nov 2025 | Budget as user control signal + contrastive SFT + RL |
| e1 | 2510.27042 | Oct 2025 | Continuous effort fraction, RL-trained |
| AdaCtrl | 2505.18822 | May 2025 | Length-triggered tags + difficulty self-awareness + RL |
| SABER | 2508.10026 | Aug 2025 | 4 tiers: NoThink/FastThink/CoreThink/DeepThink |
| Hansel | 2412.14033 | Dec 2024 | Hidden special tokens indicating remaining word count |
| LDPE | 2412.11937 | Dec 2024 | Positional encoding countdown to termination |

## Budget Injection / Control (Inference-Only)

| Paper | arxiv | Date | Mechanism |
|-------|-------|------|-----------|
| Budget Guidance | 2506.13752 | Jun 2025 | Gamma distribution predictor on hidden states |
| TALE | 2412.18547 | Dec 2024 | "Use less than N tokens" in prompt. ACL 2025. |
| NoWait | 2506.08343 | Jun 2025 | Suppress "Wait"/"Hmm" tokens. 27-51% reduction. |
| Thinking Intervention | 2503.24370 | Mar 2025 | Insert/revise tokens during reasoning |
| Overclocking | 2506.07240 | Jun 2025 | Manipulate internal progress encoding |
| Continue-Thinking Token | 2506.11274 | Jun 2025 | Learned `<|continue|>` embedding via RL |

## Length Penalties in RL (Related but Different Approach)

| Paper | arxiv | Date | Key Idea |
|-------|-------|------|----------|
| Kimi K1.5 Long2Short | 2501.12599 | Jan 2025 | Length penalty reward + reduced max rollout |
| Kimi K2.5 Toggle | 2602.02276 | Feb 2026 | Alternating budget-limited / standard RL phases |
| DeepSeek V3.2 | 2512.02556 | Dec 2025 | Length constraint reward model |
| ALP (Just Enough Thinking) | 2506.05256 | Jun 2025 | Per-prompt adaptive penalty scaled by solve rate |
| GRPO-lambda | 2505.18086 | May 2025 | Dynamic switch: length penalty when easy, accuracy when hard |
| DRPO | 2510.04474 | Oct 2025 | Decoupled correct/incorrect length signals |
| DLER | 2510.15110 | Oct 2025 | Fixing RL instability from length penalties |
| SmartThinker | 2603.08000 | Mar 2026 | Progressive calibration toward optimal length |

## Anytime Reasoning / Budget Allocation

| Paper | arxiv | Date | Key Idea |
|-------|-------|------|----------|
| BRPO/AnytimeReasoner | 2505.13438 | May 2025 | Truncate at random budgets, dense rewards. NeurIPS. |
| Elastic Reasoning | 2505.05315 | May 2025 | Separate thinking/solution budgets |
| HBPO | 2507.15844 | Jul 2025 | Hierarchical budget partitions (512-2560) |
| Predictive Scheduling | 2602.01237 | Feb 2026 | MLP predicts optimal length, batch allocator |

## Theoretical Frameworks

| Paper | arxiv | Date | Key Idea |
|-------|-------|------|----------|
| Reasoning as Compression (CIB) | 2603.08462 | Mar 2026 | Budget forcing = lossy compression under Info Bottleneck |
| Relative-Budget Theory | 2602.01523 | Feb 2026 | Three regimes: deficient/balanced/ample |
| Scaling Test-Time Compute | 2408.03314 | Aug 2024 | Optimal allocation of inference-time budget |
| Think Budget Not All You Need | 2512.19585 | Dec 2025 | Self-consistency beats raw budget increases |

## Surveys

- **Stop Overthinking** (2503.16419, TMLR 2025): First structured survey on efficient reasoning
- **Reasoning on a Budget** (2507.02076, Jul 2025): L1/L2 taxonomy (fixed vs adaptive)
- **Efficient Reasoning Models** (2504.10903, TMLR 2025): shorter/smaller/faster taxonomy

## Production Model Budget Mechanisms

| Model | Method | Model sees remaining budget? |
|-------|--------|:---:|
| Seed-OSS-36B | `<seed:cot_budget_reflect>` inline tags | YES |
| Qwen3 | Server-side "stop now" injection | No |
| Kimi K2.5 | Toggle (alternating budget/standard RL phases) | No |
| DeepSeek V3.2 | Length constraint reward model | No |
| GPT-5 | reasoning.effort parameter | No |
| Claude Opus 4.6 | Adaptive thinking | No |
| Gemini 2.5 | thinkingBudget API param | No |
| NVIDIA NIM | BudgetControlLogitsProcessor | No |
