# FineInstructions reproduction: pretraining format comparison

Re-implementation of the controlled, token-for-token experiment from *FineInstructions: Scaling Synthetic Instructions to Pre-Training Scale* (Patel, Raffel & Callison-Burch, [arXiv:2601.22146](https://arxiv.org/abs/2601.22146)), at single-GPU scale.

## 1. Data

| arm | split | tokens | source docs | instr pairs | judge>= | shards |
|---|---|---|---|---|---|---|
| pretrain | train | 1724.7M | 2,129,188 | 4,643,345 | 4 | 12 |
| pretrain | val | 166.0M | 168,422 | 400,463 | 4 | 1 |
| sft | train | 1596.8M | 2,129,188 | 4,643,345 | 4 | 12 |
| sft | val | 148.3M | 168,422 | 400,463 | 4 | 1 |

Both arms derive from the **same 2,129,188 source documents** (a document is included iff it has at least one instruction pair with Flow-Judge score >= 4, the paper's filter). Only the *format* differs.

## 2. Training runs

| run | params | token budget | steps | step reached | last train loss | tok/s | wall |
|---|---|---|---|---|---|---|---|
| A - standard pretraining on raw documents | 135M | 1.00B | 7629 | 7620 | 3.242 | 38.5k | 7.2h |
| B - SFT-format pretraining from scratch (FineInstructions) | 135M | 1.00B | 7629 | 7620 | 2.597 | 38.6k | 7.3h |
| C - pretrain then short SFT stage (classic pipeline) | 135M | 0.05B | 381 | 380 | 2.697 | 38.7k | 0.4h |

### Held-out perplexity during training (last eval)

| run | step | pretrain_val ppl | sft_val ppl |
|---|---|---|---|
| arm_a_pretrain | 7629 | 15.79 | 14.97 |
| arm_b_sft_from_scratch | 7629 | 17.90 | 9.90 |
| arm_c_pretrain_then_sft | 381 | 16.26 | 10.43 |

## 3. Cross-format held-out perplexity

**scope: all_tokens**

| model | pretrain_val | sft_val |
|---|---|---|
| arm_a_pretrain | 16.11 | 15.63 |
| arm_b_sft_from_scratch | 18.17 | 10.35 |
| arm_c_pretrain_then_sft | 16.57 | 10.92 |

**scope: answer_tokens_only**

| model | sft_val |
|---|---|
| arm_a_pretrain | 13.62 |
| arm_b_sft_from_scratch | 10.35 |
| arm_c_pretrain_then_sft | 10.38 |

## 4. Zero-shot benchmarks (length-normalized log-likelihood)

**prompt format: instruction** (acc_norm %)

| model | arc_challenge | arc_easy | boolq | hellaswag | lambada | openbookqa | piqa | sciq | winogrande | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| arm_a_pretrain | 25.4 | 42.0 | 53.6 | 36.2 | - | 24.4 | 60.4 | 51.2 | - | **41.9** |
| arm_b_sft_from_scratch | 24.7 | 37.8 | 63.0 | 36.6 | - | 25.2 | 59.8 | 45.8 | - | **41.8** |
| arm_c_pretrain_then_sft | 26.4 | 38.2 | 62.8 | 37.8 | - | 26.6 | 62.0 | 49.8 | - | **43.4** |

**prompt format: raw** (acc_norm %)

| model | arc_challenge | arc_easy | boolq | hellaswag | lambada | openbookqa | piqa | sciq | winogrande | mean |
|---|---|---|---|---|---|---|---|---|---|---|
| arm_a_pretrain | 24.1 | 40.6 | 51.8 | 37.0 | 19.0 | 22.4 | 62.4 | 50.6 | 51.4 | **39.9** |
| arm_b_sft_from_scratch | 26.1 | 39.8 | 62.8 | 35.4 | 21.2 | 26.4 | 58.0 | 46.4 | 49.8 | **40.7** |
| arm_c_pretrain_then_sft | 24.4 | 37.8 | 59.2 | 37.2 | 20.2 | 23.8 | 61.8 | 49.4 | 51.6 | **40.6** |

## 5a. Free-form instruction following (`generations_bare.json`, prompt_style=bare)

| model | token_f1 | rouge_l | terminated_frac | mean_new_tokens | distinct_3gram | echoes_instruction_frac | empty_frac |
|---|---|---|---|---|---|---|---|
| arm_a_pretrain | 0.158 | 0.139 | 0.000 | 256.000 | 0.154 | 0.000 | 0.000 |
| arm_b_sft_from_scratch | 0.208 | 0.182 | 0.020 | 251.933 | 0.234 | 0.000 | 0.000 |
| arm_c_pretrain_then_sft | 0.190 | 0.164 | 0.080 | 242.780 | 0.231 | 0.000 | 0.007 |

`terminated_frac` = fraction of generations that emitted EOS instead of running to the token cap (an instruction-following model knows when to stop). `echoes_instruction_frac` = fraction that emitted a new `Instruction:` header (a raw LM continuing the document rather than answering).

## 5b. Free-form instruction following (`generations_template.json`, prompt_style=template)

| model | token_f1 | rouge_l | terminated_frac | mean_new_tokens | distinct_3gram | echoes_instruction_frac | empty_frac |
|---|---|---|---|---|---|---|---|
| arm_a_pretrain | 0.149 | 0.135 | 0.000 | 256.000 | 0.245 | 0.000 | 0.000 |
| arm_b_sft_from_scratch | 0.191 | 0.168 | 0.260 | 198.380 | 0.415 | 0.000 | 0.000 |
| arm_c_pretrain_then_sft | 0.200 | 0.176 | 0.140 | 229.380 | 0.239 | 0.000 | 0.000 |

`terminated_frac` = fraction of generations that emitted EOS instead of running to the token cap (an instruction-following model knows when to stop). `echoes_instruction_frac` = fraction that emitted a new `Instruction:` header (a raw LM continuing the document rather than answering).

## 6. LLM-as-judge (model: claude-opus-5-5)

**Pairwise win rate** (arm_a_pretrain vs arm_b_sft_from_scratch, both orientations, n=150):

| arm_a_pretrain | arm_b_sft_from_scratch | tie | orientation agreement |
|---|---|---|---|
| 21.0% | 68.7% | 10.3% | 89% |

**Absolute helpfulness (1-10):**

| model | mean score | n |
|---|---|---|
| arm_a_pretrain | 1.63 | 150 |
| arm_b_sft_from_scratch | 2.15 | 150 |
| arm_c_pretrain_then_sft | 2.13 | 150 |

