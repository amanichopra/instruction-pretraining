# Pretraining an LLM from scratch: raw-document vs instruction-format data

A single-GPU reproduction of the controlled experiment in **FineInstructions: Scaling
Synthetic Instructions to Pre-Training Scale** (Patel, Raffel & Callison-Burch,
[arXiv:2601.22146](https://arxiv.org/abs/2601.22146)), using the authors'
[`fineinstructions/fineinstructions_nemotron`](https://huggingface.co/datasets/fineinstructions/fineinstructions_nemotron)
dataset.

The paper's claim: because that dataset provides *supervised* instruction/answer pairs at
pre-training scale, you can skip the "pretrain then SFT" pipeline and train a model from
scratch **entirely on the instruction-tuning objective** — which is far more in-distribution
with how LLMs are actually used. They validate it with token-for-token controlled runs.

This repo reproduces that comparison end to end: data pipeline, from-scratch model init,
trainer, and an evaluation suite.

## The experiment

Every arm trains a **randomly initialized 135M-parameter Llama** (the SmolLM2-135M
architecture, weights re-initialized — see "From scratch" below) on an **identical token
budget**.

| arm | training data | what it tests |
|---|---|---|
| **A** `arm_a_pretrain` | raw Nemotron-CC documents (`text`) | standard pretraining |
| **B** `arm_b_sft_from_scratch` | the instruction/answer pairs derived from *the same document pool* | the paper's proposal: SFT format for all of pretraining |
| **C** `arm_c_pretrain_then_sft` | arm A's checkpoint + a short SFT stage | the classic pipeline, as a reference point |

### Training setup at a glance

| | arms A and B (from scratch) | arm C (SFT stage on top of A) |
|---|---|---|
| **Model** | Llama architecture (SmolLM2-135M shape), random init: **134.5M params** (106.2M non-embedding, 28.3M tied embedding). 30 layers, hidden 576, MLP 1536, GQA with 9 query / 3 KV heads, vocab 49,152 | same model, initialized from arm A's final checkpoint |
| **Tokens** | **1.0B per arm** (7,629 steps × 131,072 = 999.95M), under 1 epoch of the prepared data (1.72B raw / 1.60B SFT tokens available) | **50M** more SFT-format tokens (1.05B total including A) |
| **Training steps** | **7,629** optimizer steps | **381** optimizer steps |
| **Batch** | 131,072 tokens/step = micro-batch 16 × seq len 1024 × 8 grad-accumulation steps; documents packed into 1024-token windows | same |
| **LR schedule** | **Linear warmup then cosine decay** (no WSD, no constant phase). Warmup **2%** (152 steps) from ~0 to peak **1.5e-3**, then cosine down to **10% of peak (1.5e-4)** at step 7,629 | warmup **3%** (11 steps) to peak **2e-4**, then cosine to 2e-5 |
| **Optimizer** | **AdamW** (fused), β = (0.9, 0.95), ε = 1e-8, weight decay **0.1** on 2-D weight matrices only (none on norms/biases), grad-norm clip **1.0** | same, with fresh optimizer state |
| **Loss** | next-token cross-entropy over **all** tokens, template tokens included | same |
| **Precision / compile** | bf16 autocast, fp32 master weights, `torch.compile` | same |
| **Checkpointing** | every **500 steps** plus the final step. Each save **overwrites** `runs/<arm>/ckpt/model.pt` atomically (weights + optimizer state + step + tokens seen); at the end the model is exported to `runs/<arm>/hf` (safetensors + tokenizer) | same |
| **In-training eval** | every 500 steps: perplexity on both held-out streams (25 batches × 8 seqs × 1024 tokens each); logging every 20 steps → `runs/<arm>/metrics.jsonl` | same |
| **Seed** | 0 (bit-identical init across arms; each arm's data shuffle is seeded and reproducible, but the arms see different window orders) | 0 |
| **Hardware** | **1× NVIDIA RTX 3090 (24 GB)**, ~38.5k tok/s, ~44% MFU | same |
| **Wall time** | A: 7.2 h, B: 7.3 h | 0.4 h |

The exact launch flags are in `scripts/run_experiments.sh`, and the resolved per-run configs
(including data stats) are in `runs/<arm>/config.json`. Training curves are in
`runs/<arm>/metrics.jsonl`.

### Why the comparison is fair

The dataset stores each source document together with the synthetic instruction pairs
generated from it, so both arms can be built from exactly the same documents:

* **Same document pool.** A document enters the pool iff it has ≥1 instruction pair with a
  Flow-Judge quality score ≥ 4 (the paper's filter; keeps 56.6% of pairs). Both arms are
  built from that one pool: 2,129,188 train documents each. Arm A gets their raw text
  (1.72B tokens), and arm B gets their instruction pairs (4.6M pairs, 1.60B tokens).
* **Not document-for-document during training.** Each arm cuts its own token stream into
  1024-token windows and trains on a seeded random shuffle of them
  (`src/data/loader.py`). The 1B-token budget covers about **58%** of arm A's windows and
  about **63%** of arm B's. The two shuffles are independent because the streams have
  different lengths, so each arm trains on a different random subset of the pool, and the
  two subsets only partly overlap. Both are uniform samples of the same distribution, so this
  should not bias the comparison. It is still a matched-*pool* design, not a matched-*sample*
  one.
* **Same token budget.** Both arms train on exactly the same number of tokens, with the
  same architecture, optimizer, schedule, seed, sequence length and packing. Template
  tokens (`Instruction: ` / `\n\nAnswer: `) count against arm B's budget, as in the paper.
* **Same starting weights.** Every arm is built with `--seed 0`, so all arms begin from
  *bit-identical* random weights (`tests/test_init.py` asserts `max|delta| == 0`). What
  differs is the data format, plus which random subset of the pool each arm sees (see
  above).
* **Disjoint held-out set.** Validation comes from shard 2732, far from the training
  shards (0–11), so there is no document overlap.

### From scratch

`src/model.py` takes a real pretrained release for its architecture and tokenizer, then
re-initializes every weight. `verify_randomized()` proves no pretrained knowledge survives
by comparing our tensors against the published checkpoint:

```
n_knowledge_tensors_checked: 24
max_abs_cosine_sim_knowledge_tensors: 0.00317
is_random_init: true
```

Norm weights are excluded from that verdict on purpose — initializing them to exactly 1.0
is the correct from-scratch init, and an all-ones vector trivially correlates with any
pretrained near-ones norm vector. Only tensors that actually store learned knowledge
(embeddings, attention/MLP projections) decide the verdict.

## Status (as of 2026-11-09)

**Complete.** All three arms are trained to their full budgets (A and B: 1B tokens / 7,629
steps each; C: A's final checkpoint + 50M SFT tokens / 381 steps), the full eval suite has
run, and the LLM judge (claude-opus-5-5, Batch API, ~$2.66) has scored them. Final
models are in `runs/*/hf`; every number below comes from
**[`results/REPORT.md`](results/REPORT.md)** (regenerate with `python3 src/report.py`).

Not yet done (optional): a pairwise B-vs-C judge run, a second seed, and anything at
larger scale (see "Deviations").

## Findings

**Headline: at an identical 1B-token budget, training from scratch on instruction-format
data (B) beats standard raw-document pretraining (A) on instruction following, gives up
nothing on knowledge benchmarks, and matches the classic pretrain → SFT pipeline (C).**
That is the paper's claim, reproduced directionally at 135M params.

### 1. LLM-as-judge — B clearly beats A

| | A (raw pretrain) | B (SFT-format from scratch) | C (pretrain → SFT) |
|---|---|---|---|
| pairwise win rate, A vs B (n=150, both orientations) | 21.0% | **68.7%** | — |
| absolute helpfulness, 1–10 (n=150) | 1.63 | **2.15** | 2.13 |

Ties were 10.3%, and the judge gave the same verdict in both orientations 89% of the time,
so position bias isn't driving the result. On absolute score B and C are tied (2.15 vs
2.13), even though C saw ~5% more tokens. Absolute scores are low for every arm: these are
135M-parameter models, and all three still loop and repeat a lot (see
`results/generations_template.md`).

### 2. Cross-format perplexity — each arm is best on its own format, but B's lead is bigger

| model | raw docs (pretrain_val) | instruction pairs (sft_val) | sft_val, answer tokens only |
|---|---|---|---|
| A | **16.11** | 15.63 | 13.62 |
| B | 18.17 | **10.35** | **10.35** |
| C | 16.57 | 10.92 | 10.38 |

B pays +13% PPL on raw documents but gains −34% on instruction data, versus A. Fifty
million SFT tokens let C nearly catch up with B on answer tokens (10.38 vs 10.35) while
keeping most of A's raw-document fit. Training loss is **not** comparable across arms
(different text), so use this table, not the loss curves.

### 3. Zero-shot benchmarks — no knowledge penalty for B

Mean acc_norm over the tasks scored in each format:

| model | raw-continuation prompt | instruction-template prompt |
|---|---|---|
| A | 39.9 | 41.9 |
| B | 40.7 | 41.8 |
| C | 40.6 | **43.4** |

The arms are within ~1.5 points of each other, and that is mostly noise: each task uses a
500-item subset, which gives a standard error of about ±2 points per task. The one large gap
is BoolQ, where B and C (≈63%) beat A (≈53%). Treat it with care: ~62% of BoolQ answers are
"yes", so this mostly shows that the instruction-trained arms learned to give a calibrated
yes/no answer, not that they know more. A scores better on ARC-easy and SciQ, and B on
OpenBookQA and LAMBADA, all by a few points.

### 4. Free-form generation — B answers and stops; A rambles

| model | token-F1 (bare prompt) | token-F1 (template) | emits EOS (template) |
|---|---|---|---|
| A | 0.158 | 0.149 | 0% |
| B | **0.208** | 0.191 | **26%** |
| C | 0.190 | **0.200** | 14% |

A never stops on its own and often just continues the prompt as if it were a document. B
has the highest reference overlap with the bare prompt, is the only arm that regularly
ends its answer, and has the most diverse output (distinct-3gram 0.415 vs 0.245 for A).

### Caveats

* One seed per arm. The judge and perplexity gaps are large, but the benchmark gaps are not
  significant.
* 135M params and 1B tokens, against the paper's 1.8B params and ~23B tokens. The effect's
  direction matches the paper; its size at scale is untested here.
* Arm A was interrupted and resumed once (details below). The train loss didn't move across
  the resume.

## Results

The full tables, including per-task benchmarks, perplexity during training and the
generation metrics, are in **[`results/REPORT.md`](results/REPORT.md)**. Side-by-side samples
are in `results/generations_{bare,template}.md`, and raw judge verdicts in `results/judge.json`.

## Setup

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128   # match your CUDA
pip install -r requirements.txt
```

## Running it

Run everything from this directory. Raw data, token files, checkpoints and logs are
git-ignored; the commands below regenerate them.

```bash
# 1. data: 12 train shards + 1 disjoint val shard (~12GB download)
python3 src/data/download.py --train-shards 12 --start 0 --val-shards 2732

# 2. build the two token-matched corpora (+ held-out splits)
SHARDS=$(ls data/raw/train-000{0,1}*.parquet | sort)
for arm in pretrain sft; do
  python3 src/data/prepare_data.py --arm $arm --shards $SHARDS \
    --out-dir data/tokens --split train --workers 6
  python3 src/data/prepare_data.py --arm $arm \
    --shards data/raw/train-02732-of-02755.parquet \
    --out-dir data/tokens --split val --workers 1
done

# 3. train all three arms sequentially (TOKENS sets the per-arm budget)
TOKENS=1000000000 bash scripts/run_experiments.sh

# 4. evaluate + build the report
bash scripts/run_evals.sh

# 5. optional: LLM-as-judge (needs a credential)
export ANTHROPIC_API_KEY=...
python3 src/evaluation/judge.py --generations results/generations_template.json \
  --a arm_a_pretrain --b arm_b_sft_from_scratch --mode both
python3 src/report.py          # fold judge results into the report
```

A run that dies partway can be picked up where it stopped — weights, optimizer moments,
step counter and data position are all restored (`--resume` reads `runs/<name>/ckpt`,
which is written atomically so a crash mid-save can't leave a torn checkpoint):

```bash
python3 src/train.py --run-name arm_a_pretrain --train-bin data/tokens/pretrain.train.bin \
  --train-tokens 1000000000 --resume   # ...plus the same flags as the original launch
```

Mid-training checkpoints can be evaluated without stopping the run:

```bash
python3 src/export_ckpt.py --run runs/arm_a_pretrain   # -> runs/arm_a_pretrain/hf_step<N>
```

## Evaluation suite

| script | what it measures |
|---|---|
| `src/evaluation/perplexity.py` | cross-format held-out PPL: every model × every format. The off-diagonal is the interesting part. Also reports answer-tokens-only PPL. |
| `src/evaluation/benchmarks.py` | zero-shot ARC-e/-c, SciQ, PIQA, HellaSwag, WinoGrande, OpenBookQA, BoolQ, LAMBADA by length-normalized log-likelihood — scored under **both** a raw-continuation prompt and the instruction template, since a single format would bias the comparison toward whichever arm it suits. |
| `src/evaluation/generate.py` | free-form generation on held-out instructions: token-F1 and ROUGE-L against the reference, plus reference-free signals (does it emit EOS instead of rambling? does it answer, or just continue the document with a new `Instruction:` header?). Writes side-by-side markdown. |
| `src/evaluation/judge.py` | the paper's protocol: AlpacaEval-style pairwise win rate (each item judged in both orientations so position bias cancels) and MT-Bench-style 1–10 scoring, via the Batch API. |

## Loss masking

The full hyperparameter list is under "Training setup at a glance". Arm B uses **full-sequence loss** (every token contributes), which is what makes it a true
token-for-token comparison against arm A and matches the paper counting template tokens in
its totals. `--loss-on-completion-only` runs the answer-tokens-only ablation instead; the
loss mask is built from character offsets, so the instruction/answer boundary token is
attributed exactly.

## Deviations from the paper

Honest list — this is a single-RTX-3090 reproduction, not a replication at scale:

* **Model size.** 135M params vs the paper's 1.8B.
* **Token budget.** 1B tokens per arm vs ~23B (IPT) / ~300B (Nemotron-CC). Both arms are
  well under one epoch of the prepared data, so neither repeats data.
* **Tokenizer.** SmolLM2's 49k vocab (Apache-2.0, ungated) rather than the paper's Llama-3
  tokenizer. `--model-id` swaps it; the token dtype widens to uint32 automatically above a
  64k vocab.
* **Evaluation.** MixEval / MT-Bench-101 / AlpacaEval are replaced by the suite above.
  `judge.py` implements the same judge protocols (run here with claude-opus-5-5 as the
  judge, on 150 held-out instructions).
* **Arm C token accounting.** Arm C continues from arm A's *final* checkpoint, so it sees
  1B + 50M tokens — about 5% more than arms A and B. If arm B still wins on
  instruction-following, it wins despite that handicap.
* **Arm A was interrupted once.** The machine went down at step 5020; arm A resumed from
  its step-5000 checkpoint, which predates optimizer-state saving, so Adam moments
  restarted from zero with a 100-step LR re-warmup (`--rewarm-steps`). Train loss was
  unchanged across the resume (3.25–3.28 on both sides), and data order is preserved, so
  the effect on the comparison should be negligible. Later checkpoints include optimizer
  state.
* **Data scale.** 12 of 1,909 available shards (~2.1M of ~1.2B pairs).
* **PIQA source.** Loaded from `baber/piqa` (a parquet mirror of the canonical 1,838-row
  validation split); the original `ybisk/piqa` ships a loading script that `datasets>=5`
  no longer executes.

## Layout

```
src/model.py                  architecture from a pretrained release, weights randomized
src/train.py                  trainer (packed memmap, bf16, cosine, resumable)
src/export_ckpt.py            mid-training checkpoint -> HF dir
src/report.py                 aggregate all artifacts -> results/REPORT.md
src/data/download.py          fetch parquet shards + judge score files
src/data/prepare_data.py      judge filter, matched doc pool, tokenize, pack
src/data/loader.py            memmap windows + seeded epoch-aware shuffle
src/evaluation/               perplexity, benchmarks, generation, judge
tests/test_loss.py            chunked loss == full-batch cross-entropy (loss and grads)
tests/test_scoring.py         vectorized benchmark scorer == naive reference loop
tests/test_init.py            arms share identical init; init carries no pretrained weights
scripts/run_experiments.sh    all three arms, sequentially
scripts/run_evals.sh          full evaluation + report
scripts/run_all.sh            train + eval, holding a sleep inhibitor
results/REPORT.md             all result tables (generated)
results/*.json, *.md          raw eval outputs, side-by-side generations, judge verdicts
runs/<arm>/config.json        resolved hyperparameters + data stats for each run
runs/<arm>/metrics.jsonl      training loss / LR / grad-norm / eval curves
```
