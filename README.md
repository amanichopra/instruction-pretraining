# Instruction pretraining and future learning

Research on how instruction-formatted pretraining affects subsequent supervised learning.

Start with the [research directions and annotated reading list](docs/research_directions.md). It contains 26 reference papers, four ranked project ideas, and a proposed first study.

## Experiments

| # | experiment | status | summary |
|---|---|---|---|
| 01 | [FineInstructions replication](experiments/01_fineinstructions_replication/) | complete (2026-11-05) | 135M Llama from scratch on 1B tokens per arm, 1× RTX 3090. At equal tokens, instruction-format pretraining beats raw-document pretraining on instruction following (LLM-judge win rate 68.7% vs 21.0%), with no loss on zero-shot benchmarks, and ties the pretrain → SFT pipeline. |

Each experiment directory is self-contained: code, scripts, tests, a README with the full
training setup and findings, and the result files. Data and model checkpoints are not
committed; each README explains how to regenerate them.

## Layout

```
docs/                    research planning and literature review
experiments/NN_<name>/   one directory per experiment
```
