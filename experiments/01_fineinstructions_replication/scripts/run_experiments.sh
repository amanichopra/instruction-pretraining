#!/usr/bin/env bash
# Token-matched FineInstructions format comparison on one RTX 3090.
# Runs sequentially: arm A, arm B, then arm C (classic pipeline reference).
# Every arm passes --resume, so re-running this after a crash or reboot continues each
# arm from its last checkpoint (finished arms are skipped; fresh arms start from step 0).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

TOKENS="${TOKENS:-1000000000}"      # 1B tokens per arm, matched exactly
SFT_STAGE_TOKENS="${SFT_STAGE_TOKENS:-50000000}"
SEQ=1024
MICRO=16
TPS=131072                          # tokens per optimizer step
LR=1.5e-3
COMMON="--model-id HuggingFaceTB/SmolLM2-135M --size 135m --seq-len $SEQ \
  --micro-batch $MICRO --tokens-per-step $TPS --eval-every 500 --eval-batches 25 \
  --save-every 500 --log-every 20 --seed 0 --resume \
  --val-bins pretrain_val=data/tokens/pretrain.val.bin sft_val=data/tokens/sft.val.bin"

echo "################ ARM A: standard pretraining ($TOKENS tokens) ################"
python3 src/train.py --run-name arm_a_pretrain \
  --train-bin data/tokens/pretrain.train.bin \
  --train-tokens "$TOKENS" --lr $LR $COMMON 2>&1 | tee -a logs/arm_a_pretrain.log

echo "################ ARM B: SFT-format from scratch ($TOKENS tokens) ################"
python3 src/train.py --run-name arm_b_sft_from_scratch \
  --train-bin data/tokens/sft.train.bin --train-mask data/tokens/sft.train.mask \
  --train-tokens "$TOKENS" --lr $LR $COMMON 2>&1 | tee -a logs/arm_b_sft_from_scratch.log

echo "################ ARM C: arm A + short SFT stage ($SFT_STAGE_TOKENS tokens) ################"
python3 src/train.py --run-name arm_c_pretrain_then_sft \
  --train-bin data/tokens/sft.train.bin --train-mask data/tokens/sft.train.mask \
  --init-from runs/arm_a_pretrain/ckpt \
  --train-tokens "$SFT_STAGE_TOKENS" --lr 2e-4 --warmup-frac 0.03 $COMMON 2>&1 | tee -a logs/arm_c.log

echo "ALL RUNS COMPLETE"
