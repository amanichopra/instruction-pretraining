#!/usr/bin/env bash
# Evaluate every finished arm. Pass model dirs as label=path, or let it autodetect.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
VAL_SHARD=data/raw/train-02732-of-02755.parquet
LIMIT="${LIMIT:-500}"
NPROMPTS="${NPROMPTS:-150}"

MODELS=()
for arm in arm_a_pretrain arm_b_sft_from_scratch arm_c_pretrain_then_sft; do
  if   [ -d "runs/$arm/hf" ];       then MODELS+=("$arm=runs/$arm/hf")
  else
    latest=$(ls -d runs/$arm/hf_step* 2>/dev/null | sort -V | tail -1 || true)
    [ -n "$latest" ] && MODELS+=("$arm=$latest")
  fi
done
if [ ${#MODELS[@]} -eq 0 ]; then echo "no exported models found"; exit 1; fi
echo "evaluating: ${MODELS[*]}"
mkdir -p results

echo "######## cross-format perplexity ########"
python3 src/evaluation/perplexity.py --models "${MODELS[@]}" \
  --streams pretrain_val=data/tokens/pretrain.val.bin sft_val=data/tokens/sft.val.bin \
  --out results/perplexity_matrix.json

echo "######## zero-shot benchmarks ########"
for m in "${MODELS[@]}"; do
  label="${m%%=*}"; path="${m#*=}"
  python3 src/evaluation/benchmarks.py --model "$path" --label "$label" \
    --limit "$LIMIT" --out "results/benchmarks_${label}.json"
done

echo "######## free-form instruction following ########"
python3 src/evaluation/generate.py --models "${MODELS[@]}" --val-shard "$VAL_SHARD" \
  --n-prompts "$NPROMPTS" --prompt-style template --out results/generations_template.json
python3 src/evaluation/generate.py --models "${MODELS[@]}" --val-shard "$VAL_SHARD" \
  --n-prompts "$NPROMPTS" --prompt-style bare --out results/generations_bare.json

echo "######## report ########"
python3 src/report.py --out results/REPORT.md
echo "EVALS COMPLETE -> results/REPORT.md"
