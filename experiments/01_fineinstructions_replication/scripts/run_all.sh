#!/usr/bin/env bash
# Train every arm, then evaluate. Safe to re-run: training resumes from checkpoints.
# Holds a sleep inhibitor so the machine doesn't suspend mid-run.
set -euo pipefail
cd "$(dirname "$0")/.."
exec systemd-inhibit --what=sleep:idle --who=pretrain_from_scratch \
  --why="LLM pretraining run" bash -c '
  set -euo pipefail
  bash scripts/run_experiments.sh
  bash scripts/run_evals.sh
  echo "PIPELINE COMPLETE"'
