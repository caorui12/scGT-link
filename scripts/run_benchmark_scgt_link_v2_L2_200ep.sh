#!/usr/bin/env bash
# scGT-Link v2 with gt_num_layers=2 — 44-dataset × 200-epoch suite.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

CONDA_ENV="${CONDA_ENV:-graph-transformer}"
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="$PYTHON_BIN"
elif [[ -x "$HOME/anaconda3/bin/python" ]]; then
  PY="$HOME/anaconda3/bin/python"
elif [[ -x "$HOME/anaconda3/envs/$CONDA_ENV/bin/python" ]]; then
  PY="$HOME/anaconda3/envs/$CONDA_ENV/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

if [[ -n "${BENCHMARK_ROOT:-}" ]]; then
  :
elif [[ -d "/home/rui.cao/deeplearning/benchmark/Dataset/Benchmark Dataset" ]]; then
  BENCHMARK_ROOT="/home/rui.cao/deeplearning/benchmark/Dataset/Benchmark Dataset"
else
  BENCHMARK_ROOT="${GRAPH_TRANSFORMER_BENCHMARK_ROOT:-$ROOT/data/Dataset/Benchmark Dataset}"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/out_benchmark_scgt_link_v2_L2_200ep}"
EPOCHS="${EPOCHS:-200}"
SEED="${SEED:-42}"
SPLIT_SUBDIR="${SPLIT_SUBDIR:-Train_validation_test}"
LOG="${LOG:-$ROOT/logs/benchmark_scgt_link_v2_L2_200ep.log}"

export PYTHONUNBUFFERED=1
export GRAPH_TRANSFORMER_BENCHMARK_ROOT="$BENCHMARK_ROOT"
export GRAPH_TRANSFORMER_BENCHMARK_OUTPUT_ROOT="$OUTPUT_ROOT"

{
  echo "=== scGT-Link v2 L2 (gt_num_layers=2) benchmark ==="
  echo "host:       $(hostname 2>/dev/null || true)"
  echo "started:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo:       $ROOT"
  echo "python:     $PY"
  echo "benchmark:  $BENCHMARK_ROOT"
  echo "output:     $OUTPUT_ROOT"
  echo "epochs:     $EPOCHS  seed=$SEED"
  echo "train_args: prior_graph=train_pos_pearson expr_encoder=scgpt gt_variant=directed gt_num_layers=2"
  echo "================================"
} | tee -a "$LOG"

exec "$PY" -u scripts/benchmark_train_grn_suite.py \
  --benchmark_root "$BENCHMARK_ROOT" \
  --output_root "$OUTPUT_ROOT" \
  --split_subdir "$SPLIT_SUBDIR" \
  --epochs "$EPOCHS" \
  --seed "$SEED" \
  --train_arg prior_graph=train_pos_pearson \
  --train_arg expr_encoder=scgpt \
  --train_arg gt_variant=directed \
  --train_arg gt_num_layers=2 \
  --skip_existing \
  --continue_on_error \
  2>&1 | tee -a "$LOG"
