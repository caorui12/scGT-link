#!/bin/bash
#SBATCH --job-name=scgt_gtL
#SBATCH --partition=crmh-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=/home/rui.cao/deeplearning/Graph_Transformer/logs/slurm_ablation_gt_layers_%j.out
#SBATCH --error=/home/rui.cao/deeplearning/Graph_Transformer/logs/slurm_ablation_gt_layers_%j.err

# GT depth ablation on 8 datasets (layers 0/2/4/6/8 × 50ep, scGT-Link v2).
# Pin physical GPU 7 only.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=7

REPO="/home/rui.cao/deeplearning/Graph_Transformer"
PY="${PYTHON_BIN:-$HOME/anaconda3/bin/python}"
cd "${REPO}"
mkdir -p logs

echo "=== $(date -Is) job=${SLURM_JOB_ID:-na} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true

"${PY}" -u scripts/run_ablation_gt_layers_8ds.py \
  --benchmark_root "/home/rui.cao/deeplearning/benchmark/Dataset/Benchmark Dataset" \
  --output_root "${REPO}/out_ablation_gt_layers_50ep" \
  --epochs 50 \
  --seed 42 \
  --layers 0,2,4,6,8 \
  --skip_existing \
  --continue_on_error \
  2>&1 | tee -a logs/ablation_gt_layers_50ep.log

echo "=== $(date -Is) finished job=${SLURM_JOB_ID:-na} ==="
