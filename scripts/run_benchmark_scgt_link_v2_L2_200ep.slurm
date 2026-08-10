#!/bin/bash
#SBATCH --job-name=scgt_v2_L2
#SBATCH --partition=crmh-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=200:00:00
#SBATCH --output=/home/rui.cao/deeplearning/Graph_Transformer/logs/slurm_scgt_v2_L2_200_%j.out
#SBATCH --error=/home/rui.cao/deeplearning/Graph_Transformer/logs/slurm_scgt_v2_L2_200_%j.err

# scGT-Link v2, gt_num_layers=2: 44 × 200ep. Pin GPU 7 only.
set -euo pipefail
export CUDA_VISIBLE_DEVICES=7

REPO="/home/rui.cao/deeplearning/Graph_Transformer"
cd "${REPO}"
mkdir -p logs

echo "=== $(date -Is) job=${SLURM_JOB_ID:-na} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader || true

bash scripts/run_benchmark_scgt_link_v2_L2_200ep.sh

echo "=== $(date -Is) finished job=${SLURM_JOB_ID:-na} ==="
