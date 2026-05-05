#!/bin/bash
#SBATCH --job-name=sudoku-gen-eval
#SBATCH --partition=gpu_jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --output=/data/mingyu/research/codebase/recursive/gram-sudoku-gen/logs/%x-%j.out
#SBATCH --error=/data/mingyu/research/codebase/recursive/gram-sudoku-gen/logs/%x-%j.err
set -euo pipefail

source /data/mingyu/miniconda3/etc/profile.d/conda.sh
conda activate trm_128
cd /data/mingyu/research/codebase/recursive/gram-sudoku-gen

: "${MODEL_TYPE:?MODEL_TYPE required (d3pm|trm)}"
: "${CKPT:?CKPT required}"
out_name="${OUT_NAME:-${MODEL_TYPE}_eval}"
num_samples="${NUM_SAMPLES:-200000}"
batch_size="${BATCH_SIZE:-100}"
seed="${SEED:-0}"
viz_dir="/data/mingyu/research/codebase/recursive/gram-sudoku-gen/experiments/sudoku_gen_eval_viz/${out_name}"

echo "[$(date)] eval: model=${MODEL_TYPE} ckpt=${CKPT} N=${num_samples} BS=${batch_size} seed=${seed}"
echo "viz_dir=${viz_dir}"
nvidia-smi --query-gpu=index,name --format=csv,noheader | head -8

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
  sudoku_gen_eval.py \
  --model_type "${MODEL_TYPE}" \
  --checkpoint "${CKPT}" \
  --num_samples "${num_samples}" \
  --batch_size "${batch_size}" \
  --seed "${seed}" \
  --save_viz "${viz_dir}"
