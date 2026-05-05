#!/bin/bash
#SBATCH --job-name=d3pm-sudoku-t1
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

run_name="${RUN_NAME:-d3pm-sudoku50k-absorbzero-lr1e4-dim768-l5-ep800-t1-8gpu}"
data_dir="${DATA_DIR:-data/sudoku-generation-50k}"
batch_size="${BATCH_SIZE:-64}"
nproc=8
epochs="${EPOCHS:-800}"
seed="${SEED:-42}"
dim="${DIM:-768}"
n_layers="${N_LAYERS:-5}"
n_heads="${N_HEADS:-12}"
lr="${LR:-1e-4}"
forward_type="${FORWARD_TYPE:-absorb_zero}"

train_examples=$(python -c "import numpy as np; print(np.load('${data_dir}/train/all__labels.npy', mmap_mode='r').shape[0])")
per_rank_examples=$(((train_examples + nproc - 1) / nproc))
steps_per_epoch=$(((per_rank_examples + batch_size - 1) / batch_size))
steps="${STEPS:-$((epochs * steps_per_epoch))}"

checkpoint_path="${CHECKPOINT_PATH:-checkpoints_sudoku/${run_name}}"

echo "[$(date)] turing-1 SLURM job: nproc=${nproc}, epochs=${epochs}, max_steps=${steps}, dim=${dim}, n_layers=${n_layers}, lr=${lr}, forward_type=${forward_type}"

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
torchrun --nproc-per-node ${nproc} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 d3pm/d3pm_runner_sudoku.py \
  --data-dir "${data_dir}" \
  --max-steps "${steps}" \
  --batch-size "${batch_size}" \
  --eval-samples "${EVAL_SAMPLES:-10000}" \
  --run-name "${run_name}" \
  --checkpoint-path "${checkpoint_path}" \
  --seed "${seed}" \
  --dim "${dim}" \
  --n-layers "${n_layers}" \
  --n-heads "${n_heads}" \
  --lr "${lr}" \
  --forward-type "${forward_type}" \
  ${RESUME:+--resume "${RESUME}"}
