#!/bin/bash
set -euo pipefail

data_dir="${DATA_DIR:-data/sudoku-generation-50k}"
train_size="${TRAIN_SIZE:-50000}"
test_size="${TEST_SIZE:-10000}"
num_aug="${NUM_AUG:-0}"
seed="${SEED:-42}"
epochs="${EPOCHS:-200}"
nproc="${NPROC_PER_NODE:-8}"
batch_size="${BATCH_SIZE:-64}"

if [[ -z "${RESUME:-}" ]]; then
  python dataset/build_sudoku_generation_dataset.py \
    --output-dir "${data_dir}" \
    --train-size "${train_size}" \
    --test-size "${test_size}" \
    --num-aug "${num_aug}" \
    --seed "${seed}"
else
  echo "RESUME is set; using existing dataset at ${data_dir}"
fi

train_examples=$(python -c "import numpy as np; print(np.load('${data_dir}/train/all__labels.npy', mmap_mode='r').shape[0])")
per_rank_examples=$(((train_examples + nproc - 1) / nproc))
steps_per_epoch=$(((per_rank_examples + batch_size - 1) / batch_size))
default_steps=$((epochs * steps_per_epoch))
steps="${STEPS:-${default_steps}}"
data_tag="${data_dir##*/}"
run_name="${RUN_NAME:-d3pm_sudoku_${data_tag}_aug${num_aug}_seed${seed}_${steps}steps_b${batch_size}x${nproc}}"
if [[ -n "${RESUME:-}" && -z "${CHECKPOINT_PATH:-}" ]]; then
  checkpoint_path="$(dirname "${RESUME}")"
else
  checkpoint_path="${CHECKPOINT_PATH:-checkpoints_sudoku/${run_name}}"
fi

echo "D3PM budget: train_examples=${train_examples}, epochs=${epochs}, steps_per_epoch=${steps_per_epoch}, steps=${steps}"
if [[ -n "${RESUME:-}" ]]; then
  echo "D3PM resume: source=${RESUME}, checkpoint_path=${checkpoint_path}"
fi

resume_args=()
if [[ -n "${RESUME:-}" ]]; then
  resume_args=(--resume "${RESUME}")
fi

export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
torchrun --nproc-per-node "${nproc}" --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 d3pm/d3pm_runner_sudoku.py \
  "${resume_args[@]}" \
  --data-dir "${data_dir}" \
  --max-steps "${steps}" \
  --batch-size "${batch_size}" \
  --eval-samples "${EVAL_SAMPLES:-10000}" \
  --run-name "${run_name}" \
  --checkpoint-path "${checkpoint_path}" \
  --seed "${seed}"
