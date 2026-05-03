#!/bin/bash
# Multi-seed runner for D3PM-MNIST: absorb_zero -> uniform per seed.
# Run from the d3pm/ directory (uses ./data and ./contents like the original script).
cd /data/mingyu/research/codebase/recursive/gram/d3pm
mkdir -p contents

NGPU=$(nvidia-smi -L | wc -l)

for seed in 43 44; do
  for forward_type in absorb_zero uniform; do
    run_name="d3pm_mnist_${forward_type}_seed${seed}"
    ckpt_dir="checkpoints_mnist_${forward_type}_seed${seed}"
    mkdir -p ${ckpt_dir}
    torchrun --nproc_per_node=${NGPU} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 \
      d3pm_runner_mnist.py \
      --seed ${seed} \
      --forward_type ${forward_type} \
      --run_name ${run_name} \
      --checkpoint_path ${ckpt_dir}
  done
done
