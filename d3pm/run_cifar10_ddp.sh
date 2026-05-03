#!/bin/bash
# Multi-GPU training for CIFAR10 D3PM

# Number of GPUs
NGPU=$(nvidia-smi -L | wc -l)

# Training from scratch
torchrun --nproc_per_node=8 d3pm_runner_cifar10.py

# Resume from checkpoint (uncomment to use)
# torchrun --nproc_per_node=$NGPU d3pm_runner_cifar10.py --resume checkpoints/ckpt_latest.pt
