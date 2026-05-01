#!/bin/bash
#SBATCH --job-name=training22
#SBATCH --partition=gpu_jobs
#SBATCH --nodes=1
#SBATCH --nodelist=turing-2
#SBATCH --gres=gpu:8

# Multi-GPU training for CIFAR10 D3PM

# Number of GPUs
NGPU=$(nvidia-smi -L | wc -l)

# Training from scratch
torchrun --nproc_per_node=8 d3pm_runner_mnist.py

# Resume from checkpoint (uncomment to use)
# torchrun --nproc_per_node=$NGPU d3pm_runner_mnist.py --resume checkpoints/ckpt_latest.pt
