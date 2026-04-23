#!/bin/bash
#SBATCH --job-name=looped-sudoku
#SBATCH --partition=gpu_jobs
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/work/junyeob/gram/misc/sudoku_looped.out
#SBATCH --error=/work/junyeob/gram/misc/sudoku_looped.err
# turing-2 is currently under non-slurm load (jypark crafterdojo sweep ~63 GB / GPU);
# slurm still marks it "idle" but placing an 8-GPU job there would collide on memory.
# Exclude turing-2 so we queue on turing-1 (will start when jypark slurm jobs clear).
#SBATCH --exclude=turing-2

set -euo pipefail
echo "=== start $(date -Is) on $(hostname) JOB=$SLURM_JOB_ID GPUs=${SLURM_JOB_GPUS:-?} ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/junyeob/conda_envs/gram
cd /work/junyeob/gram

export PYTHONUNBUFFERED=1
export DISABLE_COMPILE=1    # skip torch.compile (Blackwell+triton nightly can be flaky)
export WANDB_MODE=offline   # sync later if desired

bash scripts/sudoku_looped.sh
echo "=== end $(date -Is) ==="
