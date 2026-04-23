#!/bin/bash
#SBATCH --job-name=looped-sudoku-100k
#SBATCH --partition=gpu_jobs
#SBATCH --nodelist=turing-2
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/work/junyeob/gram/misc/sudoku_looped_100k.out
#SBATCH --error=/work/junyeob/gram/misc/sudoku_looped_100k.err

set -euo pipefail
echo "=== start $(date -Is) on $(hostname) JOB=$SLURM_JOB_ID GPUs=${SLURM_JOB_GPUS:-?} ==="
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /work/junyeob/conda_envs/gram
cd /work/junyeob/gram
export PYTHONUNBUFFERED=1
export DISABLE_COMPILE=1
export WANDB_MODE=offline
bash scripts/sudoku_looped_100k.sh
echo "=== end $(date -Is) ==="
