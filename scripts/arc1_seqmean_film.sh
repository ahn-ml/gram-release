#!/bin/bash
#SBATCH --job-name=arc1_seqmean_film
#SBATCH --partition=gpu_jobs
#SBATCH --nodes=1
#SBATCH --nodelist=turing-2
#SBATCH --gres=gpu:8
#SBATCH --output=/data/mingyu/research/codebase/recursive/gram/logs/gram_arc1_seqmean_film_%j.log

source /data/mingyu/miniconda3/etc/profile.d/conda.sh
conda activate trm_128
cd /data/mingyu/research/codebase/recursive/gram

run_name="pretrain_att_gram_arc1concept_seqmean_film"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=gram \
seed=42 \
global_batch_size=768 data_paths=[data/arc1concept-aug-1000] \
epochs=200000 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
arch.kl_coef=8.0 \
arch.kl_balance_alpha=0.8 \
arch.kl_seq_mean=True \
arch.film_epsilon=True \
+run_name=${run_name} ema=True checkpoint_every_eval=True
