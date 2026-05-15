#!/bin/bash
# export DISABLE_COMPILE=1
run_name="pretrain_mlp_sudoku_generation_gram_50k"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=gram \
seed=42 \
global_batch_size=128 \
data_paths="[data/sudoku-generation-50k]" \
evaluators='[{name: "sudoku@Sudoku"}]' \
epochs=200 eval_interval=20 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
arch.use_mlp_sampler=True \
arch.use_mlp_lm_head=True \
arch.kl_balance_alpha=0.8 \
arch.kl_coef=0.05 \
+run_name=${run_name} ema=True \
checkpoint_every_eval=True
