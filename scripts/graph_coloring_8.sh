#!/bin/bash
run_name="pretrain_att_gram_graph_coloring_8"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=gram \
seed=42 \
global_batch_size=768 \
data_paths="[data/graph-coloring-8]" \
evaluators="[]" \
epochs=5000 eval_interval=50 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
arch.kl_balance_alpha=0.8 \
arch.kl_coef=0.5 \
+run_name=${run_name} ema=True \
checkpoint_every_eval=True