#!/bin/bash
# Ablation B (Looped Transformer, ACT off) — Graph Coloring 8.
# T = (H_cycles*L_cycles + H_cycles) * halt_max_steps = (3*4+3)*16 = 240
# K = (H_cycles*L_cycles + H_cycles) - 1 = 14
run_name="pretrain_looped_graph_coloring_8_T240K14_noACT"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=looped \
seed=42 \
global_batch_size=768 \
data_paths="[data/graph-coloring-8]" \
evaluators="[]" \
epochs=5000 eval_interval=50 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.total_cycles=240 \
arch.supervision_cycles=14 \
arch.halt_max_steps=1 \
+run_name=${run_name} ema=True \
checkpoint_every_eval=True
