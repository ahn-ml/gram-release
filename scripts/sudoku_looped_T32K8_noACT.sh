#!/bin/bash
# Fix version — Yang pure Looped Transformer (ACT off via halt_max_steps=1).
# T=32, K=8. Each training step = one fresh input, T cycles, K readouts averaged.
run_name="pretrain_sudoku_looped_T32K8_noACT"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=looped \
seed=42 \
global_batch_size=768 \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.total_cycles=32 \
arch.supervision_cycles=8 \
arch.halt_max_steps=1 \
+run_name=${run_name} ema=True \
checkpoint_every_eval=True
