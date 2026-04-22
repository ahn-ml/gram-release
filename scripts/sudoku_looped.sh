#!/bin/bash
# Ablation B: canonical Looped Transformer on Sudoku-Extreme.
# Mirrors scripts/sudoku.sh (GRAM) for FLOP-matched comparison.
run_name="pretrain_sudoku_looped_ablation_b"
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
arch.total_cycles=21 \
+run_name=${run_name} ema=True \
checkpoint_every_eval=True
