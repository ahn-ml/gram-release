#!/bin/bash
# Short benchmark — measure step/s + peak memory at T=32, K=8 on silver 4090s.
# epochs=10, eval_interval=10 → no eval during train, single eval at end (~1 min).
run_name="bench_sudoku_looped_T32K8"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=looped \
seed=42 \
global_batch_size=768 \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=10 eval_interval=10 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.total_cycles=32 \
arch.supervision_cycles=8 \
+run_name=${run_name} ema=True \
checkpoint_every_eval=False
