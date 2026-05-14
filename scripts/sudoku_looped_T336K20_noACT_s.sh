#!/bin/bash
# noACT baseline (re-runnable with SEED env). T=336, halt=1, K=20.
# Use for fair ACT on/off A/B at same compute.
SEED=${SEED:-42}
run_name="pretrain_sudoku_looped_T336K20_noACT_s${SEED}"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=looped \
seed=${SEED} \
global_batch_size=768 \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.total_cycles=336 \
arch.supervision_cycles=20 \
arch.halt_max_steps=1 \
+run_name=${run_name} ema=True \
checkpoint_every_eval=True
