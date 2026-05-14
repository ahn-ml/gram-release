#!/bin/bash
# Smoke for ACT-on path: verify act_grad_at_halt_only chain (no_grad rollout + single grad pass
# + per-step q_halt loss) doesn't OOM/NaN. Short run, no checkpoint save.
run_name="smoke_sudoku_looped_ACT"
NPROC=${NPROC:-8}
GBS=${GBS:-768}
torchrun --nproc-per-node $NPROC --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=looped \
seed=42 \
global_batch_size=$GBS \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=2 eval_interval=2 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.total_cycles=21 \
arch.supervision_cycles=21 \
arch.halt_max_steps=16 \
arch.halt_exploration_prob=0.1 \
arch.no_ACT_continue=True \
arch.act_grad_at_halt_only=True \
+run_name=${run_name} ema=True \
checkpoint_every_eval=False
