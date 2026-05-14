#!/bin/bash
# ACT-ON variant. T_inner=21 × halt_max_steps=16 = 336 (compute match w/ T336K20_noACT).
# act_grad_at_halt_only=True → full ACT chain in one forward(): no_grad rollout + single grad
# pass at the per-sample halt step. q_halt (>0) halting, TRM/GRAM-style (no_ACT_continue=True).
# K=21 = T_inner → warmup 0, all 21 inner cycles supervised (Yang multi-iter loss = mean of 21 CEs).
# backward() fires once per puzzle; q_halt supervised at every rolled-out step.
SEED=${SEED:-42}
NPROC=${NPROC:-8}
GBS=${GBS:-768}
run_name="pretrain_sudoku_looped_T21H16K21_ACT_s${SEED}"
torchrun --nproc-per-node $NPROC --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=looped \
seed=${SEED} \
global_batch_size=$GBS \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
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
checkpoint_every_eval=True
