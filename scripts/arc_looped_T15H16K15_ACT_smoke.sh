#!/bin/bash
# Smoke for ARC ACT-on path: verify act_grad_at_halt_only chain doesn't OOM/NaN on ARC data.
# Short run, no checkpoint save. DATA env picks arc1 (default) or arc2.
run_name="smoke_arc_looped_ACT"
NPROC=${NPROC:-8}
GBS=${GBS:-768}
DATA=${DATA:-data/arc1concept-aug-1000}
torchrun --nproc-per-node $NPROC --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=looped \
seed=42 \
global_batch_size=$GBS \
data_paths="[${DATA}]" \
epochs=2 eval_interval=2 \
arch.L_layers=2 \
arch.total_cycles=15 \
arch.supervision_cycles=15 \
arch.halt_max_steps=16 \
arch.halt_exploration_prob=0.1 \
arch.no_ACT_continue=True \
arch.act_grad_at_halt_only=True \
+run_name=${run_name} ema=True \
checkpoint_every_eval=False
