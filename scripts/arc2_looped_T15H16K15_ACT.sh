#!/bin/bash
# ACT-ON variant — ARC-AGI-2 (Ablation B: Looped Transformer + ACT).
# act_grad_at_halt_only=True → full ACT chain in one forward(): no_grad rollout + single grad
# pass at the per-sample halt step. q_halt (>0) halting, TRM/GRAM-style (no_ACT_continue=True).
# T_inner=15 = H·L+H = 3·4+3 (ARC uses H=3,L=4, per scripts/arc2.sh) × halt_max=16 = 240 cycles.
# K=15 = T_inner → warmup 0, all inner cycles supervised (Yang multi-iter loss = mean of 15 CEs).
# Optimizer HPs left at cfg defaults (lr=1e-4, wd=0.1, puzzle_emb_lr=1e-2) to match scripts/arc2.sh.
# Backbone = attention + rope (looped.yaml defaults; ARC gram uses mlp_t=False too).
# evaluators left at default (arc@ARC). NOTE: needs data/arc2concept-aug-1000 (see README build cmd).
SEED=${SEED:-42}
NPROC=${NPROC:-8}
GBS=${GBS:-768}
run_name="pretrain_arc2_looped_T15H16K15_ACT_s${SEED}"
torchrun --nproc-per-node $NPROC --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=looped \
seed=${SEED} \
global_batch_size=$GBS \
data_paths="[data/arc2concept-aug-1000]" \
epochs=200000 \
arch.L_layers=2 \
arch.total_cycles=15 \
arch.supervision_cycles=15 \
arch.halt_max_steps=16 \
arch.halt_exploration_prob=0.1 \
arch.no_ACT_continue=True \
arch.act_grad_at_halt_only=True \
+run_name=${run_name} ema=True \
checkpoint_every_eval=True
