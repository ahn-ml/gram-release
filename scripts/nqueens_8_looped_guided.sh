#!/bin/bash
# Looped Transformer + GRAM-style stochastic guidance — N-Queens 8.
# T=240, P=5 (= L_cycles+1, 1 H-cycle per block) → 48 blocks total.
# Last K_b=3 blocks (15 cycles ≈ K=14) run with grad + readout + KL.
# CVAE-style z injection at every block boundary.
run_name="pretrain_looped_nqueens_8_T240P5Kb3_guided"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=looped \
seed=42 \
global_batch_size=768 \
data_paths="[data/nqueens-8]" \
evaluators="[]" \
epochs=5000 eval_interval=50 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.total_cycles=240 \
arch.supervision_cycles=14 \
arch.halt_max_steps=1 \
arch.use_stochastic_guidance=True \
arch.inject_period=5 \
arch.supervised_blocks=3 \
arch.latent_dim=256 \
arch.use_mlp_sampler=True \
arch.kl_coef=0.1 \
+run_name=${run_name} ema=True \
checkpoint_every_eval=True
