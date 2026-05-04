#!/bin/bash
# Looped Transformer + GRAM-style stochastic guidance — Sudoku-Extreme.
# T = (H_cycles*L_cycles + H_cycles) * halt_max_steps = (3*6+3)*16 = 336
# K = (H_cycles*L_cycles + H_cycles) - 1 = 20
# P = L_cycles+1 = 7 (1 H-cycle per block) → H_b = T/P = 48 blocks
# K_b = 3 → 21 supervised cycles ≈ K=20 (1 inner forward worth of grad).
# CVAE-style z injection at every block boundary; ACT off (halt_max_steps=1).
run_name="pretrain_sudoku_looped_T336P7Kb3_guided"
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
arch.total_cycles=336 \
arch.supervision_cycles=20 \
arch.halt_max_steps=1 \
arch.use_stochastic_guidance=True \
arch.inject_period=7 \
arch.supervised_blocks=3 \
arch.latent_dim=256 \
arch.use_mlp_sampler=True \
arch.kl_coef=0.1 \
+run_name=${run_name} ema=True \
checkpoint_every_eval=True
