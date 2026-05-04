#!/bin/bash
# Smoke test for looped + stochastic guidance on Sudoku.
# Verifies VariationalLayer init, _inject path, KL flow, kl_n_blocks divisor.
# Short run, no checkpoint save.
run_name="smoke_sudoku_looped_guided"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=looped \
seed=42 \
global_batch_size=768 \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=2 eval_interval=2 \
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
checkpoint_every_eval=False
