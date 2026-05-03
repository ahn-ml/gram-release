#!/bin/bash
# Multi-seed runner for N-Queens 10.
# Order per seed: TRM -> HRM -> GRAM. Direct is run last with a nested L_layers sweep (8, 32).
cd /data/mingyu/research/codebase/recursive/gram

for seed in 43 44; do
  # ---- TRM ----
  run_name="pretrain_trm_nqueens_10_seed${seed}"
  torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm \
  seed=${seed} \
  global_batch_size=768 \
  data_paths="[data/nqueens-10]" \
  evaluators="[]" \
  epochs=1000 eval_interval=5 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True \
  checkpoint_every_eval=True

  # ---- HRM ----
  run_name="pretrain_hrm_nqueens_10_seed${seed}"
  torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=hrm \
  seed=${seed} \
  global_batch_size=768 \
  data_paths="[data/nqueens-10]" \
  evaluators="[]" \
  epochs=1000 eval_interval=5 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.H_layers=4 arch.L_layers=4 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True \
  checkpoint_every_eval=True

  # ---- GRAM ----
  run_name="pretrain_gram_nqueens_10_seed${seed}"
  torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=gram \
  seed=${seed} \
  global_batch_size=768 \
  data_paths="[data/nqueens-10]" \
  evaluators="[]" \
  epochs=1000 eval_interval=5 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  arch.kl_balance_alpha=0.8 \
  arch.kl_coef=0.045 \
  +run_name=${run_name} ema=True \
  checkpoint_every_eval=True
done

# ---- Direct (L_layers sweep, runs after all the above) ----
for seed in 43 44; do
  for L_layers in 8 32; do
    run_name="pretrain_direct_nqueens_10_L${L_layers}_seed${seed}"
    torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
    arch=direct \
    seed=${seed} \
    global_batch_size=768 \
    data_paths="[data/nqueens-10]" \
    evaluators="[]" \
    epochs=1000 eval_interval=5 \
    lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
    arch.L_layers=${L_layers} \
    +run_name=${run_name} ema=True \
    checkpoint_every_eval=True
  done
done
