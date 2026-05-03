#!/bin/bash
# Multi-seed runner for Sudoku-Extreme.
# Order:
#   1) TRM, GRAM (mlp_t=True, pos_encodings=none — GRAM-matched setup) per seed
#   2) Direct sweep over L_layers in {8, 32}
#   3) HRM with the original HRM-paper Sudoku-1K config (rope pos, no mlp_t, yaml-default cycles, no ema)
cd /data/mingyu/research/codebase/recursive/gram

for seed in 43 44; do
  # ---- TRM ----
  run_name="pretrain_trm_sudoku_seed${seed}"
  torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=trm \
  seed=${seed} \
  global_batch_size=768 \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  +run_name=${run_name} ema=True \
  checkpoint_every_eval=True

  # ---- GRAM ----
  run_name="pretrain_gram_sudoku_seed${seed}"
  torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=gram \
  seed=${seed} \
  global_batch_size=768 \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=6 \
  arch.use_value_head=true \
  arch.kl_balance_alpha=0.8 \
  +run_name=${run_name} ema=True \
  checkpoint_every_eval=True
done

# ---- Direct (L_layers sweep) ----
for seed in 43 44; do
  for L_layers in 8 32; do
    run_name="pretrain_direct_sudoku_L${L_layers}_seed${seed}"
    torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
    arch=direct \
    seed=${seed} \
    global_batch_size=768 \
    data_paths="[data/sudoku-extreme-1k-aug-1000]" \
    evaluators="[]" \
    epochs=50000 eval_interval=5000 \
    lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
    arch.mlp_t=True arch.pos_encodings=none \
    arch.L_layers=${L_layers} \
    +run_name=${run_name} ema=True \
    checkpoint_every_eval=True
  done
done

# ---- HRM (original HRM-paper Sudoku-1K setup: rope pos, yaml-default cycles H=2/L=2, layers 4/4, no mlp_t) ----
# Reference: github.com/sapientinc/HRM README — Sudoku-Extreme 1K command.
# N=H_cycles=2, T=L_cycles=2 from paper Fig.4 pseudocode (`def hrm(z, x, N=2, T=2)`).
# ema=True added for parity with the other archs in this script.
for seed in 43 44; do
  run_name="pretrain_hrm_sudoku_paper_seed${seed}"
  torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
  arch=hrm \
  seed=${seed} \
  global_batch_size=768 \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=20000 eval_interval=2000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
  +run_name=${run_name} ema=True \
  checkpoint_every_eval=True
done
