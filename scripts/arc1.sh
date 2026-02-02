#!/bin/bash
run_name="pretrain_att_gram_arc1concept"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=gram \
seed=42 \
global_batch_size=768 data_paths=[data/arc1concept-aug-1000] \
epochs=200000 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 \
arch.kl_coef=0.04 \
arch.kl_balance_alpha=0.8 \
+run_name=${run_name} ema=True checkpoint_every_eval=True