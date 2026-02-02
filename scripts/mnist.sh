#!/bin/bash
run_name="pretrain_att_mnist_2"
torchrun --nproc-per-node 8 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=gram \
seed=42 \
global_batch_size=768 \
data_paths="[data/mnist_2]" \
evaluators='[{name: "mnist@MNIST", pred_vocab_size: 3}]' \
epochs=1800 eval_interval=5 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
arch.use_mlp_sampler=True \
arch.use_mlp_lm_head=True \
arch.kl_balance_alpha=0.8 \
arch.kl_coef=0.015 \
arch.use_patchify=True \
arch.patch_size=2 \
arch.image_size=28 \
arch.in_channels=1 \
+run_name=${run_name} ema=True \
checkpoint_every_eval=True
