# Generative Recursive reAsoning Models (GRAM)
Author: Junyeob Baek, Mingyu Jo, Minsu Kim, Yoshua Bengio, Sungjin Ahn

This is official codebase for the paper: "Generative Recursive reAsoning Models". 
### Abstract
We introduce Generative Recursive reAsoning Models (GRAM), a recursion-based generative model that is effective for complex planning and reasoning problems. GRAM reformulates recent latent recursive architectures as a stochastic generative process with probabilistic latent transitions, enabling efficient and stable computation entirely in latent space without relying on token-level sequences as in chain-of-thought (CoT) prompting. We optimize this generative recursion via amortized variational inference, allowing the model to represent and explore multiple plausible latent trajectories conditioned on the input. This formulation supports both conditional reasoning through p(y|x) and unconditional generative modeling through p(x). Empirically, GRAM achieves strong performance on challenging reasoning benchmarks, competitive with much larger language models on ARC-1 and ARC-2, demonstrating the effectiveness of recursion-based generative modeling for System 2 tasks.

## Overview
<div align="center">
<img src="misc/overview.png" height="400px" alt="GRAM Overview">
</div>


### Requirements

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version
pip install -r requirements.txt # install requirements
pip install --no-cache-dir --no-build-isolation adam-atan2 
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

### Dataset Preparation

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

## Note: You cannot train on both ARC-AGI-1 and ARC-AGI-2 and evaluate them both because ARC-AGI-2 training data contains some ARC-AGI-1 eval data

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples, 1000 augments

# N-Queens
# TBD

# Graph Coloring
# TBD

# MNIST
# TBD
```

## Experiments

### ARC-AGI-1 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days

### ARC-AGI-2 (assuming 4 H-100 GPUs):

```bash
run_name="pretrain_att_arc2concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc2concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Runtime:* ~3 days

### Sudoku-Extreme (assuming 1 L40S GPU):

```bash
run_name="pretrain_mlp_t_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

*Runtime:* < 36 hours

### N-Queens
TBD

### Graph Coloring
TBD

### MNIST
TBD


## Reference
TBD

This code is based on the Hierarchical Reasoning Model [code](https://github.com/sapientinc/HRM) and the Tiny Reasoning Model [code](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).
