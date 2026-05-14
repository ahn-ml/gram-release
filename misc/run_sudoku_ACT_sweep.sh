#!/bin/bash
# Sudoku ACT on/off A/B sweep. Runs chained sequentially (each run grabs all visible GPUs).
# Order: ACT-on s42 → ACT-on s1.
# GPU 0 occupied by another user → use GPUs 1-7 (7-GPU, GBS=672). noACT baseline
# (pretrain_sudoku_looped_T336K20_noACT) already exists at GBS=768 — note the 768→672 mismatch.
set -u
cd /home/junyeob/gram

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}
export NPROC=${NPROC:-7}
export GBS=${GBS:-672}

LOG=/home/junyeob/gram/misc/sudoku_ACT_sweep.log
GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' ')

# mlml-cli: register GPUs on the shared occupancy sheet; auto-clear on exit/kill (lab convention).
register_gpus () { for g in $GPUS; do mlml-cli set --gpu "$g" --name junyeob --mode Train >/dev/null 2>&1; done; }
clear_gpus    () { for g in $GPUS; do mlml-cli clear --gpu "$g" >/dev/null 2>&1; done; echo "[$(date -Iseconds)] GPUS_CLEARED" >> "$LOG"; }
trap clear_gpus SIGTERM SIGINT EXIT
register_gpus

echo "[$(date -Iseconds)] SUDOKU_ACT_SWEEP_START gpus=$CUDA_VISIBLE_DEVICES nproc=$NPROC gbs=$GBS" >> "$LOG"

run_one () {
  local LABEL="$1"; local SCRIPT="$2"; local SEED="$3"
  local OUT="/home/junyeob/gram/misc/sudoku_ACT_${LABEL}.out"
  local ERR="/home/junyeob/gram/misc/sudoku_ACT_${LABEL}.err"
  echo "[$(date -Iseconds)] $LABEL START seed=$SEED" >> "$LOG"
  SEED=$SEED NPROC=$NPROC GBS=$GBS bash "$SCRIPT" > "$OUT" 2> "$ERR"
  echo "[$(date -Iseconds)] $LABEL EXITED rc=$?" >> "$LOG"
}

run_one act_s42 /home/junyeob/gram/scripts/sudoku_looped_T21H16K21_ACT.sh 42
run_one act_s1  /home/junyeob/gram/scripts/sudoku_looped_T21H16K21_ACT.sh 1

echo "[$(date -Iseconds)] SUDOKU_ACT_SWEEP_DONE" >> "$LOG"
