"""Standalone test for the unique_answers metric path used in sudoku_gen_eval.py.

Builds synthetic boards from real training labels (all valid sudokus), injects
known duplicates and known invalids, then runs the EXACT metric code path
(batch_verify_sudoku + tobytes set) to confirm the counts match expectations.
"""
import numpy as np
import torch
from evaluators.sudoku_utils import batch_verify_sudoku


def metric(boards_np):
    """Replica of sudoku_gen_eval.py main metric block, accepts np int64 (N,81)."""
    boards_t = torch.from_numpy(boards_np)
    blanks_t = torch.ones_like(boards_t)
    valid = batch_verify_sudoku(blanks_t, boards_t).cpu().numpy()
    valid_idx = np.where(valid)[0]
    unique_answers = len({boards_np[i].tobytes() for i in valid_idx}) if len(valid_idx) else 0
    return int(valid.sum()), unique_answers


def expect(label, actual, expected):
    ok = "OK" if actual == expected else "FAIL"
    print(f"  [{ok}] {label}: got {actual}, expected {expected}")
    return actual == expected


labels = np.load("/data/mingyu/research/codebase/recursive/gram-sudoku-gen/data/sudoku-generation-50k/train/all__labels.npy").astype(np.int64).reshape(-1, 81)
real = labels[:100]  # 100 real valid sudoku boards

results = []

# Test 1: 100 unique valid boards
print("Test 1: 100 unique valid boards")
v, u = metric(real.copy())
results.append(expect("valid_count", v, 100))
results.append(expect("unique_answers", u, 100))

# Test 2: each board duplicated 2x -> 200 boards, 100 unique
print("Test 2: 200 boards (each of 100 duplicated 2x)")
dup = np.concatenate([real, real], axis=0)
v, u = metric(dup)
results.append(expect("valid_count", v, 200))
results.append(expect("unique_answers", u, 100))

# Test 3: 50 unique + 50 boards repeated 3x = 50 + 150 = 200 boards, 100 unique
print("Test 3: 50 unique + (50 boards repeated 3x) = 200 boards, 100 unique")
mix = np.concatenate([real[:50], np.tile(real[50:], (3, 1))], axis=0)
v, u = metric(mix)
results.append(expect("valid_count", v, 200))
results.append(expect("unique_answers", u, 100))

# Test 4: extreme duplication - all 200 boards identical = 1 unique
print("Test 4: 200 identical boards")
same = np.tile(real[0:1], (200, 1))
v, u = metric(same)
results.append(expect("valid_count", v, 200))
results.append(expect("unique_answers", u, 1))

# Test 5: inject INVALID boards - they should be excluded from valid_count and from unique pool
print("Test 5: 100 valid + 30 invalid (PAD/OOB/row-collision); valid pool has 1 dup pair")
mixed = real.copy()
mixed_invalid = np.tile(real[0:1], (30, 1))  # 30 copies of board0
mixed_invalid[0, 0] = 0    # PAD -> invalid
mixed_invalid[1, 5] = 99   # OOB -> invalid
mixed_invalid[2, 0] = mixed_invalid[2, 1]  # row collision (assuming row 0 had distinct)
# rows 3..29: invalid via PAD at pos 80
for k in range(3, 30):
    mixed_invalid[k, 80] = 0
combined = np.concatenate([mixed, mixed_invalid], axis=0)  # 130 boards
# Add a duplicate among valid: copy real[0] as 131st (now valid pool has 100 + 1 dup of real[0])
combined = np.concatenate([combined, real[0:1]], axis=0)  # 131 boards, 101 valid (100 real + 1 dup)
v, u = metric(combined)
results.append(expect("valid_count", v, 101))   # 100 originals + 1 dup
results.append(expect("unique_answers", u, 100))  # dup deduped -> 100

# Test 6: empty valid pool - all invalid
print("Test 6: 50 boards all invalid (PAD)")
allbad = np.zeros((50, 81), dtype=np.int64)
v, u = metric(allbad)
results.append(expect("valid_count", v, 0))
results.append(expect("unique_answers", u, 0))

print()
print(f"Result: {sum(results)}/{len(results)} passed")
exit(0 if all(results) else 1)
