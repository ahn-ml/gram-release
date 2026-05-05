"""Sudoku symmetry-aware canonical form for unique-board counting.

Drop-in: import canon_digits_batch and dedupe canonicalized boards.
To disable, delete this file and the canon block in sudoku_gen_eval.py.

Token space: PAD=0, blank=1, digits 1..9 -> tokens 2..10.

Stage 1 only: digit-relabel canonical (collapses 9! = 362,880 orbit).
First-seen digit in row-major scan -> 2, second -> 3, ..., ninth -> 10.
"""
import numpy as np


def canon_digits_batch(boards):
    """boards: (N, 81) int. Returns (N, 81) digit-canonical."""
    N, L = boards.shape
    out = np.empty_like(boards)
    for r in range(N):
        seen = {}
        nxt = 2
        b, o = boards[r], out[r]
        for i in range(L):
            v = int(b[i])
            if v < 2:
                o[i] = v
                continue
            if v not in seen:
                seen[v] = nxt
                nxt += 1
            o[i] = seen[v]
    return out
