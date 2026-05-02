"""Sudoku verification utilities.

Token encoding follows the dataset builders: PAD=0, blank=1, digits 1..9 are
stored as tokens 2..10.
"""

import numpy as np
import torch


VALID_TOKENS = set(range(2, 11))


def verify_sudoku(inputs, prediction) -> bool:
    inp = np.asarray(inputs).reshape(9, 9)
    board = np.asarray(prediction).reshape(9, 9)

    clues = inp != 1
    if np.any(clues & (board != inp)):
        return False

    for i in range(9):
        if set(board[i, :].tolist()) != VALID_TOKENS:
            return False
        if set(board[:, i].tolist()) != VALID_TOKENS:
            return False

    for r in range(0, 9, 3):
        for c in range(0, 9, 3):
            if set(board[r : r + 3, c : c + 3].reshape(-1).tolist()) != VALID_TOKENS:
                return False

    return True


@torch.compiler.disable
def batch_verify_sudoku(inputs, predictions):
    inputs_np = inputs.detach().cpu().numpy()
    preds_np = predictions.detach().cpu().numpy()
    valid = np.array(
        [verify_sudoku(inputs_np[i], preds_np[i]) for i in range(len(preds_np))],
        dtype=bool,
    )
    return torch.from_numpy(valid).to(predictions.device)
