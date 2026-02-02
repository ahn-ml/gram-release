"""N-Queens verification utilities.
Check if queens are placed without attacking each other on an N×N board.

Encoding (based on build_nqueens_dataset.py):
- 0: Empty cell
- 1: Queen
"""
import numpy as np
import torch


def verify_nqueens(prediction, n=None):
    """Check if prediction is a valid N-Queens solution.
    
    A valid solution requires:
    1. Exactly N queens on an N×N board
    2. No two queens attack each other (same row, column, or diagonal)
    
    Args:
        prediction: 1D array of length N*N (flattened board)
        n: Board size (inferred from prediction length if None)
    
    Returns:
        bool: True if valid N-Queens solution
    """
    if n is None:
        n = int(np.sqrt(len(prediction)))
    
    # Reshape to N×N board
    board = prediction.reshape(n, n)
    #board= board -1 # Adjust if necessary based on encoding
    # Find queen positions (value == 1)
    queen_positions = np.argwhere(board == 2)
    
    # Must have exactly N queens
    if len(queen_positions) != n:
        return False
    
    # Check for conflicts
    rows = set()
    cols = set()
    diag1 = set()  # row - col (main diagonal direction)
    diag2 = set()  # row + col (anti-diagonal direction)
    
    for row, col in queen_positions:
        # Check row conflict
        if row in rows:
            return False
        rows.add(row)
        
        # Check column conflict
        if col in cols:
            return False
        cols.add(col)
        
        # Check diagonal conflicts
        d1 = row - col
        d2 = row + col
        
        if d1 in diag1:
            return False
        diag1.add(d1)
        
        if d2 in diag2:
            return False
        diag2.add(d2)
    
    return True


def find_all_solutions(input_board, n=None, max_solutions=20):
    """Find all valid N-Queens solutions given initial queens.
    
    Args:
        input_board: 2D array (n, n) with initial queen positions (2=Queen)
        n: Board size (inferred if None)
        max_solutions: Maximum number of solutions to return (to avoid too many)
    
    Returns:
        list of 2D arrays: All valid solutions as (n, n) boards
    """
    if n is None:
        n = input_board.shape[0]
    
    # Extract initial queen positions
    initial_queens = [(r, c) for r, c in np.argwhere(input_board == 2)]
    
    # Track occupied rows, cols, diagonals from initial queens
    occupied_rows = set(r for r, c in initial_queens)
    occupied_cols = set(c for r, c in initial_queens)
    occupied_diag1 = set(r - c for r, c in initial_queens)
    occupied_diag2 = set(r + c for r, c in initial_queens)
    
    # Rows that still need queens
    remaining_rows = [r for r in range(n) if r not in occupied_rows]
    
    solutions = []
    
    def backtrack(row_idx, cols, diag1, diag2, current_queens):
        if len(solutions) >= max_solutions:
            return
        
        if row_idx == len(remaining_rows):
            # Found a solution - create board
            board = np.ones((n, n), dtype=np.int64)  # 1 = empty
            for r, c in initial_queens + current_queens:
                board[r, c] = 2  # 2 = queen
            solutions.append(board)
            return
        
        row = remaining_rows[row_idx]
        for col in range(n):
            if col in cols or col in occupied_cols:
                continue
            d1, d2 = row - col, row + col
            if d1 in diag1 or d1 in occupied_diag1:
                continue
            if d2 in diag2 or d2 in occupied_diag2:
                continue
            
            cols.add(col)
            diag1.add(d1)
            diag2.add(d2)
            current_queens.append((row, col))
            backtrack(row_idx + 1, cols, diag1, diag2, current_queens)
            current_queens.pop()
            cols.remove(col)
            diag1.remove(d1)
            diag2.remove(d2)
    
    backtrack(0, set(), set(), set(), [])
    return solutions


@torch.compiler.disable
def batch_verify_nqueens(inputs, predictions, n=None):
    """Verify batch of N-Queens predictions. Returns (B,) boolean tensor.

    Args:
        inputs: Input tensor (B, N*N) - not used but kept for API consistency
        predictions: Prediction tensor (B, N*N)
        n: Board size (inferred if None)

    Returns:
        torch.Tensor: Boolean tensor of shape (B,)
    """
    preds_np = predictions.cpu().numpy()

    is_correct = np.array([verify_nqueens(preds_np[i], n)
                           for i in range(len(preds_np))], dtype=bool)

    return torch.from_numpy(is_correct).to(predictions.device)


def compute_coverage_for_nqueens_puzzle(input_board, predictions, n=None, max_solutions=100):
    """Compute coverage metric for a single N-Queens puzzle.

    Coverage = (number of unique valid solutions found) / (total valid solutions)

    Args:
        input_board: 1D array of length N*N (flattened input board with initial queens)
        predictions: 2D array (K, N*N) of K predictions for this puzzle
        n: Board size (inferred if None)
        max_solutions: Maximum solutions to enumerate (for efficiency)

    Returns:
        tuple: (coverage, num_found, num_total)
    """
    if n is None:
        n = int(np.sqrt(len(input_board)))

    # Find all valid solutions for this input
    input_2d = input_board.reshape(n, n)
    all_solutions = find_all_solutions(input_2d, n, max_solutions=max_solutions)
    num_total = len(all_solutions)

    if num_total == 0:
        return 0.0, 0, 0

    # Hash all ground truth solutions
    gt_hashes = {hash(sol.tobytes()) for sol in all_solutions}

    # Check which predictions are valid and match a GT solution
    found_hashes = set()
    for pred in predictions:
        if verify_nqueens(pred, n):
            pred_2d = pred.reshape(n, n)
            pred_hash = hash(pred_2d.tobytes())
            if pred_hash in gt_hashes:
                found_hashes.add(pred_hash)

    num_found = len(found_hashes)
    coverage = num_found / num_total

    return coverage, num_found, num_total