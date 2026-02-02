"""
N-Queens Dataset Builder

Generate N-Queens solutions and create training data by removing some queens.
Input: Board with some queens removed
Label: Complete solution board

Usage:
    python dataset/build_nqueens_dataset.py --n 8 --remove-counts 3 4 5 --output-dir data/nqueens-8
"""

from typing import List, Dict, Optional
import os
import json
import numpy as np
from itertools import combinations
from collections import defaultdict
from multiprocessing import Pool, cpu_count

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata, dihedral_transform


cli = ArgParser()


class DataProcessConfig(BaseModel):
    n: int = 8
    output_dir: str = "data/nqueens-8"
    remove_counts: Optional[List[int]] = None  # Default handled in function
    test_ratio: float = 0.15
    seed: int = 42
    num_aug: int = 0
    multi_sol_ratio: Optional[float] = None


def generate_nqueens_solutions(n: int) -> List[np.ndarray]:
    """
    Generate all n-queens solutions using backtracking.
    Returns a list of (n, n) numpy arrays where 1 = queen, 0 = empty.
    """
    solutions = []
    
    def is_safe(board: List[int], row: int, col: int) -> bool:
        for r in range(row):
            c = board[r]
            if c == col or abs(c - col) == abs(r - row):
                return False
        return True
    
    def solve(board: List[int], row: int):
        if row == n:
            arr = np.zeros((n, n), dtype=np.uint8)
            for r, c in enumerate(board):
                arr[r, c] = 1
            solutions.append(arr)
            return
        
        for col in range(n):
            if is_safe(board, row, col):
                board.append(col)
                solve(board, row + 1)
                board.pop()
    
    solve([], 0)
    return solutions


def generate_all_removal_combinations(
    solution: np.ndarray, 
    remove_counts: List[int]
) -> List[tuple]:
    """Generate all possible combinations of removing queens from a solution."""
    queen_positions = list(zip(*np.where(solution == 1)))
    num_queens = len(queen_positions)
    
    results = []
    for remove_count in remove_counts:
        if remove_count > num_queens:
            continue
        for positions_to_remove in combinations(queen_positions, remove_count):
            input_board = solution.copy()
            for row, col in positions_to_remove:
                input_board[row, col] = 0
            results.append((input_board, solution.copy()))
    return results


def count_solutions(board: np.ndarray, all_solutions: List[np.ndarray]) -> int:
    """Count how many solutions are compatible with the given partial board."""
    queen_positions = set(zip(*np.where(board == 1)))
    count = 0
    for solution in all_solutions:
        solution_queens = set(zip(*np.where(solution == 1)))
        if queen_positions.issubset(solution_queens):
            count += 1
    return count


def process_single_solution(args):
    """Worker function for parallel processing."""
    solution, all_solutions, remove_counts, aug_idx = args
    
    if aug_idx > 0:
        tid = np.random.randint(1, 8)
        sol = dihedral_transform(solution, tid)
    else:
        sol = solution
    
    pairs = generate_all_removal_combinations(sol, remove_counts)
    results = []
    for input_board, label_board in pairs:
        num_solutions = count_solutions(input_board, all_solutions)
        results.append((input_board, label_board, num_solutions))
    return results


def print_solution_count_statistics(solution_counts: Dict[int, int], split_name: str):
    """Print statistics about the number of possible solutions per input."""
    print(f"\n=== {split_name} Solution Count Statistics ===")
    sorted_counts = dict(sorted(solution_counts.items()))
    print(f"Number of possible solutions -> Number of inputs: {sorted_counts}")
    
    total_inputs = sum(solution_counts.values())
    unique_solution_inputs = solution_counts.get(1, 0)
    multi_solution_inputs = total_inputs - unique_solution_inputs
    
    print(f"Total inputs: {total_inputs}")
    print(f"Unique solution inputs: {unique_solution_inputs} ({100*unique_solution_inputs/total_inputs:.2f}%)")
    print(f"Multi solution inputs: {multi_solution_inputs} ({100*multi_solution_inputs/total_inputs:.2f}%)")


def create_split(split_name: str, split_data: List[tuple], n: int, output_dir: str, multi_sol_ratio: Optional[float]):
    """Create and save a dataset split."""
    single_sol_queue = []
    multi_sol_queue = []
    solution_count_stats: Dict[int, int] = defaultdict(int)
    
    for input_board, label_board, num_solutions in split_data:
        solution_count_stats[num_solutions] += 1
        item = (input_board, label_board, num_solutions)
        if num_solutions == 1:
            single_sol_queue.append(item)
        else:
            multi_sol_queue.append(item)
    
    print_solution_count_statistics(solution_count_stats, split_name)
    
    # Apply multi_sol_ratio if specified
    if multi_sol_ratio is not None and len(multi_sol_queue) > 0:
        target_single = int(len(multi_sol_queue) * (1 - multi_sol_ratio) / multi_sol_ratio)
        if target_single < len(single_sol_queue):
            np.random.shuffle(single_sol_queue)
            single_sol_queue = single_sol_queue[:target_single]
            print(f"Adjusted single-solution examples: {target_single} (multi_sol_ratio={multi_sol_ratio})")
    
    # Combine and build results
    all_items = single_sol_queue + multi_sol_queue
    np.random.shuffle(all_items)
    
    results = {
        "inputs": [],
        "labels": [],
        "puzzle_identifiers": [],
        "puzzle_indices": [0],
        "group_indices": [0],
        "num_solutions": []
    }
    
    for idx, (input_board, label_board, num_solutions) in enumerate(all_items):
        results["inputs"].append(input_board)
        results["labels"].append(label_board)
        results["num_solutions"].append(num_solutions)
        results["puzzle_indices"].append(idx + 1)
        results["puzzle_identifiers"].append(0)
        results["group_indices"].append(idx + 1)
    
    # Convert to numpy
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        return arr + 1  # Shift: 0->1 (empty), 1->2 (queen)
    
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
        "num_solutions": np.array(results["num_solutions"], dtype=np.int32),
    }
    
    n_groups = len(results["group_indices"]) - 1
    
    # Metadata: PAD=0, empty=1, queen=2
    metadata = PuzzleDatasetMetadata(
        seq_len=n * n,
        vocab_size=3,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=n_groups,
        mean_puzzle_examples=1,
        total_puzzles=n_groups,
        sets=["all"]
    )
    
    # Save
    save_dir = os.path.join(output_dir, split_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    print(f"Saved {split_name}: {n_groups} examples")


@cli.command(singleton=True)
def build_nqueens_dataset(config: DataProcessConfig):
    """Build the N-Queens dataset."""
    np.random.seed(config.seed)
    
    # Handle default remove_counts
    remove_counts = config.remove_counts if config.remove_counts is not None else [3, 4, 5]
    
    # Generate all solutions
    print(f"Generating {config.n}-queens solutions...")
    solutions = generate_nqueens_solutions(config.n)
    print(f"Found {len(solutions)} solutions")
    print(f"Remove counts: {remove_counts}")
    
    if len(solutions) == 0:
        raise ValueError(f"No solutions found for {config.n}-queens")
    
    # Generate all data first, then split by input board
    print("Generating all input/label pairs...")
    args_list = [
        (sol, solutions, remove_counts, aug_idx)
        for sol in solutions
        for aug_idx in range(1 + config.num_aug)
    ]
    
    with Pool(cpu_count()) as pool:
        all_results = list(tqdm(
            pool.imap(process_single_solution, args_list),
            total=len(args_list),
            desc="Generating data"
        ))
    
    # Group by input board
    input_groups = defaultdict(list)
    for results in all_results:
        for input_board, label_board, num_solutions in results:
            input_key = input_board.tobytes()
            input_groups[input_key].append((input_board, label_board, num_solutions))
    
    print(f"Total unique inputs: {len(input_groups)}")
    
    # Split by input to ensure no overlap
    input_keys = list(input_groups.keys())
    np.random.shuffle(input_keys)
    test_size = int(len(input_keys) * config.test_ratio)
    
    test_input_keys = set(input_keys[:test_size])
    train_input_keys = set(input_keys[test_size:])
    
    train_data = [item for key in train_input_keys for item in input_groups[key]]
    test_data = [item for key in test_input_keys for item in input_groups[key]]
    
    print(f"Train inputs: {len(train_input_keys)} ({len(train_data)} examples)")
    print(f"Test inputs: {len(test_input_keys)} ({len(test_data)} examples)")
    
    create_split("train", train_data, config.n, config.output_dir, config.multi_sol_ratio)
    create_split("test", test_data, config.n, config.output_dir, config.multi_sol_ratio)
    
    # Save identifiers mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    
    print(f"\nDataset saved to {config.output_dir}")


if __name__ == "__main__":
    cli()
