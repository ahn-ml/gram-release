"""Sudoku generation dataset builder.

Input: blank 9x9 board.
Label: complete valid Sudoku board.
"""

import json
import os
import csv
from typing import Optional, Set

import numpy as np
from argdantic import ArgParser
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from tqdm import tqdm

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-generation-1k"
    train_size: int = 50000
    test_size: int = 1000
    min_difficulty: Optional[int] = None
    seed: int = 42
    num_aug: int = 0


def augment_solution(board: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rows = np.concatenate([band * 3 + rng.permutation(3) for band in rng.permutation(3)])
    cols = np.concatenate([stack * 3 + rng.permutation(3) for stack in rng.permutation(3)])
    digits = np.pad(rng.permutation(np.arange(1, 10, dtype=np.uint8)), (1, 0))
    board = board[rows][:, cols]
    if rng.random() < 0.5:
        board = board.T
    return digits[board]


def load_solutions(set_name: str, config: DataProcessConfig, rng: np.random.Generator):
    solutions = []
    path = hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset")
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for _source, _q, answer, rating in reader:
            if config.min_difficulty is not None and int(rating) < config.min_difficulty:
                continue
            solutions.append(np.frombuffer(answer.encode(), dtype=np.uint8).reshape(9, 9) - ord("0"))

    size = config.train_size if set_name == "train" else config.test_size
    if size < len(solutions):
        indices = rng.choice(len(solutions), size=size, replace=False)
        solutions = [solutions[i] for i in indices]
    return solutions


def make_split(solutions, rng: np.random.Generator, seen: Set[bytes], num_aug: int):
    labels = []
    progress = tqdm(total=len(solutions) * (1 + num_aug))
    for board in solutions:
        for aug_idx in range(1 + num_aug):
            out = board if aug_idx == 0 else augment_solution(board, rng)
            key = out.tobytes()
            if key in seen:
                continue
            seen.add(key)
            labels.append(out.reshape(-1))
            progress.update(1)
    progress.close()
    labels = np.stack(labels).astype(np.uint8)
    inputs = np.zeros_like(labels)
    return inputs + 1, labels + 1


def save_split(split: str, inputs: np.ndarray, labels: np.ndarray, output_dir: str):
    save_dir = os.path.join(output_dir, split)
    os.makedirs(save_dir, exist_ok=True)
    n = len(inputs)

    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=11,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=n,
        mean_puzzle_examples=1,
        total_puzzles=n,
        sets=["all"],
    )

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    np.save(os.path.join(save_dir, "all__inputs.npy"), inputs)
    np.save(os.path.join(save_dir, "all__labels.npy"), labels)
    np.save(os.path.join(save_dir, "all__puzzle_identifiers.npy"), np.zeros(n, dtype=np.int32))
    np.save(os.path.join(save_dir, "all__puzzle_indices.npy"), np.arange(n + 1, dtype=np.int32))
    np.save(os.path.join(save_dir, "all__group_indices.npy"), np.arange(n + 1, dtype=np.int32))


@cli.command(singleton=True)
def build_sudoku_generation_dataset(config: DataProcessConfig):
    train_rng = np.random.default_rng(config.seed)
    aug_rng = np.random.default_rng(config.seed + 1)
    test_rng = np.random.default_rng(config.seed + 2)
    os.makedirs(config.output_dir, exist_ok=True)

    print(f"Loading {config.train_size} train Sudoku solutions with {config.num_aug} augmentations each")
    train_inputs, train_labels = make_split(load_solutions("train", config, train_rng), aug_rng, set(), config.num_aug)
    print(f"Loading {config.test_size} test Sudoku solutions")
    test_inputs, test_labels = make_split(load_solutions("test", config, test_rng), test_rng, set(), 0)

    save_split("train", train_inputs, train_labels, config.output_dir)
    save_split("test", test_inputs, test_labels, config.output_dir)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    print(f"Saved dataset to {config.output_dir}")


if __name__ == "__main__":
    cli()
