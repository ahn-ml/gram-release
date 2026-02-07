"""Graph coloring dataset builder.

Downloads raw numpy data (adjacency matrices + colorings) from HuggingFace
and encodes into the PuzzleDataset format.

HuggingFace repo structure:
    {config_name}/train_adjacencies.npy  — (N, n, n) binary adjacency matrices
    {config_name}/train_colorings.npy    — (N, n) 0-indexed canonical colorings
    {config_name}/test_adjacencies.npy
    {config_name}/test_colorings.npy

Encoding (matching evaluators/graph_coloring_utils.py):
    Input: Upper triangle flattened, diagonal=0, off-diagonal: 2=edge, 1=no-edge
    Label: First n tokens are colors (3+color_id), rest masked with 0
    Vocab: 0=pad/ignore, 1=no-edge, 2=edge, 3+=colors
    seq_len: n*(n+1)/2
"""
from typing import Optional
import os
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "CUN-bjy/graph-coloring"
    config_name: str = "graph-coloring-8"  # "graph-coloring-8" or "graph-coloring-10"
    output_dir: str = "data/graph-coloring-8"
    num_colors: int = 3
    subsample_size: Optional[int] = None


def encode_adjacency(adj: np.ndarray) -> np.ndarray:
    """Encode adjacency matrix to upper triangle flattened sequence.

    Args:
        adj: (n, n) binary adjacency matrix

    Returns:
        (seq_len,) encoded sequence where seq_len = n*(n+1)/2
            diagonal=0, edge=2, no-edge=1
    """
    n = adj.shape[0]
    seq = []
    for i in range(n):
        for j in range(i, n):
            if i == j:
                seq.append(0)  # diagonal
            elif adj[i, j]:
                seq.append(2)  # edge
            else:
                seq.append(1)  # no-edge
    return np.array(seq, dtype=np.uint8)


def encode_coloring(coloring: np.ndarray, seq_len: int) -> np.ndarray:
    """Encode coloring to label sequence.

    Args:
        coloring: (n,) 0-indexed canonical color assignments
        seq_len: total sequence length (n*(n+1)/2)

    Returns:
        (seq_len,) encoded label where first n tokens = 3+color_id, rest = 0
    """
    n = len(coloring)
    label = np.zeros(seq_len, dtype=np.uint8)
    label[:n] = coloring + 3  # vocab offset
    return label


def convert_subset(set_name: str, config: DataProcessConfig):
    num_nodes = int(config.config_name.split("-")[-1])
    seq_len = num_nodes * (num_nodes + 1) // 2

    # Download from HuggingFace
    adj_path = hf_hub_download(
        config.source_repo,
        f"{config.config_name}/{set_name}_adjacencies.npy",
        repo_type="dataset",
    )
    col_path = hf_hub_download(
        config.source_repo,
        f"{config.config_name}/{set_name}_colorings.npy",
        repo_type="dataset",
    )

    adjacencies = np.load(adj_path)   # (N, n, n)
    colorings = np.load(col_path)     # (N, n)

    assert adjacencies.shape[1] == adjacencies.shape[2] == num_nodes
    assert colorings.shape[1] == num_nodes
    assert len(adjacencies) == len(colorings)

    # Subsample (training only)
    if set_name == "train" and config.subsample_size is not None:
        total = len(adjacencies)
        if config.subsample_size < total:
            indices = np.random.choice(total, size=config.subsample_size, replace=False)
            adjacencies = adjacencies[indices]
            colorings = colorings[indices]

    # Encode
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    puzzle_id = 0
    example_id = 0

    for adj, col in zip(tqdm(adjacencies, desc=f"Encoding {set_name}"), colorings):
        inp = encode_adjacency(adj)
        lbl = encode_coloring(col, seq_len)

        assert len(inp) == seq_len
        assert len(lbl) == seq_len

        results["inputs"].append(inp)
        results["labels"].append(lbl)
        example_id += 1
        puzzle_id += 1

        results["puzzle_indices"].append(example_id)
        results["puzzle_identifiers"].append(0)

        # Each puzzle is its own group
        results["group_indices"].append(puzzle_id)

    # To numpy
    results = {
        "inputs": np.stack(results["inputs"], axis=0),
        "labels": np.stack(results["labels"], axis=0),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=3 + config.num_colors,  # 0=pad, 1=no-edge, 2=edge, 3..=colors
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"],
    )

    # Save
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save identifiers mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    print(f"[{set_name}] Saved {len(adjacencies)} puzzles to {save_dir}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
