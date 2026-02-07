"""Reverse-extract raw adjacency matrices and colorings from encoded PuzzleDataset files.

Reads all__inputs.npy / all__labels.npy and produces:
    - {set_name}_adjacencies.npy  (N, n, n) binary adjacency matrices
    - {set_name}_colorings.npy    (N, n) 0-indexed canonical colorings

These can be uploaded to HuggingFace for use with build_graph_coloring_dataset.py.

Usage:
    python -m dataset.extract_graph_coloring_raw \
        --input_dir data/graph-coloring-8 \
        --output_dir export/graph-coloring-8 \
        --num_nodes 8
"""
import os
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm


cli = ArgParser()


class ExtractConfig(BaseModel):
    input_dir: str = "data/graph-coloring-8"
    output_dir: str = "export/graph-coloring-8"
    num_nodes: int = 8


def decode_adjacency(inp: np.ndarray, n: int) -> np.ndarray:
    """Decode upper triangle flattened sequence back to adjacency matrix.

    Args:
        inp: (seq_len,) encoded sequence (0=diagonal, 1=no-edge, 2=edge)
        n: number of nodes

    Returns:
        (n, n) binary adjacency matrix
    """
    adj = np.zeros((n, n), dtype=np.uint8)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            if i != j and inp[idx] == 2:
                adj[i, j] = adj[j, i] = 1
            idx += 1
    return adj


def decode_coloring(lbl: np.ndarray, n: int) -> np.ndarray:
    """Decode label sequence back to 0-indexed coloring.

    Args:
        lbl: (seq_len,) encoded label (first n tokens = 3+color_id, rest = 0)
        n: number of nodes

    Returns:
        (n,) 0-indexed color assignments
    """
    return (lbl[:n] - 3).astype(np.int32)


def extract_subset(set_name: str, config: ExtractConfig):
    n = config.num_nodes
    inputs_path = os.path.join(config.input_dir, set_name, f"all__inputs.npy")
    labels_path = os.path.join(config.input_dir, set_name, f"all__labels.npy")

    if not os.path.exists(inputs_path):
        print(f"[{set_name}] {inputs_path} not found, skipping")
        return

    inputs = np.load(inputs_path)   # (N, seq_len)
    labels = np.load(labels_path)   # (N, seq_len)

    assert len(inputs) == len(labels)
    print(f"[{set_name}] Extracting {len(inputs)} puzzles (n={n})")

    adjacencies = np.stack([decode_adjacency(inp, n) for inp in tqdm(inputs, desc=f"Decoding {set_name}")], axis=0)
    colorings = np.stack([decode_coloring(lbl, n) for lbl in labels], axis=0)

    assert adjacencies.shape == (len(inputs), n, n)
    assert colorings.shape == (len(inputs), n)

    # Save
    os.makedirs(config.output_dir, exist_ok=True)
    np.save(os.path.join(config.output_dir, f"{set_name}_adjacencies.npy"), adjacencies)
    np.save(os.path.join(config.output_dir, f"{set_name}_colorings.npy"), colorings)

    print(f"[{set_name}] Saved to {config.output_dir}/")


@cli.command(singleton=True)
def extract(config: ExtractConfig):
    extract_subset("train", config)
    extract_subset("test", config)


if __name__ == "__main__":
    cli()
