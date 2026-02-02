#!/usr/bin/env python3
"""Convert GNN-GCP dataset to TRM format with multi-label colorings."""
import sys
import json
import random
import numpy as np
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from pydantic import BaseModel
from argdantic import ArgParser
from itertools import product
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "GNN-GCP"))
from common import PuzzleDatasetMetadata

cli = ArgParser()

def normalize_coloring(coloring):
    """Normalize to canonical form (colors in order of first appearance)."""
    color_map, normalized, next_color = {}, [], 0
    for c in coloring:
        if c not in color_map:
            color_map[c] = next_color
            next_color += 1
        normalized.append(color_map[c])
    return tuple(normalized)

def generate_all_valid_colorings(G: nx.Graph, max_colors: int, use_canonical: bool = False, max_solutions: int = None):
    """Generate all valid colorings using backtracking with early termination.

    Args:
        G: NetworkX graph
        max_colors: Maximum number of colors to use
        use_canonical: If True, only return canonical colorings (deduplicated)
        max_solutions: Stop after finding this many solutions (None = no limit)

    Returns:
        List of valid colorings, or None if max_solutions exceeded
    """
    n = len(G.nodes())
    adj = [set(G.neighbors(node)) for node in range(n)]

    results = set() if use_canonical else []
    coloring = [-1] * n

    def backtrack(node):
        if max_solutions and len(results) > max_solutions:
            return False  # Signal to stop

        if node == n:
            # Found valid coloring
            if use_canonical:
                results.add(normalize_coloring(coloring))
            else:
                results.append(list(coloring))
            return True

        # Get colors used by neighbors
        forbidden = {coloring[nb] for nb in adj[node] if coloring[nb] >= 0}

        for color in range(max_colors):
            if color in forbidden:
                continue
            coloring[node] = color
            if not backtrack(node + 1):
                coloring[node] = -1
                return False
            coloring[node] = -1

        return True

    backtrack(0)

    if max_solutions and len(results) > max_solutions:
        return None  # Too many solutions

    return [list(c) for c in results] if use_canonical else results

def read_graph(filepath):
    with open(filepath,"r") as f:

        line = ''

        # Parse number of vertices
        while 'DIMENSION' not in line: line = f.readline();
        n = int(line.split()[1])
        Ma = np.zeros((n,n),dtype=int)
        
        # Parse edges
        while 'EDGE_DATA_SECTION' not in line: line = f.readline();
        line = f.readline()
        while '-1' not in line:
            i,j = [ int(x) for x in line.split() ]
            Ma[i,j] = 1
            line = f.readline()
        #end while

        # Parse diff edge
        while 'DIFF_EDGE' not in line: line = f.readline();
        diff_edge = [ int(x) for x in f.readline().split() ]

        # Parse target cost
        while 'CHROM_NUMBER' not in line: line = f.readline();
        chrom_number = int(f.readline().strip())

    #end
    return Ma,chrom_number,diff_edge
#end


class ConvertConfig(BaseModel):
    gnn_gcp_train_dir: str = "../GNN-GCP/data/train"
    comp_reasoning_test_dir: str = "../GNN-GCP/data/test"
    output_dir: str = "data/graph-coloring"
    max_nodes: int = 5
    max_labels_per_graph: int = 5  # Try to find up to K colorings
    max_colors: int = 3  # Maximum colors to use
    use_canonical_colorings: bool = False  # Reduce k! permutations to 1 canonical form
    seed: int = 42


def flatten_upper_triangle(mat, n):
    """Flatten upper triangle (including diagonal) row by row."""
    tokens = []
    for i in range(n):
        for j in range(i, n):
            tokens.append(mat[i, j])
    return tokens

def to_sequence(adj, coloring, max_n, is_input):
    """Convert to triangular sequence encoding.

    Input:  Upper triangle (n(n+1)/2 tokens) with diagonal=0 (uncolored)
    Label:  First n tokens are colors, rest are 0 (masked by ignore_label_id)
    """
    n = adj.shape[0]

    if is_input:
        # Build upper triangle matrix
        mat = np.ones((max_n, max_n), dtype=np.uint8)
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    mat[i, j] = 0  # Uncolored diagonal
                else:
                    mat[i, j] = 2 if adj[i, j] == 1 else 1  # Edge or no-edge

        # Flatten upper triangle
        return np.array(flatten_upper_triangle(mat, max_n), dtype=np.uint8)

    else:
        # Label: first n tokens are colors, rest are 0 (ignored)
        seq_len = max_n * (max_n + 1) // 2
        label = np.zeros(seq_len, dtype=np.uint8)
        for i in range(n):
            label[i] = 3 + coloring[i]
        return label


def convert_dataset(graph_dir: Path, output_dir: Path, config: ConvertConfig, set_name: str, existing_keys: set = None):
    """Convert graphs to TRM format."""
    random.seed(config.seed if set_name == "train" else config.seed + 1000)

    # Collect unique graphs and generate all colorings
    unique_graphs = {}
    for f in tqdm(sorted(graph_dir.glob("*.graph")), desc=set_name):
        adj, _, _ = read_graph(str(f))
        if adj.shape[0] > config.max_nodes:
            continue
        adj_key = tuple(adj.flatten())
        if adj_key not in unique_graphs:
            G = nx.from_numpy_array(adj)
            colorings = generate_all_valid_colorings(G, config.max_colors, config.use_canonical_colorings, max_solutions=config.max_labels_per_graph)
            if colorings is not None and 0 < len(colorings) <= config.max_labels_per_graph:  # Filter
                unique_graphs[adj_key] = (adj, colorings)

    # Check overlap
    if existing_keys:
        overlap = set(unique_graphs.keys()) & existing_keys
        print(f"\n{set_name}: {len(overlap)} graphs overlap with previous set (removed)")
        unique_graphs = {k: v for k, v in unique_graphs.items() if k not in existing_keys}

    # Stats
    sols = sorted([len(c) for _, c in unique_graphs.values()])
    print(f"{set_name}: {len(unique_graphs)} unique (filtered)")
    print(f"  Solutions - min: {min(sols)}, max: {max(sols)}, median: {sols[len(sols)//2]}, avg: {sum(sols)/len(sols):.1f}")
    print(f"  Distribution: {sols[:5]}...{sols[-5:]}" if len(sols) > 10 else f"  Distribution: {sols}")

    # Convert to dataset format
    inputs, labels = [], []
    for adj, colorings in tqdm(unique_graphs.values(), desc=f"{set_name} (convert)"):
        inp = to_sequence(adj, None, config.max_nodes, True)
        # Test set: only use 1 label (constraint verification accepts any valid coloring)
        # Train set: use all labels (learn multiple valid solutions)
        colorings_to_use = [random.choice(colorings)] if set_name == "test" else colorings
        for coloring in colorings_to_use:
            inputs.append(inp)
            labels.append(to_sequence(adj, coloring, config.max_nodes, False))

    # Save
    save_dir = output_dir / set_name
    save_dir.mkdir(parents=True, exist_ok=True)

    n = len(inputs)
    seq_len = config.max_nodes * (config.max_nodes + 1) // 2
    data = {
        "inputs": np.array(inputs, dtype=np.uint8),
        "labels": np.array(labels, dtype=np.uint8),
        "group_indices": np.arange(n + 1, dtype=np.int32),
        "puzzle_indices": np.arange(n + 1, dtype=np.int32),
        "puzzle_identifiers": np.zeros(n, dtype=np.int32),
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len, vocab_size=3 + config.max_colors, pad_id=0,
        ignore_label_id=0, blank_identifier_id=0, num_puzzle_identifiers=1,
        total_groups=n, mean_puzzle_examples=1.0, total_puzzles=n, sets=["all"]
    )

    json.dump(metadata.model_dump(), open(save_dir / "dataset.json", "w"), indent=2)
    for k, v in data.items():
        np.save(save_dir / f"all__{k}.npy", v)

    print(f"{set_name}: {data['inputs'].shape}")
    return set(unique_graphs.keys())


@cli.command(singleton=True)
def convert(config: ConvertConfig):
    """Convert to TRM format."""
    train_keys = convert_dataset(Path(config.gnn_gcp_train_dir), Path(config.output_dir), config, "train")
    convert_dataset(Path(config.comp_reasoning_test_dir), Path(config.output_dir), config, "test", train_keys)
    json.dump(["<blank>"], open(Path(config.output_dir) / "identifiers.json", "w"))


if __name__ == "__main__":
    cli()
