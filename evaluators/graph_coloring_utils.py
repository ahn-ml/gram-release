"""Graph coloring constraint verification utilities.

Encoding (from convert_gnn_gcp_to_trm.py):
- Input: Upper triangle flattened, diagonal=0, off-diagonal: 2=edge, 1=no-edge
- Label: First n tokens are colors (3+color_id), rest masked with 0
- Vocab: 0=pad/ignore, 1=no-edge, 2=edge, 3+=colors
"""
import numpy as np
import torch
import networkx as nx
from itertools import product


def normalize_coloring(coloring):
    """Normalize to canonical form (colors in order of first appearance).

    Args:
        coloring: Sequence of color IDs (e.g., [2, 1, 2, 0])

    Returns:
        tuple: Canonical form (e.g., (0, 1, 0, 2))
    """
    color_map, normalized, next_color = {}, [], 0
    for c in coloring:
        if c not in color_map:
            color_map[c] = next_color
            next_color += 1
        normalized.append(color_map[c])
    return tuple(normalized)


def generate_all_valid_colorings(adj_matrix, max_colors):
    """Generate all valid canonical colorings for a graph.

    Args:
        adj_matrix: (n, n) adjacency matrix (boolean or int)
        max_colors: Maximum number of colors to try

    Returns:
        set of tuples: All valid canonical colorings
    """
    n = adj_matrix.shape[0]
    G = nx.from_numpy_array(adj_matrix)

    unique = set()
    for coloring in product(range(max_colors), repeat=n):
        # Check adjacency constraint
        if all(coloring[u] != coloring[v] for u, v in G.edges()):
            # Canonicalize and add
            unique.add(normalize_coloring(coloring))

    return unique


def is_canonical_form(colors: np.ndarray) -> bool:
    """Check if colors are in canonical form (first appearance order 0,1,2,...).

    Args:
        colors: Node colors (e.g., [3,4,5] for nodes colored 0,1,2)

    Returns:
        bool: True if colors follow canonical ordering
    """
    # Decode colors: subtract 3 to get color IDs (vocab offset)
    color_ids = colors - 3

    # Check canonical form: colors appear in order 0, 1, 2, ...
    seen_colors = {}
    next_expected_color = 0

    for color_id in color_ids:
        if color_id < 0:  # Invalid (masked position)
            continue

        if color_id not in seen_colors:
            # New color must be next in sequence
            if color_id != next_expected_color:
                return False
            seen_colors[color_id] = True
            next_expected_color += 1

    return True


def count_conflicts(inp: np.ndarray, pred: np.ndarray) -> int:
    """Count edge conflicts: number of adjacent node pairs with same color.

    Args:
        inp: Input sequence (triangular encoding of adjacency matrix)
        pred: Predicted sequence (first n tokens are node colors)

    Returns:
        int: Number of edge conflicts (0 = valid coloring)
    """
    seq_len = len(inp)
    n = int((-1 + np.sqrt(1 + 8 * seq_len)) / 2)

    # Reconstruct adjacency from upper triangle
    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            if i != j and inp[idx] == 2:
                adj[i, j] = adj[j, i] = True
            idx += 1

    # Extract node colors
    colors = pred[:n]

    # Count conflicts
    conflicts = 0
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] and colors[i] == colors[j]:
                conflicts += 1

    return conflicts


@torch._dynamo.disable
def batch_count_conflicts(inputs, predictions):
    """Count conflicts for a batch of predictions.

    Args:
        inputs: Input tensor (B, seq_len)
        predictions: Prediction tensor (B, seq_len)

    Returns:
        torch.Tensor: Integer tensor of shape (B,) with conflict counts
    """
    inputs_np = inputs.cpu().numpy()
    preds_np = predictions.cpu().numpy()

    conflicts = np.array([count_conflicts(inputs_np[i], preds_np[i])
                          for i in range(len(preds_np))], dtype=np.int64)

    return torch.from_numpy(conflicts).to(predictions.device)


def verify_coloring(inp: np.ndarray, pred: np.ndarray) -> bool:
    """Check if predicted coloring satisfies graph coloring constraints in canonical form.

    Args:
        inp: Input sequence (triangular encoding of adjacency matrix)
        pred: Predicted sequence (first n tokens are node colors)

    Returns:
        bool: True if (1) adjacent nodes have different colors AND (2) colors are canonical
    """
    seq_len = len(inp)
    # Solve: n(n+1)/2 = seq_len => n = (-1 + sqrt(1 + 8*seq_len)) / 2
    n = int((-1 + np.sqrt(1 + 8 * seq_len)) / 2)

    # Reconstruct adjacency from upper triangle (row by row)
    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            if i != j and inp[idx] == 2:  # Edge token (skip diagonal)
                adj[i, j] = adj[j, i] = True
            idx += 1

    # Extract node colors from first n positions of prediction
    colors = pred[:n]

    # Verify constraint 1: adjacent nodes must have different colors
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] and colors[i] == colors[j]:
                return False

    # Verify constraint 2: colors must be in canonical form
    if not is_canonical_form(colors):
        return False

    return True


@torch._dynamo.disable
def batch_verify_coloring(inputs, predictions):
    """Verify batch of graph coloring predictions.

    Args:
        inputs: Input tensor (B, seq_len)
        predictions: Prediction tensor (B, seq_len)

    Returns:
        torch.Tensor: Boolean tensor of shape (B,)
    """
    inputs_np = inputs.cpu().numpy()
    preds_np = predictions.cpu().numpy()

    is_correct = np.array([verify_coloring(inputs_np[i], preds_np[i])
                           for i in range(len(preds_np))], dtype=bool)

    return torch.from_numpy(is_correct).to(predictions.device)


def compute_coverage_for_puzzle(inp, predictions, max_colors):
    """Compute coverage for one puzzle: fraction of all valid colorings found.

    Args:
        inp: (seq_len,) input encoding (numpy array)
        predictions: (K, seq_len) K model predictions (numpy array)
        max_colors: Maximum colors in vocab

    Returns:
        coverage: float [0-1], fraction of valid colorings found
        num_found: int, number of unique valid colorings found
        num_total: int, total valid colorings
    """
    # 1. Parse adjacency matrix from triangular encoding
    seq_len = len(inp)
    n = int((-1 + np.sqrt(1 + 8 * seq_len)) / 2)

    adj = np.zeros((n, n), dtype=bool)
    idx = 0
    for i in range(n):
        for j in range(i, n):
            if i != j and inp[idx] == 2:  # Edge token
                adj[i, j] = adj[j, i] = True
            idx += 1

    # 2. Generate all valid canonical colorings (ground truth)
    all_valid_colorings = generate_all_valid_colorings(adj, max_colors)

    # 3. Extract model's unique valid predictions
    model_colorings = set()
    for pred in predictions:
        if verify_coloring(inp, pred):  # Only count valid predictions
            colors = pred[:n] - 3  # Decode from vocab offset
            model_colorings.add(tuple(colors))

    # 4. Compute coverage
    num_found = len(model_colorings)
    num_total = len(all_valid_colorings)
    coverage = num_found / num_total if num_total > 0 else 0.0

    return coverage, num_found, num_total
