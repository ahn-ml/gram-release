"""Sampling strategies for pass@k evaluation."""
import torch


def compute_seq_correctness(inputs, labels, predictions, device, data_path=None):
    """Compute sequence-level correctness with task-specific verification.

    For graph coloring / N-Queens / Sudoku: Uses constraint verification.
    For other tasks: Uses exact label matching.
    """
    IGNORE_LABEL_ID = -100

    mask = (labels != IGNORE_LABEL_ID)
    loss_counts = mask.sum(-1)
    is_correct = mask & (predictions == labels)
    token_acc = torch.where(loss_counts > 0,
                           is_correct.sum(-1).float() / loss_counts.float(),
                           torch.zeros_like(loss_counts, dtype=torch.float32))

    # Detect task type from data_path (more reliable than vocab size)
    is_graph_coloring = False
    is_nqueens = False
    is_sudoku = False

    if data_path is not None:
        data_path_lower = data_path.lower() if isinstance(data_path, str) else ""
        is_nqueens = 'nqueens' in data_path_lower
        is_graph_coloring = ('graph-coloring' in data_path_lower or 'graph_coloring' in data_path_lower)
        is_sudoku = 'sudoku' in data_path_lower

    if is_graph_coloring:
        # Use constraint + canonical form verification for graph coloring
        from evaluators.graph_coloring_utils import batch_verify_coloring
        seq_is_correct = batch_verify_coloring(inputs, predictions)
    elif is_nqueens:
        # Use n-queens specific verification
        from evaluators.nqueens_utils import batch_verify_nqueens
        seq_is_correct = batch_verify_nqueens(inputs, predictions)
    elif is_sudoku:
        from evaluators.sudoku_utils import batch_verify_sudoku
        seq_is_correct = batch_verify_sudoku(inputs, predictions)
    else:
        # Use exact matching for other puzzles
        seq_is_correct = (is_correct.sum(-1) == loss_counts) & (loss_counts > 0)

    return seq_is_correct, token_acc, loss_counts


def sample_parallel(model, batch, num_samples, return_keys, device, extract_h_states=True, track_trajectories=False, data_path=None):
    """Simple parallel sampling: expand batch to B*K and forward once.

    Args:
        model: TRM model
        batch: Input batch dict (B, ...)
        num_samples: K samples per puzzle
        return_keys: Required outputs for evaluators
        device: torch device
        extract_h_states: Whether to extract h_states (needed for value network)
        track_trajectories: If True, track intermediate h_states at each H-cycle (for visualization)

    Returns:
        batch_samples: List of K dicts with predictions and correctness
            If track_trajectories=True, each dict also contains:
                - 'trajectory_h_states': List of h_states tensors (one per H-cycle)
                - 'trajectory_preds': List of prediction tensors (one per H-cycle)
    """
    IGNORE_LABEL_ID = -100
    B, K = batch["inputs"].shape[0], num_samples

    # Expand batch: (B) -> (B*K)
    batch_expanded = {k: v.repeat_interleave(K, dim=0) for k, v in batch.items()}

    # Initialize trajectory tracking if needed
    if track_trajectories:
        trajectories_h_states = [[] for _ in range(B * K)]
        trajectories_preds = [[] for _ in range(B * K)]

    # Forward pass for all K samples in parallel
    with torch.device(device):
        carry = model.initial_carry(batch_expanded)
    while True:
        carry, loss, metrics, preds, all_finish = model(
            carry=carry, batch=batch_expanded, return_keys=return_keys
        )

        # Track intermediate states if requested
        if track_trajectories:
            for i in range(B * K):
                trajectories_h_states[i].append(carry.inner_carry.z_H[i].clone())
                trajectories_preds[i].append(preds["logits"][i].cpu() if "logits" in preds else preds["preds"][i].cpu())

        if all_finish:
            break

    # Compute correctness (B*K)
    labels, predictions = batch_expanded["labels"], preds["preds"]
    inputs = batch_expanded["inputs"]
    seq_is_correct, token_acc, loss_counts = compute_seq_correctness(inputs, labels, predictions, device, data_path=data_path)

    # Extract h_states if needed (B*K, hidden_dim)
    h_states_final = None
    if extract_h_states:
        z_H = carry.inner_carry.z_H
        puzzle_emb_len = model.model.config.puzzle_emb_len
        h_states_final = z_H[:, puzzle_emb_len:].mean(dim=1)

    # Extract value predictions if available (from online value head)
    value_preds = preds.get("value_pred", None)

    # Reshape (B*K) -> K samples of (B) each
    # Note: repeat_interleave creates blocked layout [p0_s0, p0_s1, ..., p1_s0, p1_s1, ...]
    # So for sample k, we need indices [k, K+k, 2K+k, ...] to get k-th sample from each puzzle
    batch_samples = []
    for k in range(K):
        indices = torch.arange(B, device=device) * K + k
        sample_dict = {
            "preds": predictions[indices],
            "q_halt_logits": preds["q_halt_logits"][indices],
            "exact_correct": seq_is_correct[indices],
            "token_accuracy": token_acc[indices],
            "valid": loss_counts[indices] > 0,
            "h_states": h_states_final[indices] if h_states_final is not None else None,
        }

        # Add value predictions if available
        if value_preds is not None:
            sample_dict["value_pred"] = value_preds[indices]

        # Add trajectory data if tracked
        if track_trajectories:
            sample_dict["trajectory_h_states"] = [trajectories_h_states[b * K + k] for b in range(B)]
            sample_dict["trajectory_preds"] = [trajectories_preds[b * K + k] for b in range(B)]

        batch_samples.append(sample_dict)

    return batch_samples
