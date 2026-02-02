"""Sampling strategies for pass@k evaluation."""
import torch


def compute_seq_correctness(inputs, labels, predictions, device, data_path=None):
    """Compute sequence-level correctness with task-specific verification.

    For maze tasks: Uses BFS to verify any valid shortest path.
    For graph coloring: Uses constraint + canonical form verification.
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
    is_maze = False
    is_graph_coloring = False
    is_nqueens = False

    if data_path is not None:
        data_path_lower = data_path.lower() if isinstance(data_path, str) else ""
        is_nqueens = 'nqueens' in data_path_lower
        is_maze = 'maze' in data_path_lower
        is_graph_coloring = ('graph-coloring' in data_path_lower or 'graph_coloring' in data_path_lower)

    if is_maze:
        # Use BFS verification for mazes
        from evaluators.maze_utils import batch_verify_maze
        seq_is_correct = batch_verify_maze(inputs, predictions)
    elif is_graph_coloring:
        # Use constraint + canonical form verification for graph coloring
        from evaluators.graph_coloring_utils import batch_verify_coloring
        seq_is_correct = batch_verify_coloring(inputs, predictions)
    elif is_nqueens:
        # Use n-queens specific verification
        from evaluators.nqueens_utils import batch_verify_nqueens
        seq_is_correct = batch_verify_nqueens(inputs, predictions)
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

    # Compute correctness (B*K) with maze BFS verification
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


def sample_with_beam_search(model, batch, num_samples, beam_size, branch_every,
                             value_network, h_state_mean, h_state_std, return_keys, device,
                             use_smc=False, smc_temperature=1.0, smc_temp_schedule=None,
                             track_trajectories=False, data_path=None):
    """Value-guided beam search: periodic elite selection via value network scoring.

    Algorithm:
        1. Start with K parallel h_states
        2. Forward M cycles
        3. Score K states with value network
        4a. Beam mode: Copy top beam_size elites (round-robin duplication)
        4b. SMC mode: Multinomial resampling according to softmax weights
        5. Repeat until halt (variational sampling creates diversity)

    Args:
        model: TRM model
        batch: Input batch dict (B, ...)
        num_samples: K population size
        beam_size: K_beam elite size (only used if use_smc=False)
        branch_every: M, score/select every M H-cycles
        value_network: Value network for scoring
        h_state_mean, h_state_std: Normalization params
        return_keys: Required outputs
        device: torch device
        use_smc: If True, use SMC-style importance sampling (default: False for beam search)
        smc_temperature: Temperature for SMC (default: 1.0). Constant if smc_temp_schedule is None.
        smc_temp_schedule: Optional tuple (T_init, T_final) for cosine annealing
            - T_init: Initial temperature (exploration, early cycles)
            - T_final: Final temperature (exploitation, late cycles)
            - Example: (2.0, 0.5) anneals from T=2.0 → T=0.5 via cosine schedule
        track_trajectories: If True, track intermediate h_states and beam selections

    Returns:
        batch_samples: List of K dicts (some are evolved elites)
            If track_trajectories=True, each dict also contains:
                - 'trajectory_h_states': List of h_states tensors (one per H-cycle)
                - 'trajectory_preds': List of prediction tensors (one per H-cycle)
                - 'beam_selections': List of selection events with h_cycle, score, etc.
    """
    IGNORE_LABEL_ID = -100
    B, K = batch["inputs"].shape[0], num_samples

    if not use_smc and beam_size > K:
        raise ValueError(f"beam_size ({beam_size}) must be <= num_samples ({K})")

    # Expand batch: (B) -> (B*K)
    batch_expanded = {k: v.repeat_interleave(K, dim=0) for k, v in batch.items()}

    # Initialize trajectory tracking if needed
    if track_trajectories:
        trajectories_h_states = [[] for _ in range(B * K)]
        trajectories_preds = [[] for _ in range(B * K)]
        beam_selections = [[] for _ in range(B * K)]

    # Initialize
    with torch.device(device):
        carry = model.initial_carry(batch_expanded)

    h_cycle = 0
    max_cycles = model.model.config.halt_max_steps

    # Temperature scheduling (cosine annealing)
    def get_temperature(h_cycle):
        if not use_smc or smc_temp_schedule is None:
            return smc_temperature
        import math
        T_init, T_final = smc_temp_schedule
        progress = min(h_cycle / max(max_cycles, 1), 1.0)
        return T_final + 0.5 * (T_init - T_final) * (1 + math.cos(math.pi * progress))

    while h_cycle < max_cycles:
        # Forward M cycles
        for _ in range(branch_every):
            carry, loss, metrics, preds, all_finish = model(
                carry=carry, batch=batch_expanded, return_keys=return_keys
            )

            # Track intermediate states if requested
            if track_trajectories:
                for i in range(B * K):
                    trajectories_h_states[i].append(carry.inner_carry.z_H[i].clone())
                    trajectories_preds[i].append(preds["logits"][i].cpu() if "logits" in preds else preds["preds"][i].cpu())

            h_cycle += 1
            if all_finish or h_cycle >= max_cycles:
                break

        if all_finish or h_cycle >= max_cycles:
            break

        # CEM: Score and select elites
        with torch.no_grad():
            # Extract h_states (B*K, hidden_dim)
            z_H = carry.inner_carry.z_H
            puzzle_emb_len = model.model.config.puzzle_emb_len
            h_states = z_H[:, puzzle_emb_len:].mean(dim=1)  # (B*K, hidden_dim)

            # Score with value network (online or external)
            if "value_pred" in preds:
                # Use online value head (already computed during forward)
                scores = preds["value_pred"]  # (B*K)
            elif value_network is not None:
                # Use external value network
                h_states_norm = (h_states - h_state_mean) / h_state_std
                scores = value_network(h_states_norm).squeeze(-1)  # (B*K)
            else:
                raise ValueError("Need value predictions for guided sampling")

            # Reshape to per-puzzle: (B*K) -> (B, K)
            scores_per_puzzle = scores.view(B, K)

            # Resample per puzzle
            current_temp = get_temperature(h_cycle)
            for b in range(B):
                # SMC: multinomial resampling OR Beam: top-k + round-robin
                if use_smc:
                    weights = torch.softmax(scores_per_puzzle[b] / current_temp, dim=0)
                    elite_idx = torch.multinomial(weights, K, replacement=True)
                else:
                    _, top_k = scores_per_puzzle[b].topk(beam_size)
                    elite_idx = top_k[torch.arange(K) % beam_size]  # Round-robin duplicate

                # Clone elite states to avoid overwriting during duplication
                elite_states = {
                    ek.item(): (carry.inner_carry.z_H[b * K + ek.item()].clone(),
                               carry.inner_carry.z_L[b * K + ek.item()].clone(),
                               carry.steps[b * K + ek.item()].clone(),
                               carry.halted[b * K + ek.item()].clone())
                    for ek in elite_idx
                }

                # Duplicate elites to all K positions
                for k in range(K):
                    src_k = elite_idx[k].item()
                    dst_global = b * K + k

                    if track_trajectories:
                        beam_selections[dst_global].append({
                            'h_cycle': h_cycle,
                            'score': scores[b * K + src_k].item(),
                            'is_elite': src_k in elite_idx,
                            'copied_from': src_k if src_k != k else None
                        })

                    if src_k != k:
                        z_H, z_L, steps, halted = elite_states[src_k]
                        carry.inner_carry.z_H[dst_global] = z_H.clone()
                        carry.inner_carry.z_L[dst_global] = z_L.clone()
                        carry.steps[dst_global] = steps.clone()
                        carry.halted[dst_global] = halted.clone()

    # Compute final correctness (B*K) with maze BFS verification
    labels, predictions = batch_expanded["labels"], preds["preds"]
    inputs = batch_expanded["inputs"]
    seq_is_correct, token_acc, loss_counts = compute_seq_correctness(inputs, labels, predictions, device, data_path=data_path)

    # Extract final h_states
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
            sample_dict["beam_selections"] = [beam_selections[b * K + k] for b in range(B)]

        batch_samples.append(sample_dict)

    return batch_samples
