#!/usr/bin/env python3
"""
Standalone pass@k evaluation script for variational models.

Loads a checkpoint and runs evaluation K times to generate K samples per input.
Uses aggregated_voting to accumulate predictions and compute pass@k metrics.

Optionally supports best-of-N selection using a trained value network.

Usage:
    # Standard evaluation
    python eval_pass_at_k.py \\
        --checkpoint_path checkpoints/run_name/step_50000 \\
        --num_samples 100 \\
        --data_paths "[data/sudoku-extreme-1k-aug-1000]"

    # With value network for best-of-N selection
    python eval_pass_at_k.py \\
        --checkpoint_path checkpoints/run_name/step_50000 \\
        --num_samples 100 \\
        --value_network_path value_models/value_net/best_model.pt
"""
import argparse
import os
import random
import numpy as np
import torch
import torch.distributed as dist
import tqdm
from datetime import timedelta

from pretrain import PretrainConfig, create_dataloader, create_evaluators, ArchConfig, LossConfig, EvaluatorConfig
from dataset.puzzle_dataset import PuzzleDatasetMetadata
from evaluators.sampling_strategies import sample_parallel

def make_pass_ks(num_samples, filter_augmentation=False):
    if not filter_augmentation:
        # For ARC with augmentation, always use full set of pass@ks
        pass_Ks = [1, 2, 5, 10, 20, 100, 250, 500, 1000]
        return pass_Ks
    assert num_samples in [1, 2, 5, 10, 20, 100, 250, 500, 1000]
    if num_samples == 1:
        pass_Ks = [1]
    elif num_samples == 2:
        pass_Ks = [1, 2]
    elif num_samples == 5:
        pass_Ks = [1, 2, 5]
    elif num_samples == 10:
        pass_Ks = [1, 2, 5, 10]
    elif num_samples == 20:
        pass_Ks = [1, 2, 5, 10, 20]
    elif num_samples == 100:
        pass_Ks = [1, 2, 5, 10, 20, 100]
    elif num_samples == 250:
        pass_Ks= [1, 2, 5, 10, 20, 100, 250]
    elif num_samples == 500:
        pass_Ks = [1, 2, 5, 10, 20, 100, 250, 500]
    elif num_samples == 1000:
        pass_Ks = [1, 2, 5, 10, 20, 100, 250, 500, 1000]
    return pass_Ks

def viz_nqueens_samples(batch, samples, batch_idx, out_dir="nqueens_viz", num_puzzles=5, multi_sol_only=False):
    """Visualize N-Queens puzzles with deduplicated samples.
    
    Args:
        batch: Input batch
        samples: List of K sample dicts
        batch_idx: Batch index for naming
        out_dir: Output directory
        num_puzzles: Max puzzles to visualize per batch
        multi_sol_only: If True, only show puzzles with multiple unique samples
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from evaluators.nqueens_utils import verify_nqueens, find_all_solutions

    os.makedirs(out_dir, exist_ok=True)
    B, K = batch["inputs"].shape[0], len(samples)
    n = int(np.sqrt(batch["inputs"].shape[1]))
    
    saved_count = 0
    for puzzle_idx in range(B):
        if saved_count >= num_puzzles:
            break
            
        inp = batch["inputs"][puzzle_idx].cpu().numpy().reshape(n, n)
        
        # Deduplicate predictions: hash -> [board, count, is_valid]
        pred_map = {}
        valid_count = 0
        for k in range(K):
            pred = samples[k]["preds"][puzzle_idx].cpu().numpy().reshape(n, n)
            h = hash(pred.tobytes())
            is_valid = verify_nqueens(pred, n)
            valid_count += is_valid
            if h in pred_map:
                pred_map[h][1] += 1
            else:
                pred_map[h] = [pred, 1, is_valid]
        
        unique_preds = sorted(pred_map.values(), key=lambda x: -x[1])
        num_unique = len(unique_preds)
        
        # Skip if multi_sol_only and only 1 unique sample
        if multi_sol_only and num_unique <= 1:
            continue
        
        # Find all GT solutions
        all_gts = find_all_solutions(inp, n, max_solutions=10)
        num_gts = len(all_gts)
        
        # Create figure
        total = 1 + num_gts + num_unique  # Input + GTs + Samples
        ncols = min(6, total)
        nrows = (total + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
        axes = np.array(axes).reshape(-1)
        
        for ax in axes[total:]:
            ax.axis('off')
        
        # Draw Input
        _draw_nqueens_board(axes[0], n, inp, title="Input")
        
        # Draw GTs
        for i, gt in enumerate(all_gts):
            title = f"GT{i+1}" if num_gts > 1 else "GT"
            _draw_nqueens_board(axes[1 + i], n, gt, inp, title=title)
        
        # Draw Samples
        for i, (pred, count, is_valid) in enumerate(unique_preds):
            ax_idx = 1 + num_gts + i
            title = f"S{i+1}" + (f" ({count})" if count > 1 else "") + (" ✓" if is_valid else " ✗")
            _draw_nqueens_board(axes[ax_idx], n, pred, inp, title=title, is_valid=is_valid)
        
        gt_str = f"{num_gts} GT{'s' if num_gts > 1 else ''}"
        fig.suptitle(f"Puzzle {batch_idx * num_puzzles + puzzle_idx}: {valid_count}/{K} valid, {num_unique} unique ({gt_str})", 
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        
        save_path = f"{out_dir}/puzzle_{batch_idx:03d}_{puzzle_idx:02d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path} ({valid_count}/{K} valid, {num_unique} unique)")
        saved_count += 1


def _draw_nqueens_board(ax, n, board, input_board=None, title="", is_valid=None):
    """Draw N-Queens board. Green=given queen, Blue=predicted queen, Red circle=conflict."""
    import matplotlib.patches as patches
    
    # Checkerboard
    for i in range(n):
        for j in range(n):
            color = '#F0D9B5' if (i + j) % 2 == 0 else '#B58863'
            ax.add_patch(patches.Rectangle((j, n-1-i), 1, 1, facecolor=color))
    
    # Queens
    queens = np.argwhere(board == 2)
    input_queens = set(map(tuple, np.argwhere(input_board == 2))) if input_board is not None else set()
    
    for r, c in queens:
        color = '#2E7D32' if (r, c) in input_queens else '#1565C0'
        ax.text(c + 0.5, n-1-r + 0.5, '♛', fontsize=min(24, 200//n), 
                ha='center', va='center', color=color, fontweight='bold')
    
    # Highlight conflicts
    if is_valid == False:
        conflicts = _find_conflicts(queens)
        for idx in conflicts:
            r, c = queens[idx]
            ax.add_patch(patches.Circle((c+0.5, n-1-r+0.5), 0.45, fill=False, edgecolor='red', linewidth=2))
    
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title color
    color = 'green' if is_valid == True else ('red' if is_valid == False else 'black')
    ax.set_title(title, fontsize=10, color=color, fontweight='bold')
    
    # Border
    if is_valid is not None:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('green' if is_valid else 'red')
            spine.set_linewidth(3)


def _find_conflicts(queens):
    """Return indices of queens in conflict."""
    rows, cols, d1, d2 = {}, {}, {}, {}
    for i, (r, c) in enumerate(queens):
        rows.setdefault(r, []).append(i)
        cols.setdefault(c, []).append(i)
        d1.setdefault(r-c, []).append(i)
        d2.setdefault(r+c, []).append(i)
    
    conflicts = set()
    for d in [rows, cols, d1, d2]:
        for indices in d.values():
            if len(indices) > 1:
                conflicts.update(indices)
    return conflicts


def load_value_network(value_network_path, device):
    """Load trained value network from checkpoint.

    Returns:
        model: Loaded value network
        h_state_mean: Normalization mean
        h_state_std: Normalization std
    """
    from train_value_network import ValueNetwork

    print(f"Loading value network from {value_network_path}")
    checkpoint = torch.load(value_network_path, map_location=device, weights_only=False)

    config = checkpoint['config']
    normalization = checkpoint['normalization']

    model = ValueNetwork(
        hidden_dim=config['hidden_dim'],
        use_mlp=config['use_mlp'],
        num_layers=config.get('num_layers', 2),
        expansion=config.get('expansion', 4)
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    h_state_mean = normalization['h_state_mean'].to(device)
    h_state_std = normalization['h_state_std'].to(device)

    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  Mode: {'MLP' if config['use_mlp'] else 'Linear'}")

    return model, h_state_mean, h_state_std


def main():
    parser = argparse.ArgumentParser(description='Pass@k evaluation for checkpoints')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to checkpoint directory')
    parser.add_argument('--num_samples', type=int, default=20,
                       help='Number of samples per input for pass@k')
    parser.add_argument('--data_paths', type=str, default=None,
                       help='Override data paths (optional)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Sampling temperature (0.0=greedy, >0.0=sample)')
    parser.add_argument('--filter_augmented', action='store_true',
                       help='Filter out augmented examples (for ARC task)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')
    parser.add_argument('--max_h_cycles', type=int, default=None,
                       help='Maximum H-cycles (default: None for unlimited)')
    parser.add_argument('--value_network_path', type=str, default=None,
                       help='Path to trained value network for best-of-N selection (optional)')
    parser.add_argument('--use_online_value_head', action='store_true',
                       help='Use model\'s built-in value head instead of external network')
    parser.add_argument('--effective_batch_size', type=int, default=None,
                       help='Effective batch size B*K for GPU (default: config.global_batch_size * num_samples)')
    parser.add_argument('--visualize', action='store_true',
                       help='Save prediction visualizations')
    parser.add_argument('--visualize_multi_sol', action='store_true',
                       help='Only visualize puzzles with multiple unique samples')
    parser.add_argument('--enable_logging', action='store_true',
                       help='Enable per-puzzle logging for graph coloring (slower)')
    args = parser.parse_args()

    # Load checkpoint and config
    checkpoint_file = os.path.join(args.checkpoint_path, 'checkpoint.pt')
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    rank = 0
    world_size = 1
    cpu_group = None

    if "LOCAL_RANK" in os.environ:
        # Initialize distributed
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=48))

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        # CPU GLOO process group for CPU operations
        cpu_group = dist.new_group(backend="gloo", timeout=timedelta(hours=48))
        assert (
            dist.get_rank(cpu_group) == rank and dist.get_world_size(cpu_group) == world_size
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Seed RNGs to ensure consistency
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
        # For full reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if rank == 0:
        print(f"Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    config: PretrainConfig = checkpoint['config']

    # Override data paths if provided (for test set evaluation)
    if args.data_paths:
        # Use provided paths as test data
        config.data_paths_test = eval(args.data_paths)

    # Ensure we have test data configured
    if not config.data_paths_test:
        config.data_paths_test = config.data_paths

    # Override temperature if provided
    if hasattr(config.arch, 'temperature'):
        config.arch.temperature = args.temperature

    # Determine which data to use
    data_paths = config.data_paths_test if config.data_paths_test else config.data_paths

    # Compute batch size from effective_batch_size
    if args.effective_batch_size is not None:
        eval_batch_size = args.effective_batch_size // args.num_samples
        if eval_batch_size == 0:
            raise ValueError(f"effective_batch_size ({args.effective_batch_size}) must be >= num_samples ({args.num_samples})")
    else:
        eval_batch_size = config.global_batch_size

    # max_h_cycles override
    if args.max_h_cycles is not None:
        config.arch.halt_max_steps = args.max_h_cycles

    # Load external value network if provided (online value head checked after model load)
    value_network = None
    h_state_mean = None
    h_state_std = None

    if args.value_network_path:
        if rank == 0:
            print(f"\nLoading external value network for best-of-N selection...")
        value_network, h_state_mean, h_state_std = load_value_network(args.value_network_path, device)
        # Broadcast to all ranks
        if world_size > 1:
            with torch.no_grad():
                for param in list(value_network.parameters()) + list(value_network.buffers()):
                    dist.broadcast(param, src=0)

    if rank == 0:
        print(f"\nConfiguration:")
        print(f"  Architecture: {config.arch.name}")
        print(f"  Train data: {config.data_paths}")
        print(f"  Eval data: {data_paths} (TEST SET)")
        print(f"  Batch size: {eval_batch_size} (B), Samples: {args.num_samples} (K) -> Effective: {eval_batch_size * args.num_samples} (B*K)")
        print(f"  Temperature: {args.temperature}")
        print(f"  Filter augmented: {args.filter_augmented}")
        print(f"  Max H-cycles: {args.max_h_cycles if args.max_h_cycles else 'unlimited'}")
        if args.use_online_value_head:
            print(f"  Value: online value head (built-in)")
        elif value_network:
            print(f"  Value: external network (loaded)")
        else:
            print(f"  Value: disabled")
        print(f"  World size: {world_size}")
        print()

    # Load eval data
    if rank == 0:
        print("Loading evaluation data...")
    eval_loader, eval_metadata = create_dataloader(
        config, "test", test_set_mode=True,
        epochs_per_iter=1, global_batch_size=eval_batch_size,
        rank=rank, world_size=world_size
    )

    # Create evaluators (manually create Sudoku evaluator for pass@k)
    if rank == 0:
        print("Creating pass@k evaluator...")
    
    # Detect task type from data path
    data_path = data_paths[0] if data_paths else config.data_paths[0]
    is_arc = 'arc' in data_path.lower()
    is_sudoku = 'sudoku' in data_path.lower()

    pass_Ks= make_pass_ks(args.num_samples, filter_augmentation=args.filter_augmented if is_arc else True)
    if is_arc:
        from evaluators.arc_pass_k import ARC
        from dataset.build_arc_dataset import inverse_aug
        evaluators = [
            ARC(
                data_path=data_path,
                eval_metadata=eval_metadata,
                submission_K=2,
                pass_Ks=pass_Ks,
                aggregated_voting=True,
            )
        ]
        if rank == 0:
            mode_str = "with augmentation aggregation" if not args.filter_augmented else "original test cases only"
            print(f"Using ARC pass@k evaluator ({mode_str})")
    elif is_sudoku:
        from evaluators.sudoku import Sudoku
        inverse_aug = None
        evaluators = [
            Sudoku(
                data_path=data_path,
                eval_metadata=eval_metadata,
                pass_Ks=pass_Ks,
                aggregated_voting=True
            )
        ]
    elif "nqueen" in data_path.lower():
        if rank == 0:
            print("Creating N-Queen pass@k evaluator...")
        from evaluators.nqueens import NQueens
        evaluators = [
            NQueens(
                data_path=data_path,
                eval_metadata=eval_metadata,
                pass_Ks=pass_Ks,
                aggregated_voting=True
            )
        ]
    elif "graph-coloring" in data_path.lower() or "graph_coloring" in data_path.lower():
        if rank == 0:
            print("Creating Graph Coloring pass@k evaluator...")
        from evaluators.graph_coloring import GraphColoring
        # Extract max_colors from vocab_size (vocab: 0=ignore, 1=no-edge, 2=edge, 3+=colors)
        max_colors = eval_metadata.vocab_size - 3
        evaluators = [
            GraphColoring(
                data_path=data_path,
                eval_metadata=eval_metadata,
                pass_Ks=pass_Ks,
                aggregated_voting=True,
                max_colors=max_colors,
                enable_logging=args.enable_logging
            )
        ]
    else:
        raise ValueError(f"Unknown dataset type in path: {data_path}. Expected 'sudoku', 'arc', 'nqueen', or 'graph-coloring'.")

    # Load model
    if rank == 0:
        print("Loading model...")
    from utils.functions import load_model_class

    # Use checkpoint's arch config and update metadata fields
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size,
        vocab_size=eval_metadata.vocab_size,
        seq_len=eval_metadata.seq_len,
        num_puzzle_identifiers=eval_metadata.num_puzzle_identifiers,
        causal=False,
        temperature=args.temperature,  # Explicitly pass temperature for sampling
    )

    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)

    # Load state dict (includes puzzle_emb weights)
    # Remove _orig_mod prefix from torch.compile
    state_dict = checkpoint['model']
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, assign=True)
    
    # Load EMA if available
    if 'ema' in checkpoint:
        if rank == 0:
            print('  ✓ Loading EMA parameters from checkpoint')
        from models.ema import EMAHelper
        ema_helper = EMAHelper(mu=0.9999)  # mu doesn't matter for loading
        ema_helper.register(model)
        state_dict = checkpoint['ema']
        state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        ema_helper.load_state_dict(state_dict)
        ema_helper.ema(model)  # Apply EMA parameters to model

    model.eval()

    # Ensure all parameters and buffers are on device
    model = model.to(device)

    # Broadcast model parameters from rank 0
    if world_size > 1:
        with torch.no_grad():
            for param in list(model.parameters()) + list(model.buffers()):
                dist.broadcast(param, src=0)

    # Check for online value head and validate requirements (after model is loaded)
    if args.use_online_value_head:
        if not hasattr(model.model.inner, 'value_head') or model.model.inner.value_head is None:
            raise ValueError("Model doesn't have value_head. Train with use_online_value_head=True")
        if rank == 0:
            print(f"\nUsing model's online value head")

    if rank == 0:
        print(f"\nRunning pass@{args.num_samples} evaluation...")
        print("=" * 60)

    # Initialize evaluators
    for evaluator in evaluators:
        evaluator.begin_eval()

    # Collect required outputs
    return_keys = set()
    for evaluator in evaluators:
        return_keys.update(evaluator.required_outputs)

    # Add value_pred to return_keys if using online value head
    if args.use_online_value_head:
        return_keys.add("value_pred")

    # Run evaluation: iterate dataset once, sample K times per batch
    with torch.inference_mode():
        batch_iterator = eval_loader
        total_batches_processed = 0
        total_original_examples = 0
        
        if rank == 0:
            # Progress bar (no total - varies per rank in distributed mode)
            eval_loader.dataset._lazy_load_dataset()
            for set_i, (set_name, dataset) in enumerate(eval_loader.dataset._data.items()):  # type: ignore
                total_examples = len(dataset["inputs"])
                print(f"Total examples in {set_name}: {total_examples}")
            batch_iterator = tqdm.tqdm(batch_iterator, desc=f"Eval K={args.num_samples}", unit="batch")

        for set_name, batch, global_batch_size in batch_iterator:
            # To device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # For ARC: Filter to only original test cases BEFORE forward
            if is_arc and inverse_aug is not None and args.filter_augmented:
                # Get identifier map from evaluator
                identifier_map = evaluators[0].identifier_map
                
                # Find original indices
                puzzle_identifiers = batch["puzzle_identifiers"].cpu().numpy()
                
                original_mask = []
                for identifier in puzzle_identifiers:
                    if identifier == eval_metadata.blank_identifier_id:
                        original_mask.append(False)
                    else:
                        name = identifier_map[int(identifier)]
                        orig_name, _ = inverse_aug(name)
                        original_mask.append(orig_name == name)
                
                original_mask = np.array(original_mask)
                
                # Skip batch if no original examples
                if not np.any(original_mask):
                    continue
                
                # Count originals for logging
                total_original_examples += np.sum(original_mask)
                
                # Filter batch to only original examples
                filtered_batch = {}
                for k, v in batch.items():
                    filtered_batch[k] = v[torch.from_numpy(original_mask).to(device)]
                
                batch = filtered_batch
            
            total_batches_processed += 1

            # Sample K candidates in parallel
            batch_samples = sample_parallel(
                model, batch, args.num_samples, return_keys, device,
                extract_h_states=(value_network is not None or args.use_online_value_head), data_path=data_path.lower()
            )

            # Visualize N-Queens predictions (if enabled)
            if (args.visualize or args.visualize_multi_sol) and rank == 0 and "nqueen" in data_path.lower():
                if not hasattr(viz_nqueens_samples, '_saved_count'):
                    viz_nqueens_samples._saved_count = 0
                    viz_nqueens_samples._batch_count = 0
                if viz_nqueens_samples._saved_count < 10:
                    viz_nqueens_samples(batch, batch_samples, viz_nqueens_samples._batch_count,
                                        multi_sol_only=args.visualize_multi_sol)
                    viz_nqueens_samples._batch_count += 1

            # Best-of-N selection using value network (if enabled)
            best_of_n_metrics = None
            if value_network is not None or args.use_online_value_head:
                with torch.no_grad():
                    # Get value predictions
                    if args.use_online_value_head and "value_pred" in batch_samples[0]:
                        # Online value head: predictions already computed
                        value_preds = torch.stack([s["value_pred"] for s in batch_samples], dim=0)  # (K, B)
                    elif value_network is not None:
                        # External value network: extract h_states and normalize
                        all_h_states = torch.stack([s["h_states"] for s in batch_samples], dim=0)
                        all_h_states_norm = (all_h_states - h_state_mean) / h_state_std
                        K, B, hidden_dim = all_h_states.shape
                        value_preds = value_network(all_h_states_norm.view(K * B, hidden_dim)).view(K, B)
                    else:
                        raise ValueError("No value predictions available")

                    # Select best sample per puzzle (higher value = better)
                    best_indices = value_preds.argmax(dim=0)  # (B,)

                    # Compute best-of-N accuracy
                    B = batch["inputs"].shape[0]
                    best_exact = torch.zeros(B, dtype=torch.bool, device=device)
                    best_token = torch.zeros(B, dtype=torch.float32, device=device)
                    best_valid = torch.zeros(B, dtype=torch.bool, device=device)

                    for b in range(B):
                        idx = best_indices[b].item()
                        best_exact[b] = batch_samples[idx]["exact_correct"][b]
                        best_token[b] = batch_samples[idx]["token_accuracy"][b]
                        best_valid[b] = batch_samples[idx]["valid"][b]

                    best_of_n_metrics = {
                        "exact_correct": best_exact,
                        "token_accuracy": best_token,
                        "valid": best_valid,
                        "best_indices": best_indices,
                        "value_preds": value_preds  # (K, B) tensor of value predictions per sample
                    }

            # Update evaluators with all K samples for this batch
            for evaluator in evaluators:
                evaluator.update_batch_samples(batch, batch_samples, args.num_samples, best_of_n_metrics)

    if rank == 0:
        print("\n" + "=" * 60)
        if is_arc:
            print(f"Processed {total_batches_processed} batches with {total_original_examples} original test cases")
        print("Computing pass@k metrics...")

    # Compute final metrics (evaluator handles cross-rank reduction)
    final_metrics = {}
    for evaluator in evaluators:
        metrics = evaluator.result(
            save_path=args.checkpoint_path,
            rank=rank,
            world_size=world_size,
            group=cpu_group
        )
        if rank == 0 and metrics:
            final_metrics.update(metrics)

    # Print and log results (rank 0 only)
    if rank == 0:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        for key in sorted(final_metrics.keys()):
            value = final_metrics[key]
            print(f"{key:40s}: {value:.4f}")

        # Save results to text log
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cycles_suffix = f"_cycles{args.max_h_cycles}" if args.max_h_cycles else ""
        log_filename = f"eval_results{cycles_suffix}_{timestamp}.txt"
        log_path = os.path.join(args.checkpoint_path, log_filename)

        with open(log_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("EVALUATION RESULTS\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Checkpoint: {args.checkpoint_path}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Num samples: {args.num_samples}\n")
            f.write(f"Temperature: {args.temperature}\n")
            f.write(f"Max H-cycles: {args.max_h_cycles if args.max_h_cycles else 'unlimited'}\n")
            f.write(f"Eval data: {data_paths}\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("METRICS\n")
            f.write("=" * 60 + "\n\n")
            for key in sorted(final_metrics.keys()):
                value = final_metrics[key]
                f.write(f"{key:40s}: {value:.4f}\n")

        print(f"\nResults saved to: {log_path}")
        print("\nDone!")

    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()