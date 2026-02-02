"""
Visualize latent space trajectories for a SINGLE puzzle.

Samples multiple trajectories from one puzzle and visualizes them in 2D latent space using PCA.
Supports both independent sampling and value-guided beam search.

Usage:
    # Default: Independent sampling (visualize both h_states and decoded spaces)
    python visualize_single_puzzle.py \
        --checkpoint_path checkpoints/run_name/step_50000 \
        --data_paths "['data/maze-30x30-hard-1k']" \
        --puzzle_idx 0 \
        --num_samples 20

    # Visualize only h_states
    python visualize_single_puzzle.py \
        --checkpoint_path checkpoints/run_name/step_50000 \
        --data_paths "['data/maze-30x30-hard-1k']" \
        --puzzle_idx 0 \
        --num_samples 20 \
        --space h_states

    # With value-guided beam search (requires model with value_head)
    python visualize_single_puzzle.py \
        --checkpoint_path checkpoints/run_name/step_50000 \
        --data_paths "['data/maze-30x30-hard-1k']" \
        --puzzle_idx 0 \
        --num_samples 20 \
        --beam_size 5 \
        --branch_every 2 \
        --use_online_value_head
"""

import os
import argparse
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataclasses import replace
from sklearn.decomposition import PCA
from scipy.spatial import cKDTree

from pretrain import (
    PretrainConfig, create_dataloader, TrainState, evaluate,
    ArchConfig, LossConfig, EvaluatorConfig
)
from utils.functions import load_model_class
from sampling_strategies import sample_parallel, sample_with_beam_search


def sample_trajectories(model, batch, device, num_samples=20, max_h_cycles=50,
                        beam_size=None, branch_every=1, use_online_value_head=False,
                        samples_per_batch=None):
    """Sample trajectories with optional batching to reduce memory."""
    if max_h_cycles is not None:
        model.model.config.halt_max_steps = max_h_cycles

    return_keys = {"logits", "preds", "q_halt_logits"}
    if use_online_value_head:
        return_keys.add("value_pred")

    trajectories = []
    spb = samples_per_batch or num_samples

    for i in range(0, num_samples, spb):
        k = min(spb, num_samples - i)
        if beam_size is not None:
            batch_samples = sample_with_beam_search(
                model, batch, num_samples, beam_size, branch_every,
                None, None, None, return_keys, device,
                track_trajectories=True, use_online_value_head=use_online_value_head)
        else:
            batch_samples = sample_parallel(
                model, batch, k, return_keys, device,
                extract_h_states=use_online_value_head, track_trajectories=True)

        for sample in batch_samples:
            trajectories.append({
                'h_states': [h.cpu() for h in sample['trajectory_h_states'][0]],
                'h_preds': sample['trajectory_preds'][0],
                'steps': len(sample['trajectory_h_states'][0]),
                'beam_selections': sample.get('beam_selections', [[]])[0]})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if beam_size is not None:
            break

    return trajectories


def extract_features(trajectories, target_labels=None, target_mask=None, lm_head=None, device=None, use_decoded=False, vocab_size=None):
    """Extract features from trajectories (mean pooling over content tokens).

    Args:
        use_decoded: If True, extract features from decoded logits space instead of h_states
        vocab_size: Vocabulary size (needed for one-hot encoding target in decoded space)

    Returns:
        all_features: (num_h_states, feature_dim) array of features
        all_h_states: list of h_states tensors
        trajectory_info: list of dicts with trajectory metadata
        target_feature: (feature_dim,) array of target one-hot if use_decoded, else None
    """
    all_features = []
    all_h_states = []
    trajectory_info = []
    target_feature = None

    # Compute target feature if in decoded space (will compute after getting logit scale)
    target_feature = None

    for traj_idx, traj in enumerate(trajectories):
        h_states = traj['h_states']

        # Extract features for this trajectory
        # h_states from sampling_strategies.py are shape (seq_len, hidden_dim) - no batch dim
        h_states_content = [z_H[16:].cpu() for z_H in h_states]

        if use_decoded and lm_head is not None:
            # Decode h_states to logit space and use as features
            with torch.no_grad():
                decoded_features = []
                all_logits_for_scale = []
                for idx, h_state in enumerate(h_states_content):
                    logits = lm_head(h_state.to(device).unsqueeze(0))[0]  # (seq_len, vocab_size)
                    # Mean pool over content tokens (use raw logits)
                    mean_logits = logits.float().cpu().numpy().mean(axis=0)  # (vocab_size,)
                    decoded_features.append(mean_logits)
                    # Collect all logits for computing scale
                    all_logits_for_scale.append(logits.float().cpu().numpy())
                features = np.array(decoded_features)

                # Compute logit scale from all trajectories (only once)
                if traj_idx == 0 and target_feature is None and use_decoded:
                    # Compute typical logit magnitude across all H-states
                    all_logits_stacked = np.concatenate(all_logits_for_scale, axis=0)  # (total_tokens, vocab_size)
                    # Use standard deviation as scale (represents typical variation)
                    logit_scale = float(all_logits_stacked.std())

                    # Create target logits using this scale (on same device as target_mask)
                    target_logits = torch.full((len(target_labels), vocab_size), -logit_scale,
                                               dtype=torch.float32, device=target_mask.device)
                    target_logits[torch.arange(len(target_labels), device=target_mask.device), target_labels.long()] = logit_scale
                    # Mask and mean pool
                    target_logits = target_logits * target_mask.unsqueeze(-1).float()
                    target_feature = (target_logits.sum(dim=0) / target_mask.sum()).cpu().numpy()
        else:
            # Use h_states directly (current behavior)
            # z_H is (seq_len, hidden_dim), so z_H[16:] gives content tokens
            features = np.array([z_H[16:].float().cpu().numpy().mean(axis=0) for z_H in h_states])

        # Compute correctness for each H-state if possible
        h_correct = []
        if lm_head is not None and target_labels is not None:
            with torch.no_grad():
                for h_state in h_states_content:
                    logits = lm_head(h_state.to(device).unsqueeze(0))[0]
                    preds = torch.argmax(logits, dim=-1)
                    is_correct = ((preds == target_labels) & target_mask).all().item()
                    h_correct.append(is_correct)

        all_features.append(features)
        all_h_states.extend(h_states_content)
        trajectory_info.append({
            'traj_idx': traj_idx,
            'features': features,
            'h_states': h_states_content,
            'h_correct': h_correct,
            'steps': traj['steps']
        })

    return np.vstack(all_features), all_h_states, trajectory_info, target_feature


def fit_reducer(features, target_feature=None):
    """Fit PCA on features.

    Args:
        features: (N, D) array of trajectory features
        target_feature: Optional (D,) array to include in fitting (e.g., one-hot target)

    Returns:
        reducer: Fitted PCA object
        embeddings: (N, 2) transformed trajectory features
        target_embedding: (2,) transformed target feature if provided, else None
    """
    reducer = PCA(n_components=2)
    target_embedding = None

    if target_feature is not None:
        # Include target in fitting
        combined = np.vstack([features, target_feature.reshape(1, -1)])
        combined_embeddings = reducer.fit_transform(combined)
        embeddings = combined_embeddings[:-1]
        target_embedding = combined_embeddings[-1]
    else:
        # Fit on features only
        embeddings = reducer.fit_transform(features)

    print(f"PCA variance explained: PC1={reducer.explained_variance_ratio_[0]:.1%}, PC2={reducer.explained_variance_ratio_[1]:.1%}")
    return reducer, embeddings, target_embedding


def count_nqueens_violations(board):
    """Count number of attacking queen pairs in N-Queens board."""
    n = int(np.sqrt(len(board)))
    board = board.reshape(n, n)
    queens = np.argwhere(board == 2)
    violations = 0
    for i, (r1, c1) in enumerate(queens):
        for r2, c2 in queens[i+1:]:
            if r1 == r2 or c1 == c2 or abs(r1-r2) == abs(c1-c2):
                violations += 1
    # Also penalize wrong queen count
    violations += abs(len(queens) - n)
    return violations


def compute_grid_metric(embeddings, h_states, xlim, ylim, target_labels, target_mask, lm_head, device, grid_size=50, k_neighbors=15, metric='ce_loss'):
    """Compute metric (CE loss or constraint satisfaction) on interpolated grid."""
    import torch.nn.functional as F

    x_grid = np.linspace(xlim[0], xlim[1], grid_size)
    y_grid = np.linspace(ylim[0], ylim[1], grid_size)

    tree = cKDTree(embeddings)
    h_states_tensor = torch.stack(h_states).to(device)
    target_labels = target_labels.long().to(device)
    target_mask = target_mask.to(device)

    grid_coords = []
    grid_values = []

    for y in y_grid:
        for x in x_grid:
            dists, idxs = tree.query([x, y], k=k_neighbors)
            weights = 1.0 / (dists + 1e-6)
            weights /= weights.sum()
            h_interp = sum(w * h_states_tensor[idx] for w, idx in zip(weights, idxs))

            with torch.no_grad():
                logits = lm_head(h_interp.unsqueeze(0))[0]
                preds = torch.argmax(logits, dim=-1)

                if metric == 'constraint':
                    # Constraint satisfaction (lower = better, like loss)
                    violations = count_nqueens_violations(preds.cpu().numpy())
                    value = float(violations)
                else:
                    # CE loss (default)
                    loss_per_token = F.cross_entropy(logits, target_labels, reduction='none')
                    value = (loss_per_token * target_mask.float()).sum() / target_mask.sum()
                    value = value.item()

            grid_coords.append([x, y])
            grid_values.append(value)

    return np.array(grid_coords), np.array(grid_values)


def plot_trajectories(embeddings, trajectory_info, all_h_states, target_labels, target_mask, lm_head, device, output_path, k_neighbors=15, reducer=None, show_gradients=False, use_decoded=False, target_embedding=None, log_scale=False, theme='default', cmap=None, metric='ce_loss'):
    """Plot all trajectories in 2D latent space with loss/constraint landscape.

    Args:
        use_decoded: If True, plot is in decoded logits space, else h_states space
        target_embedding: Optional (2,) array with target position in 2D space
        log_scale: If True, use logarithmic scale for colormap
        theme: 'default', 'paper', or 'density'
        cmap: Optional colormap name (overrides theme default)
        metric: 'ce_loss' or 'constraint' (N-Queens violations)
    """

    # For density/kde theme, compute shared origin first to include in bounds
    shared_origin = None
    if theme in ['density', 'kde']:
        temp_start_idx = 0
        h0_fakes = []
        for info in trajectory_info:
            num_h = len(info['features'])
            traj_emb = embeddings[temp_start_idx:temp_start_idx + num_h]
            temp_start_idx += num_h
            if num_h >= 2:
                h1, h2 = traj_emb[0], traj_emb[1]
                h0_fake = h1 - 0.5 * (h2 - h1)
            else:
                h0_fake = traj_emb[0]
            h0_fakes.append(h0_fake)
        shared_origin = np.array(h0_fakes).mean(axis=0)

    # Compute bounds (make square for undistorted view)
    # Include shared_origin in bounds for density theme
    all_points = embeddings if shared_origin is None else np.vstack([embeddings, shared_origin])
    data_range = all_points.max(axis=0) - all_points.min(axis=0)

    # If variance is too small (all points nearly identical), use default bounds
    if data_range[0] < 1e-6 and data_range[1] < 1e-6:
        print("Warning: All embeddings are identical, using default bounds [-5, 5]")
        xlim = (-5, 5)
        ylim = (-5, 5)
    else:
        margin = 0.1 * data_range
        # Use minimum margin of 0.1 to avoid zero-width plots
        margin = np.maximum(margin, 0.1)
        xlim = (all_points[:, 0].min() - margin[0], all_points[:, 0].max() + margin[0])
        ylim = (all_points[:, 1].min() - margin[1], all_points[:, 1].max() + margin[1])

    # Print loss at trajectory points
    import torch.nn.functional as F
    for i, h_state in enumerate(all_h_states[:10]):  # Print first 10
        with torch.no_grad():
            logits = lm_head(h_state.to(device).unsqueeze(0))[0]
            loss_per_token = F.cross_entropy(logits, target_labels.long().to(device), reduction='none')
            loss = (loss_per_token * target_mask.to(device).float()).sum() / target_mask.sum()

    # Compute grid metric (higher resolution for smooth contours)
    metric_name = "constraint violations" if metric == 'constraint' else "CE loss"
    print(f"\nComputing {metric_name} landscape...")
    grid_coords, grid_values = compute_grid_metric(
        embeddings, all_h_states, xlim, ylim, target_labels, target_mask, lm_head, device,
        grid_size=80, k_neighbors=k_neighbors, metric=metric
    )
    print(f"Grid {metric_name} range: [{grid_values.min():.4f}, {grid_values.max():.4f}]")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot contours (invert colormap: low loss = green, high loss = red)
    grid_size = int(np.sqrt(len(grid_coords)))
    gX = grid_coords[:, 0].reshape(grid_size, grid_size)
    gY = grid_coords[:, 1].reshape(grid_size, grid_size)
    gloss = grid_values.reshape(grid_size, grid_size)

    # Apply Gaussian smoothing for smoother contours
    from scipy.ndimage import gaussian_filter
    gloss = gaussian_filter(gloss, sigma=2.0)

    # Set color scale normalization
    if log_scale:
        # Use logarithmic scale for better gradient visibility
        from matplotlib.colors import LogNorm
        vmin = gloss[gloss > 0].min() if (gloss > 0).any() else 1e-10
        vmax = gloss.max()
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        # Use percentile for better contrast (default)
        vmin, vmax = np.percentile(gloss, [5, 95])
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

    # Theme-specific settings
    if theme in ['paper', 'density', 'kde']:
        # Dark theme: purple/blue gradient, white lines
        default_cmap = 'plasma'  # Dark purple to yellow
        contour_colors = 'white'
        contour_alpha = 0.5
        contourf_alpha = 1.0
        show_colorbar = False
        show_grid_points = False
        num_levels = 50  # More levels for smoother gradient
        num_contour_lines = 15
    else:
        # Default theme: red-yellow-green
        default_cmap = 'RdYlGn_r'
        contour_colors = 'gray'
        contour_alpha = 0.3
        contourf_alpha = 0.6
        show_colorbar = True
        show_grid_points = True
        num_levels = 30
        num_contour_lines = 10

    # Use provided cmap or theme default
    cmap_to_use = cmap if cmap is not None else default_cmap

    # Skip contours for kde theme (uses solid background)
    if theme != 'kde':
        contourf = ax.contourf(gX, gY, gloss, levels=num_levels, cmap=cmap_to_use, norm=norm, alpha=contourf_alpha, zorder=0)
        ax.contour(gX, gY, gloss, levels=num_contour_lines, colors=contour_colors, linewidths=0.5, alpha=contour_alpha, zorder=1)

    # Add colorbar (default theme only)
    if show_colorbar and theme != 'kde':
        cbar = plt.colorbar(contourf, ax=ax)
        label = 'Constraint Violations' if metric == 'constraint' else 'Cross-Entropy Loss'
        cbar.set_label(label, fontsize=12)

    # Visualize grid points (default theme only)
    if show_grid_points:
        ax.scatter(grid_coords[:, 0], grid_coords[:, 1], c='blue', s=1, alpha=0.1, zorder=2)

    # Show gradients
    if show_gradients and reducer is not None:
        print("Computing gradients...")
        import torch.nn.functional as F

        # Sample points on a coarser grid for gradient arrows
        grad_grid_size = 10
        x_grad = np.linspace(xlim[0], xlim[1], grad_grid_size)
        y_grad = np.linspace(ylim[0], ylim[1], grad_grid_size)

        target_labels_device = target_labels.long().to(device)
        target_mask_device = target_mask.to(device)

        for xi, x in enumerate(x_grad):
            for yi, y in enumerate(y_grad):
                # Get 2D point
                point_2d = np.array([x, y]).reshape(1, -1)

                # Inverse transform to 512D space (one vector per content token)
                # Expand to full sequence: (900, 512) where all tokens share same 512D representation
                point_512d_single = torch.from_numpy(reducer.inverse_transform(point_2d)[0]).float().to(device)
                point_512d = point_512d_single.unsqueeze(0).expand(900, -1).clone()
                point_512d.requires_grad_(True)

                # Decode and compute loss
                logits = lm_head(point_512d.unsqueeze(0))[0]  # (seq_len, vocab_size)
                loss_per_token = F.cross_entropy(logits, target_labels_device, reduction='none')
                loss = (loss_per_token * target_mask_device.float()).sum() / target_mask_device.sum()

                # Compute gradient in 512D (average over sequence)
                loss.backward()
                grad_512d = point_512d.grad.mean(dim=0).detach().cpu().numpy()  # Average gradient across sequence

                # Project gradient to 2D PCA space
                grad_2d = reducer.transform(grad_512d.reshape(1, -1))[0] - reducer.transform(np.zeros((1, 512)))[0]

                # Plot gradient arrow (pointing downhill = negative gradient)
                arrow_scale = 0.1
                ax.arrow(x, y, -grad_2d[0] * arrow_scale, -grad_2d[1] * arrow_scale,
                        head_width=0.005, head_length=0.05, fc='black', ec='black', alpha=0.3, zorder=2)

    # Plot all trajectories
    start_idx = 0

    if theme == 'paper':
        # Paper theme: white lines, red start, green end
        line_color = 'white'
        line_alpha = 0.8
        line_width = 2.0
        start_color = 'red'
        end_color = 'lime'
    elif theme == 'density':
        # Density theme: visible lines with density effect from overlap
        line_color = 'white'
        num_samples = len(trajectory_info)
        # Higher alpha so lines are visible, overlap creates density
        line_alpha = min(0.7, max(0.1, 15.0 / num_samples))
        line_width = 1.5
        start_color = 'red'
        end_color = 'lime'

        # Collect trajectory embeddings and end points
        temp_start_idx = 0
        end_points = []
        all_traj_embs = []
        for info in trajectory_info:
            num_h = len(info['features'])
            traj_emb = embeddings[temp_start_idx:temp_start_idx + num_h]
            temp_start_idx += num_h
            all_traj_embs.append(traj_emb)
            end_points.append(traj_emb[-1])
        end_points = np.array(end_points)

        # Draw all trajectories from shared origin (pre-computed above)
        for traj_emb in all_traj_embs:
            # Prepend shared origin to trajectory
            traj_with_origin = np.vstack([shared_origin, traj_emb])
            ax.plot(traj_with_origin[:, 0], traj_with_origin[:, 1], '-', linewidth=line_width,
                    alpha=line_alpha, color=line_color, zorder=3, solid_capstyle='round')

        # Plot shared origin (red) and end points (green)
        point_alpha = min(0.8, max(0.3, 15.0 / num_samples))
        ax.scatter(shared_origin[0], shared_origin[1], s=150, color=start_color,
                   edgecolors='white', linewidths=2, zorder=7)
        ax.scatter(end_points[:, 0], end_points[:, 1], s=80, color=end_color,
                   edgecolors='none', zorder=6, alpha=point_alpha)
    elif theme == 'kde':
        # KDE theme: dark background + trajectories + endpoint KDE
        from scipy.stats import gaussian_kde

        # Dark background (low end of colormap for contrast with white lines)
        cmap_obj = plt.get_cmap(cmap_to_use)
        ax.set_facecolor(cmap_obj(0.0))  # Dark purple for plasma

        # Collect trajectories and endpoints
        num_samples = len(trajectory_info)
        line_alpha = min(0.8, max(0.2, 20.0 / num_samples))
        temp_start_idx = 0
        end_points = []
        all_traj_embs = []
        for info in trajectory_info:
            num_h = len(info['features'])
            traj_emb = embeddings[temp_start_idx:temp_start_idx + num_h]
            temp_start_idx += num_h
            all_traj_embs.append(traj_emb)
            end_points.append(traj_emb[-1])
        end_points = np.array(end_points)

        # Draw KDE of endpoints (with error handling for singular matrices)
        if len(end_points) > 2:
            try:
                kde = gaussian_kde(end_points.T)
                xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 100),
                                      np.linspace(ylim[0], ylim[1], 100))
                zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                ax.contourf(xx, yy, zz, levels=20, cmap='Greens', alpha=0.7, zorder=2)
                ax.contour(xx, yy, zz, levels=10, colors='white', linewidths=0.5, alpha=0.5, zorder=3)
            except np.linalg.LinAlgError:
                print("Warning: KDE failed (endpoints may be colinear), skipping density plot")

        # Draw trajectories (white lines on dark background)
        for traj_emb in all_traj_embs:
            traj_with_origin = np.vstack([shared_origin, traj_emb])
            ax.plot(traj_with_origin[:, 0], traj_with_origin[:, 1], '-', linewidth=2.0,
                    alpha=line_alpha, color='white', zorder=4)

        # Start point (red) and end points (lime)
        ax.scatter(*shared_origin, s=150, color='red', edgecolors='white', lw=2, zorder=7)
        ax.scatter(end_points[:, 0], end_points[:, 1], s=60, color='lime',
                   edgecolors='white', lw=0.5, alpha=0.8, zorder=6)
    else:
        # Default theme: colorful per-trajectory
        colors = plt.cm.tab20(np.linspace(0, 1, len(trajectory_info)))
        line_alpha = 0.5
        line_width = 1.5

    # Plot trajectories (skip for density/kde - already handled above)
    if theme not in ['density', 'kde']:
        for idx, info in enumerate(trajectory_info):
            num_h = len(info['features'])
            traj_emb = embeddings[start_idx:start_idx + num_h]
            start_idx += num_h

            if theme == 'paper':
                # Paper theme: clean white lines
                ax.plot(traj_emb[:, 0], traj_emb[:, 1], '-', linewidth=line_width, alpha=line_alpha, color=line_color, zorder=3)
                ax.scatter(traj_emb[0, 0], traj_emb[0, 1], s=120, color=start_color, edgecolors='white', linewidths=1.5, zorder=6)
                ax.scatter(traj_emb[-1, 0], traj_emb[-1, 1], s=120, color=end_color, edgecolors='white', linewidths=1.5, zorder=6)
            else:
                # Default theme: colorful with labels
                ax.plot(traj_emb[:, 0], traj_emb[:, 1], '-', linewidth=line_width, alpha=line_alpha, color=colors[idx], zorder=3)

                # Plot points with H-cycle labels (mark correct ones with green border)
                for h_idx, (x, y) in enumerate(traj_emb):
                    is_correct = info['h_correct'][h_idx] if h_idx < len(info['h_correct']) else False
                    edge_color = 'lime' if is_correct else 'black'
                    edge_width = 3 if is_correct else 1

                    ax.scatter(x, y, s=300, color=colors[idx], edgecolors=edge_color, linewidths=edge_width, zorder=4)
                    ax.text(x, y, str(h_idx), fontsize=8, ha='center', va='center', color='black', weight='bold', zorder=5)

                # Mark start and end
                ax.scatter(traj_emb[0, 0], traj_emb[0, 1], s=150, marker='s', facecolors='none', edgecolors=colors[idx], linewidths=3, zorder=6, label=f'Sample {idx}')
                ax.scatter(traj_emb[-1, 0], traj_emb[-1, 1], s=150, marker='*', color=colors[idx], edgecolors='black', linewidths=2, zorder=6)

    # Plot target point if available (decoded space only)
    if target_embedding is not None:
        if theme in ['paper', 'density']:
            ax.scatter(target_embedding[0], target_embedding[1], s=200, marker='o',
                      color='lime', edgecolors='white', linewidths=2, zorder=7)
        else:
            ax.scatter(target_embedding[0], target_embedding[1], s=500, marker='X',
                      color='gold', edgecolors='black', linewidths=3, zorder=7,
                      label='Target (Ideal Logits)')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if theme in ['paper', 'density', 'kde']:
        # Clean style: no grid, no axis labels, no title
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
    else:
        ax.grid(True, alpha=0.3)
        space_name = "Decoded Logits" if use_decoded else "H-States"
        ax.set_title(f'Single Puzzle: {len(trajectory_info)} Trajectories (PCA, {space_name})', fontsize=14)
        # Only show legend if 50 or fewer samples
        if len(trajectory_info) <= 50:
            ax.legend(fontsize=8, loc='upper right', ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize single puzzle latent trajectories')
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--data_paths', type=str, required=True)
    parser.add_argument('--puzzle_idx', type=int, default=None,
                       help='Puzzle index to visualize (default: None for all puzzles)')
    parser.add_argument('--max_puzzles', type=int, default=None,
                       help='Maximum number of puzzles to process (default: None for all)')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--samples_per_batch', type=int, default=None,
                       help='Sample in batches to reduce memory (default: all at once)')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--k_neighbors', type=int, default=15)
    parser.add_argument('--show_gradients', action='store_true', help='Show loss gradients (PCA only)')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')
    parser.add_argument('--space', type=str, default='both', choices=['h_states', 'decoded', 'both'],
                       help='Which space to visualize: h_states, decoded, or both (default: both)')
    parser.add_argument('--max_h_cycles', type=int, default=50,
                       help='Maximum number of H-cycles before stopping (default: 50)')
    parser.add_argument('--log_scale', action='store_true',
                       help='Use logarithmic scale for loss colormap (shows gradients more clearly)')
    parser.add_argument('--theme', type=str, default='default', choices=['default', 'paper', 'density', 'kde'],
                       help='Visualization theme: default (colorful), paper (dark, clean lines), density (all trajectories), kde (endpoint distribution)')
    parser.add_argument('--cmap', type=str, default=None,
                       help='Colormap for loss landscape (e.g., plasma, viridis, coolwarm, RdYlGn_r). Overrides theme default.')
    parser.add_argument('--metric', type=str, default='ce_loss', choices=['ce_loss', 'constraint'],
                       help='Metric for landscape: ce_loss (cross-entropy) or constraint (N-Queens violations)')
    # Beam search arguments
    parser.add_argument('--beam_size', type=int, default=None,
                       help='Beam size for value-guided search (requires --use_online_value_head)')
    parser.add_argument('--branch_every', type=int, default=1,
                       help='Beam selection frequency in H-cycles (default: 1)')
    parser.add_argument('--use_online_value_head', action='store_true',
                       help='Use model\'s built-in value head for beam search')
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Seed RNGs to ensure consistency
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        # For full reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}")
    checkpoint = torch.load(os.path.join(args.checkpoint_path, 'checkpoint.pt'), map_location=device, weights_only=False)
    config: PretrainConfig = checkpoint['config']
    config.data_paths_test = eval(args.data_paths)
    if hasattr(config.arch, 'temperature'):
        config.arch.temperature = args.temperature

    # Load data
    eval_loader, eval_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=1, rank=0, world_size=1
    )

    # Load model
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device(device):
        model = model_cls(dict(
            **config.arch.__pydantic_extra__,
            batch_size=1,
            vocab_size=eval_metadata.vocab_size,
            seq_len=eval_metadata.seq_len,
            num_puzzle_identifiers=eval_metadata.num_puzzle_identifiers,
            causal=False
        ))
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)

    state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state_dict, assign=True)
    model.eval()
    model = model.to(device)

    # Get lm_head for decoding
    lm_head = model.model.inner.lm_head

    # Check for online value head if requested
    if args.use_online_value_head:
        if not hasattr(model.model.inner, 'value_head') or model.model.inner.value_head is None:
            raise ValueError("Model doesn't have value_head. Train with arch.use_value_head=True")
        print(f"\nUsing model's online value head")

    # Validate beam search arguments
    if args.beam_size is not None and not args.use_online_value_head:
        raise ValueError("Beam search (--beam_size) requires --use_online_value_head")

    print(f"\nConfiguration:")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Max H-cycles: {args.max_h_cycles}")
    print(f"  Temperature: {args.temperature}")
    if args.beam_size:
        print(f"  Beam search: beam_size={args.beam_size}, branch_every={args.branch_every}")
    else:
        print(f"  Sampling mode: independent (no beam search)")
    print()

    # Determine puzzles to process
    if args.puzzle_idx is not None:
        puzzle_indices = [args.puzzle_idx]
    else:
        eval_loader.dataset._lazy_load_dataset()
        total = sum(len(d["inputs"]) for _, d in eval_loader.dataset._data.items())
        puzzle_indices = range(args.max_puzzles if args.max_puzzles else total)
        print(f"Processing {len(puzzle_indices)} puzzles")

    # Spaces to visualize
    spaces = [('h_states', False)] if args.space == 'h_states' else \
             [('decoded', True)] if args.space == 'decoded' else \
             [('h_states', False), ('decoded', True)]

    output_dir = args.output_dir or os.path.join(args.checkpoint_path, 'all_puzzles' if args.puzzle_idx is None else f'puzzle_{args.puzzle_idx}')
    os.makedirs(output_dir, exist_ok=True)

    # Process each puzzle
    for puzzle_idx in puzzle_indices:
        print(f"\n{'='*60}\nPuzzle {puzzle_idx}\n{'='*60}")

        # Get batch
        for idx, (_, batch, _) in enumerate(eval_loader):
            if idx == puzzle_idx:
                batch = {k: v.to(device) for k, v in batch.items()}
                target_labels, target_mask = batch["labels"][0], (batch["labels"][0] != -100)
                break

        # Sample trajectories
        trajectories = sample_trajectories(
            model, batch, device, args.num_samples, args.max_h_cycles,
            beam_size=args.beam_size, branch_every=args.branch_every,
            use_online_value_head=args.use_online_value_head,
            samples_per_batch=args.samples_per_batch
        )

        # Analyze solution diversity
        print("\nAnalyzing solution diversity...")
        final_preds = []
        for traj in trajectories:
            h_final = traj['h_states'][-1].to(device)
            with torch.no_grad():
                logits = lm_head(h_final.unsqueeze(0))[0]
                pred = torch.argmax(logits, dim=-1).cpu().numpy()
            final_preds.append(pred)

        # Count unique solutions
        unique_solutions = {hash(p.tobytes()) for p in final_preds}

        # Check validity based on task (detect from data_path)
        data_path = args.data_paths.lower()
        valid_count = 0
        total_solutions = None
        puzzle_emb_len = 16  # Skip puzzle embedding tokens
        if 'nqueen' in data_path:
            from evaluators.nqueens_utils import find_all_solutions
            for pred in final_preds:
                pred_content = pred[puzzle_emb_len:]
                if count_nqueens_violations(pred_content) == 0:
                    valid_count += 1
            validity_type = "N-Queens constraint"
            # Try to compute total solutions
            try:
                input_board = batch["inputs"][0, puzzle_emb_len:].cpu().numpy()
                n = int(np.sqrt(len(input_board)))
                if n * n == len(input_board):
                    all_sols = find_all_solutions(input_board.reshape(n, n), n, max_solutions=1000)
                    total_solutions = len(all_sols)
            except Exception:
                pass  # Skip if encoding doesn't match
        elif 'graph' in data_path or 'coloring' in data_path:
            from evaluators.graph_coloring_utils import batch_verify_coloring, compute_coverage_for_puzzle
            inputs_content = batch["inputs"][0, puzzle_emb_len:].cpu().numpy()
            preds_content = np.array([p[puzzle_emb_len:] for p in final_preds])
            _, _, total_solutions = compute_coverage_for_puzzle(inputs_content, preds_content, max_colors=3)
            preds_tensor = torch.tensor(preds_content)
            inputs_tensor = batch["inputs"][:, puzzle_emb_len:].expand(len(final_preds), -1)
            valid_mask = batch_verify_coloring(inputs_tensor, preds_tensor)
            valid_count = valid_mask.sum().item()
            validity_type = "Graph coloring constraint"
        else:
            label_np = target_labels.cpu().numpy()
            mask_np = target_mask.cpu().numpy()
            for pred in final_preds:
                if np.all(pred[mask_np] == label_np[mask_np]):
                    valid_count += 1
            validity_type = "Exact match"
            total_solutions = 1

        print(f"  Total samples: {len(final_preds)}")
        print(f"  Unique predictions: {len(unique_solutions)}")
        print(f"  Valid solutions ({validity_type}): {valid_count} ({100*valid_count/len(final_preds):.1f}%)")
        if total_solutions is not None:
            print(f"  Total possible solutions: {total_solutions}")

        # Process each space
        for space_name, use_decoded in spaces:
            print(f"\nProcessing {space_name}...")
            features, h_states, traj_info, target_feat = extract_features(
                trajectories, target_labels, target_mask, lm_head, device, use_decoded, eval_metadata.vocab_size)
            reducer, embeddings, target_emb = fit_reducer(features, target_feat)
            output_path = os.path.join(output_dir, f'puzzle_{puzzle_idx}_{space_name}.png')
            plot_trajectories(embeddings, traj_info, h_states, target_labels, target_mask, lm_head, device,
                            output_path, args.k_neighbors, reducer, args.show_gradients, use_decoded, target_emb, args.log_scale, args.theme, args.cmap, args.metric)

        # Reload for next puzzle
        if puzzle_idx != puzzle_indices[-1]:
            eval_loader, _ = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1,
                                              global_batch_size=1, rank=0, world_size=1)

    print("\nDone!")


if __name__ == '__main__':
    main()
