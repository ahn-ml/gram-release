from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn
import math

IGNORE_LABEL_ID = -100

def s(x, epsilon=1e-30):
    return torch.where(
        x<0,
        1/(1-x+ epsilon),
        x + 1
    )


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x/torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index: int = -100, valid_mask=None):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)

    if valid_mask is None:
        valid_mask = (labels != ignore_index)
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)

    return -torch.where(valid_mask, prediction_logprobs, 0)


def softmax_cross_entropy(logits, labels, ignore_index: int = -100):
    # Cast logits to f32
    # Flatten logits
    return F.cross_entropy(logits.to(torch.float32).view(-1, logits.shape[-1]), labels.to(torch.long).view(-1), ignore_index=ignore_index, reduction="none").view(labels.shape)


def softmax_focal_loss(logits, labels, ignore_index: int = -100, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, valid_mask=None):
    """
    Focal Loss for multi-class classification (softmax version).
    Based on RetinaNet paper: https://arxiv.org/abs/1708.02002 (Lin et al., 2017)
    Adapted from detectron2's sigmoid_focal_loss to softmax.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Args:
        logits: (B, SeqLen, VocabSize) unnormalized predictions
        labels: (B, SeqLen) ground truth class indices
        ignore_index: label value to ignore (default: -100)
        gamma: focusing parameter (default: 2.0). Higher = more focus on hard examples
        alpha: (VocabSize,) optional class weights. If None, no class weighting
        valid_mask: optional boolean mask for valid positions

    Returns:
        loss: (B, SeqLen) per-token focal loss
    """
    logits = logits.float()

    # Compute probabilities
    p = F.softmax(logits, dim=-1)  # (B, SeqLen, VocabSize)

    # Create mask
    if valid_mask is None:
        valid_mask = (labels != ignore_index)

    # Get probability of correct class p_t
    labels_clamped = torch.where(valid_mask, labels.long(), 0)
    p_t = torch.gather(p, dim=-1, index=labels_clamped.unsqueeze(-1)).squeeze(-1)  # (B, SeqLen)

    # Focal weight: (1 - p_t)^gamma
    focal_weight = (1 - p_t) ** gamma

    # Cross-entropy: -log(p_t)
    ce_loss = F.cross_entropy(
        logits.view(-1, logits.shape[-1]),
        labels_clamped.view(-1),
        reduction='none'
    ).view(labels.shape)

    # Focal loss
    loss = focal_weight * ce_loss

    # Optional class weighting (alpha_t)
    if alpha is not None:
        alpha_t = torch.gather(alpha.to(logits.device), dim=0, index=labels_clamped.view(-1)).view(labels.shape)
        loss = alpha_t * loss

    # Mask invalid positions
    loss = torch.where(valid_mask, loss, torch.zeros_like(loss))

    return loss


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, kl_coef: float = 0.0, kl_balance_alpha: Optional[float] = None, free_bits: Optional[float] = None, value_coef: float = 1.0, focal_gamma: float = 2.0, focal_alpha: Optional[torch.Tensor] = None, reinforce_coef: float = 0.0, length_penalty_coef: float = 0.0, data_path: Optional[str] = None):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.loss_type = loss_type
        self.kl_coef = kl_coef
        self.kl_balance_alpha = kl_balance_alpha  # If not None, use KL balancing (e.g., 0.8)
        self.free_bits = free_bits  # If not None, apply free bits to KL (e.g., 0.1)
        self.value_coef = value_coef  # Coefficient for value loss
        self.focal_gamma = focal_gamma  # Focal loss gamma parameter
        self.focal_alpha = focal_alpha  # Focal loss class weights
        self.reinforce_coef = reinforce_coef  # Coefficient for REINFORCE loss (global)
        self.length_penalty_coef = length_penalty_coef  # Length penalty for path efficiency

        # Detect task type from data_path for task-specific verification
        self.is_nqueen = False
        self.is_graph_coloring = False
        if data_path is not None:
            data_path_lower = data_path.lower()
            self.is_nqueen = 'nqueen' in data_path_lower
            self.is_graph_coloring = 'graph-coloring' in data_path_lower or 'graph_coloring' in data_path_lower

        # REINFORCE baseline for variance reduction
        if self.reinforce_coef > 0:
            self.register_buffer('reward_baseline', torch.tensor(0.5))
            self.baseline_momentum = 0.99

    def initial_carry(self, *args, **kwargs):
        return self.model.initial_carry(*args, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        # Model args
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        # Model logits
        # B x SeqLen x D
        new_carry, outputs = self.model(**model_kwargs)
        labels = new_carry.current_data["labels"]

        with torch.no_grad():
            # Preds - sample if temperature > 0 during inference, else greedy
            temperature = getattr(self.model.config, 'temperature', 0.0)
            if not self.model.training and temperature > 0.0:
                probs = torch.softmax(outputs["logits"], dim=-1)
                outputs["preds"] = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(probs.shape[:-1])
            else:
                outputs["preds"] = torch.argmax(outputs["logits"], dim=-1)

            # Correctness
            mask = (labels != IGNORE_LABEL_ID)
            loss_counts = mask.sum(-1)
            loss_divisor = loss_counts.clamp_min(1).unsqueeze(-1)  # Avoid NaNs in division

            # Get predictions
            preds = outputs["preds"]
            
            # Compute sequence-level correctness with task-specific verification
            # For N-Queens and Graph Coloring: verify constraints instead of label matching
            if self.is_nqueen:
                from evaluators.nqueens_utils import batch_verify_nqueens
                inputs = new_carry.current_data["inputs"]
                seq_is_correct = batch_verify_nqueens(inputs, preds)
            elif self.is_graph_coloring:
                from evaluators.graph_coloring_utils import batch_verify_coloring
                inputs = new_carry.current_data["inputs"]
                seq_is_correct = batch_verify_coloring(inputs, preds)
            else:
                # Fallback to label matching for other tasks
                is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
                seq_is_correct = is_correct.sum(-1) == loss_counts
            
            # Token-level accuracy (always computed for metrics)
            is_correct = mask & (torch.argmax(outputs["logits"], dim=-1) == labels)
            
            # Metrics (halted)
            valid_metrics = new_carry.halted & (loss_counts > 0)
            metrics = {
                "count": valid_metrics.sum(),
                
                "accuracy":       torch.where(valid_metrics, (is_correct.to(torch.float32) / loss_divisor).sum(-1), 0).sum(),
                "exact_accuracy": (valid_metrics & seq_is_correct).sum(),

                "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
                "steps":          torch.where(valid_metrics, new_carry.steps, 0).sum(),
            }

        # Losses

        # Pass focal loss parameters if using focal loss
        if self.loss_type == "softmax_focal_loss":
            lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, gamma=self.focal_gamma, alpha=self.focal_alpha, valid_mask=mask) / loss_divisor).sum()
        else:
            lm_loss = (self.loss_fn(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID, valid_mask=mask) / loss_divisor).sum()
        q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")
        metrics.update({
            "lm_loss": lm_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })
        # Q continue (bootstrapping target loss); Alexia: This fits Q-learning, but seems totally unecessary
        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(outputs["q_continue_logits"], outputs["target_q_continue"], reduction="sum")
            metrics["q_continue_loss"] = q_continue_loss.detach()

        # KL divergence (if variational)
        kl_loss = 0
        if "post_mu" in outputs and "prior_mu" in outputs:
            # Skip KL computation if fix_std is enabled (logvar ≈ -1e10)
            # This avoids division by exp(-1e10) ≈ 0 which causes NaN
            if outputs["prior_logvar"].min() < -1e5:
                kl_loss = torch.tensor(0.0, device=outputs["prior_mu"].device)
                metrics["kl_loss"] = kl_loss.detach()
            else:
                def compute_kl(post_mu, post_logvar, prior_mu, prior_logvar):
                    return torch.sum(0.5 * (
                        prior_logvar - post_logvar - 1
                        + torch.exp(post_logvar - prior_logvar)
                        + ((post_mu - prior_mu) ** 2) / torch.exp(prior_logvar)
                    ))

                if self.kl_balance_alpha is not None:
                    # KL balancing: alpha * KL(sg(post) || prior) + (1-alpha) * KL(post || sg(prior))
                    alpha = self.kl_balance_alpha
                    kl_loss = (
                        alpha * compute_kl(outputs["post_mu"].detach(), outputs["post_logvar"].detach(),
                                          outputs["prior_mu"], outputs["prior_logvar"])
                        + (1 - alpha) * compute_kl(outputs["post_mu"], outputs["post_logvar"],
                                                   outputs["prior_mu"].detach(), outputs["prior_logvar"].detach())
                    )
                else:
                    # Standard KL
                    kl_loss = compute_kl(outputs["post_mu"], outputs["post_logvar"],
                                        outputs["prior_mu"], outputs["prior_logvar"])
                # Apply free bits if specified (per sample)
                if self.free_bits is not None:
                    kl_loss = torch.clamp(kl_loss, min=self.free_bits)
                metrics["kl_loss"] = kl_loss.detach()

        # Value loss (if value head is enabled)
        value_loss = 0
        if "value_pred" in outputs:
            # Compute current accuracy (what the value should predict)
            current_accuracy = (is_correct.sum(-1).float() / loss_counts.clamp_min(1).float())
            # MSE loss between predicted value and current accuracy
            value_loss = F.mse_loss(outputs["value_pred"], current_accuracy, reduction="sum")
            metrics["value_loss"] = value_loss.detach()
            metrics["value_mae"] = (outputs["value_pred"] - current_accuracy).abs().mean().detach()

        # REINFORCE loss (global sequence-level correctness)
        reinforce_loss = 0
        if self.reinforce_coef > 0 and self.model.training:
            # Import BFS verification
            from evaluators.maze_utils import batch_verify_maze

            # Use greedy predictions
            logits = outputs["logits"]  # (B, N, V)
            preds = outputs["preds"]  # (B, N) - argmax predictions

            # Get log probabilities of greedy predictions
            log_probs = F.log_softmax(logits, dim=-1)  # (B, N, V)
            preds_log_probs = torch.gather(
                log_probs,
                dim=-1,
                index=preds.unsqueeze(-1)
            ).squeeze(-1)  # (B, N)

            # Get reward from BFS verification (entire path correct?)
            inputs = new_carry.current_data["inputs"]
            validity = batch_verify_maze(inputs, preds).float()  # (B,) - 1.0 if valid, 0.0 otherwise

            # Count path length (number of path cells: tokens 3, 4, 5 = start/goal/path)
            # For mazes: vocab is [empty=0, wall=1, obstacle=2, start=3, goal=4, path=5]
            is_path_cell = (preds >= 3) & (preds <= 5)  # (B, N)
            path_lengths = is_path_cell.sum(dim=-1).float()  # (B,)

            # Compute reward: validity - length_penalty
            rewards = validity - self.length_penalty_coef * path_lengths  # (B,)

            # Update baseline (EMA)
            with torch.no_grad():
                self.reward_baseline.mul_(self.baseline_momentum).add_(rewards.mean() * (1 - self.baseline_momentum))

            # REINFORCE: -log_prob * (reward - baseline)
            advantages = rewards - self.reward_baseline  # (B,)
            reinforce_loss = -(preds_log_probs * mask.float()).sum(-1) * advantages  # (B,)
            reinforce_loss = reinforce_loss.sum()  # Sum over batch

            metrics["reinforce_loss"] = reinforce_loss.detach()
            metrics["reward_mean"] = rewards.mean().detach()
            metrics["reward_validity"] = validity.mean().detach()
            metrics["reward_path_length"] = path_lengths.mean().detach()
            metrics["reward_baseline"] = self.reward_baseline.detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + self.kl_coef * kl_loss + self.value_coef * value_loss + self.reinforce_coef * reinforce_loss, metrics, detached_outputs, new_carry.halted.all()

