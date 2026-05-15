from typing import Any, Tuple, Dict, Sequence, Optional

import torch
import torch.nn.functional as F
from torch import nn

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


class ACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, loss_type: str, kl_coef: float = 0.0, kl_balance_alpha: Optional[float] = None, free_bits: Optional[float] = None, value_coef: float = 1.0, data_path: Optional[str] = None):
        super().__init__()
        self.model = model
        self.loss_fn = globals()[loss_type]
        self.kl_coef = kl_coef
        self.kl_balance_alpha = kl_balance_alpha  # If not None, use KL balancing (e.g., 0.8)
        self.free_bits = free_bits  # If not None, apply free bits to KL (e.g., 0.1)
        self.value_coef = value_coef  # Coefficient for value loss

        # Detect task type from data_path for task-specific verification
        self.is_nqueen = False
        self.is_graph_coloring = False
        self.is_sudoku = False
        if data_path is not None:
            data_path_lower = data_path.lower()
            self.is_nqueen = 'nqueen' in data_path_lower
            self.is_graph_coloring = 'graph-coloring' in data_path_lower or 'graph_coloring' in data_path_lower
            self.is_sudoku = 'sudoku' in data_path_lower

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
            elif self.is_sudoku:
                from evaluators.sudoku_utils import batch_verify_sudoku
                inputs = new_carry.current_data["inputs"]
                seq_is_correct = batch_verify_sudoku(inputs, preds)
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
            current_accuracy = seq_is_correct.float() if self.is_sudoku else (is_correct.sum(-1).float() / loss_counts.clamp_min(1).float())
            # MSE loss between predicted value and current accuracy
            value_loss = F.mse_loss(outputs["value_pred"], current_accuracy, reduction="sum")
            metrics["value_loss"] = value_loss.detach()
            metrics["value_mae"] = (outputs["value_pred"] - current_accuracy).abs().mean().detach()

        # Filter outputs for return
        detached_outputs = {k: outputs[k].detach() for k in return_keys if k in outputs}

        return new_carry, lm_loss + 0.5 * (q_halt_loss + q_continue_loss) + self.kl_coef * kl_loss + self.value_coef * value_loss, metrics, detached_outputs, new_carry.halted.all()

