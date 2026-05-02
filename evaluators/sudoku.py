from typing import Dict, Sequence, Optional
import os
import json

import torch
import numpy as np
import torch.distributed as dist

from dataset.common import PuzzleDatasetMetadata


class Sudoku:
    """Pass@k and majority voting evaluator for Sudoku puzzles.

    Collects multiple predictions per puzzle and computes two types of metrics:

    1. Majority voting: Is the most frequent prediction correct?
       - Ranks predictions by frequency (count), with q_halt as tiebreaker
       - Reports: Sudoku/majority_exact, Sudoku/majority_token

    2. Pass@k: Is the correct answer in ANY of k samples (unranked)?
       - No ranking or deduplication - checks raw samples in order
       - Reports: Sudoku/exact_pass@k, Sudoku/pass@k for various k values

    Note: These are different! Majority voting ranks, pass@k doesn't.
    """
    required_outputs = {"inputs", "labels", "preds", "q_halt_logits"}

    def __init__(
        self,
        data_path: str,
        eval_metadata: PuzzleDatasetMetadata,
        pass_Ks: Sequence[int] = (1,),
        aggregated_voting: bool = True
    ):
        super().__init__()
        self.pass_Ks = pass_Ks
        self.aggregated_voting = aggregated_voting

        # Accumulators for counts (computed on-the-fly)
        self._exact_correct = [0.0 for _ in range(len(self.pass_Ks))]
        self._token_correct = [0.0 for _ in range(len(self.pass_Ks))]
        self._majority_exact_correct = 0.0
        self._majority_token_correct = 0.0
        self._majority_ratio_sum = 0.0  # Sum of (majority_count / num_samples)
        self._total_puzzles = 0

        # NEW: Track correctness distribution in K samples
        # correctness_counts[i] = how many puzzles have exactly i correct answers in K samples
        self._correctness_counts = [0 for _ in range(max(self.pass_Ks) + 1)]

        # Track uniqueness: sum of unique predictions per puzzle
        self._total_unique_samples = 0

    def set_train_vocab_size(self, vocab_size: int): pass
    def set_step(self, step: int): pass
    def scale_batch(self, batch): return batch, None
    def precompute_real_features(self, eval_loader, device): pass

    def begin_eval(self):
        # Reset counters
        self._exact_correct = [0.0 for _ in range(len(self.pass_Ks))]
        self._token_correct = [0.0 for _ in range(len(self.pass_Ks))]
        self._majority_exact_correct = 0.0
        self._majority_token_correct = 0.0
        self._majority_ratio_sum = 0.0
        self._total_puzzles = 0
        self._correctness_counts = [0 for _ in range(max(self.pass_Ks) + 1)]
        self._total_unique_samples = 0
        # Best-of-N metrics (value network)
        self._best_of_n_exact = 0.0
        self._best_of_n_token = 0.0

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor], original_labels=None):
        from evaluators.sudoku_utils import batch_verify_sudoku

        labels = batch["labels"]
        predictions = preds["preds"]
        mask = labels != -100
        loss_counts = mask.sum(-1)
        self.update_batch_samples(batch, [{
            "preds": predictions,
            "q_halt_logits": preds["q_halt_logits"],
            "exact_correct": batch_verify_sudoku(batch["inputs"], predictions),
            "token_accuracy": torch.where(
                loss_counts > 0,
                ((predictions == labels) & mask).sum(-1).float() / loss_counts.float(),
                torch.zeros_like(loss_counts, dtype=torch.float32),
            ),
            "valid": loss_counts > 0,
        }], 1)

    def update_batch_samples(self, batch: Dict[str, torch.Tensor], batch_samples: list, num_samples: int, best_of_n_metrics: Optional[Dict] = None):
        """Process K samples for a batch and compute metrics on-the-fly.

        Args:
            batch: Original batch dict with inputs/labels
            batch_samples: List of K dicts with precomputed correctness metrics
            num_samples: K (number of samples)
            best_of_n_metrics: Optional dict with best-of-N selection results from value network
        """
        if num_samples >= len(self._correctness_counts):
            self._correctness_counts.extend([0] * (num_samples + 1 - len(self._correctness_counts)))

        # Collect correctness metrics (already computed in eval_pass_at_k.py)
        all_exact_correct = []
        all_token_accuracy = []
        all_valid = []
        all_q_values = []
        all_preds_hash = []

        for sample in batch_samples:
            all_exact_correct.append(sample["exact_correct"].cpu().numpy())  # (B,)
            all_token_accuracy.append(sample["token_accuracy"].cpu().numpy())  # (B,)
            all_valid.append(sample["valid"].cpu().numpy())  # (B,)
            all_q_values.append(sample["q_halt_logits"].to(torch.float64).sigmoid().cpu().numpy())  # (B,)
            # Hash predictions for deduplication
            preds_np = sample["preds"].cpu().numpy()  # (B, N)
            preds_hash = np.array([hash(preds_np[i].tobytes()) for i in range(len(preds_np))])
            all_preds_hash.append(preds_hash)  # (B,)

        # Process each puzzle in this batch
        batch_size = len(all_valid[0])
        for puzzle_idx in range(batch_size):
            # Skip puzzles with no valid tokens (following losses.py line 81: loss_counts > 0)
            if not all_valid[0][puzzle_idx]:
                continue

            # Collect K samples for this puzzle
            puzzle_samples = [
                (all_exact_correct[sample_idx][puzzle_idx],  # exact correctness
                 all_token_accuracy[sample_idx][puzzle_idx],  # token accuracy
                 all_preds_hash[sample_idx][puzzle_idx],  # prediction hash
                 float(all_q_values[sample_idx][puzzle_idx]))  # q_halt value
                for sample_idx in range(num_samples)
            ]

            # === Majority Voting (ranked by frequency) ===
            p_map = {}
            pred_examples = {}  # Store one example (exact_correct, token_acc) per hash
            for exact_correct, token_acc, pred_hash, q in puzzle_samples:
                p_map.setdefault(pred_hash, [0, 0])
                p_map[pred_hash][0] += 1  # COUNT (frequency)
                p_map[pred_hash][1] += q  # SUM of q_halt scores
                if pred_hash not in pred_examples:
                    pred_examples[pred_hash] = (exact_correct, token_acc)

            # Convert sum to average q-value
            for pred_hash, stats in p_map.items():
                stats[1] /= stats[0]  # avg_q = sum_q / count

            # Sort by [count, avg_q] - frequency first, q_halt as tiebreaker
            sorted_preds = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

            # Check if most frequent prediction is correct
            if sorted_preds:
                majority_pred_hash, (majority_count, _) = sorted_preds[0]
                majority_exact, majority_token_acc = pred_examples[majority_pred_hash]

                if majority_exact:
                    self._majority_exact_correct += 1
                self._majority_token_correct += majority_token_acc

                # Track majority ratio (consensus strength)
                majority_ratio = majority_count / num_samples
                self._majority_ratio_sum += majority_ratio

            # === Pass@k (unranked - check first k raw samples) ===
            for i, k in enumerate(self.pass_Ks):
                # Take first k samples (no ranking)
                k_samples = puzzle_samples[:k]

                # Exact pass@k: is ANY of the k samples exactly correct?
                found_exact = any(exact_correct for exact_correct, _, _, _ in k_samples)
                if found_exact:
                    self._exact_correct[i] += 1

                # Token-level pass@k: best token accuracy in k samples
                if k_samples:
                    best_token_acc = max(token_acc for _, token_acc, _, _ in k_samples)
                    self._token_correct[i] += best_token_acc

            # Count how many correct answers in ALL K samples
            num_correct_in_k = sum(1 for exact_correct, _, _, _ in puzzle_samples if exact_correct)
            self._correctness_counts[num_correct_in_k] += 1

            # Count unique predictions (using p_map which already deduplicated by hash)
            num_unique = len(p_map)
            self._total_unique_samples += num_unique

            # Track best-of-N metrics if provided
            if best_of_n_metrics is not None:
                self._best_of_n_exact += best_of_n_metrics["exact_correct"][puzzle_idx].item()
                self._best_of_n_token += best_of_n_metrics["token_accuracy"][puzzle_idx].item()

            # Increment puzzle count
            self._total_puzzles += 1

    def result(
        self,
        save_path: Optional[str],
        rank: int,
        world_size: int,
        group: Optional[torch.distributed.ProcessGroup] = None
    ) -> Optional[Dict[str, float]]:
        # ===== Aggregate counts across all ranks =====
        # Convert local counts to tensors for reduction
        local_counts = torch.tensor(
            [self._total_puzzles, self._majority_exact_correct, self._majority_token_correct,
             self._majority_ratio_sum, self._total_unique_samples,
             self._best_of_n_exact, self._best_of_n_token] +
            self._exact_correct + self._token_correct + self._correctness_counts,
            dtype=torch.float32
        )

        # Sum across all ranks
        if world_size > 1:
            global_counts = local_counts.clone()
            dist.all_reduce(global_counts, op=dist.ReduceOp.SUM, group=group)
        else:
            global_counts = local_counts

        # ===== Rank 0 computes final metrics =====
        if rank != 0:
            return None

        # Extract aggregated counts
        total_puzzles = int(global_counts[0].item())
        global_majority_exact = global_counts[1].item()
        global_majority_token = global_counts[2].item()
        global_majority_ratio_sum = global_counts[3].item()
        global_total_unique = global_counts[4].item()
        global_best_of_n_exact = global_counts[5].item()
        global_best_of_n_token = global_counts[6].item()
        offset = 7
        global_exact_correct = [global_counts[offset + i].item() for i in range(len(self.pass_Ks))]
        global_token_correct = [global_counts[offset + len(self.pass_Ks) + i].item() for i in range(len(self.pass_Ks))]
        offset = 7 + len(self.pass_Ks) * 2
        global_correctness_counts = [int(global_counts[offset + i].item()) for i in range(len(self._correctness_counts))]

        # Compute final metrics (percentages from aggregated counts)
        results = {}

        # Majority voting (most frequent prediction)
        majority_exact_acc = global_majority_exact / total_puzzles if total_puzzles > 0 else 0.0
        majority_token_acc = global_majority_token / total_puzzles if total_puzzles > 0 else 0.0
        majority_ratio = global_majority_ratio_sum / total_puzzles if total_puzzles > 0 else 0.0
        results["Sudoku/majority_exact"] = majority_exact_acc
        results["Sudoku/majority_token"] = majority_token_acc
        results["Sudoku/majority_ratio"] = majority_ratio

        # Pass@k (correct answer in top-k ranked by frequency)
        for i, k in enumerate(self.pass_Ks):
            exact_acc = global_exact_correct[i] / total_puzzles if total_puzzles > 0 else 0.0
            token_acc = global_token_correct[i] / total_puzzles if total_puzzles > 0 else 0.0
            results[f"Sudoku/exact_pass@{k}"] = exact_acc
            results[f"Sudoku/pass@{k}"] = exact_acc
            results[f"Sudoku/token_best@{k}"] = token_acc

        # Best-of-N (value network selection)
        if global_best_of_n_exact > 0 or global_best_of_n_token > 0:
            best_of_n_exact_acc = global_best_of_n_exact / total_puzzles if total_puzzles > 0 else 0.0
            best_of_n_token_acc = global_best_of_n_token / total_puzzles if total_puzzles > 0 else 0.0
            results["Sudoku/best_of_n_exact"] = best_of_n_exact_acc
            results["Sudoku/best_of_n_token"] = best_of_n_token_acc

        # Add total count
        results["Sudoku/total_puzzles"] = total_puzzles

        # Compute average correctness ratio across all samples
        total_correct_samples = sum(i * count for i, count in enumerate(global_correctness_counts))
        total_samples = total_puzzles * max(self.pass_Ks)  # Assuming we use max K
        avg_correctness_ratio = total_correct_samples / total_samples if total_samples > 0 else 0.0
        results["Sudoku/avg_correctness_ratio"] = avg_correctness_ratio

        # Compute average uniqueness (avg unique samples per puzzle)
        avg_unique_samples = global_total_unique / total_puzzles if total_puzzles > 0 else 0.0
        results["Sudoku/avg_unique_samples"] = avg_unique_samples

        # Save results if path provided
        if save_path is not None:
            with open(os.path.join(save_path, "sudoku_results.json"), "w") as f:
                json.dump(results, f, indent=2)

        return results
