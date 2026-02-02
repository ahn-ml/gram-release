from typing import Dict, Sequence, Optional
import os
import json

import torch
import numpy as np
import torch.distributed as dist

from dataset.common import PuzzleDatasetMetadata
from evaluators.graph_coloring_utils import batch_verify_coloring, batch_count_conflicts, compute_coverage_for_puzzle


class GraphColoring:
    """Pass@k and majority voting evaluator for Graph Coloring puzzles.

    Collects multiple predictions per puzzle and computes two types of metrics:

    1. Majority voting: Is the most frequent prediction correct?
       - Ranks predictions by frequency (count), with q_halt as tiebreaker
       - Reports: GraphColoring/majority_exact, GraphColoring/majority_token

    2. Pass@k: Is the correct answer in ANY of k samples (unranked)?
       - No ranking or deduplication - checks raw samples in order
       - Reports: GraphColoring/exact_pass@k, GraphColoring/pass@k for various k values

    Note: These are different! Majority voting ranks, pass@k doesn't.
    """
    required_outputs = {"inputs", "labels", "preds", "q_halt_logits"}

    def __init__(
        self,
        data_path: str,
        eval_metadata: PuzzleDatasetMetadata,
        pass_Ks: Sequence[int] = (1, 5, 10, 50, 100),
        aggregated_voting: bool = True,
        max_colors: int = 3,
        enable_logging: bool = False
    ):
        super().__init__()
        self.pass_Ks = pass_Ks
        self.aggregated_voting = aggregated_voting
        self.max_colors = max_colors  # For coverage computation
        self.enable_logging = enable_logging  # Enable per-puzzle logging

        # Accumulators for counts (computed on-the-fly)
        self._exact_correct = [0.0 for _ in range(len(self.pass_Ks))]
        self._token_correct = [0.0 for _ in range(len(self.pass_Ks))]
        self._constraint_correct = [0.0 for _ in range(len(self.pass_Ks))]
        self._majority_exact_correct = 0.0
        self._majority_token_correct = 0.0
        self._majority_constraint_correct = 0.0
        self._majority_ratio_sum = 0.0  # Sum of (majority_count / num_samples)
        self._majority_conflicts_sum = 0  # Sum of conflict edges in majority voted samples
        self._total_puzzles = 0

        # Track correctness distribution in K samples
        # correctness_counts[i] = how many puzzles have exactly i correct answers in K samples
        self._correctness_counts = [0 for _ in range(101)]  # Support up to K=100

        # Track uniqueness: sum of unique predictions per puzzle
        self._total_unique_samples = 0

        # Track coverage: fraction of all valid colorings found (WEIGHTED)
        self._total_solutions_found = 0  # Sum of num_found across all puzzles
        self._total_solutions_available = 0  # Sum of num_total across all puzzles
        self._solutions_found_by_k = [0 for _ in range(len(self.pass_Ks))]
        self._solutions_available_by_k = [0 for _ in range(len(self.pass_Ks))]

        # Per-puzzle logging for analysis
        self._puzzle_logs = []  # [{num_solutions, correct, coverage}, ...]

    def begin_eval(self):
        # Reset counters
        self._exact_correct = [0.0 for _ in range(len(self.pass_Ks))]
        self._token_correct = [0.0 for _ in range(len(self.pass_Ks))]
        self._constraint_correct = [0.0 for _ in range(len(self.pass_Ks))]
        self._majority_exact_correct = 0.0
        self._majority_token_correct = 0.0
        self._majority_constraint_correct = 0.0
        self._majority_ratio_sum = 0.0
        self._majority_conflicts_sum = 0
        self._total_puzzles = 0
        self._correctness_counts = [0 for _ in range(101)]
        self._total_unique_samples = 0
        # Best-of-N metrics (value network)
        self._best_of_n_exact = 0.0
        self._best_of_n_token = 0.0
        self._best_of_n_constraint = 0.0
        # Coverage metrics (weighted)
        self._total_solutions_found = 0
        self._total_solutions_available = 0
        self._solutions_found_by_k = [0 for _ in range(len(self.pass_Ks))]
        self._solutions_available_by_k = [0 for _ in range(len(self.pass_Ks))]
        # Per-puzzle logs
        self._puzzle_logs = []

    def update_batch_samples(self, batch: Dict[str, torch.Tensor], batch_samples: list, num_samples: int, best_of_n_metrics: Optional[Dict] = None):
        """Process K samples for a batch and compute metrics on-the-fly.

        Args:
            batch: Original batch dict with inputs/labels
            batch_samples: List of K dicts with precomputed correctness metrics
            num_samples: K (number of samples)
            best_of_n_metrics: Optional dict with best-of-N selection results from value network
        """
        # Collect correctness metrics (already computed in eval_pass_at_k.py)
        all_exact_correct = []
        all_token_accuracy = []
        all_valid = []
        all_q_values = []
        all_preds_hash = []
        all_constraint_correct = []

        for sample in batch_samples:
            all_exact_correct.append(sample["exact_correct"].cpu().numpy())  # (B,)
            all_token_accuracy.append(sample["token_accuracy"].cpu().numpy())  # (B,)
            all_valid.append(sample["valid"].cpu().numpy())  # (B,)
            all_q_values.append(sample["q_halt_logits"].to(torch.float64).sigmoid().cpu().numpy())  # (B,)
            # Hash predictions for deduplication
            preds_np = sample["preds"].cpu().numpy()  # (B, N)
            preds_hash = np.array([hash(preds_np[i].tobytes()) for i in range(len(preds_np))])
            all_preds_hash.append(preds_hash)  # (B,)
            # Verify coloring constraints
            constraint_correct = batch_verify_coloring(batch["inputs"], sample["preds"]).cpu().numpy()
            all_constraint_correct.append(constraint_correct)  # (B,)

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
                 all_constraint_correct[sample_idx][puzzle_idx],  # constraint satisfaction
                 all_preds_hash[sample_idx][puzzle_idx],  # prediction hash
                 float(all_q_values[sample_idx][puzzle_idx]))  # q_halt value
                for sample_idx in range(num_samples)
            ]

            # === Majority Voting (ranked by frequency) ===
            p_map = {}
            pred_examples = {}  # Store one example (exact_correct, token_acc, constraint, sample_idx) per hash
            for sample_idx, (exact_correct, token_acc, constraint_ok, pred_hash, q) in enumerate(puzzle_samples):
                p_map.setdefault(pred_hash, [0, 0])
                p_map[pred_hash][0] += 1  # COUNT (frequency)
                p_map[pred_hash][1] += q  # SUM of q_halt scores
                if pred_hash not in pred_examples:
                    pred_examples[pred_hash] = (exact_correct, token_acc, constraint_ok, sample_idx)

            # Convert sum to average q-value
            for pred_hash, stats in p_map.items():
                stats[1] /= stats[0]  # avg_q = sum_q / count

            # Sort by [count, avg_q] - frequency first, q_halt as tiebreaker
            sorted_preds = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

            # Check if most frequent prediction is correct
            majority_conflicts = 0  # Initialize for logging
            majority_constraint = False  # Initialize for logging
            if sorted_preds:
                majority_pred_hash, (majority_count, _) = sorted_preds[0]
                majority_exact, majority_token_acc, majority_constraint, majority_sample_idx = pred_examples[majority_pred_hash]

                if majority_exact:
                    self._majority_exact_correct += 1
                if majority_constraint:
                    self._majority_constraint_correct += 1
                self._majority_token_correct += majority_token_acc

                # Track majority ratio (consensus strength)
                majority_ratio = majority_count / num_samples
                self._majority_ratio_sum += majority_ratio

                # Count conflicts in majority voted prediction
                majority_pred_tensor = batch_samples[majority_sample_idx]["preds"][puzzle_idx:puzzle_idx+1]
                majority_conflicts = batch_count_conflicts(
                    batch["inputs"][puzzle_idx:puzzle_idx+1],
                    majority_pred_tensor
                )[0].item()
                self._majority_conflicts_sum += majority_conflicts

            # === Pass@k (unranked - check first k raw samples) ===
            for i, k in enumerate(self.pass_Ks):
                # Take first k samples (no ranking)
                k_samples = puzzle_samples[:k]

                # Exact pass@k: is ANY of the k samples exactly correct?
                found_exact = any(exact_correct for exact_correct, _, _, _, _ in k_samples)
                if found_exact:
                    self._exact_correct[i] += 1

                # Constraint pass@k: is ANY of the k samples valid?
                found_valid = any(constraint_ok for _, _, constraint_ok, _, _ in k_samples)
                if found_valid:
                    self._constraint_correct[i] += 1

                # Token-level pass@k: best token accuracy in k samples
                if k_samples:
                    best_token_acc = max(token_acc for _, token_acc, _, _, _ in k_samples)
                    self._token_correct[i] += best_token_acc

            # Count how many correct answers in ALL K samples
            num_correct_in_k = sum(1 for exact_correct, _, _, _, _ in puzzle_samples if exact_correct)
            self._correctness_counts[num_correct_in_k] += 1

            # Count unique predictions (using p_map which already deduplicated by hash)
            num_unique = len(p_map)
            self._total_unique_samples += num_unique

            # Track best-of-N metrics if provided
            if best_of_n_metrics is not None:
                self._best_of_n_exact += best_of_n_metrics["exact_correct"][puzzle_idx].item()
                self._best_of_n_token += best_of_n_metrics["token_accuracy"][puzzle_idx].item()
                # Check constraint for best-of-N selection
                best_idx = best_of_n_metrics["best_indices"][puzzle_idx].item()
                best_constraint = all_constraint_correct[best_idx][puzzle_idx]
                self._best_of_n_constraint += float(best_constraint)

            # Compute coverage: fraction of all valid colorings found (WEIGHTED)
            inp_np = batch["inputs"][puzzle_idx].cpu().numpy()

            # Coverage with all K samples
            all_preds_np = np.stack([batch_samples[k]["preds"][puzzle_idx].cpu().numpy() for k in range(num_samples)])
            coverage_all, num_found, num_total = compute_coverage_for_puzzle(inp_np, all_preds_np, self.max_colors)
            self._total_solutions_found += num_found
            self._total_solutions_available += num_total

            # Coverage@k: using first k samples
            for i, k in enumerate(self.pass_Ks):
                k_preds_np = np.stack([batch_samples[j]["preds"][puzzle_idx].cpu().numpy() for j in range(min(k, num_samples))])
                coverage_k, num_found_k, num_total_k = compute_coverage_for_puzzle(inp_np, k_preds_np, self.max_colors)
                self._solutions_found_by_k[i] += num_found_k
                self._solutions_available_by_k[i] += num_total_k

            # Log per-puzzle metrics for analysis (only if enabled)
            if self.enable_logging:
                self._puzzle_logs.append({
                    "num_valid_solutions": num_total,
                    "correct": bool(majority_constraint),
                    "coverage": coverage_all,
                    "num_found": num_found,
                    "majority_conflicts": majority_conflicts,
                    "uniqueness": num_unique
                })

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
             self._majority_constraint_correct, self._majority_ratio_sum, self._majority_conflicts_sum,
             self._total_unique_samples,
             self._best_of_n_exact, self._best_of_n_token, self._best_of_n_constraint,
             self._total_solutions_found, self._total_solutions_available] +
            self._exact_correct + self._token_correct + self._constraint_correct +
            self._solutions_found_by_k + self._solutions_available_by_k + self._correctness_counts,
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
        global_majority_constraint = global_counts[3].item()
        global_majority_ratio_sum = global_counts[4].item()
        global_majority_conflicts_sum = int(global_counts[5].item())
        global_total_unique = global_counts[6].item()
        global_best_of_n_exact = global_counts[7].item()
        global_best_of_n_token = global_counts[8].item()
        global_best_of_n_constraint = global_counts[9].item()
        global_solutions_found = int(global_counts[10].item())
        global_solutions_available = int(global_counts[11].item())
        offset = 12
        global_exact_correct = [global_counts[offset + i].item() for i in range(len(self.pass_Ks))]
        global_token_correct = [global_counts[offset + len(self.pass_Ks) + i].item() for i in range(len(self.pass_Ks))]
        global_constraint_correct = [global_counts[offset + 2 * len(self.pass_Ks) + i].item() for i in range(len(self.pass_Ks))]
        global_solutions_found_by_k = [int(global_counts[offset + 3 * len(self.pass_Ks) + i].item()) for i in range(len(self.pass_Ks))]
        global_solutions_available_by_k = [int(global_counts[offset + 4 * len(self.pass_Ks) + i].item()) for i in range(len(self.pass_Ks))]
        offset = 12 + len(self.pass_Ks) * 5
        global_correctness_counts = [int(global_counts[offset + i].item()) for i in range(101)]

        # Compute final metrics (percentages from aggregated counts)
        results = {}

        # Majority voting (most frequent prediction)
        majority_exact_acc = global_majority_exact / total_puzzles if total_puzzles > 0 else 0.0
        majority_token_acc = global_majority_token / total_puzzles if total_puzzles > 0 else 0.0
        majority_constraint_acc = global_majority_constraint / total_puzzles if total_puzzles > 0 else 0.0
        majority_ratio = global_majority_ratio_sum / total_puzzles if total_puzzles > 0 else 0.0
        results["GraphColoring/majority_exact"] = majority_exact_acc
        results["GraphColoring/majority_token"] = majority_token_acc
        results["GraphColoring/majority_constraint"] = majority_constraint_acc
        results["GraphColoring/majority_ratio"] = majority_ratio
        results["GraphColoring/majority_conflicts"] = global_majority_conflicts_sum  # Total conflicts (sum, not average)

        # Pass@k (correct answer in top-k ranked by frequency)
        for i, k in enumerate(self.pass_Ks):
            exact_acc = global_exact_correct[i] / total_puzzles if total_puzzles > 0 else 0.0
            token_acc = global_token_correct[i] / total_puzzles if total_puzzles > 0 else 0.0
            constraint_acc = global_constraint_correct[i] / total_puzzles if total_puzzles > 0 else 0.0
            results[f"GraphColoring/exact_pass@{k}"] = exact_acc
            results[f"GraphColoring/pass@{k}"] = token_acc
            results[f"GraphColoring/constraint_pass@{k}"] = constraint_acc

        # Best-of-N (value network selection)
        if global_best_of_n_exact > 0 or global_best_of_n_token > 0:
            best_of_n_exact_acc = global_best_of_n_exact / total_puzzles if total_puzzles > 0 else 0.0
            best_of_n_token_acc = global_best_of_n_token / total_puzzles if total_puzzles > 0 else 0.0
            best_of_n_constraint_acc = global_best_of_n_constraint / total_puzzles if total_puzzles > 0 else 0.0
            results["GraphColoring/best_of_n_exact"] = best_of_n_exact_acc
            results["GraphColoring/best_of_n_token"] = best_of_n_token_acc
            results["GraphColoring/best_of_n_constraint"] = best_of_n_constraint_acc

        # Add total count
        results["GraphColoring/total_puzzles"] = total_puzzles

        # Compute average correctness ratio across all samples
        total_correct_samples = sum(i * count for i, count in enumerate(global_correctness_counts))
        total_samples = total_puzzles * max(self.pass_Ks)  # Assuming we use max K
        avg_correctness_ratio = total_correct_samples / total_samples if total_samples > 0 else 0.0
        results["GraphColoring/avg_correctness_ratio"] = avg_correctness_ratio

        # Compute average uniqueness (avg unique samples per puzzle)
        avg_unique_samples = global_total_unique / total_puzzles if total_puzzles > 0 else 0.0
        results["GraphColoring/avg_unique_samples"] = avg_unique_samples

        # Compute coverage metrics (WEIGHTED: fraction of all valid colorings found)
        weighted_coverage = global_solutions_found / global_solutions_available if global_solutions_available > 0 else 0.0
        results["GraphColoring/coverage"] = weighted_coverage

        # Coverage@k (weighted)
        for i, k in enumerate(self.pass_Ks):
            coverage_k = global_solutions_found_by_k[i] / global_solutions_available_by_k[i] if global_solutions_available_by_k[i] > 0 else 0.0
            results[f"GraphColoring/coverage@{k}"] = coverage_k

        # Gather per-puzzle logs from all ranks for analysis (only if logging enabled)
        if self.enable_logging:
            if world_size > 1:
                # Gather puzzle logs from all ranks
                gathered_logs = [None] * world_size
                dist.all_gather_object(gathered_logs, self._puzzle_logs, group=group)
                if rank == 0:
                    # Flatten list of lists
                    all_puzzle_logs = []
                    for rank_logs in gathered_logs:
                        all_puzzle_logs.extend(rank_logs)
                else:
                    all_puzzle_logs = []
            else:
                all_puzzle_logs = self._puzzle_logs if rank == 0 else []
        else:
            all_puzzle_logs = []

        # Save results if path provided (rank 0 only)
        if save_path is not None and rank == 0:
            # Save summary metrics
            with open(os.path.join(save_path, "graph_coloring_results.json"), "w") as f:
                json.dump(results, f, indent=2)

            # Save per-puzzle logs for analysis (only if logging enabled)
            if self.enable_logging and len(all_puzzle_logs) > 0:
                with open(os.path.join(save_path, "graph_coloring_puzzle_logs.json"), "w") as f:
                    json.dump({
                        "puzzle_metrics": all_puzzle_logs,
                        "total_puzzles": len(all_puzzle_logs)
                    }, f, indent=2)
                print(f"Saved {len(all_puzzle_logs)} puzzle logs to {save_path}/graph_coloring_puzzle_logs.json")

        return results
