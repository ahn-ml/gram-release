from typing import Dict, Sequence, Optional
import os
import json

import torch
import numpy as np
from numba import njit
import torch.distributed as dist

from dataset.build_arc_dataset import inverse_aug, grid_hash, arc_grid_to_np
from dataset.common import PuzzleDatasetMetadata

@njit
def _crop(grid: np.ndarray):
    """Find maximum-sized rectangle without any EOS token inside. """
    grid = grid.reshape(30, 30)

    max_area = 0
    max_size = (0, 0)
    nr, nc = grid.shape
    
    num_c = nc
    for num_r in range(1, nr + 1):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if (x < 2) | (x > 11):
                num_c = c - 1
                break
        
        area = num_r * num_c
        if area > max_area:
            max_area = area
            max_size = (num_r, num_c)

    return (grid[:max_size[0], :max_size[1]] - 2).astype(np.uint8)


class ARC:
    """Pass@k and majority voting evaluator for ARC puzzles.
    
    Aggregates all predictions (including augmented) per original puzzle using inverse_aug.
    Works for both original-only and augmented data.
    
    Metrics:
    1. Majority voting: Is the most frequent prediction correct?
    2. Pass@k: Is the correct answer in ANY of top-k voted samples?
    """
    required_outputs = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds", "labels"}
    
    def __init__(self, data_path: str, 
        eval_metadata: PuzzleDatasetMetadata, 
        submission_K: int = 2, 
        pass_Ks: Sequence[int] = (1, 2, 5, 10, 20), 
        aggregated_voting: bool = True,
        ):
        super().__init__()
        self.pass_Ks = pass_Ks
        self.submission_K = submission_K
        self.aggregated_voting = aggregated_voting
        self.blank_identifier_id = eval_metadata.blank_identifier_id

        with open(os.path.join(data_path, "identifiers.json"), "r") as f:
            self.identifier_map = json.load(f)
        
        # Aggregation state: {orig_name: {input_hash: {"label_hash": ..., "samples": [...]}}}
        self._global_preds = {}
        self._global_hmap = {}  # {pred_hash: pred_grid}
        
    def begin_eval(self):
        self._global_preds = {}
        self._global_hmap = {}

    def update_batch_samples(self, batch: Dict[str, torch.Tensor], batch_samples: list, num_samples: int, best_of_n_metrics: Optional[Dict] = None):
        """Accumulate K samples for a batch."""
        inputs = batch["inputs"].cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        puzzle_identifiers = batch["puzzle_identifiers"].cpu().numpy()
        
        # Remove padding
        mask = puzzle_identifiers != self.blank_identifier_id
        if not np.any(mask):
            return
            
        inputs = inputs[mask]
        labels = labels[mask]
        puzzle_identifiers = puzzle_identifiers[mask]
        
        # Extract value predictions if available (K, B) -> filter by mask
        has_value = best_of_n_metrics is not None and "value_preds" in best_of_n_metrics
        if has_value:
            # value_preds: (K, B) -> (K, num_valid)
            all_value_preds = best_of_n_metrics["value_preds"][:, torch.from_numpy(mask).to(best_of_n_metrics["value_preds"].device)].cpu().numpy()
        
        # Collect all K samples
        all_predictions = []
        all_q_values = []
        for sample_preds in batch_samples:
            preds = sample_preds["preds"].cpu().numpy()[mask]
            q_vals = sample_preds["q_halt_logits"].to(torch.float64).sigmoid().cpu().numpy()[mask]
            all_predictions.append(preds)
            all_q_values.append(q_vals)
        
        # Process each puzzle
        for puzzle_idx in range(len(labels)):
            identifier = int(puzzle_identifiers[puzzle_idx])
            name = self.identifier_map[identifier]
            orig_name, _inverse_fn = inverse_aug(name)
            
            input_grid = _inverse_fn(_crop(inputs[puzzle_idx]))
            label_grid = _inverse_fn(_crop(labels[puzzle_idx]))
            input_hash = grid_hash(input_grid)
            label_hash = grid_hash(label_grid)
            
            # Accumulate samples
            self._global_preds.setdefault(orig_name, {})
            self._global_preds[orig_name].setdefault(input_hash, {"label_hash": label_hash, "samples": []})
            
            for sample_idx in range(num_samples):
                pred_grid = _inverse_fn(_crop(all_predictions[sample_idx][puzzle_idx]))
                pred_hash = grid_hash(pred_grid)
                q = float(all_q_values[sample_idx][puzzle_idx])
                # Include value prediction if available
                v = float(all_value_preds[sample_idx][puzzle_idx]) if has_value else None
                self._global_preds[orig_name][input_hash]["samples"].append((pred_hash, q, v))
                if pred_hash not in self._global_hmap:
                    self._global_hmap[pred_hash] = pred_grid

    def result(
        self,
        save_path: Optional[str],
        rank: int,
        world_size: int,
        group: Optional[torch.distributed.ProcessGroup] = None
    ) -> Optional[Dict[str, float]]:
        """Gather predictions from all ranks and compute metrics."""
        
        # Gather to rank 0
        global_hmap_preds = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object((self._global_hmap, self._global_preds), global_hmap_preds, dst=0, group=group)
        
        if rank != 0:
            return None
        
        # Merge from all ranks
        merged_preds = {}
        for hmap, preds in global_hmap_preds:  # type: ignore
            for orig_name, input_dict in preds.items():
                merged_preds.setdefault(orig_name, {})
                for input_hash, data in input_dict.items():
                    if input_hash not in merged_preds[orig_name]:
                        merged_preds[orig_name][input_hash] = {"label_hash": data["label_hash"], "samples": []}
                    merged_preds[orig_name][input_hash]["samples"].extend(data["samples"])
        
        # Compute metrics
        total_puzzles = 0
        majority_exact_correct = 0.0
        majority_ratio_sum = 0.0
        exact_correct = [0.0 for _ in range(len(self.pass_Ks))]
        value_exact_correct = [0.0 for _ in range(len(self.pass_Ks))]  # Value-based pass@k
        value_exact_correct_available = True  # Track if value predictions are available
        total_unique_samples = 0
        total_samples_count = 0
        total_correct_samples = 0
        
        for orig_name, input_dict in merged_preds.items():
            for input_hash, data in input_dict.items():
                label_hash = data["label_hash"]
                samples = data["samples"]
                
                if not samples:
                    print(f"Puzzle {orig_name} has no predictions.")
                    continue
                
                # Check if value predictions are available (samples now have 3 elements: pred_hash, q, v)
                has_value = len(samples[0]) == 3 and samples[0][2] is not None
                if not has_value:
                    value_exact_correct_available = False
                
                # Build frequency map: {pred_hash: [count, sum_q, sum_v]}
                p_map = {}
                for sample in samples:
                    pred_hash = sample[0]
                    q = sample[1]
                    v = sample[2] if has_value else 0.0
                    p_map.setdefault(pred_hash, [0, 0.0, 0.0])
                    p_map[pred_hash][0] += 1
                    p_map[pred_hash][1] += q
                    p_map[pred_hash][2] += v
                
                # Average q-value and v-value
                for stats in p_map.values():
                    stats[1] /= stats[0]  # avg_q
                    stats[2] /= stats[0]  # avg_v
                
                # Sort by [count, avg_q] for majority voting
                sorted_preds = sorted(p_map.items(), key=lambda kv: (kv[1][0], kv[1][1]), reverse=True)
                
                # Sort by [avg_v, count] for value-based ranking
                if has_value:
                    sorted_preds_by_value = sorted(p_map.items(), key=lambda kv: (kv[1][2], kv[1][0]), reverse=True)
                
                # Majority voting
                majority_pred_hash, (majority_count, _, _) = sorted_preds[0]
                if majority_pred_hash == label_hash:
                    majority_exact_correct += 1
                
                total_num_samples = sum(s[0] for _, s in sorted_preds)
                majority_ratio_sum += majority_count / total_num_samples
                
                # Pass@k (majority voting based)
                for i, k in enumerate(self.pass_Ks):
                    if any(ph == label_hash for ph, _ in sorted_preds[:k]):
                        exact_correct[i] += 1
                
                # Pass@k (value based)
                if has_value:
                    for i, k in enumerate(self.pass_Ks):
                        if any(ph == label_hash for ph, _ in sorted_preds_by_value[:k]):
                            value_exact_correct[i] += 1
                
                # Stats
                total_unique_samples += len(p_map)
                total_samples_count += total_num_samples
                total_correct_samples += sum(s[0] for ph, s in sorted_preds if ph == label_hash)
                total_puzzles += 1
        
        if total_puzzles == 0:
            print("WARNING: No puzzles found!")
            return {}
        
        # Build results
        results = {
            "ARC/majority_exact": majority_exact_correct / total_puzzles,
            "ARC/majority_avg_ratio": majority_ratio_sum / total_puzzles,
            "ARC/total_puzzles": total_puzzles,
            "ARC/avg_correctness_ratio": total_correct_samples / total_samples_count if total_samples_count > 0 else 0.0,
            "ARC/avg_unique_samples": total_unique_samples / total_puzzles,
            "ARC/total_samples": total_samples_count,
        }
        
        # Majority voting based pass@k
        for i, k in enumerate(self.pass_Ks):
            results[f"ARC/exact_pass@{k}"] = exact_correct[i] / total_puzzles
        
        # Value-based pass@k (only if value predictions were available)
        if value_exact_correct_available:
            for i, k in enumerate(self.pass_Ks):
                results[f"ARC/value_pass@{k}"] = value_exact_correct[i] / total_puzzles
        
        if save_path is not None:
            with open(os.path.join(save_path, "arc_results.json"), "w") as f:
                json.dump(results, f, indent=2)
        
        return results