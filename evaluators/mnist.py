"""MNIST evaluator with FID/IS metrics and sample visualization.

Supports train/eval vocabulary mismatch (e.g., train with 16 levels, eval with 256 levels).
Token encoding: PAD=0, pixel values -> tokens 1 to vocab_size-1
"""
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from dataset.common import PuzzleDatasetMetadata

try:
    import wandb
except ImportError:
    wandb = None

try:
    from torchmetrics.image.inception import InceptionScore
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    FrechetInceptionDistance = None
    InceptionScore = None


def tokens_to_pixels(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Convert tokens to pixel values [0, 255]."""
    num_levels = vocab_size - 1
    if num_levels == 256:
        return (tokens.float() - 1).clamp(0, 255)
    return ((tokens.float() - 1) / max(num_levels - 1, 1) * 255).clamp(0, 255)


class MNIST:
    """MNIST image evaluator with FID/IS metrics."""
    required_outputs = {"preds", "labels"}

    def __init__(
        self,
        data_path: str,
        eval_metadata: PuzzleDatasetMetadata,
        img_size: int = 28,
        channels: int = 1,
        max_samples_to_log: int = 10,
        pred_vocab_size: Optional[int] = None,
    ):
        self.data_path = data_path
        self.img_size = img_size
        self.channels = channels
        self.max_samples_to_log = max_samples_to_log
        
        self.label_vocab_size = eval_metadata.vocab_size
        self.pred_vocab_size = pred_vocab_size or self.label_vocab_size
        
        self._fid_metric = None
        self._is_metric = None
        self._samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._step = 0
        
    def _init_metrics(self, device: torch.device):
        if self._fid_metric is not None or FrechetInceptionDistance is None:
            return
        self._fid_metric = FrechetInceptionDistance(
            feature=2048, normalize=False, reset_real_features=False
        ).to(device)
        self._is_metric = InceptionScore(
            feature="logits_unbiased", splits=10, normalize=False
        ).to(device)

    def begin_eval(self):
        """Reset state for new evaluation run."""
        self._samples = []
        if self._fid_metric is not None:
            self._fid_metric.reset()
        if self._is_metric is not None:
            self._is_metric.reset()

    def set_train_vocab_size(self, vocab_size: int):
        """Set training vocab size for prediction scaling."""
        self.pred_vocab_size = vocab_size
    
    def set_step(self, step: int):
        """Set current training step for logging."""
        self._step = step
    
    def scale_batch(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Scale batch tokens from eval vocab to train vocab range.
        
        Returns:
            Tuple of (scaled_batch, original_labels)
        """
        original_labels = batch["labels"].clone()
        
        if self.pred_vocab_size != self.label_vocab_size:
            eval_levels = self.label_vocab_size - 1
            train_levels = self.pred_vocab_size - 1
            for key in ["inputs", "labels"]:
                if key in batch:
                    tokens = batch[key].float()
                    scaled = ((tokens - 1) / max(eval_levels - 1, 1) * (train_levels - 1) + 1).round().long()
                    batch[key] = scaled.clamp(0, self.pred_vocab_size - 1)
        
        return batch, original_labels
    
    def precompute_real_features(self, eval_loader, device: torch.device):
        """Precompute real image statistics from eval set."""
        self._init_metrics(device)
        if self._fid_metric is None:
            return
            
        with torch.inference_mode():
            for _, batch, _ in eval_loader:
                labels = batch["labels"].to(device)
                real_imgs = self._to_inception_format(labels, self.label_vocab_size)
                self._fid_metric.update(real_imgs, real=True)

    def _to_inception_format(self, tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
        """Convert tokens to Inception-compatible image format."""
        imgs = tokens_to_pixels(tokens, vocab_size)
        imgs = imgs.view(-1, self.channels, self.img_size, self.img_size)
        if self.channels == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        return F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False).to(torch.uint8)
                
    def update_batch(
        self, 
        batch: Dict[str, torch.Tensor], 
        preds: Dict[str, torch.Tensor],
        original_labels: Optional[torch.Tensor] = None,
    ):
        """Update metrics with a batch of predictions."""
        device = preds["preds"].device
        self._init_metrics(device)
        
        predictions = preds["preds"]
        
        # Collect samples for visualization
        if len(self._samples) < self.max_samples_to_log:
            labels = original_labels if original_labels is not None else batch["labels"]
            n = min(self.max_samples_to_log - len(self._samples), predictions.shape[0])
            for i in range(n):
                self._samples.append((labels[i].cpu(), predictions[i].cpu()))
        
        # Update FID/IS
        if self._fid_metric is not None:
            fake_imgs = self._to_inception_format(predictions, self.pred_vocab_size)
            self._fid_metric.update(fake_imgs, real=False)
            self._is_metric.update(fake_imgs)

    def _log_samples(self):
        """Log sample images to wandb."""
        if not self._samples or wandb is None or wandb.run is None:
            return
            
        images = []
        for label, pred in self._samples:
            label_img = tokens_to_pixels(label, self.label_vocab_size).numpy().astype(np.uint8)
            pred_img = tokens_to_pixels(pred, self.pred_vocab_size).numpy().astype(np.uint8)
            label_img = label_img.reshape(self.img_size, self.img_size)
            pred_img = pred_img.reshape(self.img_size, self.img_size)
            combined = np.concatenate([label_img, pred_img], axis=1)
            images.append(wandb.Image(combined, caption="Label | Pred"))
            
        wandb.log({f"eval/{self.data_path.split('/')[-1]}_samples": images}, step=self._step)

    def result(
        self,
        save_path: Optional[str],
        rank: int,
        world_size: int,
        group=None,
    ) -> Optional[Dict[str, float]]:
        """Compute final metrics and log samples."""
        results = {}
        
        if self._fid_metric is not None:
            if world_size > 1:
                import torch.distributed as dist
                dist.barrier()
            
            try:
                fid = self._fid_metric.compute().item()
                is_mean, is_std = self._is_metric.compute()
                
                if rank == 0:
                    results["fid"] = fid
                    results["is_mean"] = is_mean.item()
                    results["is_std"] = is_std.item()
                    print(f"FID: {fid:.2f}, IS: {is_mean:.2f} ± {is_std:.2f}")
            except RuntimeError as e:
                if rank == 0:
                    print(f"Warning: Could not compute metrics: {e}")
        
        if rank == 0:
            self._log_samples()
            
        return results if rank == 0 and results else None


MNISTImage = MNIST
