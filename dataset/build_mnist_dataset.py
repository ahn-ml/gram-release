"""
MNIST Dataset Builder for TRM/HRM

Discretize MNIST images into a dataset for generation tasks.
- Input: all 0 (blank canvas)
- Label: discretized MNIST image

Usage:
    # binary (default)
    python -m dataset.build_mnist_dataset --output-dir data/mnist --vocab-levels 2 --threshold 0.5
    
    # train binary, test 256 level
    python -m dataset.build_mnist_dataset --output-dir data/mnist --vocab-levels 2 --test-vocab-levels 256
    
    # custom level (e.g. 16 level)
    python -m dataset.build_mnist_dataset --output-dir data/mnist_16 --vocab-levels 16
"""

from typing import Optional
import os
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from torchvision import datasets

from common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    output_dir: str = "data/mnist"
    vocab_levels: int = 2  
    test_vocab_levels: Optional[int] = None  
    threshold: float = 0.5  
    subsample_size: Optional[int] = None
    seed: int = 42


def discretize_mnist(image: np.ndarray, vocab_levels: int, threshold: float = 0.5) -> np.ndarray:
    if vocab_levels == 2:
        return (image > threshold).astype(np.uint8)
    elif vocab_levels == 256:
        return (image * 255).astype(np.uint8)
    else:
        return (image * (vocab_levels - 1)).astype(np.uint8)


def convert_subset(set_name: str, config: DataProcessConfig):
    np.random.seed(config.seed)
    
    # Determine vocab_levels for this subset
    if set_name == "test" and config.test_vocab_levels is not None:
        vocab_levels = config.test_vocab_levels
    else:
        vocab_levels = config.vocab_levels
    
    # Load MNIST
    is_train = (set_name == "train")
    mnist = datasets.MNIST(root="dataset/mnist_raw", train=is_train, download=True)
    images = mnist.data.numpy() / 255.0  # (N, 28, 28), normalize to [0,1]
    
    # Subsample if specified
    if config.subsample_size is not None and config.subsample_size < len(images):
        indices = np.random.choice(len(images), size=config.subsample_size, replace=False)
        images = images[indices]
    
    # Build dataset
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    seq_len = 28 * 28  # 784
    
    for idx, img in enumerate(tqdm(images, desc=f"Processing {set_name}")):
        inp = np.zeros((28, 28), dtype=np.uint8)
        label = discretize_mnist(img, vocab_levels, config.threshold)
        
        results["inputs"].append(inp)
        results["labels"].append(label)
        results["puzzle_identifiers"].append(0)
        results["puzzle_indices"].append(idx + 1)
        results["group_indices"].append(idx + 1)
    
    # To Numpy (vocab: PAD=0, 0=1, 1=2)
    def _seq_to_numpy(seq):
        arr = np.stack(seq).reshape(len(seq), -1)
        return arr + 1  # shift by 1 for PAD token
    
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }
    
    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=seq_len,
        vocab_size=vocab_levels + 1,  # PAD + vocab_levels
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )
    
    # Save
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
    
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    print(f"{set_name}: {len(images)} samples (vocab_levels={vocab_levels}) saved to {save_dir}")


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    # Save identifiers mapping
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
