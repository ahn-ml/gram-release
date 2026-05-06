"""
Image Evaluation Script for D3PM, TRM, and VAE models.
Computes FID/IS metrics and visualizes the generation process.

Usage:
    torchrun --nproc_per_node=8 image_eval.py --model trm --checkpoint ...
    python image_eval.py --model d3pm --checkpoint ...
    python image_eval.py --model vae --checkpoint checkpoints_vae_mnist/ckpt_latest.pt
"""

import os
import sys
import re
import argparse

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "d3pm"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "trm"))


# ============================================================================
# Utils
# ============================================================================

def tokens_to_pixels(tokens: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Tokens -> pixels [0, 255]. Token encoding: PAD=0, pixels -> tokens 1~vocab_size-1."""
    n = vocab_size - 1
    if n == 256:
        return (tokens.float() - 1).clamp(0, 255)
    else:
        return ((tokens.float() - 1) / max(n - 1, 1) * 255).clamp(0, 255)


def tokens_to_images(tokens: torch.Tensor, vocab_size: int, shape: tuple) -> torch.Tensor:
    """Tokens -> images [0,1] for visualization."""
    n = vocab_size - 1
    return ((tokens.float() - 1).clamp(0, n - 1) / max(n - 1, 1)).view(-1, *shape)


def images_to_tokens(images: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Images [0,1] -> Tokens (1~vocab_size-1). Adds 1 for PAD offset."""
    n = vocab_size - 1
    if n == 2:  # Binary
        return ((images > 0.5).long() + 1).view(images.shape[0], -1)
    return ((images * (n - 1)).round().long() + 1).view(images.shape[0], -1)


def prepare_for_fid(images: torch.Tensor, channels: int, size: int) -> torch.Tensor:
    """Prepare images (in [0,255]) for FID/IS: 3-channel, 299x299, uint8."""
    images = images.view(-1, channels, size, size)
    if channels == 1:
        images = images.repeat(1, 3, 1, 1)
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    return images.clamp(0, 255).to(torch.uint8)


def setup_distributed():
    """Setup DDP. Returns (rank, world_size, device, is_distributed)."""
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        return dist.get_rank(), dist.get_world_size(), f"cuda:{rank}", True
    return 0, 1, "cuda" if torch.cuda.is_available() else "cpu", False


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# VAE
# ============================================================================

def load_vae_model(ckpt_path: str, device: str):
    """Load compact discrete VAE from vae.py."""
    from vae import load_vae

    model, cfg = load_vae(ckpt_path, device)
    cfg.update({"channels": 1, "size": 28})
    return model, cfg


def sample_vae(model, n: int, cfg: dict, device: str):
    return model.sample(n, device=device, binary=cfg.get("sample_binary", False))


# ============================================================================
# D3PM
# ============================================================================

def load_d3pm(ckpt_path: str, device: str):
    """Load D3PM model."""
    from d3pm_runner import D3PM
    from dit import DiT_Llama
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    channels, size = (3, 32) if cfg.get("data") == "cifar10" else (1, 28)
    
    model = D3PM(
        DiT_Llama(channels, cfg["N"], input_size=size, n_layers=cfg["n_layers"],
                  n_heads=cfg["n_heads"], dim=cfg["dim"],
                  class_dropout_prob=cfg.get("class_dropout_prob", 1.0)),
        cfg["timesteps"], num_classes=cfg["N"],
        hybrid_loss_coeff=cfg.get("hybrid_loss_coeff", 0.0),
        forward_type=cfg["forward_type"]
    ).to(device)
    
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, {"N": cfg["N"], "forward_type": cfg["forward_type"], 
                   "channels": channels, "size": size}


def sample_d3pm(model, n: int, cfg: dict, device: str, stride: int = 40):
    """Generate n samples. Returns (final_images, step_images) in [0,1]."""
    N, c, s = cfg["N"], cfg["channels"], cfg["size"]
    
    if cfg["forward_type"] == "absorb":
        init = torch.ones(n, c, s, s, device=device).long() * (N // 2)
    elif cfg["forward_type"] == "absorb_zero":
        init = torch.zeros(n, c, s, s, device=device).long()
    else:
        init = torch.randint(0, N, (n, c, s, s), device=device)
    
    conds = torch.randint(0, 10, (n,), device=device)
    steps = model.sample_with_image_sequence(init, conds, stride=stride)
    steps = [img.float() / (N - 1) for img in steps]
    return steps[-1], steps


# ============================================================================
# TRM
# ============================================================================

def load_trm(ckpt_dir: str, device: str, batch_size: int = 100, max_steps: int = None):
    """Load TRM model with EMA weights if available."""
    import yaml
    from utils.functions import load_model_class
    from pretrain import PretrainConfig, ArchConfig, LossConfig, EvaluatorConfig
    
    # Load config
    cfg_path = os.path.join(ckpt_dir, "all_config.yaml")
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(os.path.dirname(ckpt_dir), "all_config.yaml")
    
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    arch = cfg["arch"]
    size = arch.get("image_size", 28)
    channels = arch.get("in_channels", 1)
    
    # Derive vocab_size from data_paths (e.g., "mnist_2" -> 3)
    paths = cfg.get("data_paths", [])
    if paths:
        m = re.search(r'_(\d+)$', paths[0] if isinstance(paths, list) else paths)
        vocab_size = int(m.group(1)) + 1 if m else 257
    else:
        vocab_size = 257
    
    # Build model
    model_cfg = {k: v for k, v in arch.items() if k not in ["loss", "name"]}
    model_cfg.update({"batch_size": batch_size, "vocab_size": vocab_size,
                      "seq_len": size * size * channels, "num_puzzle_identifiers": 1})
    
    model_cls = load_model_class(arch["name"])
    loss_cls = load_model_class(arch["loss"]["name"])
    model = loss_cls(model_cls(model_cfg), **{k: v for k, v in arch["loss"].items() if k != "name"})
    
    # Load checkpoint
    ckpt_file = os.path.join(ckpt_dir, "checkpoint.pt")
    if not os.path.exists(ckpt_file):
        step_dirs = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("step_")],
                           key=lambda x: int(x.split("_")[1]))
        if step_dirs:
            ckpt_file = os.path.join(ckpt_dir, step_dirs[-1], "checkpoint.pt")
    
    # Inject classes for pickle
    for cls in [PretrainConfig, ArchConfig, LossConfig, EvaluatorConfig]:
        setattr(sys.modules['__main__'], cls.__name__, cls)
    
    ckpt = torch.load(ckpt_file, map_location=device, weights_only=False)
    clean = lambda d: {k.replace("_orig_mod.", ""): v for k, v in d.items()}
    model.load_state_dict(clean(ckpt["model"]), strict=False)
    
    # Load EMA if available
    if "ema" in ckpt:
        from models.ema import EMAHelper
        ema = EMAHelper()
        ema.register(model)
        ema.load_state_dict(clean(ckpt["ema"]))
        ema.ema(model)
        print("  Using EMA weights")
    
    model.to(device).eval()
    
    # Override max_steps in model's internal config if provided
    actual_max_steps = arch.get("halt_max_steps", 16)
    if max_steps is not None:
        model.model.config.halt_max_steps = max_steps  # Modify internal config
        actual_max_steps = max_steps
    
    return model, {"size": size, "channels": channels, "vocab_size": vocab_size,
                   "max_steps": actual_max_steps}


def sample_trm(model, batch: dict, device: str, max_steps: int = 16):
    """Generate from blank canvas. Returns (final_tokens, step_tokens)."""
    steps = []
    with torch.inference_mode(), torch.device(device):
        carry = model.initial_carry(batch)
        for _ in range(max_steps):
            carry, _, _, preds, done = model(carry=carry, batch=batch, return_keys={"preds"})
            steps.append(preds["preds"].clone())
            if done:
                break
    return steps[-1], steps


def make_trm_batch(images: torch.Tensor, vocab_size: int, device: str):
    """Create TRM batch from images. Input is blank canvas (all 1s)."""
    tokens = images_to_tokens(images, vocab_size)
    return {
        "inputs": torch.ones_like(tokens),  # Blank canvas
        "labels": tokens,
        "puzzle_identifiers": torch.zeros(images.shape[0], dtype=torch.long, device=device)
    }


# ============================================================================
# FID/IS
# ============================================================================

def compute_fid_is(model, cfg: dict, device: str, rank: int, world_size: int,
                   num_samples: int, batch_size: int, loader: DataLoader,
                   model_type: str, real_vocab: int = 256):
    """Compute FID/IS for either model type."""
    fid = FrechetInceptionDistance(feature=2048, normalize=False, reset_real_features=False).to(device)
    iscore = InceptionScore(feature="logits_unbiased", splits=10, normalize=False).to(device)
    
    if model_type == "d3pm":
        channels, size = cfg["channels"], cfg["size"]
        N = cfg["N"]
    elif model_type == "trm":
        channels, size = cfg["channels"], cfg["size"]
        vocab_size = cfg["vocab_size"]
    else:
        channels, size = cfg["channels"], cfg["size"]
    
    is_main = (rank == 0)
    
    # Real images: convert [0,1] -> [0,255] with optional quantization
    if is_main:
        print("Processing real images...")
    for x, _ in tqdm(loader, desc=f"[GPU{rank}] Real", disable=not is_main):
        x = x.to(device)
        # Quantize to real_vocab levels, then scale to [0,255]
        if real_vocab < 256:
            x = (x * (real_vocab - 1)).round() / max(real_vocab - 1, 1)
        real_pixels = (x * 255).clamp(0, 255)
        fid.update(prepare_for_fid(real_pixels, channels, size), real=True)
    
    # Fake images
    local_n = (num_samples + world_size - 1) // world_size
    if is_main:
        print(f"Generating {num_samples} samples...")
    
    generated = 0
    loader_iter = iter(loader)
    pbar = tqdm(total=local_n, desc=f"[GPU{rank}] Fake", disable=not is_main)
    
    while generated < local_n:
        try:
            x, _ = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            x, _ = next(loader_iter)
        
        x = x.to(device)
        n = min(x.shape[0], local_n - generated)
        
        with torch.inference_mode():
            if model_type == "d3pm":
                fake, _ = sample_d3pm(model, n, cfg, device, stride=1000)
                # Apply same quantization as real images for fair comparison
                if real_vocab < 256:
                    fake = (fake * (real_vocab - 1)).round() / max(real_vocab - 1, 1)
                fake_pixels = (fake * 255).clamp(0, 255)  # D3PM outputs [0,1]
            elif model_type == "trm":
                batch = make_trm_batch(x[:n], vocab_size, device)
                preds, _ = sample_trm(model, batch, device, cfg["max_steps"])
                fake_pixels = tokens_to_pixels(preds, vocab_size)  # TRM: tokens -> [0,255]
            else:
                fake = sample_vae(model, n, cfg, device)
                if real_vocab < 256:
                    fake = (fake * (real_vocab - 1)).round() / max(real_vocab - 1, 1)
                fake_pixels = (fake * 255).clamp(0, 255)
        
        fake_prepared = prepare_for_fid(fake_pixels, channels, size)
        fid.update(fake_prepared, real=False)
        iscore.update(fake_prepared)
        generated += n
        pbar.update(n)
    
    pbar.close()
    if world_size > 1:
        dist.barrier()
    
    return fid.compute().item(), *[x.item() for x in iscore.compute()]


# ============================================================================
# Visualization
# ============================================================================

def save_grid(step_images: list, path: str, nrows: int = 10):
    """Save generation steps as grid (rows=samples, cols=steps)."""
    nrows = min(nrows, step_images[0].shape[0])
    rows = [torch.stack([step_images[j][i] for j in range(len(step_images))]) 
            for i in range(nrows)]
    grid = make_grid(torch.cat(rows), nrow=len(step_images), padding=2)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    save_image(grid, path)
    print(f"Saved: {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["d3pm", "trm", "vae"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--step_stride", type=int, default=40, help="D3PM visualization stride")
    parser.add_argument("--nrows", type=int, default=10)
    parser.add_argument("--output_dir", default="eval_outputs")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--vis_only", action="store_true")
    parser.add_argument("--real_split", default="test", choices=["train", "test"])
    parser.add_argument("--real_vocab", type=int, default=256, help="Real image quantization")
    parser.add_argument("--max_steps", type=int, default=None, help="TRM max inference steps (default: from config)")
    parser.add_argument("--vae_mean_sample", action="store_true", help="Use VAE decoder probabilities instead of Bernoulli samples")
    args = parser.parse_args()
    
    rank, world_size, device, is_dist = setup_distributed()
    is_main = (rank == 0)
    
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"GPUs: {world_size} | Real: {args.real_split} | Quantization: {args.real_vocab}")
    
    # Data is only needed for FID/IS or TRM visualization. VAE/D3PM vis-only can run offline.
    need_loader = (not args.skip_fid and not args.vis_only) or args.model == "trm"
    loader = None
    if need_loader:
        transform = transforms.ToTensor()
        if is_dist:
            if is_main:
                MNIST("./data", train=(args.real_split == "train"), download=True, transform=transform)
            dist.barrier()
            mnist = MNIST("./data", train=(args.real_split == "train"), download=False, transform=transform)
        else:
            mnist = MNIST("./data", train=(args.real_split == "train"), download=True, transform=transform)
        loader = DataLoader(mnist, args.batch_size, num_workers=4,
                            sampler=DistributedSampler(mnist, world_size, rank) if is_dist else None)
    
    # Load model
    if args.model == "d3pm":
        if is_main:
            print(f"\nLoading D3PM: {args.checkpoint}")
        model, cfg = load_d3pm(args.checkpoint, device)
    elif args.model == "trm":
        if is_main:
            print(f"\nLoading TRM: {args.checkpoint}")
        model, cfg = load_trm(args.checkpoint, device, args.batch_size, args.max_steps)
        if is_main:
            print(f"  vocab: {cfg['vocab_size']}, steps: {cfg['max_steps']}")
    else:
        if is_main:
            print(f"\nLoading VAE: {args.checkpoint}")
        model, cfg = load_vae_model(args.checkpoint, device)
        cfg["sample_binary"] = cfg.get("binary", True) and not args.vae_mean_sample
    
    # FID/IS
    if not args.skip_fid and not args.vis_only:
        fid, is_mean, is_std = compute_fid_is(
            model, cfg, device, rank, world_size,
            args.num_samples, args.batch_size, loader, args.model, args.real_vocab
        )
        if is_main:
            print(f"\nFID: {fid:.2f} | IS: {is_mean:.2f}±{is_std:.2f}")
            with open(f"{args.output_dir}/results.txt", "w") as f:
                f.write(f"Model: {args.model.upper()}\nFID: {fid:.4f}\nIS: {is_mean:.4f}±{is_std:.4f}\n")
    
    # Visualization
    if is_main:
        with torch.no_grad():
            if args.model == "d3pm":
                final, steps = sample_d3pm(model, args.nrows, cfg, device, stride=args.step_stride)
                # Save individual samples
                os.makedirs(f"{args.output_dir}/samples", exist_ok=True)
                for i, img in enumerate(final[:10]):
                    save_image(img, f"{args.output_dir}/samples/{i:02d}.png")
                print(f"  Saved {min(10, len(final))} samples")
            elif args.model == "trm":
                x, _ = next(iter(loader))
                x = x[:args.nrows].to(device)
                batch = make_trm_batch(x, cfg["vocab_size"], device)
                final, steps = sample_trm(model, batch, device, cfg["max_steps"])
                shape = (cfg["channels"], cfg["size"], cfg["size"])
                steps = [tokens_to_images(p, cfg["vocab_size"], shape).cpu() for p in steps]
                
                # Save individual samples
                os.makedirs(f"{args.output_dir}/samples", exist_ok=True)
                final_imgs = tokens_to_images(final, cfg["vocab_size"], shape).cpu()
                for i, img in enumerate(final_imgs[:10]):
                    save_image(img, f"{args.output_dir}/samples/{i:02d}.png")
                print(f"  Saved {min(10, len(final_imgs))} samples")
            else:
                final = sample_vae(model, max(args.nrows, 10), cfg, device)
                os.makedirs(f"{args.output_dir}/samples", exist_ok=True)
                for i, img in enumerate(final[:10].cpu()):
                    save_image(img, f"{args.output_dir}/samples/{i:02d}.png")
                save_image(final.cpu(), f"{args.output_dir}/generation.png", nrow=args.nrows)
                print(f"  Saved {min(10, len(final))} samples")
                print(f"\nDone! → {args.output_dir}/")
                cleanup()
                return
            
            save_grid(steps, f"{args.output_dir}/generation.png", args.nrows)
    
    if is_main:
        print(f"\nDone! → {args.output_dir}/")
    cleanup()


if __name__ == "__main__":
    main()
