import argparse
import math
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
from tqdm import tqdm


class DiscreteVAE(nn.Module):
    def __init__(self, z=32, k=16, h=64):
        super().__init__()
        self.z, self.k, self.h = z, k, h
        self.enc = nn.Sequential(
            nn.Conv2d(1, h, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(h, 2 * h, 4, 2, 1), nn.SiLU(),
            nn.Flatten(),
        )
        self.q = nn.Linear(2 * h * 7 * 7, z * k)
        self.fc = nn.Linear(z * k, 2 * h * 7 * 7)
        self.dec = nn.Sequential(
            nn.Unflatten(1, (2 * h, 7, 7)),
            nn.ConvTranspose2d(2 * h, h, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(h, 1, 4, 2, 1),
        )

    def encode(self, x):
        return self.q(self.enc(x)).view(x.size(0), self.z, self.k)

    def decode(self, z):
        return self.dec(self.fc(z.flatten(1)))

    def forward(self, x, tau=1.0, hard=True):
        q_logits = self.encode(x)
        z = F.gumbel_softmax(q_logits, tau=tau, hard=hard, dim=-1)
        return self.decode(z), q_logits

    @torch.no_grad()
    def sample(self, n, device=None, binary=True):
        device = device or next(self.parameters()).device
        idx = torch.randint(self.k, (n, self.z), device=device)
        z = F.one_hot(idx, self.k).float()
        x = torch.sigmoid(self.decode(z))
        return torch.bernoulli(x) if binary else x


def ddp():
    if "LOCAL_RANK" not in os.environ:
        return 0, 1, "cuda" if torch.cuda.is_available() else "cpu", False
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), f"cuda:{local_rank}", True


def save_ckpt(path, model, opt, step, args):
    raw = model.module if hasattr(model, "module") else model
    torch.save({
        "model": raw.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "config": {
            "vae_type": "discrete",
            "z": args.z,
            "k": args.k,
            "h": args.h,
            "binary": True,
            "image_size": 28,
            "channels": 1,
        },
    }, path)


def load_vae(path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model = DiscreteVAE(cfg.get("z", 32), cfg.get("k", 16), cfg.get("h", 64)).to(device)
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    return model.eval(), cfg


def binarize(x, args):
    return torch.bernoulli(x) if args.dynamic_binarize else (x > args.threshold).float()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="checkpoints_vae_mnist")
    p.add_argument("--resume")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=256, help="per-GPU batch size")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--z", type=int, default=32, help="number of categorical latent variables")
    p.add_argument("--k", type=int, default=16, help="categories per latent variable")
    p.add_argument("--h", type=int, default=64)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0, help="Gumbel-Softmax temperature")
    p.add_argument("--soft", action="store_true", help="use soft Gumbel samples instead of straight-through hard samples")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--dynamic_binarize", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--sample_every", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    rank, world, device, is_dist = ddp()
    main_rank = rank == 0
    random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    if main_rank:
        os.makedirs(args.out, exist_ok=True)

    transform = transforms.ToTensor()
    if is_dist:
        if main_rank:
            MNIST("./data", train=True, download=True, transform=transform)
        dist.barrier()
        ds = MNIST("./data", train=True, download=False, transform=transform)
    else:
        ds = MNIST("./data", train=True, download=True, transform=transform)
    sampler = DistributedSampler(ds, world, rank, shuffle=True) if is_dist else None
    loader = DataLoader(ds, args.batch_size, sampler=sampler, shuffle=sampler is None,
                        num_workers=args.num_workers, pin_memory=device.startswith("cuda"))

    step = 0
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {})
        args.z, args.k, args.h = cfg.get("z", args.z), cfg.get("k", args.k), cfg.get("h", args.h)

    model = DiscreteVAE(args.z, args.k, args.h).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if ckpt is not None:
        model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt["model"].items()})
        opt.load_state_dict(ckpt["opt"])
        step = ckpt.get("step", 0)
    if is_dist:
        model = DDP(model, device_ids=[int(os.environ["LOCAL_RANK"])])

    pbar = tqdm(total=args.steps, initial=step, disable=not main_rank)
    epoch = 0
    while step < args.steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        for x, _ in loader:
            if step >= args.steps:
                break
            x = binarize(x.to(device, non_blocking=True), args)
            logits, q_logits = model(x, tau=args.tau, hard=not args.soft)
            bce = F.binary_cross_entropy_with_logits(logits, x, reduction="sum") / x.size(0)
            log_q = F.log_softmax(q_logits, dim=-1)
            q = log_q.exp()
            kl = (q * (log_q + math.log(args.k))).sum(dim=(1, 2)).mean()
            loss = bce + args.beta * kl
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            step += 1
            if main_rank:
                pbar.update(1)
                pbar.set_description(f"loss {loss.item():.1f} bce {bce.item():.1f} kl {kl.item():.1f}")
                raw = model.module if hasattr(model, "module") else model
                if step % args.sample_every == 0:
                    save_image(raw.sample(64, device).cpu(), f"{args.out}/sample_{step:07d}.png", nrow=8)
                if step % args.save_every == 0:
                    save_ckpt(f"{args.out}/ckpt_{step:07d}.pt", model, opt, step, args)
                    save_ckpt(f"{args.out}/ckpt_latest.pt", model, opt, step, args)
        epoch += 1

    if main_rank:
        save_ckpt(f"{args.out}/ckpt_final.pt", model, opt, step, args)
        save_ckpt(f"{args.out}/ckpt_latest.pt", model, opt, step, args)
        pbar.close()
    if is_dist:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
