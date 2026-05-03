import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torchvision.utils import make_grid
from tqdm import tqdm
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance

import wandb
from d3pm_runner import D3PM
from dit import DiT_Llama



def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def save_checkpoint(config, d3pm, optim, global_step, path):
    state = {
        "config": config,
        "model": d3pm.module.state_dict() if hasattr(d3pm, "module") else d3pm.state_dict(),
        "optim": optim.state_dict(),
        "global_step": global_step,
    }
    torch.save(state, path)


def load_checkpoint(d3pm, optim, path, device):
    ckpt = torch.load(path, map_location=device)
    model = d3pm.module if hasattr(d3pm, "module") else d3pm
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    return ckpt["global_step"]


@torch.no_grad()
def evaluate_fid_is(model, fid_metric, is_metric, N, forward_type, device, rank, world_size, in_channels, img_size, num_samples=5000, batch_size=100):
    """Evaluate FID and IS."""
    model.eval()
    
    per_proc = (num_samples + world_size - 1) // world_size
    start = rank * per_proc
    end = min(start + per_proc, num_samples)
    local_size = end - start
    
    num_batches = (local_size + batch_size - 1) // batch_size
    
    pbar = tqdm(range(num_batches), desc="FID/IS eval", disable=(rank != 0))
    for i in pbar:
        cur_size = min(batch_size, local_size - i * batch_size)
        if cur_size <= 0:
            break
        
        conds = torch.randint(0, 10, (cur_size,), device=device)
        if forward_type == "absorb":
            init_noise = (torch.ones(cur_size, in_channels, img_size, img_size, device=device) * (N // 2)).long()
        elif forward_type == "absorb_zero":
            init_noise = torch.zeros(cur_size, in_channels, img_size, img_size, device=device).long()
        elif forward_type == "uniform":
            init_noise = torch.randint(0, N, (cur_size, in_channels, img_size, img_size), device=device)
        else:
            raise NotImplementedError()
        
        images = model.sample_with_image_sequence(init_noise, conds, stride=40)
        fake = images[-1].float() / (N - 1)
        # Convert 1-channel to 3-channel for FID/IS
        if in_channels == 1:
            fake = fake.repeat(1, 3, 1, 1)
        fake = (fake * 255).clamp(0, 255)
        fake = F.interpolate(fake, size=(299, 299), mode="bilinear", align_corners=False).to(torch.uint8)
        
        fid_metric.update(fake, real=False)
        is_metric.update(fake)
    
    dist.barrier()
    fid_value = fid_metric.compute()
    is_mean, is_std = is_metric.compute()
    
    fid_metric.reset()
    is_metric.reset()
    
    return fid_value.item(), is_mean.item(), is_std.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--forward_type", type=str, default="absorb_zero",
                        choices=["absorb", "absorb_zero", "uniform"], help="D3PM forward process")
    parser.add_argument("--run_name", type=str, default=None,
                        help="wandb run_name; defaults to d3pm_mnist_<forward_type>_seed<seed>")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="checkpoint dir; defaults to checkpoints_mnist_<forward_type>_seed<seed>")
    args = parser.parse_args()

    # Seed everything (per-rank offset done after DDP init below)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    run_name = args.run_name or f"d3pm_mnist_{args.forward_type}_seed{args.seed}"
    checkpoint_path = args.checkpoint_path or f"checkpoints_mnist_{args.forward_type}_seed{args.seed}"

    config = {
        "data": "mnist",  # "cifar10" or "mnist"
        "N": 2,
        "n_layers": 3,
        "n_heads": 8,
        "dim": 512,
        "forward_type": args.forward_type,
        "timesteps": 1000,
        "class_dropout_prob": 1.0,
        "lr": 2e-5,
        "hybrid_loss_coeff": 0.0,
        "batch_size": 64,
        "max_steps": 450_000,
        "save_interval": 10000,
        "eval_interval": 10000,
        "eval_samples": 5000,
        "seed": args.seed,
        "run_name": run_name,
        "project_name": "v_trm_d3pm_mnist",
        "checkpoint_path": checkpoint_path,
    }
    
    # Dataset-specific settings
    if config["data"] == "cifar10":
        in_channels = 3
        img_size = 32
    elif config["data"] == "mnist":
        in_channels = 1
        img_size = 28
    else:
        raise ValueError(f"Unknown data: {config['data']}")

    # DDP setup
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"
    is_main = (rank == 0)

    if is_main:
        wandb.init(project="v_trm_d3pm_mnist", name=config["run_name"])
        os.makedirs(config["checkpoint_path"], exist_ok=True)
        if is_main:
            # 현재 작업 디렉토리 전체를 코드로 업로드(원하는 확장자만)
            wandb.run.log_code(
                root=".",
                include_fn=lambda path: path.endswith((".py", "sh", ".yaml", ".yml", ".json", ".txt", ".md"))
            )
            wandb.config.update(config)

    N = config["N"]  # number of classes for discretized state per pixel
    forward_type = config["forward_type"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    dim = config["dim"]
    timesteps = config["timesteps"]
    hybrid_loss_coeff = config["hybrid_loss_coeff"]
    learning_rate = config["lr"]
    class_dropout_prob = config["class_dropout_prob"]
    
    
    d3pm = D3PM(
        DiT_Llama(in_channels, N, input_size=img_size, class_dropout_prob=class_dropout_prob, n_layers=n_layers, n_heads=n_heads, dim=dim), timesteps, num_classes=N, hybrid_loss_coeff=hybrid_loss_coeff, forward_type=forward_type
    ).to(device)
    
    if is_main:
        num_param = sum([p.numel() for p in d3pm.x0_model.parameters()])
        print(f"Total Param Count: {sum([p.numel() for p in d3pm.x0_model.parameters()])}")
        wandb.log({"num_params": num_param}, step=0)
    

    
    # Wrap with DDP
    d3pm = DDP(d3pm, device_ids=[local_rank])

    # Dataset setup
    if config["data"] == "cifar10":
        dataset = CIFAR10(
            "./data",
            train=True,
            download=is_main,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]),
        )
        val_dataset = CIFAR10("./data", train=False, download=False, transform=transforms.ToTensor())
    elif config["data"] == "mnist":
        dataset = MNIST(
            "./data",
            train=True,
            download=is_main,
            transform=transforms.ToTensor(),
        )
        val_dataset = MNIST("./data", train=False, download=False, transform=transforms.ToTensor())
    dist.barrier()  # wait for download
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler, num_workers=16)
    optim = torch.optim.AdamW(d3pm.module.x0_model.parameters(), lr=learning_rate)
    d3pm.train()

    max_steps = config["max_steps"]
    save_interval = config["save_interval"]
    eval_interval = config["eval_interval"]
    eval_samples = config["eval_samples"]
    global_step = 0
    
    # Setup FID/IS metrics
    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False, reset_real_features=False).to(device)
    is_metric = InceptionScore(feature="logits_unbiased", splits=10, normalize=False).to(device)
    
    # Precompute real image statistics for FID
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)
    for x_real, _ in val_loader:
        x_real = x_real.to(device)
        # Convert 1-channel to 3-channel for FID
        if in_channels == 1:
            x_real = x_real.repeat(1, 3, 1, 1)
        x_real = (x_real * 255).clamp(0, 255)
        x_real = F.interpolate(x_real, size=(299, 299), mode="bilinear", align_corners=False).to(torch.uint8)
        fid_metric.update(x_real, real=True)
    dist.barrier()
    
    # Resume from checkpoint
    if args.resume and os.path.exists(args.resume):
        global_step = load_checkpoint(d3pm, optim, args.resume, device)
        if is_main:
            print(f"Resumed from step {global_step}")

    while global_step < max_steps:
        sampler.set_epoch(global_step)
        pbar = tqdm(dataloader, disable=not is_main)
        loss_ema = None
        for x, cond in pbar:
            if global_step >= max_steps:
                break
                
            optim.zero_grad()
            x = x.to(device)
            cond = cond.to(device)

            # discritize x to N bins
            x_cat = (x * (N - 1)).round().long().clamp(0, N - 1)
            loss, info = d3pm(x_cat, cond)

            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(d3pm.module.x0_model.parameters(), 5.0)

            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.99 * loss_ema + 0.01 * loss.item()

            if is_main and global_step % 10 == 0:
                wandb.log(
                    {
                        "train_loss": loss,
                        "train_grad_norm": norm,
                    },
                    step=global_step,
                )
            pbar.set_description(
                f"step: {global_step}, loss: {loss_ema:.4f}, vb: {info['vb_loss']:.4f}, ce: {info['ce_loss']:.4f}"
            )
            optim.step()
            

            
            global_step += 1

            # Save checkpoint
            if is_main and global_step % save_interval == 0:
                save_checkpoint(config, d3pm, optim, global_step, f"{config['checkpoint_path']}/ckpt_{global_step}.pt")
                save_checkpoint(config, d3pm, optim, global_step, f"{config['checkpoint_path']}/ckpt_latest.pt")

            if is_main and global_step % 600 == 1:
                d3pm.module.eval()

                with torch.no_grad():
                    cond_sample = torch.arange(0, 16).to(device) % 10
                    if forward_type == "absorb":
                        init_noise = torch.ones(16, in_channels, img_size, img_size).to(device) * N//2
                        init_noise = init_noise.to(torch.long)
                    elif forward_type == "absorb_zero":
                        init_noise = torch.zeros(16, in_channels, img_size, img_size, device=device).long()
                    elif forward_type == "uniform":                    
                        init_noise = torch.randint(0, N, (16, in_channels, img_size, img_size)).to(device)
                    else:
                        raise NotImplementedError()
                        

                    images = d3pm.module.sample_with_image_sequence(
                        init_noise, cond_sample, stride=40
                    )
                    # image sequences to gif
                    gif = []
                    for image in images:
                        x_from_dataloader = x_cat[:16].cpu() / (N - 1)
                        this_image = image.float().cpu() / (N - 1)
                        all_images = torch.cat([x_from_dataloader, this_image], dim=0)
                        x_as_image = make_grid(all_images, nrow=4)
                        img = x_as_image.permute(1, 2, 0).cpu().numpy()
                        img = (img * 255).astype(np.uint8)
                        gif.append(Image.fromarray(img))

                    gif[0].save(
                        f"contents/sample_{global_step}.gif",
                        save_all=True,
                        append_images=gif[1:],
                        duration=100,
                        loop=0,
                    )

                    last_img = gif[-1]
                    last_img.save(f"contents/sample_{global_step}_last.png")

                    # log images
                    wandb.log(
                        {
                            "sample": wandb.Image(last_img),
                        },
                        step=global_step,
                    )

            # Evaluate FID/IS
            if global_step % eval_interval == 0:
                if is_main:
                    print(f"\nEvaluating FID/IS at step {global_step}...")
                fid_value, is_mean, is_std = evaluate_fid_is(
                    d3pm.module, fid_metric, is_metric, N, forward_type, device, rank, world_size, in_channels, img_size, eval_samples
                )
                if is_main:
                    print(f"FID: {fid_value:.2f}, IS: {is_mean:.2f} ± {is_std:.2f}")
                    wandb.log(
                        {"val/fid": fid_value, "val/is_mean": is_mean, "val/is_std": is_std},
                        step=global_step,
                    )
                d3pm.train()

    # Final save
    if is_main:
        save_checkpoint(config, d3pm, optim, global_step, f"checkpoints_uniform/ckpt_final_{global_step}.pt")
        wandb.finish()
    
    cleanup_ddp()
