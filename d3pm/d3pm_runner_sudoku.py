import os
import argparse
import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import wandb
from d3pm_runner import D3PM
from dit import DiT_Llama
from evaluators.sudoku_utils import batch_verify_sudoku


def sudoku_conflicts(board):
    """board: (9,9) token space (PAD=0, blank=1, digits 2..10).
    Returns (9,9) bool mask of cells violating sudoku rules or holding invalid tokens."""
    board = np.asarray(board)
    mask = np.zeros((9, 9), dtype=bool)
    invalid = (board < 2) | (board > 10)  # PAD or blank or OOB
    mask |= invalid
    digits = np.where(invalid, 0, board)
    for v in range(2, 11):
        for i in range(9):
            idx = np.where(digits[i] == v)[0]
            if len(idx) > 1:
                mask[i, idx] = True
            idx = np.where(digits[:, i] == v)[0]
            if len(idx) > 1:
                mask[idx, i] = True
        for br in range(0, 9, 3):
            for bc in range(0, 9, 3):
                box = digits[br:br+3, bc:bc+3]
                rr, cc = np.where(box == v)
                if len(rr) > 1:
                    mask[br + rr, bc + cc] = True
    return mask


_VIZ_FONT = None
def _viz_font(size=15):
    global _VIZ_FONT
    if _VIZ_FONT is None:
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            if os.path.exists(p):
                _VIZ_FONT = ImageFont.truetype(p, size)
                break
        if _VIZ_FONT is None:
            _VIZ_FONT = ImageFont.load_default()
    return _VIZ_FONT


def render_sudoku_grid(boards, conflicts_list=None, title=None, cell=48, gap=12, rows=2, cols=2):
    """boards: (B,9,9) token space (PAD=0, blank=1, digits 2..10 = "1".."9").
    conflicts_list: list of (9,9) bool masks or None. rows×cols layout (default 2x2).
    Conflicting cells get red bg + red digit; non-digit token→'?'."""
    bs = 9 * cell
    title_h = 24 if title else 0
    W = cols * bs + (cols + 1) * gap
    H = rows * bs + (rows + 1) * gap + title_h
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font = _viz_font(size=int(cell * 0.7))
    if title:
        draw.text((gap, 4), title, fill="black", font=_viz_font(size=15))
    for b_idx in range(min(rows * cols, len(boards))):
        r, c = divmod(b_idx, cols)
        x0 = gap + c * (bs + gap)
        y0 = title_h + gap + r * (bs + gap)
        board = np.asarray(boards[b_idx])
        conf = conflicts_list[b_idx] if conflicts_list is not None else None
        for i in range(9):
            for j in range(9):
                cx, cy = x0 + j * cell, y0 + i * cell
                bg = "#ffd6d6" if (conf is not None and conf[i, j]) else "white"
                draw.rectangle([cx, cy, cx + cell, cy + cell], fill=bg, outline="#bbbbbb")
                v = int(board[i, j])
                if 2 <= v <= 10:
                    txt, color = str(v - 1), ("red" if (conf is not None and conf[i, j]) else "black")
                elif v == 1:
                    txt, color = "", "black"  # blank cell (incomplete)
                else:
                    txt, color = "?", "red"  # PAD or OOB
                if txt:
                    bbox = draw.textbbox((0, 0), txt, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    draw.text((cx + (cell - tw) / 2 - bbox[0],
                               cy + (cell - th) / 2 - bbox[1]),
                              txt, fill=color, font=font)
        for k in range(0, 10, 3):
            draw.line([x0 + k * cell, y0, x0 + k * cell, y0 + 9 * cell], fill="black", width=2)
            draw.line([x0, y0 + k * cell, x0 + 9 * cell, y0 + k * cell], fill="black", width=2)
    return img


def render_sample(sample_boards, mark_conflicts=True):
    """sample_boards: (B,9,9) token space. Render up to 4 boards (2x2)."""
    confs = [sudoku_conflicts(b) for b in sample_boards] if mark_conflicts else None
    return render_sudoku_grid(sample_boards, confs, title="Sample (red = rule violation or blank)")


def make_init_noise(forward_type, batch_size, in_channels, img_size, N, device):
    """Initial state for D3PM reverse sampling. Sudoku uses absorb_zero (PAD=0) or uniform."""
    shape = (batch_size, in_channels, img_size, img_size)
    if forward_type == "absorb_zero":
        return torch.zeros(shape, dtype=torch.long, device=device)
    if forward_type == "uniform":
        return torch.randint(0, N, shape, dtype=torch.long, device=device)
    raise NotImplementedError(f"forward_type={forward_type} not supported for sudoku")


class DiT_Sudoku(DiT_Llama):
    """Small-board variant: nn.Embedding instead of conv+patchify (patch_size=1)."""

    def __init__(self, in_channels, N, input_size, dim, **kwargs):
        super().__init__(
            in_channels=in_channels, N=N, input_size=input_size,
            patch_size=1, dim=dim, num_classes=N, **kwargs,
        )
        del self.init_conv_seq
        del self.x_embedder
        self.token_embedder = nn.Embedding(N, dim)

    def forward(self, x, t, y):
        self.freqs_cis = self.freqs_cis.to(x.device)
        x_onehot = nn.functional.one_hot(x, self.N).float().to(x.device)
        h = self.token_embedder(x.view(x.shape[0], -1))

        adaln_input = self.t_embedder(t) + self.y_embedder(y, self.training)
        for layer in self.layers:
            h = layer(h, self.freqs_cis[: h.size(1)], adaln_input=adaln_input)

        h = self.final_layer(h, adaln_input)
        h = self.unpatchify(h)

        x_out, gate = (
            h.reshape(h.shape[0], -1, self.N * 2, *h.shape[2:])
             .transpose(2, -1).contiguous()
        ).chunk(2, dim=-1)
        return x_out + x_onehot * (1 + gate).abs()


def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def save_checkpoint(config, d3pm, optim, global_step, path, loader_epoch=0, batch_offset=0):
    state = {
        "config": config,
        "model": d3pm.module.state_dict() if hasattr(d3pm, "module") else d3pm.state_dict(),
        "optim": optim.state_dict(),
        "global_step": global_step,
        "loader_epoch": loader_epoch,
        "batch_offset": batch_offset,
    }
    torch.save(state, path)


def load_checkpoint(d3pm, optim, path, device, config):
    ckpt = torch.load(path, map_location=device)
    for key in [
        "data",
        "data_dir",
        "N",
        "n_layers",
        "n_heads",
        "dim",
        "forward_type",
        "timesteps",
        "class_dropout_prob",
        "lr",
        "hybrid_loss_coeff",
        "batch_size",
        "seed",
        "world_size",
        "train_examples",
        "labels_sha256",
    ]:
        if ckpt["config"].get(key) != config.get(key):
            raise ValueError(f"Resume config mismatch for {key}: {ckpt['config'].get(key)} != {config.get(key)}")
    model = d3pm.module if hasattr(d3pm, "module") else d3pm
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    return ckpt["global_step"], ckpt.get("loader_epoch"), ckpt.get("batch_offset", 0)


def rank_state_path(path, rank):
    return f"{path}.rank{rank}.rng"


def save_rank_state(path, rank, global_step, loader_epoch, batch_offset, world_size):
    torch.save(
        {
            "global_step": global_step,
            "loader_epoch": loader_epoch,
            "batch_offset": batch_offset,
            "rank": rank,
            "world_size": world_size,
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
            "numpy": np.random.get_state(),
        },
        rank_state_path(path, rank),
    )


def load_rank_state(path, rank, global_step, loader_epoch, batch_offset, world_size):
    state_path = rank_state_path(path, rank)
    if not os.path.exists(state_path):
        raise FileNotFoundError(state_path)
    state = torch.load(state_path)
    for key, expected in {
        "global_step": global_step,
        "loader_epoch": loader_epoch,
        "batch_offset": batch_offset,
        "rank": rank,
        "world_size": world_size,
    }.items():
        if state.get(key) != expected:
            raise ValueError(f"Resume RNG state mismatch for {key}: {state.get(key)} != {expected}")
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])
    np.random.set_state(state["numpy"])


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sudoku_dataset(data_dir, split):
    labels = np.load(os.path.join(data_dir, split, "all__labels.npy"))   # token space {2..10}
    x = labels.reshape(-1, 1, 9, 9).astype(np.int64)
    return TensorDataset(torch.from_numpy(x), torch.zeros(len(x), dtype=torch.long))


def seed_for_sampling(seed, step, rank, offset):
    value = seed + offset + step * 1000003 + rank
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)


@torch.no_grad()
def evaluate_sudoku(model, N, forward_type, device, rank, world_size, in_channels, img_size, num_samples=10000, batch_size=100, seed=0, step=0):
    model.eval()

    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        seed_for_sampling(seed, step, rank, 200000000)
        per_proc = (num_samples + world_size - 1) // world_size
        start = rank * per_proc
        end = min(start + per_proc, num_samples)
        local_size = end - start

        num_batches = (local_size + batch_size - 1) // batch_size
        local_valid = torch.zeros((), device=device)
        local_total = torch.zeros((), device=device)

        pbar = tqdm(range(num_batches), desc="Sudoku eval", disable=(rank != 0))
        for i in pbar:
            cur_size = min(batch_size, local_size - i * batch_size)
            if cur_size <= 0:
                break

            conds = torch.full((cur_size,), N, dtype=torch.long, device=device)
            init_noise = make_init_noise(forward_type, cur_size, in_channels, img_size, N, device)
            images = model.sample_with_image_sequence(init_noise, conds, stride=40)
            pred = images[-1][:, 0].reshape(cur_size, 81)  # already token space
            blank = torch.ones(cur_size, 81, dtype=torch.long, device=device)
            local_valid += batch_verify_sudoku(blank, pred).sum()
            local_total += cur_size

    dist.all_reduce(local_valid, op=dist.ReduceOp.SUM)
    dist.all_reduce(local_total, op=dist.ReduceOp.SUM)
    dist.barrier()
    return (local_valid / local_total.clamp_min(1)).item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume")
    parser.add_argument("--data-dir", type=str, default="data/sudoku-generation-50k")
    parser.add_argument("--max-steps", type=int, default=78000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-samples", type=int, default=10000)
    parser.add_argument("--run-name", type=str, default="d3pm_sudoku_generation")
    parser.add_argument("--checkpoint-path", type=str, default="checkpoints_sudoku")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--forward-type", type=str, default="absorb_zero")
    args = parser.parse_args()

    config = {
        "data": "sudoku",
        "data_dir": args.data_dir,
        "N": 11,  # PAD=0, blank=1, digits 1..9 = token 2..10
        "n_layers": args.n_layers,
        "n_heads": args.n_heads,
        "dim": args.dim,
        "forward_type": args.forward_type,
        "timesteps": 1000,
        "class_dropout_prob": 1.0,
        "lr": args.lr,
        "hybrid_loss_coeff": 0.0,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "save_interval": 10000,
        "eval_interval": 10000,
        "eval_samples": args.eval_samples,
        "run_name": args.run_name,
        "project_name": "v_trm_d3pm_sudoku",
        "checkpoint_path": args.checkpoint_path,
        "seed": args.seed
    }

    # Dataset-specific settings
    if config["data"] == "sudoku":
        in_channels = 1
        img_size = 9
    else:
        raise ValueError(f"Unknown data: {config['data']}")

    # DDP setup
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{local_rank}"
    is_main = (rank == 0)
    config["world_size"] = world_size
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])

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
        DiT_Sudoku(in_channels, N, input_size=img_size, class_dropout_prob=class_dropout_prob, n_layers=n_layers, n_heads=n_heads, dim=dim), timesteps, num_classes=N, hybrid_loss_coeff=hybrid_loss_coeff, forward_type=forward_type
    ).to(device)

    num_param = sum([p.numel() for p in d3pm.x0_model.parameters()])

    # Wrap with DDP
    d3pm = DDP(d3pm, device_ids=[local_rank])
    np.random.seed(config["seed"] + rank)
    torch.manual_seed(config["seed"] + rank)
    torch.cuda.manual_seed_all(config["seed"] + rank)

    # Dataset setup
    if config["data"] == "sudoku":
        dataset = load_sudoku_dataset(config["data_dir"], "train")
    config["train_examples"] = len(dataset)
    config["labels_sha256"] = file_sha256(os.path.join(config["data_dir"], "train", "all__labels.npy"))
    dist.barrier()

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=config["seed"])
    loader_generator = torch.Generator().manual_seed(config["seed"] + rank)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], sampler=sampler, num_workers=16, generator=loader_generator)
    optim = torch.optim.AdamW(d3pm.module.x0_model.parameters(), lr=learning_rate)
    d3pm.train()

    max_steps = config["max_steps"]
    save_interval = config["save_interval"]
    eval_interval = config["eval_interval"]
    eval_samples = config["eval_samples"]
    global_step = 0
    resume_loader_epoch = None
    resume_batch_offset = 0

    # Resume from checkpoint
    if args.resume and not os.path.exists(args.resume):
        raise FileNotFoundError(args.resume)
    if args.resume and os.path.exists(args.resume):
        global_step, resume_loader_epoch, resume_batch_offset = load_checkpoint(d3pm, optim, args.resume, device, config)
        load_rank_state(args.resume, rank, global_step, resume_loader_epoch, resume_batch_offset, world_size)
        if global_step >= max_steps:
            raise ValueError(f"Resume checkpoint step {global_step} is already >= max_steps {max_steps}")
        config["resumed_from"] = args.resume
        config["resumed_from_step"] = global_step
        config["run_name"] = f"{config['run_name']}_resume{global_step}_to{max_steps}"
        if is_main:
            print(f"Resumed from step {global_step}")

    if is_main:
        wandb.init(project=config["project_name"], name=config["run_name"])
        os.makedirs(config["checkpoint_path"], exist_ok=True)
        wandb.run.log_code(
            root=".",
            include_fn=lambda path: path.endswith((".py", "sh", ".yaml", ".yml", ".json", ".txt", ".md"))
        )
        wandb.config.update(config)
        print(f"Total Param Count: {num_param}")
        wandb.log({"num_params": num_param}, step=0)
    dist.barrier()

    last_loader_epoch = resume_loader_epoch if resume_loader_epoch is not None else global_step
    last_batch_offset = resume_batch_offset
    while global_step < max_steps:
        loader_epoch = resume_loader_epoch if resume_loader_epoch is not None else global_step
        sampler.set_epoch(loader_epoch)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), disable=not is_main)
        loss_ema = None
        for batch_idx, (x, cond) in pbar:
            if resume_loader_epoch is not None and batch_idx < resume_batch_offset:
                continue
            if global_step >= max_steps:
                break

            optim.zero_grad()
            x = x.to(device)
            cond = cond.to(device)

            x_cat = x.long()  # token space, no normalize
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
            last_loader_epoch = loader_epoch
            last_batch_offset = batch_idx + 1

            if is_main and global_step % 600 == 1:
                d3pm.module.eval()

                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    seed_for_sampling(config["seed"], global_step, rank, 100000000)
                    with torch.no_grad():
                        cond_sample = torch.full((16,), N, dtype=torch.long, device=device)
                        init_noise = make_init_noise(forward_type, 16, in_channels, img_size, N, device)

                        images = d3pm.module.sample_with_image_sequence(
                            init_noise, cond_sample, stride=40
                        )
                        # sudoku-format gif: 4 samples (2x2), conflicts in red, blank cells empty
                        gif = []
                        for image in images:
                            sample_boards = image[:4, 0].cpu().numpy().astype(np.int64)
                            gif.append(render_sample(sample_boards))

                        os.makedirs("contents", exist_ok=True)
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
                d3pm.train()

            # Evaluate Sudoku validity
            if global_step % eval_interval == 0:
                if is_main:
                    print(f"\nEvaluating Sudoku validity at step {global_step}...")
                sudoku_valid = evaluate_sudoku(
                    d3pm.module, N, forward_type, device, rank, world_size, in_channels, img_size, eval_samples, seed=config["seed"], step=global_step
                )
                if is_main:
                    print(f"Sudoku valid: {sudoku_valid:.4f}")
                    wandb.log(
                        {"val/sudoku_valid": sudoku_valid},
                        step=global_step,
                    )
                d3pm.train()

            # Save checkpoint
            if is_main and global_step % save_interval == 0:
                save_checkpoint(config, d3pm, optim, global_step, f"{config['checkpoint_path']}/ckpt_{global_step}.pt", loader_epoch, batch_idx + 1)
                save_checkpoint(config, d3pm, optim, global_step, f"{config['checkpoint_path']}/ckpt_latest.pt", loader_epoch, batch_idx + 1)
            if global_step % save_interval == 0:
                save_rank_state(f"{config['checkpoint_path']}/ckpt_{global_step}.pt", rank, global_step, loader_epoch, batch_idx + 1, world_size)
                save_rank_state(f"{config['checkpoint_path']}/ckpt_latest.pt", rank, global_step, loader_epoch, batch_idx + 1, world_size)
        resume_loader_epoch = None
        resume_batch_offset = 0

    # Final save
    final_eval_needed = global_step % eval_interval != 0
    if is_main and final_eval_needed:
        print(f"\nEvaluating Sudoku validity at final step {global_step}...")
    
    if final_eval_needed:
        sudoku_valid = evaluate_sudoku(
            d3pm.module, N, forward_type, device, rank, world_size, in_channels, img_size, eval_samples, seed=config["seed"], step=global_step
        )
        if is_main:
            print(f"Final Sudoku valid: {sudoku_valid:.4f}")
            wandb.log(
                {"val/sudoku_valid": sudoku_valid},
                step=global_step,
            )
        d3pm.train()

    if is_main:
        save_checkpoint(config, d3pm, optim, global_step, f"{config['checkpoint_path']}/ckpt_final_{global_step}.pt", last_loader_epoch, last_batch_offset)
        save_checkpoint(config, d3pm, optim, global_step, f"{config['checkpoint_path']}/ckpt_latest.pt", last_loader_epoch, last_batch_offset)
    save_rank_state(f"{config['checkpoint_path']}/ckpt_final_{global_step}.pt", rank, global_step, last_loader_epoch, last_batch_offset, world_size)
    save_rank_state(f"{config['checkpoint_path']}/ckpt_latest.pt", rank, global_step, last_loader_epoch, last_batch_offset, world_size)
    if is_main:
        wandb.finish()

    cleanup_ddp()
