"""Unconditioned sudoku generation eval for D3PM and TRM checkpoints.

Generates N boards from a trained model and reports:
  - accuracy:        fraction of generated boards that are valid sudokus
  - unique_answers:  unique boards / total valid boards  (diversity among correct answers)

Usage:
  python sudoku_gen_eval.py --model_type d3pm \
      --checkpoint checkpoints_sudoku/<run>/ckpt_470400.pt --num_samples 10000

  python sudoku_gen_eval.py --model_type trm \
      --checkpoint checkpoints/Sudoku-generation-50k-ACT-torch/<run>/step_78120 \
      --num_samples 10000
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from evaluators.sudoku_utils import batch_verify_sudoku


def gen_d3pm(ckpt_path, num_samples, batch_size, device):
    # d3pm/d3pm_runner_sudoku.py uses absolute imports (`from d3pm_runner ...`),
    # so the d3pm/ directory must be on sys.path.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "d3pm"))
    from d3pm_runner import D3PM
    from d3pm_runner_sudoku import DiT_Sudoku, make_init_noise

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    N = cfg["N"]
    in_channels, img_size = 1, 9
    forward_type = cfg["forward_type"]

    model = D3PM(
        DiT_Sudoku(in_channels, N, input_size=img_size,
                   class_dropout_prob=cfg["class_dropout_prob"],
                   n_layers=cfg["n_layers"], n_heads=cfg["n_heads"], dim=cfg["dim"]),
        cfg["timesteps"], num_classes=N,
        hybrid_loss_coeff=cfg["hybrid_loss_coeff"], forward_type=forward_type,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    boards = []
    disable_tqdm = os.environ.get("LOCAL_RANK", "0") != "0"
    with torch.inference_mode():
        for i in tqdm(range(0, num_samples, batch_size), desc="d3pm gen", disable=disable_tqdm):
            cur = min(batch_size, num_samples - i)
            init_noise = make_init_noise(forward_type, cur, in_channels, img_size, N, device)
            conds = torch.full((cur,), N, dtype=torch.long, device=device)
            seq = model.sample_with_image_sequence(init_noise, conds, stride=40)
            boards.append(seq[-1][:, 0].reshape(cur, 81).cpu())
    return torch.cat(boards, dim=0)


def gen_trm(ckpt_dir, num_samples, batch_size, device, temperature=0.0):
    import __main__
    import pretrain  # checkpoint was pickled when pretrain.py was __main__
    for name in ("PretrainConfig", "ArchConfig", "LossConfig", "EvaluatorConfig"):
        if hasattr(pretrain, name):
            setattr(__main__, name, getattr(pretrain, name))
    from utils.functions import load_model_class
    from evaluators.sampling_strategies import sample_parallel

    ckpt = torch.load(os.path.join(ckpt_dir, "checkpoint.pt"),
                      map_location=device, weights_only=False)
    config = ckpt["config"]

    seq_len, vocab_size = 81, 11
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=batch_size,
        vocab_size=vocab_size,
        seq_len=seq_len,
        num_puzzle_identifiers=1,
        causal=False,
        temperature=temperature,
    )
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    with torch.device(device):
        model = model_cls(model_cfg)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)

    sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, assign=True)
    if "ema" in ckpt:
        from models.ema import EMAHelper
        helper = EMAHelper(mu=0.9999)
        helper.register(model)
        helper.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ckpt["ema"].items()})
        helper.ema(model)
    model.eval().to(device)

    boards = []
    disable_tqdm = os.environ.get("LOCAL_RANK", "0") != "0"
    with torch.inference_mode():
        for i in tqdm(range(0, num_samples, batch_size), desc="trm gen", disable=disable_tqdm):
            cur = min(batch_size, num_samples - i)
            batch = {
                "inputs": torch.ones(cur, seq_len, dtype=torch.long, device=device),
                "labels": torch.full((cur, seq_len), -100, dtype=torch.long, device=device),
                "puzzle_identifiers": torch.zeros(cur, dtype=torch.long, device=device),
            }
            samples = sample_parallel(
                model, batch, num_samples=1,
                return_keys={"preds", "q_halt_logits"},
                device=device, extract_h_states=False, data_path="sudoku",
            )
            boards.append(samples[0]["preds"].cpu())
    return torch.cat(boards, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["d3pm", "trm"], required=True)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="d3pm: ckpt .pt file. trm: dir containing checkpoint.pt")
    p.add_argument("--num_samples", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="TRM only: >0 enables multinomial sampling for diversity")
    p.add_argument("--save_viz", type=str, default=None,
                   help="If set, save correct/incorrect sample visualizations under this dir")
    p.add_argument("--viz_n", type=int, default=16, help="boards per viz panel (4x4 default)")
    args = p.parse_args()

    if args.num_samples <= 0 or args.batch_size <= 0:
        p.error("--num_samples and --batch_size must be positive")

    # Optional DDP: torchrun --nproc-per-node N sudoku_gen_eval.py ...
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    else:
        rank, world = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)

    per_rank = (args.num_samples + world - 1) // world
    local_n = max(0, min(per_rank, args.num_samples - rank * per_rank))

    if local_n > 0:
        if args.model_type == "d3pm":
            local_boards = gen_d3pm(args.checkpoint, local_n, args.batch_size, device)
        else:
            local_boards = gen_trm(args.checkpoint, local_n, args.batch_size, device,
                                   temperature=args.temperature)
    else:
        local_boards = torch.zeros(0, 81, dtype=torch.long)

    if is_ddp:
        # Pad to per_rank for all_gather (requires equal sizes), then trim on rank 0
        if local_boards.shape[0] < per_rank:
            pad = torch.zeros(per_rank - local_boards.shape[0], 81, dtype=local_boards.dtype)
            local_boards = torch.cat([local_boards, pad], dim=0)
        local_gpu = local_boards.to(device)
        gathered = [torch.zeros_like(local_gpu) for _ in range(world)]
        dist.all_gather(gathered, local_gpu)
        if rank != 0:
            dist.destroy_process_group()
            return
        boards = torch.cat(gathered, dim=0).cpu()[:args.num_samples]
        dist.destroy_process_group()
    else:
        boards = local_boards

    # accuracy: verify against all-blank inputs (token 1 = blank)
    blanks = torch.ones_like(boards)
    valid = batch_verify_sudoku(blanks, boards).cpu().numpy()
    accuracy = float(valid.mean())

    arr = boards.numpy()
    valid_idx = np.where(valid)[0]
    unique_answers = len({arr[i].tobytes() for i in valid_idx}) if len(valid_idx) else 0

    # --- Symmetry-aware unique (digit-relabel canonical, drop-in module) ---
    from sudoku_canon import canon_digits_batch
    canon_unique = (len({b.tobytes() for b in canon_digits_batch(arr[valid_idx])})
                    if len(valid_idx) else 0)

    print(f"model_type:      {args.model_type}")
    print(f"checkpoint:      {args.checkpoint}")
    print(f"num_samples:     {args.num_samples}")
    print(f"accuracy:        {accuracy:.4f}  ({int(valid.sum())}/{len(valid)})")
    if len(valid_idx):
        print(f"unique_answers:  {unique_answers/len(valid_idx):.4f}  ({unique_answers}/{len(valid_idx)})")
        print(f"unique_canon:    {canon_unique/len(valid_idx):.4f}  ({canon_unique}/{len(valid_idx)})  [digit-relabel]")
    else:
        print(f"unique_answers:  n/a (no valid boards)")

    if args.save_viz:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "d3pm"))
        from d3pm_runner_sudoku import render_sudoku_grid, sudoku_conflicts
        os.makedirs(args.save_viz, exist_ok=True)
        n = args.viz_n
        side = int(np.ceil(np.sqrt(n)))
        invalid_idx = np.where(~valid)[0]
        for label, idx in [("correct", valid_idx), ("incorrect", invalid_idx)]:
            if len(idx) == 0:
                continue
            picks = idx[:n]
            boards_2d = arr[picks].reshape(-1, 9, 9)
            confs = [sudoku_conflicts(b) for b in boards_2d]
            img = render_sudoku_grid(boards_2d, confs,
                                     title=f"{label} samples ({len(picks)}/{n})",
                                     rows=side, cols=side)
            out = os.path.join(args.save_viz, f"{args.model_type}_{label}.png")
            img.save(out)
            print(f"viz:           saved {out}")


if __name__ == "__main__":
    main()
