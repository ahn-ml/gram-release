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
    with torch.inference_mode():
        for i in range(0, num_samples, batch_size):
            cur = min(batch_size, num_samples - i)
            init_noise = make_init_noise(forward_type, cur, in_channels, img_size, N, device)
            conds = torch.full((cur,), N, dtype=torch.long, device=device)
            seq = model.sample_with_image_sequence(init_noise, conds, stride=40)
            boards.append(seq[-1][:, 0].reshape(cur, 81).cpu())
    return torch.cat(boards, dim=0)


def gen_trm(ckpt_dir, num_samples, batch_size, device, temperature=0.0):
    from pretrain import PretrainConfig  # noqa: F401  (used by checkpoint pickle)
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
    with torch.inference_mode():
        for i in range(0, num_samples, batch_size):
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
    args = p.parse_args()

    if args.num_samples <= 0 or args.batch_size <= 0:
        p.error("--num_samples and --batch_size must be positive")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "d3pm":
        boards = gen_d3pm(args.checkpoint, args.num_samples, args.batch_size, device)
    else:
        boards = gen_trm(args.checkpoint, args.num_samples, args.batch_size, device,
                         temperature=args.temperature)  # (N, 81) long

    # accuracy: verify against all-blank inputs (token 1 = blank)
    blanks = torch.ones_like(boards)
    valid = batch_verify_sudoku(blanks, boards).cpu().numpy()
    accuracy = float(valid.mean())

    arr = boards.numpy()
    valid_idx = np.where(valid)[0]
    unique_answers = len({arr[i].tobytes() for i in valid_idx}) if len(valid_idx) else 0

    print(f"model_type:      {args.model_type}")
    print(f"checkpoint:      {args.checkpoint}")
    print(f"num_samples:     {args.num_samples}")
    print(f"accuracy:        {accuracy:.4f}  ({int(valid.sum())}/{len(valid)})")
    if len(valid_idx):
        print(f"unique_answers:  {unique_answers/len(valid_idx):.4f}  ({unique_answers}/{len(valid_idx)})")
    else:
        print(f"unique_answers:  n/a (no valid boards)")


if __name__ == "__main__":
    main()
