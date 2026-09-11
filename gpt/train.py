
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from model import GPT, GPTConfig

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
CKPT_PATH = os.path.join(MODEL_DIR, "gpt_model.pt")


def load_split(name):
    path = os.path.join(MODEL_DIR, f"{name}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def lr_schedule(step, total_steps, base_lr, warmup_steps=200, min_lr_ratio=0.1):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio * base_lr + coeff * (1 - min_lr_ratio) * base_lr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500, help="Additional steps to train this run.")
    parser.add_argument("--total-steps", type=int, default=8000, help="Total steps the LR schedule is planned over.")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=250)
    args = parser.parse_args()

    device = "cpu"
    torch.manual_seed(1337)

    train_data = load_split("train")
    val_data = load_split("val")

    with open(os.path.join(MODEL_DIR, "gpt_config.json")) as f:
        cfg_dict = json.load(f)
    cfg = GPTConfig(**cfg_dict)

    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )

    start_step = 0
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"Resumed from checkpoint at step {start_step}")
    else:
        print(f"Starting fresh. Non-embedding params: {model.num_params():,} | total: {model.num_params(False):,}")

    model.train()
    t0 = time.time()
    target_step = start_step + args.steps

    for step in range(start_step, target_step):
        lr = lr_schedule(step, args.total_steps, args.lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, cfg.block_size, args.batch_size, device)
        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if (step + 1) % args.log_every == 0 or step == start_step:
            elapsed = time.time() - t0
            print(f"step {step+1}/{target_step} | loss {loss.item():.4f} | lr {lr:.2e} | {elapsed:.1f}s elapsed")

        if (step + 1) % args.eval_every == 0 or step + 1 == target_step:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, cfg.block_size, args.batch_size, device)
                _, vloss = model(vx, vy)
            print(f"  [val] step {step+1} | val loss {vloss.item():.4f} | val ppl {math.exp(vloss.item()):.2f}")
            model.train()

    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": target_step, "config": cfg_dict},
        CKPT_PATH,
    )
    print(f"Saved checkpoint at step {target_step} -> {CKPT_PATH}")


if __name__ == "__main__":
    main()
