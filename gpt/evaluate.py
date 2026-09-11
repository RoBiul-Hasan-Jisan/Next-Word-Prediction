import argparse
import json
import math
import os

import numpy as np
import torch

from model import GPT, GPTConfig

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def load_split(name):
    path = os.path.join(MODEL_DIR, f"{name}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")


@torch.no_grad()
def evaluate(model, data, block_size, batch_size, n_batches, device="cpu"):
    model.eval()
    total_loss, total_top1, total_top3, total_top5, total_tokens = 0.0, 0, 0, 0, 0

    torch.manual_seed(42)  # fixed seed -> reproducible eval batches across runs
    for _ in range(n_batches):
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]).to(device)

        logits, loss = model(x, y)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

        topk = logits.topk(5, dim=-1).indices  # [B, T, 5]
        y_exp = y.unsqueeze(-1)
        hits = (topk == y_exp)
        total_top1 += hits[..., :1].any(dim=-1).sum().item()
        total_top3 += hits[..., :3].any(dim=-1).sum().item()
        total_top5 += hits[..., :5].any(dim=-1).sum().item()

    avg_loss = total_loss / total_tokens
    return {
        "loss": avg_loss,
        "perplexity": math.exp(avg_loss),
        "top1_acc": total_top1 / total_tokens,
        "top3_acc": total_top3 / total_tokens,
        "top5_acc": total_top5 / total_tokens,
        "n_tokens_evaluated": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default=os.path.join(MODEL_DIR, "gpt_model.pt"))
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--n-batches", type=int, default=40)
    args = parser.parse_args()

    with open(os.path.join(MODEL_DIR, "gpt_config.json")) as f:
        cfg = GPTConfig(**json.load(f))

    model = GPT(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    data = load_split(args.split)
    metrics = evaluate(model, data, cfg.block_size, args.batch_size, args.n_batches)

    print(f"\n== {args.split} split ({metrics['n_tokens_evaluated']:,} tokens evaluated) ==")
    print(f"loss        {metrics['loss']:.4f}")
    print(f"perplexity  {metrics['perplexity']:.2f}")
    print(f"top-1 acc   {metrics['top1_acc']*100:.1f}%")
    print(f"top-3 acc   {metrics['top3_acc']*100:.1f}%")
    print(f"top-5 acc   {metrics['top5_acc']*100:.1f}%")


if __name__ == "__main__":
    main()
