import json
import os

import torch

from evaluate import evaluate, load_split
from model import GPT, GPTConfig, DequantizingEmbedding

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
FP32_PATH = os.path.join(MODEL_DIR, "gpt_model.pt")
INT8_PATH = os.path.join(MODEL_DIR, "gpt_model_int8.pt")

# If top-5 accuracy drops by more than this many percentage points after
# quantization, keep shipping fp32 instead. This is the gate mentioned in
# the project README -- a decision made by measurement, not by default.
TOP5_DROP_GATE_PP = 1.0


def model_size_mb(state_dict):
    total_bytes = 0
    for v in state_dict.values():
        if torch.is_tensor(v):
            total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024 * 1024)


def main():
    with open(os.path.join(MODEL_DIR, "gpt_config.json")) as f:
        cfg = GPTConfig(**json.load(f))

    model = GPT(cfg)
    ckpt = torch.load(FP32_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Dynamic int8 quantization of every Linear layer (attention + MLP
    # projections). This is the "matmul" quantization variant -- weights
    # for the big matrix multiplies are stored as int8, activations are
    # quantized on the fly at inference time. The tied embedding/lm_head
    # is left in fp32 (quantizing it barely changes size and QAT-free
    # embedding quantization tends to hurt quality more than it saves).
    q_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    val_data = load_split("val")
    test_data = load_split("test")

    print("Evaluating fp32 ...")
    fp32_val = evaluate(model, val_data, cfg.block_size, batch_size=24, n_batches=40)
    fp32_test = evaluate(model, test_data, cfg.block_size, batch_size=24, n_batches=40)

    print("Evaluating int8 (matmul only) ...")
    int8_val = evaluate(q_model, val_data, cfg.block_size, batch_size=24, n_batches=40)
    int8_test = evaluate(q_model, test_data, cfg.block_size, batch_size=24, n_batches=40)

    fp32_deploy_path = os.path.join(MODEL_DIR, "gpt_model_fp32.pt")
    torch.save({"model": model.state_dict(), "config": ckpt.get("config")}, fp32_deploy_path)
    fp32_size = os.path.getsize(fp32_deploy_path) / (1024 * 1024)

    torch.save({"model": q_model.state_dict(), "config": ckpt.get("config")}, INT8_PATH)
    int8_size = os.path.getsize(INT8_PATH) / (1024 * 1024)

    # -- Variant 2: matmul + embedding quantization --------------------
    # torch.quantization.quantize_dynamic only touches nn.Linear layers,
    # so the tied embedding table (the single largest tensor at this
    # model size) stays fp32 above. To measure the "matmul+embedding"
    # variant, symmetric per-tensor int8 quantization is applied to the
    # embedding weight by hand and the model is monkey-patched to
    # dequantize on every lookup -- exactly the trade this project's
    # README describes: does quantizing the *embedding too* actually
    # cost accuracy, or just save space?
    emb_weight = model.tok_emb.weight.data
    quantized_emb = DequantizingEmbedding.quantize(emb_weight)

    q_model_full = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    q_model_full.tok_emb = quantized_emb

    print("Evaluating int8 (matmul + embedding) ...")
    int8f_val = evaluate(q_model_full, val_data, cfg.block_size, batch_size=24, n_batches=40)
    int8f_test = evaluate(q_model_full, test_data, cfg.block_size, batch_size=24, n_batches=40)
    int8f_size = int8_size - (emb_weight.numel() * 4 / (1024 * 1024)) + (emb_weight.numel() * 1 / (1024 * 1024))

    INT8_FULL_PATH = os.path.join(MODEL_DIR, "gpt_model_int8_full.pt")
    torch.save(
        {
            "linear_state_dict": q_model_full.state_dict(),
            "emb_int8_weight": quantized_emb.int8_weight,
            "emb_scale": quantized_emb.scale,
            "config": ckpt.get("config"),
        },
        INT8_FULL_PATH,
    )

    top5_drop_pp = (fp32_test["top5_acc"] - int8_test["top5_acc"]) * 100
    top5_drop_pp_full = (fp32_test["top5_acc"] - int8f_test["top5_acc"]) * 100

    if top5_drop_pp_full <= TOP5_DROP_GATE_PP:
        decision = "int8 (matmul+embedding)"
    elif top5_drop_pp <= TOP5_DROP_GATE_PP:
        decision = "int8 (matmul only)"
    else:
        decision = "fp32"

    print("\n" + "=" * 66)
    print(f"{'variant':<24}{'size (MB)':<12}{'test ppl':<10}{'test top-5':<12}{'\u0394top-5':<8}")
    print(f"{'fp32':<24}{fp32_size:<12.1f}{fp32_test['perplexity']:<10.2f}{fp32_test['top5_acc']*100:<12.1f}{'-':<8}")
    print(
        f"{'int8 (matmul)':<24}{int8_size:<12.1f}{int8_test['perplexity']:<10.2f}"
        f"{int8_test['top5_acc']*100:<12.1f}{-top5_drop_pp:<8.2f}"
    )
    print(
        f"{'int8 (matmul+embedding)':<24}{int8f_size:<12.1f}{int8f_test['perplexity']:<10.2f}"
        f"{int8f_test['top5_acc']*100:<12.1f}{-top5_drop_pp_full:<8.2f}"
    )
    print("=" * 66)
    print(f"gate: ship the smallest variant whose top-5 drop is <= {TOP5_DROP_GATE_PP}pp")
    print(f"DECISION: ship {decision} (see model/quantization_report.json)")

    report = {
        "fp32": {"size_mb": fp32_size, **fp32_test, "val_perplexity": fp32_val["perplexity"]},
        "int8_matmul": {"size_mb": int8_size, **int8_test, "val_perplexity": int8_val["perplexity"]},
        "int8_matmul_embedding": {"size_mb": int8f_size, **int8f_test, "val_perplexity": int8f_val["perplexity"]},
        "top5_drop_pp_matmul": top5_drop_pp,
        "top5_drop_pp_matmul_embedding": top5_drop_pp_full,
        "gate_pp": TOP5_DROP_GATE_PP,
        "decision": decision,
    }
    with open(os.path.join(MODEL_DIR, "quantization_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    shipped_file = {
        "int8 (matmul+embedding)": "gpt_model_int8_full.pt",
        "int8 (matmul only)": "gpt_model_int8.pt",
        "fp32": "gpt_model_fp32.pt",
    }[decision]
    with open(os.path.join(MODEL_DIR, "shipped_variant.json"), "w") as f:
        json.dump({"decision": decision, "file": shipped_file}, f, indent=2)
    print(f"\nWrote model/shipped_variant.json -> predict.py will load {shipped_file} by default.")


if __name__ == "__main__":
    main()
