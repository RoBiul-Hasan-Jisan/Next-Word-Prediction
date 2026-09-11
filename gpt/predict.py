import json
import os
import sys

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import GPT, GPTConfig, DequantizingEmbedding

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
SPACE_MARKER = "\u0120"  # 'Ġ' -- how byte-level BPE represents a leading space
MAX_EXTRA_TOKENS = 4     # cap on how many extra sub-word pieces to assemble per word


class GPTPredictor:
    def __init__(self, model_dir: str = MODEL_DIR, checkpoint: str = None):
        with open(os.path.join(model_dir, "gpt_config.json")) as f:
            self.cfg = GPTConfig(**json.load(f))

        checkpoint_path, variant = self._resolve_checkpoint(model_dir, checkpoint)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"No trained checkpoint at {checkpoint_path}. Run train_tokenizer.py, "
                "prepare_data.py, and train.py (and optionally quantize.py) first."
            )

        self.model = GPT(self.cfg)
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        if variant == "int8_full":
            # Linear layers were dynamically quantized; the embedding was
            # quantized by hand -- reconstruct both pieces, embedding
            # swap first so its keys match before load_state_dict runs.
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            self.model.tok_emb = DequantizingEmbedding(ckpt["emb_int8_weight"], ckpt["emb_scale"])
            self.model.load_state_dict(ckpt["linear_state_dict"])
        elif variant == "int8_matmul":
            self.model = torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)
            self.model.load_state_dict(ckpt["model"])
        else:
            self.model.load_state_dict(ckpt["model"])

        self.model.eval()
        self.variant = variant
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))

    @staticmethod
    def _resolve_checkpoint(model_dir: str, checkpoint: str):
        """Pick a checkpoint: an explicit path, or whatever quantize.py decided to ship,
        or the plain fp32 checkpoint if quantize.py hasn't been run yet."""
        if checkpoint:
            name = os.path.basename(checkpoint)
            variant = "int8_full" if "int8_full" in name else "int8_matmul" if "int8" in name else "fp32"
            return checkpoint, variant

        decision_path = os.path.join(model_dir, "shipped_variant.json")
        if os.path.exists(decision_path):
            with open(decision_path) as f:
                decision = json.load(f)
            name = decision["file"]
            variant = "int8_full" if "int8_full" in name else "int8_matmul" if "int8" in name else "fp32"
            return os.path.join(model_dir, name), variant

        return os.path.join(model_dir, "gpt_model.pt"), "fp32"

    def _token_str(self, token_id: int) -> str:
        return self.tokenizer.id_to_token(token_id) or ""

    def _decode_word(self, token_ids):
        text = self.tokenizer.decode(token_ids)
        return text.strip()

    @torch.no_grad()
    def predict(self, text: str, top_k: int = 5, first_token_candidates: int = 20):
        ids = self.tokenizer.encode(text).ids
        if not ids:
            return []
        context = torch.tensor([ids], dtype=torch.long)

        logits = self.model.next_token_logits(context)[0]
        probs = F.log_softmax(logits, dim=-1)
        top = torch.topk(probs, first_token_candidates)

        candidates = {}  # word -> best log-prob score
        for first_id, first_logp in zip(top.indices.tolist(), top.values.tolist()):
            first_str = self._token_str(first_id)
            if not first_str or first_str.startswith("<|"):
                continue

            token_path = [first_id]
            logp_sum = first_logp

            # If this token doesn't already start a new word, it can't be
            # offered as "the next word" on its own -- skip it. (Given the
            # UI's assumption that input ends at a word boundary, the vast
            # majority of top candidates already start with the space
            # marker, so this mostly filters mid-word continuation noise.)
            if not first_str.startswith(SPACE_MARKER):
                continue

            # Greedily extend until the next token would start a *new*
            # word, or we hit the extension cap.
            cur_context = torch.cat([context, torch.tensor([token_path])], dim=1)
            for _ in range(MAX_EXTRA_TOKENS):
                word_so_far = self._decode_word(token_path)
                if word_so_far and not word_so_far.isalpha():
                    break  # hit punctuation / non-word content, stop extending

                next_logits = self.model.next_token_logits(cur_context)[0]
                next_logp = F.log_softmax(next_logits, dim=-1)
                next_id = int(torch.argmax(next_logp).item())
                next_str = self._token_str(next_id)

                if not next_str or next_str.startswith(SPACE_MARKER) or next_str.startswith("<|"):
                    break  # next token starts a new word -> current word is complete

                token_path.append(next_id)
                logp_sum += next_logp[next_id].item()
                cur_context = torch.cat([cur_context, torch.tensor([[next_id]])], dim=1)

            word = self._decode_word(token_path)
            if not word or not word.replace("'", "").isalpha():
                continue

            word_lower = word.lower()
            if word_lower not in candidates or logp_sum > candidates[word_lower][0]:
                candidates[word_lower] = (logp_sum, word)

        ranked = sorted(candidates.values(), key=lambda x: x[0], reverse=True)[:top_k]
        results = [{"word": w, "probability": float(torch.exp(torch.tensor(lp)))} for lp, w in ranked]

        # Drop noisy low-confidence tail candidates (e.g. odd assembled
        # sub-word remnants) rather than padding out to top_k regardless.
        if results:
            floor = 0.05 * results[0]["probability"]
            results = [r for r in results if r["probability"] >= floor]
        return results


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "the weather today is"
    predictor = GPTPredictor()
    for r in predictor.predict(text, top_k=5):
        print(f"{r['word']:<15} {r['probability']:.4f}")
