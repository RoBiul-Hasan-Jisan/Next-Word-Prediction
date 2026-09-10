
import string
import sys

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"


class TransformerPredictor:
    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).eval()

    def predict(self, text: str, top_k: int = 5):
        """Return the top_k predicted next words for the given input text."""
        text = text.strip()
        masked_text = f"{text} {self.tokenizer.mask_token}"

        inputs = self.tokenizer(masked_text, return_tensors="pt")
        mask_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
        if len(mask_index) == 0:
            return []
        mask_index = mask_index.item()

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits[0, mask_index], dim=-1)
        top = torch.topk(probs, top_k * 3)  # over-fetch, then filter punctuation/subwords

        results = []
        for score, idx in zip(top.values.tolist(), top.indices.tolist()):
            token = self.tokenizer.decode([idx]).strip()
            if not token or token in string.punctuation or token.startswith("##"):
                continue
            if not token.isalpha():
                continue
            results.append({"word": token, "probability": float(score)})
            if len(results) == top_k:
                break
        return results


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "The weather today is"
    predictor = TransformerPredictor()
    for r in predictor.predict(text, top_k=5):
        print(f"{r['word']:<15} {r['probability']:.4f}")
