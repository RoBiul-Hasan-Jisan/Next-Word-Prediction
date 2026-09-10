import json
import os
import pickle
import sys

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


class LSTMPredictor:
    def __init__(self, model_dir: str = MODEL_DIR):
        model_path = os.path.join(model_dir, "lstm_model.h5")
        tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
        config_path = os.path.join(model_dir, "config.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                "Run `python train_lstm.py` first to train and save a model."
            )

        self.model = load_model(model_path)
        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)
        with open(config_path) as f:
            self.config = json.load(f)

        # index -> word lookup, built once
        self.index_word = {idx: w for w, idx in self.tokenizer.word_index.items()}

    def predict(self, text: str, top_k: int = 5):
        """Return the top_k predicted next words for the given input text."""
        seq_len = self.config["max_len"] - 1
        token_list = self.tokenizer.texts_to_sequences([text])[0]
        token_list = pad_sequences([token_list], maxlen=seq_len, padding="pre")

        preds = self.model.predict(token_list, verbose=0)[0]
        top_indices = np.argsort(preds)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            word = self.index_word.get(idx, "")
            if word and word != "<OOV>":
                results.append({"word": word, "probability": float(preds[idx])})
        return results


if __name__ == "__main__":
    text = " ".join(sys.argv[1:]) or "to be or not to"
    predictor = LSTMPredictor()
    for r in predictor.predict(text, top_k=5):
        print(f"{r['word']:<15} {r['probability']:.4f}")
