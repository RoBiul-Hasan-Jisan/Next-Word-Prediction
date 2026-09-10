
import argparse
import json
import os
import pickle

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def build_sequences(text: str, num_words: int, max_seq_len: int):
    """Tokenize text and build overlapping n-gram training sequences."""
    # Split into lines/sentences so the model doesn't learn nonsense
    # transitions that jump across unrelated chunks of text.
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(lines)

    input_sequences = []
    for line in lines:
        token_list = tokenizer.texts_to_sequences([line])[0]
        # build every prefix >= 2 tokens: [w1,w2], [w1,w2,w3], ...
        for i in range(1, len(token_list)):
            n_gram_sequence = token_list[: i + 1]
            input_sequences.append(n_gram_sequence)

    # Cap sequence length so training stays fast; keep the END of long
    # sequences (most relevant recent context for predicting next word).
    input_sequences = [seq[-max_seq_len:] for seq in input_sequences]
    max_len = max(len(seq) for seq in input_sequences)
    padded = pad_sequences(input_sequences, maxlen=max_len, padding="pre")

    X = padded[:, :-1]
    y = padded[:, -1]
    return X, y, tokenizer, max_len


def build_model(vocab_size: int, seq_len: int, embedding_dim: int = 100):
    model = Sequential(
        [
            Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=seq_len),
            LSTM(150, return_sequences=True),
            Dropout(0.2),
            LSTM(100),
            Dropout(0.2),
            Dense(vocab_size, activation="softmax"),
        ]
    )
    # sparse_categorical_crossentropy keeps labels as plain integers (shape
    # [N]) instead of one-hot vectors (shape [N, vocab_size]), which avoids
    # allocating a huge one-hot array for large corpora/vocabularies.
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Train an LSTM next-word-prediction model.")
    parser.add_argument("--data", default="../data/corpus.txt", help="Path to a plain-text training corpus.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-words", type=int, default=8000, help="Max vocabulary size.")
    parser.add_argument("--max-seq-len", type=int, default=20, help="Max tokens of context kept per training sample.")
    parser.add_argument("--embedding-dim", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"Loading corpus from {args.data} ...")
    text = load_text(args.data)
    print(f"Corpus length: {len(text)} characters")

    print("Building training sequences (this can take a minute for large corpora)...")
    X, y, tokenizer, max_len = build_sequences(text, args.num_words, args.max_seq_len)
    vocab_size = min(args.num_words, len(tokenizer.word_index) + 1) + 1
    print(f"Training samples: {X.shape[0]}, sequence length: {max_len}, vocab size: {vocab_size}")

    model = build_model(vocab_size, X.shape[1], args.embedding_dim)
    model.summary()

    callbacks = [
        EarlyStopping(monitor="loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(os.path.join(MODEL_DIR, "lstm_model.h5"), monitor="loss", save_best_only=True),
    ]

    model.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=1)

    model.save(os.path.join(MODEL_DIR, "lstm_model.h5"))
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)
    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(
            {
                "max_len": max_len,
                "vocab_size": vocab_size,
                "num_words": args.num_words,
                "embedding_dim": args.embedding_dim,
            },
            f,
            indent=2,
        )

    print(f"\nDone. Model + tokenizer + config saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
