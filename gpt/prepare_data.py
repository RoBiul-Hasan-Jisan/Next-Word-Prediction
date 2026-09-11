import argparse
import os

import numpy as np
from tokenizers import Tokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/corpus_large.txt")
    args = parser.parse_args()

    tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.json")
    tokenizer = Tokenizer.from_file(tokenizer_path)

    with open(args.data, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    print(f"Tokenizing {len(text)} characters ...")
    ids = tokenizer.encode(text).ids
    ids = np.array(ids, dtype=np.uint16)
    print(f"Total tokens: {len(ids)}")

    n = len(ids)
    n_train = int(n * 0.90)
    n_val = int(n * 0.05)

    train_ids = ids[:n_train]
    val_ids = ids[n_train : n_train + n_val]
    test_ids = ids[n_train + n_val :]

    train_ids.tofile(os.path.join(MODEL_DIR, "train.bin"))
    val_ids.tofile(os.path.join(MODEL_DIR, "val.bin"))
    test_ids.tofile(os.path.join(MODEL_DIR, "test.bin"))

    print(f"train: {len(train_ids)} tokens, val: {len(val_ids)}, test: {len(test_ids)}")
    print(f"Saved .bin files to {MODEL_DIR}")


if __name__ == "__main__":
    main()
