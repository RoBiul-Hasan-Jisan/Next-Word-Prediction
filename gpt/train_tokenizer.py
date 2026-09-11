
import argparse
import os

from tokenizers import Tokenizer, decoders, pre_tokenizers, trainers
from tokenizers.models import BPE

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/corpus_large.txt")
    parser.add_argument("--vocab-size", type=int, default=6000)
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
        show_progress=True,
    )

    print(f"Training BPE tokenizer (vocab_size={args.vocab_size}) on {args.data} ...")
    tokenizer.train([args.data], trainer)

    out_path = os.path.join(MODEL_DIR, "tokenizer.json")
    tokenizer.save(out_path)
    print(f"Saved tokenizer to {out_path}. Actual vocab size: {tokenizer.get_vocab_size()}")


if __name__ == "__main__":
    main()
