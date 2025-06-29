#!/usr/bin/env python3
import argparse
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True, help="Tokenised text file")
    p.add_argument("--output",     required=True, help="Path to save vocab (WordPiece JSON or txt)")
    p.add_argument("--vocab_size", type=int, default=30000)
    args = p.parse_args()

    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["[PAD]","[UNK]","[CLS]","[SEP]","[MASK]"]
    )
    tokenizer.train([args.input], trainer)
    tokenizer.save(args.output)
    print(f"Vocabulary saved to {args.output}")

if __name__ == "__main__":
    main()