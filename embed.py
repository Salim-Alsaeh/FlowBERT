#!/usr/bin/env python3
import os
import argparse
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from src.masked_lm import FlowBERTForMaskedLM, FlowBERTConfig

class FlowDataset(Dataset):
    def __init__(self, token_file, tokenizer, seq_len=32):
        with open(token_file, "r") as f:
            self.lines = f.read().splitlines()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_id = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.lines[idx]).ids
        if len(ids) > self.seq_len:
            ids = ids[:self.seq_len]
        else:
            ids += [self.pad_id] * (self.seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser(description="Extract 128-D CLS embeddings with FlowBERT")
    parser.add_argument("--tokens",     required=True, help="Path to tokenised text file")
    parser.add_argument("--vocab",      required=True, help="Path to tokenizer JSON")
    parser.add_argument("--ckpt",       required=True, help="Path to flowbert.ckpt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len",    type=int, default=32)
    parser.add_argument("--output",     required=True, help="Path prefix for saving embeddings (.npz)")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f">>> Loading model on {device}")

    # Load tokenizer and model
    tokenizer = Tokenizer.from_file(args.vocab)
    cfg = FlowBERTConfig(vocab_size=tokenizer.get_vocab_size())
    model = FlowBERTForMaskedLM(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # Prepare dataset & loader
    ds = FlowDataset(args.tokens, tokenizer, seq_len=args.seq_len)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # Collect CLS embeddings
    embs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)             # (batch_size, seq_len)
            # Token + position embeddings
            x = model.token_embeddings(batch)
            pos_ids = torch.arange(args.seq_len, device=device) \
                        .unsqueeze(0).expand(batch.size(0), -1)
            x = x + model.position_embeddings(pos_ids)
            # Encoder forward
            enc = model.encoder(x.transpose(0,1)).transpose(0,1)
            # Extract CLS (position 0)
            cls_emb = enc[:, 0, :].cpu().numpy()
            embs.append(cls_emb)

    embeddings = np.vstack(embs)  # shape (N, hidden_size)
    output_path = args.output if args.output.endswith(".npz") else args.output + ".npz"
    np.savez_compressed(output_path, embeddings=embeddings)
    print(f"✔ Saved embeddings to {output_path}, shape={embeddings.shape}")

if __name__ == "__main__":
    main()
