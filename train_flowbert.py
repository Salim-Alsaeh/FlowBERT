#!/usr/bin/env python3
import os
import argparse
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from tokenizers import Tokenizer
from src.masked_lm import FlowBERTForMaskedLM, FlowBERTConfig

# Enable MPS fallback on macOS M1 if needed
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

class FlowDataset(Dataset):
    def __init__(self, token_file, tokenizer, seq_len=128):
        with open(token_file, "r") as f:
            self.lines = f.read().splitlines()
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_id = tokenizer.token_to_id("[PAD]")
        self.mask_id = tokenizer.token_to_id("[MASK]")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        ids = self.tokenizer.encode(self.lines[idx]).ids
        if len(ids) > self.seq_len:
            ids = ids[:self.seq_len]
        else:
            ids += [self.pad_id] * (self.seq_len - len(ids))
        input_ids = torch.tensor(ids)
        labels = input_ids.clone()
        # 15% Masking
        rand = torch.rand(input_ids.shape)
        mask_arr = (rand < 0.15) & (input_ids != self.pad_id)
        input_ids[mask_arr] = self.mask_id
        attention_mask = (input_ids != self.pad_id).long()
        return input_ids, attention_mask, labels

def train(args):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f">>> Training on device: {device}")

    tokenizer = Tokenizer.from_file(args.vocab)
    model = FlowBERTForMaskedLM(FlowBERTConfig(vocab_size=tokenizer.get_vocab_size())).to(device)

    ds = FlowDataset(args.tokens, tokenizer, seq_len=args.seq_len)
    total_rows = len(ds)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))

    os.makedirs(args.output_dir, exist_ok=True)
    metrics = []

    for epoch in range(1, args.epochs + 1):
        start_time = time.perf_counter()
        total_loss = 0.0

        model.train()
        pbar = tqdm(total=total_rows, desc=f"Epoch {epoch}/{args.epochs}", unit="flow")
        for input_ids, attn_mask, labels in loader:
            input_ids, attn_mask, labels = [t.to(device) for t in (input_ids, attn_mask, labels)]
            optimizer.zero_grad()
            logits = model(input_ids, attn_mask)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            pbar.update(input_ids.size(0))
        pbar.close()

        epoch_time = time.perf_counter() - start_time
        avg_loss = total_loss / len(loader)
        print(
            f"Epoch {epoch}/{args.epochs} — "
            f"Loss: {avg_loss:.4f} — "
            f"Time: {epoch_time:.1f}s — "
            f"Rows processed: {total_rows}"
        )

        metrics.append({
            "epoch":   epoch,
            "loss":    round(avg_loss, 4),
            "duration": round(epoch_time, 3),
            "rows":    total_rows
        })

    ckpt_path = os.path.join(args.output_dir, "flowbert.ckpt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"✔ Saved checkpoint to {ckpt_path}")

    metrics_path = os.path.join(args.output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch","loss","duration","rows"])
        writer.writeheader()
        writer.writerows(metrics)
    print(f"✔ Saved metrics to {metrics_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens",      required=True, help="Path to tokenised text file")
    parser.add_argument("--vocab",       required=True, help="Path to tokenizer JSON")
    parser.add_argument("--output-dir",  default="outputs", help="Where to save checkpoints and metrics")
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--seq-len",     type=int, default=128)
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
