#!/usr/bin/env python3
import argparse
import pandas as pd

def bucketize(value):
    # TODO: refine numeric bucketing logic
    return str(value)

def row_to_tokens(row):
    tokens = []
    for k, v in row.to_dict().items():
        tokens.append(f"{k}={bucketize(v)}")
    return tokens

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="Path to .parquet or .csv NetFlow file")
    p.add_argument("--output", required=True, help="Path to write token sequences")
    args = p.parse_args()

    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)

    with open(args.output, "w") as f:
        for _, row in df.iterrows():
            f.write(" ".join(row_to_tokens(row)) + "\n")

if __name__ == "__main__":
    main()
