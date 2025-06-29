#!/usr/bin/env python3
import argparse
import glob
import pandas as pd
from src.tokenise import row_to_tokens


def main():
    parser = argparse.ArgumentParser(description="Tokenise CICIDS-2017 benign flows")
    parser.add_argument("--input-dir", required=True, help="Path to CICIDS-2017 CSV folder")
    parser.add_argument("--output",    required=True, help="Path to write token sequences")
    args = parser.parse_args()

    csv_files = glob.glob(f"{args.input_dir}/*.csv")
    with open(args.output, "w") as out:
        for fn in csv_files:
            df = pd.read_csv(fn)
            # strip whitespace from column names
            df.columns = [col.strip() for col in df.columns]
            # filter only benign flows (CICIDS-2017 uses 'Benign' label)
            if "Label" not in df.columns:
                raise ValueError(f"No 'Label' column in {fn}")
            # ensure case-insensitive match for benign
            benign = df[df["Label"].astype(str).str.lower() == "benign"]
            for _, row in benign.iterrows():
                tokens = row_to_tokens(row)
                out.write(" ".join(tokens) + "\n")

if __name__ == "__main__":
    main()
