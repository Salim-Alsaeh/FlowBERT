#!/usr/bin/env python3
import numpy as np

def main():
    # Load the full test embeddings + labels
    arr = np.load("embeddings/unsw_test.npz")
    emb, lab = arr["embeddings"], arr["labels"]

    # Take the first 200,000 examples
    emb2, lab2 = emb[:200_000], lab[:200_000]

    # Save them
    np.savez_compressed(
        "embeddings/unsw_test_200k.npz",
        embeddings=emb2,
        labels=lab2
    )
    print("✔ Saved subsampled test set:",
          emb2.shape, lab2.shape)

if __name__ == "__main__":
    main()
