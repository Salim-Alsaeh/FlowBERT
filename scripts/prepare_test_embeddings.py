#!/usr/bin/env python3
import numpy as np
import pandas as pd

# 1) Load labels from the original Parquet (must match token order)
df = pd.read_parquet('data/UNSW/NF-UNSW-NB15.parquet')
labels = df['Label'].astype(int).values

# 2) Load your embeddings
arr = np.load('embeddings/unsw_benign.npz')
embeddings = arr['embeddings']

# 3) Save both in one NPZ
np.savez_compressed(
    'embeddings/unsw_test.npz',
    embeddings=embeddings,
    labels=labels
)
print(f"Saved test embeddings with shape {embeddings.shape} and labels {labels.shape}")
