#!/usr/bin/env python3
import argparse
import time
import numpy as np
import joblib
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

def main():
    p = argparse.ArgumentParser(
        description="Fit k-NN on benign embeddings and evaluate with progress/ETA"
    )
    p.add_argument("--benign-emb", required=True,
                   help="NPZ with key 'embeddings' (benign train)")
    p.add_argument("--test-emb",   required=True,
                   help="NPZ with keys 'embeddings' and 'labels' (test)")
    p.add_argument("--output",     default="detectors/knn.joblib",
                   help="Where to save the fitted k-NN model")
    p.add_argument("--k",          type=int,   default=5,
                   help="Number of neighbors")
    p.add_argument("--test-batch-size", type=int, default=10000,
                   help="How many test points per batch")
    p.add_argument("--n-jobs",     type=int,   default=1,
                   help="Parallel jobs for neighbor search")
    args = p.parse_args()

    # 1) Fit on benign embeddings
    train_arr = np.load(args.benign_emb)
    X_train = train_arr["embeddings"]
    nn = NearestNeighbors(
        n_neighbors=args.k,
        metric="euclidean",
        algorithm="brute",
        n_jobs=args.n_jobs
    )
    nn.fit(X_train)
    joblib.dump(nn, args.output)
    print(f"✔ Saved k-NN to {args.output}")

    # 2) Load & score test set in batches with ETA
    test_arr = np.load(args.test_emb)
    X_test, y_true = test_arr["embeddings"], test_arr["labels"]
    n = X_test.shape[0]
    scores = np.empty(n, dtype=float)
    start = time.time()

    for i in range(0, n, args.test_batch_size):
        j = min(i + args.test_batch_size, n)
        dists, _ = nn.kneighbors(X_test[i:j])
        scores[i:j] = dists[:, -1]

        elapsed = time.time() - start
        done = j
        remaining = (elapsed / done) * (n - done)
        print(f"Processed {done}/{n} rows — Elapsed: {elapsed:.1f}s — ETA: {remaining:.1f}s")

    # 3) Compute ROC-AUC
    auc = roc_auc_score(y_true, scores)
    print(f"✚ Test ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    main()
