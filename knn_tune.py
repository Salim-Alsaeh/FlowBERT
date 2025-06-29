#!/usr/bin/env python3
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_auc_score

def score_knn(X_train, X_test, y_test, k, batch_size=50000, n_jobs=4):
    nn = NearestNeighbors(
        n_neighbors=k, algorithm="brute", metric="euclidean", n_jobs=n_jobs
    )
    nn.fit(X_train)
    n = X_test.shape[0]
    scores = np.empty(n, float)
    t0 = time.time()
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        d, _ = nn.kneighbors(X_test[i:j])
        scores[i:j] = d[:, -1]
    auc = roc_auc_score(y_test, scores)
    return auc, time.time() - t0

def main():
    # load embeddings
    train = np.load("embeddings/unsw_benign.npz")["embeddings"]
    test  = np.load("embeddings/unsw_test_200k.npz")
    X_test, y_test = test["embeddings"], test["labels"]

    best_k, best_auc = None, 0.0
    print(f"Grid‐search on 200k samples (n={X_test.shape[0]})\n")
    for k in range(1, 21):
        auc, dt = score_knn(train, X_test, y_test, k)
        print(f"k={k:2d} → AUC={auc:.4f} (time {dt:.1f}s)")
        if auc > best_auc:
            best_k, best_auc = k, auc

    print(f"\n✔ Best k={best_k} with AUC={best_auc:.4f}")

if __name__ == "__main__":
    main()
