# src/check_features.py
import json
import sys
import numpy as np

def check_features(path):
    with open(path) as f:
        records = [json.loads(line) for line in f]

    print(f"--- {path} ---")
    print("Number of records:", len(records))

    lengths = set(len(r["features"]) for r in records)
    print("Feature vector lengths found:", lengths)

    all_vals = np.array([v for r in records for v in r["features"]])
    print("Any NaNs:", np.isnan(all_vals).any())
    print("Min/max attention value:", all_vals.min(), all_vals.max())

    pos_means = np.mean([np.mean(r["features"]) for r in records if r["label"] == 1])
    neg_means = np.mean([np.mean(r["features"]) for r in records if r["label"] == 0])
    print("Mean attention (positives):", pos_means)
    print("Mean attention (negatives):", neg_means)
    print()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/dev_features.jsonl"
    check_features(path)