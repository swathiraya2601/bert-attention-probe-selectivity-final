# src/probes.py
import json
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from config import SEEDS


def load_features(path):
    """Load cached BERT attention features + labels from a *_features.jsonl file."""
    X, y_exist, y_type = [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            X.append(r["features"])
            y_exist.append(r["label"])
            y_type.append(r["deprel"] if r["label"] == 1 else None)
    return np.array(X), np.array(y_exist), y_type


def build_type_labels(y_type, top_n=20, fixed_label_set=None):
    """Map raw deprel strings down to top-N most frequent + 'other'.
    Pass fixed_label_set (from a prior call on train) to reuse the same
    label set on dev/test, rather than recomputing top-N independently."""
    if fixed_label_set is None:
        counts = Counter(l for l in y_type if l is not None)
        top_labels = {l for l, _ in counts.most_common(top_n)}
        label_set = sorted(top_labels | {"other"})
    else:
        label_set = fixed_label_set
        top_labels = set(label_set) - {"other"}
    mapped = [l if l in top_labels else ("other" if l is not None else None) for l in y_type]
    return mapped, label_set


def train_existence_probe(X_train, y_train, X_test, y_test, probe_type="linear", seed=42, hidden_size=None):
    if probe_type == "linear":
        clf = LogisticRegression(max_iter=1000, random_state=seed)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(hidden_size,), max_iter=500, random_state=seed)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    return acc, clf


def run_seeded(X_train, y_train, X_test, y_test, probe_type="linear", hidden_size=None):
    accs = [train_existence_probe(X_train, y_train, X_test, y_test, probe_type, s, hidden_size)[0] for s in SEEDS]
    return np.mean(accs), np.std(accs)


if __name__ == "__main__":
    # --- Existence probe: real task, full 144-dim vector ---
    X_train, y_exist_train, y_type_train = load_features("data/processed/train_features.jsonl")
    X_test, y_exist_test, y_type_test = load_features("data/processed/test_features.jsonl")

    exist_linear_mean, exist_linear_std = run_seeded(X_train, y_exist_train, X_test, y_exist_test, "linear")
    print(f"Existence (linear): {exist_linear_mean:.4f} ± {exist_linear_std:.4f}")

    exist_mlp_mean, exist_mlp_std = run_seeded(X_train, y_exist_train, X_test, y_exist_test, "mlp", hidden_size=128)
    print(f"Existence (mlp_h128): {exist_mlp_mean:.4f} ± {exist_mlp_std:.4f}")

    # --- Type probe: real task, positive pairs only ---
    mapped_train, label_set = build_type_labels(y_type_train, top_n=20)
    mapped_test, _ = build_type_labels(y_type_test, top_n=20, fixed_label_set=label_set)

    pos_train_mask = y_exist_train == 1
    pos_test_mask = y_exist_test == 1

    X_type_train = X_train[pos_train_mask]
    y_type_train_mapped = np.array([l for l, m in zip(mapped_train, pos_train_mask) if m])

    X_type_test = X_test[pos_test_mask]
    y_type_test_mapped = np.array([l for l, m in zip(mapped_test, pos_test_mask) if m])

    type_linear_mean, type_linear_std = run_seeded(X_type_train, y_type_train_mapped, X_type_test, y_type_test_mapped, "linear")
    print(f"Type (linear): {type_linear_mean:.4f} ± {type_linear_std:.4f}")