# src/run_selectivity.py
import json
import numpy as np
from probes import load_features, build_type_labels, run_seeded


def load_control_features(path):
    """Like load_features, but pulls control_label / control_deprel instead of label/deprel."""
    X, y_exist_control, y_type_control = [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            X.append(r["features"])
            y_exist_control.append(r["control_label"])
            # type control label only meaningful where the REAL pair was positive,
            # mirroring how the real type probe only trains on real positive pairs
            y_type_control.append(r["control_deprel"] if r["label"] == 1 else None)
    return np.array(X), np.array(y_exist_control), y_type_control


if __name__ == "__main__":
    # --- Real task numbers (rerun here for side-by-side printing; reuses cached features, fast) ---
    X_train, y_exist_train, y_type_train = load_features("data/processed/train_features.jsonl")
    X_test, y_exist_test, y_type_test = load_features("data/processed/test_features.jsonl")

    real_exist_mean, real_exist_std = run_seeded(X_train, y_exist_train, X_test, y_exist_test, "linear")

    mapped_train, label_set = build_type_labels(y_type_train, top_n=20)
    mapped_test, _ = build_type_labels(y_type_test, top_n=20, fixed_label_set=label_set)
    pos_train_mask = y_exist_train == 1
    pos_test_mask = y_exist_test == 1
    X_type_train = X_train[pos_train_mask]
    y_type_train_mapped = np.array([l for l, m in zip(mapped_train, pos_train_mask) if m])
    X_type_test = X_test[pos_test_mask]
    y_type_test_mapped = np.array([l for l, m in zip(mapped_test, pos_test_mask) if m])
    real_type_mean, real_type_std = run_seeded(X_type_train, y_type_train_mapped, X_type_test, y_type_test_mapped, "linear")

    # --- Control task numbers ---
    Xc_train, yc_exist_train, yc_type_train = load_control_features("data/processed/train_features_control.jsonl")
    Xc_test, yc_exist_test, yc_type_test = load_control_features("data/processed/test_features_control.jsonl")

    control_exist_mean, control_exist_std = run_seeded(Xc_train, yc_exist_train, Xc_test, yc_exist_test, "linear")

    yc_type_train_mapped = np.array([l for l, m in zip(yc_type_train, pos_train_mask) if m])
    yc_type_test_mapped = np.array([l for l, m in zip(yc_type_test, pos_test_mask) if m])
    control_type_mean, control_type_std = run_seeded(X_type_train, yc_type_train_mapped, X_type_test, yc_type_test_mapped, "linear")

    # --- Report ---
    print("=== Existence probe (linear) ===")
    print(f"Real:    {real_exist_mean:.4f} ± {real_exist_std:.4f}")
    print(f"Control: {control_exist_mean:.4f} ± {control_exist_std:.4f}")
    print(f"Selectivity: {real_exist_mean - control_exist_mean:.4f}")

    print("\n=== Type probe (linear) ===")
    print(f"Real:    {real_type_mean:.4f} ± {real_type_std:.4f}")
    print(f"Control: {control_type_mean:.4f} ± {control_type_std:.4f}")
    print(f"Selectivity: {real_type_mean - control_type_mean:.4f}")