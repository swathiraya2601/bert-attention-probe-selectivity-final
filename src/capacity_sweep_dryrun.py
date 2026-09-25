# src/capacity_sweep_dryrun.py
import numpy as np

from probes import load_features, build_type_labels
from run_selectivity import load_control_features
from capacity_sweep import run_capacity_sweep, PROBE_CONFIGS

SUBSAMPLE_SIZE = 5000
SEED_FOR_SUBSAMPLE = 0

if __name__ == "__main__":
    # --- Load real-task features ---
    X_train, y_exist_train, y_type_train = load_features("data/processed/train_features.jsonl")
    X_test, y_exist_test, y_type_test = load_features("data/processed/test_features.jsonl")

    # --- Load control-task labels ---
    Xc_train, yc_exist_train, yc_type_train = load_control_features("data/processed/train_features_control.jsonl")
    Xc_test, yc_exist_test, yc_type_test = load_control_features("data/processed/test_features_control.jsonl")

    assert np.allclose(X_train, Xc_train), "Mismatch: real and control train features differ!"
    assert np.allclose(X_test, Xc_test), "Mismatch: real and control test features differ!"

    # --- Subsample train and test down to a small size for a fast smoke test ---
    rng = np.random.RandomState(SEED_FOR_SUBSAMPLE)
    train_idx = rng.choice(len(X_train), size=min(SUBSAMPLE_SIZE, len(X_train)), replace=False)
    test_idx = rng.choice(len(X_test), size=min(SUBSAMPLE_SIZE // 5, len(X_test)), replace=False)

    X_train_s, y_exist_train_s, yc_exist_train_s = X_train[train_idx], y_exist_train[train_idx], yc_exist_train[train_idx]
    X_test_s, y_exist_test_s, yc_exist_test_s = X_test[test_idx], y_exist_test[test_idx], yc_exist_test[test_idx]

    print("=== DRY RUN: existence task, subsample sizes ===")
    print("train:", len(X_train_s), " test:", len(X_test_s))

    df_exist_dry = run_capacity_sweep(
        X_train_s, y_exist_train_s, yc_exist_train_s,
        X_test_s, y_exist_test_s, yc_exist_test_s,
        task_name="existence-dryrun"
    )
    print(df_exist_dry)
    print("\nIf this ran without errors and numbers look sane (accuracy between 0 and 1, "
          "selectivity roughly consistent with your earlier full-data run), the pipeline is confirmed. "
          "Numbers themselves will differ from the full run since this uses far less data.")