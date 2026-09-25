# src/layerwise.py
import os
import time
import numpy as np
import pandas as pd

from probes import load_features, build_type_labels, train_existence_probe
from run_selectivity import load_control_features
from config import SEEDS

HEADS_PER_LAYER = 12
N_LAYERS = 12
RANDOM_DIMS_SEED = 42


def get_layer_slice(X, layer_idx, heads_per_layer=HEADS_PER_LAYER):
    start = layer_idx * heads_per_layer
    end = start + heads_per_layer
    return X[:, start:end]


def get_random_dims_slice(X, n_dims=12, seed=RANDOM_DIMS_SEED):
    rng = np.random.RandomState(seed)
    dims = rng.choice(X.shape[1], size=n_dims, replace=False)
    return X[:, dims]


def run_one(X_train, y_train, X_test, y_test, seed, label):
    t0 = time.time()
    acc, _ = train_existence_probe(X_train, y_train, X_test, y_test, "linear", seed, None)
    elapsed = time.time() - t0
    print(f"    [{label}] seed={seed} acc={acc:.4f}  ({elapsed:.1f}s)")
    return acc


def run_layerwise_sweep(X_train_full, y_train_real, y_train_control,
                         X_test_full, y_test_real, y_test_control,
                         task_name):
    results = []

    # --- per-layer ---
    for layer in range(N_LAYERS):
        X_train_l = get_layer_slice(X_train_full, layer)
        X_test_l = get_layer_slice(X_test_full, layer)

        print(f"\n[{task_name}] === layer {layer} ===")

        print("  -- real labels --")
        real_accs = [
            run_one(X_train_l, y_train_real, X_test_l, y_test_real, s, f"{task_name}/layer{layer}/real")
            for s in SEEDS
        ]

        print("  -- control labels --")
        control_accs = [
            run_one(X_train_l, y_train_control, X_test_l, y_test_control, s, f"{task_name}/layer{layer}/control")
            for s in SEEDS
        ]

        real_mean, real_std = np.mean(real_accs), np.std(real_accs)
        control_mean, control_std = np.mean(control_accs), np.std(control_accs)
        selectivity = real_mean - control_mean

        print(f"  >> layer {layer} SUMMARY: real={real_mean:.4f}±{real_std:.4f}  control={control_mean:.4f}±{control_std:.4f}  selectivity={selectivity:.4f}")

        results.append({
            "layer": layer,
            "real_mean": real_mean, "real_std": real_std,
            "control_mean": control_mean, "control_std": control_std,
            "selectivity": selectivity,
        })

    # --- random-12-dims baseline ---
    X_train_r = get_random_dims_slice(X_train_full)
    X_test_r = get_random_dims_slice(X_test_full)

    print(f"\n[{task_name}] === random_12_dims baseline ===")

    print("  -- real labels --")
    real_accs = [
        run_one(X_train_r, y_train_real, X_test_r, y_test_real, s, f"{task_name}/random12/real")
        for s in SEEDS
    ]

    print("  -- control labels --")
    control_accs = [
        run_one(X_train_r, y_train_control, X_test_r, y_test_control, s, f"{task_name}/random12/control")
        for s in SEEDS
    ]

    real_mean, real_std = np.mean(real_accs), np.std(real_accs)
    control_mean, control_std = np.mean(control_accs), np.std(control_accs)
    selectivity = real_mean - control_mean

    print(f"  >> random_12_dims SUMMARY: real={real_mean:.4f}±{real_std:.4f}  control={control_mean:.4f}±{control_std:.4f}  selectivity={selectivity:.4f}")

    results.append({
        "layer": "random_12_dims",
        "real_mean": real_mean, "real_std": real_std,
        "control_mean": control_mean, "control_std": control_std,
        "selectivity": selectivity,
    })

    return pd.DataFrame(results)


if __name__ == "__main__":
    os.makedirs("results/tables", exist_ok=True)

    print("Loading real-task features...")
    X_train, y_exist_train, y_type_train = load_features("data/processed/train_features.jsonl")
    X_test, y_exist_test, y_type_test = load_features("data/processed/test_features.jsonl")

    print("Loading control-task features...")
    Xc_train, yc_exist_train, yc_type_train = load_control_features("data/processed/train_features_control.jsonl")
    Xc_test, yc_exist_test, yc_type_test = load_control_features("data/processed/test_features_control.jsonl")

    print("Verifying real/control feature alignment...")
    assert np.allclose(X_train, Xc_train), "Mismatch: real and control train features differ!"
    assert np.allclose(X_test, Xc_test), "Mismatch: real and control test features differ!"
    print("OK.\n")

    # ===================== EXISTENCE TASK =====================
    print("########## EXISTENCE LAYERWISE SWEEP ##########")
    df_exist = run_layerwise_sweep(
        X_train, y_exist_train, yc_exist_train,
        X_test, y_exist_test, yc_exist_test,
        task_name="existence"
    )
    df_exist.to_csv("results/tables/layerwise_existence.csv", index=False)
    print("\nExistence layerwise sweep saved.\n")
    print(df_exist)

    # ===================== TYPE TASK =====================
    print("\n########## TYPE LAYERWISE SWEEP ##########")
    mapped_train, label_set = build_type_labels(y_type_train, top_n=20)
    mapped_test, _ = build_type_labels(y_type_test, top_n=20, fixed_label_set=label_set)

    pos_train_mask = y_exist_train == 1
    pos_test_mask = y_exist_test == 1

    X_type_train = X_train[pos_train_mask]
    y_type_train_real = np.array([l for l, m in zip(mapped_train, pos_train_mask) if m])
    y_type_train_control = np.array([l for l, m in zip(yc_type_train, pos_train_mask) if m])

    X_type_test = X_test[pos_test_mask]
    y_type_test_real = np.array([l for l, m in zip(mapped_test, pos_test_mask) if m])
    y_type_test_control = np.array([l for l, m in zip(yc_type_test, pos_test_mask) if m])

    df_type = run_layerwise_sweep(
        X_type_train, y_type_train_real, y_type_train_control,
        X_type_test, y_type_test_real, y_type_test_control,
        task_name="type"
    )
    df_type.to_csv("results/tables/layerwise_type.csv", index=False)
    print("\nType layerwise sweep saved.\n")
    print(df_type)

    print("\n\nALL DONE. CSVs saved to results/tables/")