# src/capacity_sweep.py
import os
import time
import numpy as np
import pandas as pd

from probes import load_features, build_type_labels, train_existence_probe
from run_selectivity import load_control_features
from config import SEEDS

PROBE_CONFIGS = [
    {"name": "linear",   "type": "linear", "hidden_size": None},
    {"name": "mlp_h8",   "type": "mlp",    "hidden_size": 8},
    {"name": "mlp_h32",  "type": "mlp",    "hidden_size": 32},
]


def run_one(X_train, y_train, X_test, y_test, probe_type, hidden, seed, label):
    t0 = time.time()
    acc, _ = train_existence_probe(X_train, y_train, X_test, y_test, probe_type, seed, hidden)
    elapsed = time.time() - t0
    print(f"    [{label}] seed={seed} acc={acc:.4f}  ({elapsed:.1f}s)")
    return acc


def run_capacity_sweep(X_train, y_train_real, y_train_control, X_test, y_test_real, y_test_control, task_name):
    results = []
    for cfg in PROBE_CONFIGS:
        probe_type = cfg["type"]
        hidden = cfg["hidden_size"]

        print(f"\n[{task_name}] === {cfg['name']} ===")

        print(f"  -- real labels --")
        real_accs = [
            run_one(X_train, y_train_real, X_test, y_test_real, probe_type, hidden, s, f"{task_name}/{cfg['name']}/real")
            for s in SEEDS
        ]

        print(f"  -- control labels --")
        control_accs = [
            run_one(X_train, y_train_control, X_test, y_test_control, probe_type, hidden, s, f"{task_name}/{cfg['name']}/control")
            for s in SEEDS
        ]

        real_mean, real_std = np.mean(real_accs), np.std(real_accs)
        control_mean, control_std = np.mean(control_accs), np.std(control_accs)
        selectivity = real_mean - control_mean

        print(f"  >> {cfg['name']} SUMMARY: real={real_mean:.4f}±{real_std:.4f}  control={control_mean:.4f}±{control_std:.4f}  selectivity={selectivity:.4f}")

        results.append({
            "probe": cfg["name"],
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
    
    print("########## EXISTENCE CAPACITY SWEEP ##########")
    df_exist = run_capacity_sweep(
        X_train, y_exist_train, yc_exist_train,
        X_test, y_exist_test, yc_exist_test,
        task_name="existence"
    )
    df_exist.to_csv("results/tables/capacity_sweep_existence.csv", index=False)
    print("\nExistence sweep saved.\n")
    print(df_exist)


        # ===================== TYPE TASK =====================
    print("\n########## TYPE CAPACITY SWEEP ##########")
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

    df_type = run_capacity_sweep(
        X_type_train, y_type_train_real, y_type_train_control,
        X_type_test, y_type_test_real, y_type_test_control,
        task_name="type"
    )
    df_type.to_csv("results/tables/capacity_sweep_type.csv", index=False)
    print("\nType sweep saved.\n")
    print(df_type)

    print("\n\nALL DONE. CSVs saved to results/tables/")