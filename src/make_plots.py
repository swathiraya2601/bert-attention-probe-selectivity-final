# src/make_plots.py
import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("results/figures", exist_ok=True)

# ============ Capacity sweep plots ============

def plot_capacity_sweep(csv_path, title, out_path):
    df = pd.read_csv(csv_path)
    x = range(len(df))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(x, df["real_mean"], yerr=df["real_std"], label="Real task", marker="o", capsize=3)
    ax.errorbar(x, df["control_mean"], yerr=df["control_std"], label="Control task", marker="s", capsize=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["probe"], rotation=20)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Probe capacity")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

def plot_selectivity_only(csv_path, title, out_path, layer_axis=False):
    df = pd.read_csv(csv_path)
    x = range(len(df))
    xcol = "layer" if layer_axis else "probe"

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, df["selectivity"], marker="o", color="darkred")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df[xcol], rotation=20)
    ax.set_ylabel("Selectivity (real - control)")
    ax.set_xlabel("Layer" if layer_axis else "Probe capacity")
    ax.set_title(title)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

# ============ Layer-wise plots ============

def plot_layerwise(csv_path, title, out_path):
    df = pd.read_csv(csv_path)
    numeric_rows = df[df["layer"].apply(lambda x: str(x).isdigit())].copy()
    numeric_rows["layer"] = numeric_rows["layer"].astype(int)
    numeric_rows = numeric_rows.sort_values("layer")

    baseline_row = df[df["layer"] == "random_12_dims"]
    baseline_real = baseline_row["real_mean"].values[0] if len(baseline_row) else None
    baseline_control = baseline_row["control_mean"].values[0] if len(baseline_row) else None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(numeric_rows["layer"], numeric_rows["real_mean"], yerr=numeric_rows["real_std"],
                label="Real task (per layer)", marker="o", capsize=3)
    ax.errorbar(numeric_rows["layer"], numeric_rows["control_mean"], yerr=numeric_rows["control_std"],
                label="Control task (per layer)", marker="s", capsize=3)
    if baseline_real is not None:
        ax.axhline(baseline_real, color="green", linestyle="--", label="Random 12-dim baseline (real)")
    if baseline_control is not None:
        ax.axhline(baseline_control, color="orange", linestyle=":", label="Random 12-dim baseline (control)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xticks(numeric_rows["layer"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")

def plot_layerwise_selectivity(csv_path, title, out_path):
    df = pd.read_csv(csv_path)
    numeric_rows = df[df["layer"].apply(lambda x: str(x).isdigit())].copy()
    numeric_rows["layer"] = numeric_rows["layer"].astype(int)
    numeric_rows = numeric_rows.sort_values("layer")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(numeric_rows["layer"], numeric_rows["selectivity"], marker="o", color="darkred")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Selectivity (real - control)")
    ax.set_title(title)
    ax.set_xticks(numeric_rows["layer"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    # Capacity sweep - existence
    plot_capacity_sweep("results/tables/capacity_sweep_existence.csv",
                         "Existence probe: accuracy vs. capacity",
                         "results/figures/capacity_sweep_existence.png")
    plot_selectivity_only("results/tables/capacity_sweep_existence.csv",
                           "Existence probe: selectivity vs. capacity",
                           "results/figures/capacity_selectivity_existence.png")

    # Capacity sweep - type
    plot_capacity_sweep("results/tables/capacity_sweep_type.csv",
                         "Type probe: accuracy vs. capacity",
                         "results/figures/capacity_sweep_type.png")
    plot_selectivity_only("results/tables/capacity_sweep_type.csv",
                           "Type probe: selectivity vs. capacity",
                           "results/figures/capacity_selectivity_type.png")

    # Layer-wise - existence
    plot_layerwise("results/tables/layerwise_existence.csv",
                    "Existence probe: accuracy by layer",
                    "results/figures/layerwise_existence.png")
    plot_layerwise_selectivity("results/tables/layerwise_existence.csv",
                                "Existence probe: selectivity by layer",
                                "results/figures/layerwise_selectivity_existence.png")

    # Layer-wise - type
    plot_layerwise("results/tables/layerwise_type.csv",
                    "Type probe: accuracy by layer",
                    "results/figures/layerwise_type.png")
    plot_layerwise_selectivity("results/tables/layerwise_type.csv",
                                "Type probe: selectivity by layer",
                                "results/figures/layerwise_selectivity_type.png")

    print("\nAll plots saved to results/figures/")