import os
import pandas as pd
import matplotlib.pyplot as plt


plots_dir = "outputs/tuning/plots"


def plot_knn_results():
    custom_path = "outputs/tuning/custom_knn_tuning.csv"
    sklearn_path = "outputs/tuning/sklearn_knn_tuning.csv"

    for file_path in [custom_path, sklearn_path]:
        if not os.path.exists(file_path):
            print("File not found:", file_path)
            return

    custom = pd.read_csv(custom_path).sort_values("k")
    sklearn = pd.read_csv(sklearn_path).sort_values("k value")

    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(custom["k"], custom["mae"], marker="o", label="Custom KNN")
    plt.plot(sklearn["k value"], sklearn["mae"], marker="o", label="sklearn KNN")
    plt.axvline(2, color="gray", linestyle="--", label="Selected k=2")
    plt.xticks(custom["k"])
    plt.title("KNN validation MAE vs k")
    plt.xlabel("k")
    plt.ylabel("Validation MAE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    file_path = os.path.join(plots_dir, "knn_mae_vs_k.png")
    plt.savefig(file_path)
    plt.close()
    print("Plot saved:", file_path)


def plot_dt_results():
    custom_path = "outputs/tuning/custom_dt_tuning.csv"
    sklearn_path = "outputs/tuning/sklearn_dt_tuning.csv"

    for file_path in [custom_path, sklearn_path]:
        if not os.path.exists(file_path):
            print("File not found:", file_path)
            return

    custom = pd.read_csv(custom_path)
    sklearn = pd.read_csv(sklearn_path)
    custom = custom[
        (custom["Min samples split"] == 5)
        & (custom["Min samples leaf"] == 2)
    ].sort_values("Max depth")
    sklearn = sklearn[
        (sklearn["Min samples split"] == 5)
        & (sklearn["Min samples leaf"] == 2)
    ].sort_values("Max depth")

    custom_selected = custom[custom["Max depth"] == 35]
    sklearn_selected = sklearn[sklearn["Max depth"] == 30]

    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(custom["Max depth"], custom["MAE"], marker="o",
             color="tab:blue", label="Custom Decision Tree")
    plt.plot(sklearn["Max depth"], sklearn["MAE"], marker="o",
             color="tab:orange", label="sklearn Decision Tree")
    plt.scatter(custom_selected["Max depth"], custom_selected["MAE"],
                marker="*", s=200, color="tab:blue", zorder=3,
                label="Selected Custom DT: depth=35")
    plt.scatter(sklearn_selected["Max depth"], sklearn_selected["MAE"],
                marker="*", s=200, color="tab:orange", zorder=3,
                label="Selected sklearn DT: depth=30")
    plt.xticks(custom["Max depth"])
    plt.title("Decision Tree validation MAE (split=5, leaf=2)")
    plt.xlabel("max_depth")
    plt.ylabel("Validation MAE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    file_path = os.path.join(plots_dir, "decision_tree_mae_vs_depth.png")
    plt.savefig(file_path)
    plt.close()
    print("Plot saved:", file_path)


def plot_rf_results():
    file_path = "outputs/tuning/random_forest_tuning.csv"
    if not os.path.exists(file_path):
        print("File not found:", file_path)
        return

    results = pd.read_csv(file_path)

    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for depth in [10, 20, 30, 40, None]:
        if depth is None:
            # None is saved as an empty CSV field and read as NaN.
            depth_results = results[results["max_depth"].isna()]
        else:
            depth_results = results[results["max_depth"] == depth]

        depth_results = depth_results.sort_values("N_estimators")
        plt.plot(depth_results["N_estimators"], depth_results["MAE"],
                 marker="o", label=f"depth={depth}")

    selected = results[
        (results["N_estimators"] == 300)
        & results["max_depth"].isna()
    ]
    plt.scatter(selected["N_estimators"], selected["MAE"],
                marker="*", s=200, color="black", zorder=3,
                label="Selected: 300 trees, depth=None")
    plt.xticks([50, 100, 200, 300])
    plt.title("Random Forest validation MAE")
    plt.xlabel("n_estimators")
    plt.ylabel("Validation MAE")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    file_path = os.path.join(plots_dir, "random_forest_mae.png")
    plt.savefig(file_path)
    plt.close()
    print("Plot saved:", file_path)


def main():
    plot_knn_results()
    plot_dt_results()
    plot_rf_results()


if __name__ == "__main__":
    main()
