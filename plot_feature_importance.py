import matplotlib.pyplot as plt
import pandas as pd

# ============================================================
# PLOT FEATURE IMPORTANCE EVOLUTION
# ============================================================



def plot_feature_importance_evolution(importance_compare_df, features_to_plot):
    plot_df = (
        importance_compare_df[
            importance_compare_df["feature"].isin(features_to_plot)
        ]
        .groupby(["dataset", "DOY", "feature"])["importance"]
        .mean()
        .reset_index()
    )

    for dataset_name in plot_df["dataset"].unique():

        plt.figure(figsize=(12, 7))

        ds = plot_df[plot_df["dataset"] == dataset_name]

        for feat in ds["feature"].unique():

            feat_df = ds[ds["feature"] == feat]

            plt.plot(
                feat_df["DOY"],
                feat_df["importance"],
                marker="o",
                linewidth=2,
                label=feat
            )

        plt.xlabel("DOY")
        plt.ylabel("Mean Feature Importance")
        plt.title(f"Feature Importance Evolution - {dataset_name}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f"feature_importance_{dataset_name.replace(' ', '_')}.png", dpi=300)
        print(f"Saved feature importance plot for {dataset_name}")
