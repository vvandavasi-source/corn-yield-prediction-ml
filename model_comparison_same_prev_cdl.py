import feature_building as fb
import plot_feature_importance as pfi
import train_test_split as tts
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ============================================================
# MODEL COMPARISON:
# SAME-YEAR CDL VS PREVIOUS-YEAR CDL
# ============================================================


df_same_cdl = fb.same_year_df.copy() # sets the same year CDL dataframe
#df_prev_cdl = fb.lagged_cdl_df.copy() # sets the lagged CDL dataframe


test_years = list(range(2018, 2025))

comparison_results = []

for doy_cut in range(65, 274, 16):
    
    model_same_cdl, pred_same_cdl, r2_same_cdl, importance_same_df = tts.train_test_split(
        df_same_cdl,
        test_years,
        doy_cut,
        "year",
        "hist_5yr",
        "yield_filled",
        "yield_bu_acre",
        True
    )
    comparison_results.append(
        {
            "DOY": doy_cut,
            "Same_Year_CDL": np.mean(r2_same_cdl) if r2_same_cdl else np.nan,
            #"Previous_Year_CDL": np.mean(r2_prev_cdl) if r2_prev_cdl else np.nan,
        }
    )

comparison_df = pd.DataFrame(comparison_results)

print("\n===== FINAL COMPARISON =====")
# ============================================================
# PLOT SAME-YEAR VS PREVIOUS-YEAR CDL
# ============================================================



plt.figure(figsize=(10, 6))

plt.plot(
    comparison_df["DOY"],
    comparison_df["Same_Year_CDL"],
    marker="o",
    linewidth=2,
    label="Same-Year CDL"
)
'''
plt.plot(
    comparison_df["DOY"],
    comparison_df["Previous_Year_CDL"],
    marker="o",
    linewidth=2,
    label="Previous-Year CDL"
)
'''
plt.axvspan(
    145,
    225,
    alpha=0.15,
    label="Peak Growing Season"
)

plt.xlabel("DOY Cutoff")
plt.ylabel("Mean Test R²")
plt.title("Same-Year CDL")
plt.xticks(comparison_df["DOY"])
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("same_year_cdl_comparison.png", dpi=300)
print("Saved r^2 plot for Same-Year CDL")

# ============================================================
# FEATURE IMPORTANCE COMPARISON
# SAME-YEAR CDL VS PREVIOUS-YEAR CDL
# ============================================================

'''importance_compare_df = pd.concat(
    importance_same_df + importance_prev_df,
    ignore_index=True
)
'''
print("Done building feature importance table")

features_to_plot = [
    "hist_5yr",
    "year",
    "EVI2_mean",
    "EVI2_last",
    "NIRv_mean",
    "NIRv_max",
    "EVI_scaled_mean",
    "EVI_scaled_auc"
]

# PLOT FEATURE IMPORTANCE EVOLUTION through plot_feature_importance.py function
pfi.plot_feature_importance_evolution(importance_same_df, features_to_plot)
print("Done plotting feature importance evolution")
