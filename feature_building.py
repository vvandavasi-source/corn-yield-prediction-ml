import data_preprocessing as dp
import pandas as pd
import numpy as np
from numpy import nanmean
from scipy import integrate
from numpy import trapezoid
same_year_df= dp.merged_df()

# ============================================================
# BUILD HISTORICAL FEATURES FOR BOTH
# ============================================================

for temp_df in [same_year_df]:

    temp_df["hist_5yr"] = (
        temp_df.groupby("GEOID")["yield_bu_acre"]
        .transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )
    )

    temp_df["yield_filled"] = temp_df["yield_bu_acre"]

    missing_mask = temp_df["yield_filled"].isna()

    temp_df.loc[missing_mask, "yield_filled"] = temp_df.loc[
        missing_mask,
        "hist_5yr"
    ]

    temp_df["yield_filled"] = temp_df["yield_filled"].fillna(
        temp_df["yield_bu_acre"].median()
    )

    temp_df["yield_anomaly"] = (
        temp_df["yield_bu_acre"] - temp_df["hist_5yr"]
    )

print("Historical features created")
# ============================================================
#CREATE ANOMALY FEATURES FOR BOTH
# ============================================================

veg_indices = ["EVI_scaled", "EVI2", "NDWI", "NIRv", "GCI"]

for temp_df in [same_year_df]:

    for idx in veg_indices:

        idx_cols = [
            c for c in temp_df.columns
            if c.startswith(idx + "_DOY_")
            and not c.endswith("_anom")
        ]

        for col in idx_cols:

            hist_mean = (
                temp_df.groupby("GEOID")[col]
                .transform(
                    lambda x: x.shift(1).expanding().mean()
                )
            )

            temp_df[col + "_anom"] = temp_df[col] - hist_mean

print("Vegetation anomaly features created")

# ============================================================
# DROP ROWS WITHOUT HISTORICAL INFORMATION
# ============================================================

same_year_df = same_year_df[
    same_year_df["hist_5yr"].notna()
].copy()


print("Same-year rows:", len(same_year_df))


"""build historical features
"""
def build_features(df_sub, doy_cut):

    feature_list = []
    veg_indices = ["EVI_scaled", "EVI2", "NDWI", "NIRv", "GCI"]

    for idx in veg_indices:

        # ------------------------------------------------
        # RAW DOY FEATURES
        # ------------------------------------------------
        raw_cols = [
            c
            for c in df_sub.columns
            if c.startswith(f"{idx}_DOY_") and not c.endswith("_anom")
        ]

        raw_cols = [
            c for c in raw_cols
            if int(c.split("_DOY_")[-1]) <= doy_cut
        ]

        raw_cols = sorted(
            raw_cols,
            key=lambda x: int(x.split("_DOY_")[-1])
        )

        if len(raw_cols) > 0:

            X = df_sub[raw_cols].values
            feat_df = pd.DataFrame(index=df_sub.index)

            doy_values = np.array([
                int(c.split("_DOY_")[-1]) for c in raw_cols
            ])

            # ------------------------------------------------
            # CORE FEATURES
            # ------------------------------------------------
            feat_df[f"{idx}_mean"] = np.mean(X, axis=1)
            feat_df[f"{idx}_max"] = np.max(X, axis=1)
            feat_df[f"{idx}_std"] = np.std(X, axis=1)

            # ------------------------------------------------
            #FIX 1: SMOOTHED LAST (reduces CDL noise)
            # ------------------------------------------------
            if X.shape[1] >= 3:
                feat_df[f"{idx}_last"] = np.nanmean(X[:, -3:], axis=1)
            else:
                feat_df[f"{idx}_last"] = X[:, -1]

            # ------------------------------------------------
            #  FIX 2: TRAPEZOIDAL AUC (true biomass proxy)
            # ------------------------------------------------
            feat_df[f"{idx}_auc"] = integrate.trapezoid(X, axis=1)

            # -----------------------------------------------
            #  FIX 3: PEAK TIMING (THIS IS BIG)
            # ------------------------------------------------
            peak_idx = np.nan_to_num(X, nan=X.max(axis=1, keepdims=True)).argmax(axis=1)
            feat_df[f"{idx}_peak_doy"] = doy_values[peak_idx]

            # ------------------------------------------------
            #  FIX 4: STABLE SLOPE (less noisy)
            # ------------------------------------------------
            if X.shape[1] >= 4:
                feat_df[f"{idx}_slope"] = (
                    np.nanmean(X[:, -2:], axis=1) -
                    np.nanmean(X[:, -4:-2], axis=1)
                )

            # ------------------------------------------------
            # EARLY SIGNAL
            # ------------------------------------------------
            if X.shape[1] >= 3:
                feat_df[f"{idx}_early_mean"] = np.nanmean(X[:, :3], axis=1)

            feature_list.append(feat_df)

        # ------------------------------------------------
        # ANOMALY FEATURES
        # ------------------------------------------------
        anom_cols = [
            c
            for c in df_sub.columns
            if c.startswith(f"{idx}_DOY_") and c.endswith("_anom")
        ]

        anom_cols = [
            c for c in anom_cols
            if int(c.split("_DOY_")[-1].replace("_anom", "")) <= doy_cut
        ]

        anom_cols = sorted(
            anom_cols,
            key=lambda x: int(
                x.split("_DOY_")[-1].replace("_anom", "")
            )
        )

        if len(anom_cols) > 0:

            X_anom = df_sub[anom_cols].values
            feat_df_anom = pd.DataFrame(index=df_sub.index)

            feat_df_anom[f"{idx}_anom_mean"] = np.nanmean(X_anom, axis=1)
            feat_df_anom[f"{idx}_anom_max"] = np.nanmax(X_anom, axis=1)

            # smoother anomaly last
            if X_anom.shape[1] >= 3:
                feat_df_anom[f"{idx}_anom_last"] = np.nanmean(X_anom[:, -3:], axis=1)
            else:
                feat_df_anom[f"{idx}_anom_last"] = X_anom[:, -1]

            # stable anomaly slope
            if X_anom.shape[1] >= 4:
                feat_df_anom[f"{idx}_anom_slope"] = (
                    np.nanmean(X_anom[:, -2:], axis=1) -
                    np.nanmean(X_anom[:, -4:-2], axis=1)
                )

            feature_list.append(feat_df_anom)

    if not feature_list:
        return pd.DataFrame(index=df_sub.index)

    return pd.concat(feature_list, axis=1)





