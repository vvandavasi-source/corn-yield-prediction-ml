import pandas as pd
ROOT = "/home/logan/Downloads/summary/"
SUMMARY_CSV_PATH = ROOT+"seed_summary_mean_sd.csv"
PREDICTIONS_CSV_PATH = ROOT+"ensemble_county_predictions.csv"
YEARS = [2022, 2023, 2024, 2025]
TRAINING_MASK = "SAME_YEAR_CDL"

SCENARIO_ROWS = [
    ("Previous-season CDL (control)", "PREVIOUS_CDL"),
    ("Our in-season ICDL", "CLEAN_REBUILT"),
    ("Published in-season ICDL", "STUDY"),
    ("Final CDL (hindsight)", "FINAL_CDL"),
]


def build_summary_table(
    summary_path: str = SUMMARY_CSV_PATH,
    predictions_path: str = PREDICTIONS_CSV_PATH,
) -> pd.DataFrame:
    summary_df = pd.read_csv(summary_path)
    pred_df = pd.read_csv(predictions_path)

    sum_filtered = summary_df[
        (summary_df["training_mask"] == TRAINING_MASK)
        & (summary_df["year"].isin(YEARS))
    ].copy()

    pred_filtered = pred_df[
        (pred_df["training_mask"] == TRAINING_MASK)
        & (pred_df["year"].isin(YEARS))
    ].copy()
    pred_filtered["bias"] = pred_filtered["predicted_yield"] - pred_filtered["yield_bu_acre"]

    rows = []
    for label, testing_mask in SCENARIO_ROWS:
        s_sub = sum_filtered[sum_filtered["testing_mask"] == testing_mask]
        p_sub = pred_filtered[pred_filtered["testing_mask"] == testing_mask]

        n_years = s_sub["year"].nunique()
        year_range = f"{YEARS[0]}–{YEARS[-1]}"
        n_periods = s_sub["doy"].nunique() * n_years

        rows.append(
            {
                "Crop-mask scenario": label,
                "RMSE (bu ac⁻¹)": round(s_sub["rmse_mean"].mean(), 2),
                "MAE (bu ac⁻¹)": round(s_sub["mae_mean"].mean(), 2),
                "R² (–)": round(s_sub["r2_mean"].mean(), 3),
                "Mean bias (bu ac⁻¹)": round(p_sub["bias"].mean(), 2),
                "Periods, n (years)": f"{n_periods} ({n_years} y, {year_range})",
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    table_df = build_summary_table()
    print(table_df.to_markdown(index=False))