"""Create large-format, poster-ready graphics from the final 50-seed county model outputs."""

import csv
import math
import statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "v5_50_seed_all_in_one" / "v5_50_seed_all_in_one"
SUMMARY_DIR = DATA_ROOT / "summary"
PREDICTIONS_FILE = SUMMARY_DIR / "ensemble_county_predictions.csv"
IMPORTANCE_FILE = SUMMARY_DIR / "SAME_YEAR_CDL_mean_feature_importance.csv"
OUTPUT_DIR = ROOT / "final_poster_graphics"
TARGET_DOYS = (193, 209, 225, 241, 257, 273)
TRAINING_MASK = "SAME_YEAR_CDL"
TESTING_MASK = "STUDY"

COLORS = {
    "navy": "#123746",
    "blue": "#19718C",
    "blue_light": "#E8F2F6",
    "teal": "#2E806E",
    "teal_light": "#E9F3EF",
    "gold": "#D18A2E",
    "gold_light": "#F8F1E5",
    "coral": "#B9503D",
    "ink": "#203744",
    "muted": "#506674",
    "grid": "#DEE7EC",
    "gray": "#82929D",
    "white": "#FFFFFF",
}
YEAR_COLORS = {
    2021: "#19718C",
    2022: "#D18A2E",
    2023: "#2E806E",
    2024: "#8065A6",
    2025: "#B9503D",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 24,
    "axes.titlesize": 28,
    "axes.labelsize": 25,
    "xtick.labelsize": 22,
    "ytick.labelsize": 22,
    "legend.fontsize": 21,
    "figure.facecolor": "white",
    "axes.facecolor": "#FBFCFD",
    "axes.edgecolor": "#AFBEC7",
    "axes.labelcolor": COLORS["ink"],
    "text.color": COLORS["ink"],
    "xtick.color": COLORS["muted"],
    "ytick.color": COLORS["muted"],
    "savefig.facecolor": "white",
})


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required source file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def load_predictions():
    rows = read_csv(PREDICTIONS_FILE)
    required = {
        "training_mask", "testing_mask", "year", "doy", "FIPS",
        "yield_bu_acre", "hist_yield_trend", "predicted_yield", "n_seeds",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Ensemble prediction file is empty or missing required columns.")
    records = []
    for row in rows:
        if (row["training_mask"], row["testing_mask"]) != (TRAINING_MASK, TESTING_MASK):
            continue
        records.append({
            "year": int(row["year"]),
            "doy": int(row["doy"]),
            "fips": row["FIPS"].zfill(5),
            "actual": float(row["yield_bu_acre"]),
            "trend": float(row["hist_yield_trend"]),
            "predicted": float(row["predicted_yield"]),
            "n_seeds": int(row["n_seeds"]),
        })
    if not records:
        raise ValueError(f"No {TRAINING_MASK}/{TESTING_MASK} ensemble predictions were found.")
    if any(row["actual"] <= 0 for row in records):
        raise ValueError("Found nonpositive observed yields; percentage errors cannot be computed.")
    if any(row["n_seeds"] != 50 for row in records):
        raise ValueError("Expected exactly 50 contributing seeds in the ensemble.")
    return records


def scores(records):
    actual = [row["actual"] for row in records]
    predicted = [row["predicted"] for row in records]
    residuals = [prediction - observed for observed, prediction in zip(actual, predicted)]
    rmse = math.sqrt(math.fsum(error * error for error in residuals) / len(residuals))
    mae = math.fsum(abs(error) for error in residuals) / len(residuals)
    mape = 100 * math.fsum(abs(error) / observed for observed, error in zip(actual, residuals)) / len(actual)
    mean_actual = math.fsum(actual) / len(actual)
    total = math.fsum((observed - mean_actual) ** 2 for observed in actual)
    r2 = 1 - math.fsum(error * error for error in residuals) / total
    bias = math.fsum(residuals) / len(residuals)
    return {"n": len(records), "rmse": rmse, "mae": mae, "mape": mape, "r2": r2, "bias": bias}


def date_label(doy):
    return (date(2025, 1, 1) + timedelta(days=doy - 1)).strftime("%b %d")


def save_figure(fig, stem):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    return fig


def add_card(ax, x, y, width, height, number, title, body, face, accent):
    card = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        transform=ax.transAxes, facecolor=face, edgecolor="#B7C6CE", linewidth=1.7,
    )
    ax.add_patch(card)
    ax.add_patch(FancyBboxPatch(
        (x, y), 0.013, height,
        boxstyle="round,pad=0,rounding_size=0.01",
        transform=ax.transAxes, facecolor=accent, edgecolor=accent, linewidth=0,
    ))
    ax.text(x + 0.035, y + height - 0.075, number, transform=ax.transAxes,
            fontsize=21, fontweight="bold", color=accent, va="top")
    ax.text(x + 0.035, y + height - 0.145, title, transform=ax.transAxes,
            fontsize=24, fontweight="bold", color=COLORS["navy"], va="top")
    ax.text(x + 0.035, y + height - 0.24, body, transform=ax.transAxes,
            fontsize=21, color=COLORS["ink"], va="top", linespacing=1.3)


def add_arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(
        start, end, transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=24, linewidth=2.2, color=COLORS["gray"],
    ))


def make_workflow_figure(records):
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.axis("off")
    counties = len({row["fips"] for row in records})
    states = len({row["fips"][:2] for row in records})
    years = len({row["year"] for row in records})
    fig.suptitle("From satellite observations to county yield estimates", x=0.045, y=0.96,
                 ha="left", fontsize=31, fontweight="bold", color=COLORS["navy"])
    fig.text(
        0.047, 0.885,
        f"Same-year CDL training • published ICDL test mask • 50 seeds • "
        f"{states} states • {counties:,} counties • {years} test years",
        fontsize=18, color=COLORS["muted"],
    )
    cards = [
        (0.045, 0.43, 0.20, 0.38, "01", "Observe crops",
         "MODIS crop-health\nsignals + seasonal\nconditions", COLORS["blue_light"], COLORS["blue"]),
        (0.29, 0.43, 0.20, 0.38, "02", "Learn patterns",
         "Learn yield patterns\nfrom historical +\nsatellite data", COLORS["teal_light"], COLORS["teal"]),
        (0.535, 0.43, 0.20, 0.38, "03", "Predict yield",
         "Average 50 runs\nfor county yields\nat each checkpoint", COLORS["gold_light"], COLORS["gold"]),
        (0.78, 0.43, 0.20, 0.38, "04", "Check accuracy",
         "Compare estimates\nto reported yield:\nRMSE, R², MAPE", "#F8ECE8", COLORS["coral"]),
    ]
    for card in cards:
        add_card(ax, *card)
    add_arrow(ax, (0.25, 0.70), (0.285, 0.70))
    add_arrow(ax, (0.495, 0.70), (0.53, 0.70))
    add_arrow(ax, (0.74, 0.70), (0.775, 0.70))

    doy_first, doy_last = min(row["doy"] for row in records), max(row["doy"] for row in records)
    checkpoints = sorted({row["doy"] for row in records})
    fig.text(0.06, 0.28, "Seasonal checkpoints", fontsize=24, fontweight="bold", color=COLORS["navy"])
    ax.plot([0.08, 0.92], [0.19, 0.19], transform=ax.transAxes, color="#C9D5DC", linewidth=7,
            solid_capstyle="round", zorder=0)
    for doy in checkpoints:
        x = 0.08 + 0.84 * (doy - doy_first) / (doy_last - doy_first)
        ax.scatter([x], [0.19], transform=ax.transAxes, s=250, color=COLORS["blue"],
                   edgecolors="white", linewidths=2, zorder=3)
        ax.text(x, 0.115, f"{date_label(doy)}\nDOY {doy}", transform=ax.transAxes,
                ha="center", va="top", fontsize=21, color=COLORS["ink"])
    fig.text(
        0.5, 0.015,
        "The archived estimates are 16 days apart. They demonstrate seasonal checkpoints, not weekly forecast coverage.",
        ha="center", fontsize=20, color=COLORS["coral"], fontweight="bold",
    )
    fig.tight_layout(rect=(0.02, 0.04, 0.98, 0.90))
    return save_figure(fig, "01_model_workflow")


def make_validation_scatter(records):
    case = [row for row in records if row["year"] == 2025 and row["doy"] == 273]
    if not case:
        raise ValueError("No 2025 DOY 273 predictions available for poster parity plot.")
    metrics = scores(case)
    actual = np.array([row["actual"] for row in case])
    predicted = np.array([row["predicted"] for row in case])

    fig, ax = plt.subplots(figsize=(11.5, 11))
    ax.grid(color=COLORS["grid"], linewidth=1.0)
    ax.set_axisbelow(True)
    ax.set_facecolor("#FBFCFD")
    minimum = 0
    maximum = math.ceil(max(actual.max(), predicted.max(), 250) / 20) * 20
    grid = np.linspace(minimum, maximum, 300)
    ax.fill_between(grid, 0.9 * grid, 1.1 * grid, color="#DCEFE7", alpha=0.60, zorder=0,
                    label="Within ±10% of reported yield")
    ax.plot(grid, grid, color=COLORS["navy"], linewidth=3, label="Exact agreement (1:1)")
    ax.scatter(actual, predicted, s=45, alpha=0.42, color=COLORS["blue"],
               edgecolors="white", linewidths=0.3, zorder=2, rasterized=True)
    ax.set_xlim(0, maximum)
    ax.set_ylim(0, maximum)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reported county corn yield (bu/ac)", labelpad=14, fontweight="bold")
    ax.set_ylabel("Predicted county corn yield (bu/ac)", labelpad=14, fontweight="bold")
    ax.set_title("The county forecasts track reported yield", loc="left", pad=20,
                 fontsize=27, fontweight="bold", color=COLORS["navy"])
    ax.text(
        0.055, 0.94,
        f"2025 DOY 273 • Study mask • N={metrics['n']:,} • 50-seed mean",
        transform=ax.transAxes, fontsize=18, color=COLORS["muted"], va="top",
    )
    metric_text = (
        f"R²  {metrics['r2']:.2f}\n"
        f"MAPE  {metrics['mape']:.1f}%\n"
        f"RMSE  {metrics['rmse']:.1f} bu/ac"
    )
    ax.text(
        0.975, 0.06, metric_text, transform=ax.transAxes,
        ha="right", va="bottom", fontsize=21, fontweight="bold", color=COLORS["navy"],
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": "#B7C6CE", "linewidth": 1.5},
    )
    ax.legend(loc="upper left", bbox_to_anchor=(0.04, 0.855), frameon=True, framealpha=0.94,
              facecolor="white", edgecolor="#D7E0E5", fontsize=21)
    ax.tick_params(axis="both", labelsize=22, pad=8)
    fig.text(
        0.06, 0.015,
        "Each dot = one county; ±10% band. MAPE is an average, not a per-county guarantee.",
        fontsize=18, color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.98, 0.98))
    return save_figure(fig, "02_county_yield_validation")


def make_seasonal_figure(records):
    by_year_date = defaultdict(list)
    for row in records:
        by_year_date[(row["year"], row["doy"])].append(row)
    years = sorted({row["year"] for row in records})
    doys = sorted({row["doy"] for row in records})
    seasonal = {}
    for doy in doys:
        for year in years:
            sample = by_year_date.get((year, doy), [])
            if sample:
                seasonal[year, doy] = scores(sample)

    mean_mape = [statistics.mean(seasonal[y, d]["mape"] for y in years if (y, d) in seasonal) for d in doys]
    min_mape = [min(seasonal[y, d]["mape"] for y in years if (y, d) in seasonal) for d in doys]
    max_mape = [max(seasonal[y, d]["mape"] for y in years if (y, d) in seasonal) for d in doys]
    mean_r2 = [statistics.mean(seasonal[y, d]["r2"] for y in years if (y, d) in seasonal) for d in doys]
    min_r2 = [min(seasonal[y, d]["r2"] for y in years if (y, d) in seasonal) for d in doys]
    max_r2 = [max(seasonal[y, d]["r2"] for y in years if (y, d) in seasonal) for d in doys]

    fig, (ax_mape, ax_r2) = plt.subplots(1, 2, figsize=(19, 8.8))
    for ax in (ax_mape, ax_r2):
        ax.grid(axis="y", color=COLORS["grid"], linewidth=1.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
    ax_mape.axhspan(0, 10, color="#E9F3EF", zorder=0)
    ax_mape.axhline(10, color=COLORS["coral"], linestyle="--", linewidth=2.5)
    ax_mape.vlines(doys, min_mape, max_mape, color=COLORS["gray"], linewidth=5, alpha=0.55, zorder=2)
    ax_mape.plot(doys, mean_mape, color=COLORS["blue"], marker="o", markersize=12,
                 linewidth=4, zorder=3)
    ax_mape.scatter(doys, mean_mape, s=210, color=COLORS["blue"], edgecolors="white", linewidth=2, zorder=4)
    ax_mape.set_ylabel("MAPE (%)", fontweight="bold")
    ax_mape.set_title("Prediction error", loc="left", fontsize=24, fontweight="bold")
    ax_mape.text(0.98, 0.95, "Green zone = 10% target", transform=ax_mape.transAxes,
                 ha="right", va="top", fontsize=18, color=COLORS["teal"])
    ax_mape.text(0.98, 0.87, "Whisker = lowest to highest year", transform=ax_mape.transAxes,
                 ha="right", va="top", fontsize=17, color=COLORS["muted"])

    ax_r2.axhspan(0.6, 1, color="#E9F3EF", zorder=0)
    ax_r2.axhline(0.6, color=COLORS["coral"], linestyle="--", linewidth=2.5)
    ax_r2.vlines(doys, min_r2, max_r2, color=COLORS["gray"], linewidth=5, alpha=0.55, zorder=2)
    ax_r2.plot(doys, mean_r2, color=COLORS["teal"], marker="o", markersize=12,
               linewidth=4, zorder=3)
    ax_r2.scatter(doys, mean_r2, s=210, color=COLORS["teal"], edgecolors="white", linewidth=2, zorder=4)
    ax_r2.set_ylabel("R² (unitless)", fontweight="bold")
    ax_r2.set_title("County-to-county fit", loc="left", fontsize=24, fontweight="bold")
    ax_r2.set_ylim(0.5, 1.0)
    for ax in (ax_mape, ax_r2):
        ax.set_xticks(doys, [f"{date_label(d)}\n{d}" for d in doys])
        ax.tick_params(axis="both", labelsize=18, pad=8)
        ax.margins(x=0.08)
    ax_mape.set_xlabel("Observation date (calendar; DOY below)", fontsize=19, labelpad=12)
    ax_r2.set_xlabel("Observation date (calendar; DOY below)", fontsize=19, labelpad=12)

    mean_early = mean_mape[0]
    mean_late = mean_mape[-1]
    overall_mape = scores(records)["mape"]
    fig.suptitle("Accuracy improves as more of the growing season is observed", x=0.055, y=0.98,
                 ha="left", fontsize=29, fontweight="bold", color=COLORS["navy"])
    fig.text(
        0.055, 0.895,
        f"Mean annual MAPE: {mean_early:.1f}% at first checkpoint → {mean_late:.1f}% at last  |  "
        f"all-date county-record MAPE: {overall_mape:.1f}%  |  {len(years)} test years",
        fontsize=21, color=COLORS["muted"],
    )
    fig.text(
        0.055, 0.015,
        f"Same-year CDL training / published ICDL testing. Means are calculated across year-specific county cohorts ({min(years)}–{max(years)}); "
        "vertical whiskers show the observed year-to-year range, not a confidence interval. "
        "Data dates are 16 days apart.",
        fontsize=14.5, color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0.03, 0.065, 0.98, 0.86), w_pad=3.1)
    return save_figure(fig, "03_seasonal_accuracy")


def display_feature_name(name):
    replacements = {
        "hist_yield_trend": "Historical yield trend",
        "GCI_anom_mean": "Mean GCI anomaly",
        "EVI2_anom_last": "Late-season EVI2 anomaly",
        "NIRv_anom_mean": "Mean NIRv anomaly",
        "NDMI_anom_last": "Late-season NDMI anomaly",
        "season_ppt": "Seasonal precipitation",
        "season_vpdmean": "Seasonal mean VPD",
        "year": "Crop year",
    }
    return replacements.get(name, name.replace("_", " ").strip())


def make_feature_figure():
    rows = read_csv(IMPORTANCE_FILE)
    if not rows or not {"feature", "mean_gain_share"}.issubset(rows[0]):
        raise ValueError("Feature importance file is missing its expected columns.")
    ranked = sorted(rows, key=lambda row: float(row["mean_gain_share"]), reverse=True)[:8]
    labels = [display_feature_name(row["feature"]) for row in reversed(ranked)]
    shares = [100 * float(row["mean_gain_share"]) for row in reversed(ranked)]
    palette = [COLORS["gray"] if "trend" in label.lower() or label == "Crop year" else COLORS["blue"] for label in labels]

    fig, ax = plt.subplots(figsize=(17, 9))
    bars = ax.barh(np.arange(len(labels)), shares, color=palette, height=0.68, edgecolor="white", linewidth=1.2)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Share of model split gain (%)", fontweight="bold", labelpad=12)
    ax.set_xlim(0, max(shares) * 1.25)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("The model combines prior yield trends with crop-health signals", loc="left",
                 fontsize=27, fontweight="bold", color=COLORS["navy"], pad=22)
    for bar, share in zip(bars, shares):
        ax.text(share + max(shares) * 0.018, bar.get_y() + bar.get_height() / 2,
                f"{share:.1f}%", va="center", fontsize=22, fontweight="bold", color=COLORS["ink"])
    ax.tick_params(axis="x", labelsize=22, pad=8)
    ax.tick_params(axis="y", labelsize=22, pad=10)
    fig.text(
        0.06, 0.015,
        "50 same-year CDL-trained models; top 8 of 241 features. Importance reflects model use, not causation.\n"
        "GCI, EVI2, NIRv, and NDMI are satellite crop-health signals; VPD measures atmospheric dryness.",
        fontsize=17.5, color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0.045, 0.07, 0.98, 0.94))
    return save_figure(fig, "04_model_inputs_and_importance")


def write_poster_metrics(records):
    by_year_doy = defaultdict(list)
    by_doy = defaultdict(list)
    for row in records:
        by_year_doy[row["year"], row["doy"]].append(row)
        by_doy[row["doy"]].append(row)
    metrics_path = OUTPUT_DIR / "poster_key_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["training_mask", "testing_mask", "scope", "year", "doy", "county_records", "r2", "mape_percent", "rmse_bu_acre", "mae_bu_acre"])
        for (year, doy), sample in sorted(by_year_doy.items()):
            result = scores(sample)
            writer.writerow([TRAINING_MASK, TESTING_MASK, "year_date", year, doy, result["n"], result["r2"], result["mape"], result["rmse"], result["mae"]])
        for doy, sample in sorted(by_doy.items()):
            result = scores(sample)
            writer.writerow([TRAINING_MASK, TESTING_MASK, "pooled_date", f"{min(row['year'] for row in records)}-{max(row['year'] for row in records)}", doy, result["n"], result["r2"], result["mape"], result["rmse"], result["mae"]])
        result = scores(records)
        writer.writerow([TRAINING_MASK, TESTING_MASK, "all_county_date_records", f"{min(row['year'] for row in records)}-{max(row['year'] for row in records)}", "all", result["n"], result["r2"],
                         result["mape"], result["rmse"], result["mae"]])
    return metrics_path


def main():
    records = load_predictions()
    figures = [
        make_workflow_figure(records),
        make_validation_scatter(records),
        make_seasonal_figure(records),
        make_feature_figure(),
    ]
    metrics_path = write_poster_metrics(records)
    pdf_path = OUTPUT_DIR / "final_poster_graphics.pdf"
    with PdfPages(pdf_path, metadata={"Title": "Poster-ready corn yield model graphics"}) as pdf:
        for figure in figures:
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    print(f"Generated 4 poster graphics from {len(records):,} county-date predictions.")
    print(f"Saved 400-dpi PNGs, scalable SVGs, and four-page PDF to {OUTPUT_DIR}")
    print(f"Saved metrics table to {metrics_path}")
    final_rows = [row for row in records if row["year"] == 2025 and row["doy"] == 273]
    final_scores = scores(final_rows)
    print(
        f"2025 DOY 273 ({TRAINING_MASK}/{TESTING_MASK}): N={final_scores['n']}, R2={final_scores['r2']:.3f}, "
        f"MAPE={final_scores['mape']:.2f}%, RMSE={final_scores['rmse']:.2f} bu/ac."
    )


if __name__ == "__main__":
    main()
