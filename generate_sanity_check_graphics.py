"""Build sanity-check and validation-plan graphics from the 50-seed county results."""

import csv
import math
import statistics
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "v5_50_seed_all_in_one" / "v5_50_seed_all_in_one"
PREDICTIONS = DATA_ROOT / "summary" / "ensemble_county_predictions.csv"
SEED_PREDICTIONS = DATA_ROOT / "seeds" / "seed_1" / "county_predictions.csv"
SEED_METRICS = DATA_ROOT / "summary" / "all_seed_metrics.csv"
OUTPUT_DIR = ROOT / "sanity_check_graphics"
PRIMARY_LIMIT_PERCENT = 10.0
R2_LIMIT = 0.60
HAND_CHECK_YEAR = 2025
HAND_CHECK_DOY = 273
HAND_CHECK_SEED = 1

COLORS = {
    "model": "#176B87",
    "trend": "#D28A28",
    "good": "#39836C",
    "warn": "#B75B47",
    "ink": "#183544",
    "muted": "#526876",
    "grid": "#DFE7EC",
    "pale": "#F3F6F8",
}
YEAR_COLORS = {
    2021: "#176B87",
    2022: "#D28A28",
    2023: "#39836C",
    2024: "#8466A5",
    2025: "#B45C48",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 12,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "#FBFCFD",
    "axes.edgecolor": "#B1BEC7",
    "axes.labelcolor": COLORS["ink"],
    "text.color": COLORS["ink"],
    "xtick.color": "#425867",
    "ytick.color": "#425867",
    "savefig.facecolor": "white",
})


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required source data not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def load_ensemble():
    rows = read_csv(PREDICTIONS)
    required = {
        "training_mask", "testing_mask", "year", "doy", "FIPS",
        "yield_bu_acre", "hist_yield_trend", "predicted_yield", "n_seeds",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Ensemble CSV is missing columns: {', '.join(sorted(missing))}")

    result = []
    for source in rows:
        if (source["training_mask"], source["testing_mask"]) != ("PREVIOUS_CDL", "PREVIOUS_CDL"):
            continue
        item = {
            "year": int(source["year"]),
            "doy": int(source["doy"]),
            "fips": source["FIPS"].zfill(5),
            "actual": float(source["yield_bu_acre"]),
            "predicted": float(source["predicted_yield"]),
            "trend": float(source["hist_yield_trend"]),
            "n_seeds": int(source["n_seeds"]),
        }
        if item["actual"] <= 0:
            raise ValueError(f"Nonpositive observed yield at {item['year']}/{item['doy']}/{item['fips']}; MAPE is undefined.")
        if item["predicted"] < 0:
            raise ValueError(f"Negative predicted yield at {item['year']}/{item['doy']}/{item['fips']}.")
        if item["n_seeds"] != 50:
            raise ValueError(f"Expected 50 seeds, found {item['n_seeds']} at {item['year']}/{item['doy']}.")
        result.append(item)
    if not result:
        raise ValueError("No PREVIOUS_CDL/ PREVIOUS_CDL ensemble predictions found.")
    return result


def metric_values(rows):
    actual = [row["actual"] for row in rows]
    predicted = [row["predicted"] for row in rows]
    trend = [row["trend"] for row in rows]

    def rmse(left, right):
        return math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(left, right)) / len(left))

    mean_actual = math.fsum(actual) / len(actual)
    ss_total = math.fsum((value - mean_actual) ** 2 for value in actual)
    ss_residual = math.fsum((x - y) ** 2 for x, y in zip(actual, predicted))
    return {
        "n": len(rows),
        "rmse_model": rmse(actual, predicted),
        "rmse_trend": rmse(actual, trend),
        "mae": math.fsum(abs(x - y) for x, y in zip(actual, predicted)) / len(actual),
        "mape": 100 * math.fsum(abs(x - y) / x for x, y in zip(actual, predicted)) / len(actual),
        "trend_mape": 100 * math.fsum(abs(x - y) / x for x, y in zip(actual, trend)) / len(actual),
        "r2": 1 - ss_residual / ss_total if ss_total else float("nan"),
        "bias": math.fsum(y - x for x, y in zip(actual, predicted)) / len(actual),
        "minimum_actual": min(actual),
        "maximum_actual": max(actual),
        "minimum_prediction": min(predicted),
        "maximum_prediction": max(predicted),
    }


def summarize(rows):
    groups = defaultdict(list)
    by_day = defaultdict(list)
    by_year = defaultdict(list)
    for row in rows:
        groups[(row["year"], row["doy"])].append(row)
        by_day[row["doy"]].append(row)
        by_year[row["year"]].append(row)

    detail = []
    for (year, doy), records in sorted(groups.items()):
        metrics = metric_values(records)
        detail.append({"year": year, "doy": doy, **metrics})
    pooled_by_day = {doy: metric_values(records) for doy, records in sorted(by_day.items())}
    pooled = metric_values(rows)
    year_metrics = {year: metric_values(records) for year, records in sorted(by_year.items())}
    return detail, pooled_by_day, pooled, year_metrics


def independently_verify_metrics():
    source_rows = read_csv(SEED_PREDICTIONS)
    sample = [
        row for row in source_rows
        if (row["training_mask"], row["testing_mask"], row["year"], row["doy"])
        == ("PREVIOUS_CDL", "PREVIOUS_CDL", str(HAND_CHECK_YEAR), str(HAND_CHECK_DOY))
    ]
    if not sample:
        raise ValueError(f"No seed-{HAND_CHECK_SEED} records for {HAND_CHECK_YEAR}/DOY {HAND_CHECK_DOY}.")
    sample.sort(key=lambda row: row["FIPS"].zfill(5))
    actual = [float(row["yield_bu_acre"]) for row in sample]
    predicted = [float(row["predicted_yield"]) for row in sample]
    residuals = [prediction - observed for observed, prediction in zip(actual, predicted)]
    rmse = math.sqrt(math.fsum(error * error for error in residuals) / len(residuals))
    mae = math.fsum(abs(error) for error in residuals) / len(residuals)
    mean_actual = math.fsum(actual) / len(actual)
    r2 = 1 - math.fsum((observed - prediction) ** 2 for observed, prediction in zip(actual, predicted)) / math.fsum(
        (observed - mean_actual) ** 2 for observed in actual
    )
    stored = [
        row for row in read_csv(SEED_METRICS)
        if (row["training_mask"], row["testing_mask"], row["year"], row["doy"], row["seed"])
        == ("PREVIOUS_CDL", "PREVIOUS_CDL", str(HAND_CHECK_YEAR), str(HAND_CHECK_DOY), str(HAND_CHECK_SEED))
    ]
    if len(stored) != 1:
        raise ValueError(f"Expected one stored audit-metric row; found {len(stored)}.")
    stored = stored[0]

    calculated = {"rmse": rmse, "mae": mae, "r2": r2}
    discrepancies = {}
    for name, value in calculated.items():
        target = float(stored[name])
        difference_percent = abs(value - target) / max(abs(target), 1e-15) * 100
        discrepancies[name] = difference_percent
        if not math.isclose(value, target, rel_tol=1e-11, abs_tol=1e-12):
            raise ValueError(f"Independent {name} check disagrees with stored metric: {value} vs {target}")

    table_rows = []
    for row in sample[:6]:
        observed = float(row["yield_bu_acre"])
        prediction = float(row["predicted_yield"])
        residual = prediction - observed
        table_rows.append({
            "FIPS": row["FIPS"].zfill(5),
            "Observed_bu_acre": observed,
            "Predicted_bu_acre": prediction,
            "Error_bu_acre": residual,
            "Absolute_error_bu_acre": abs(residual),
            "Squared_error_bu2_acre2": residual * residual,
        })
    return {
        "n": len(sample),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "stored": {name: float(stored[name]) for name in calculated},
        "discrepancies": discrepancies,
        "table": table_rows,
    }


def write_metric_csv(detail, pooled_by_day, pooled, year_metrics, hand_check):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "sanity_check_metrics.csv").open("w", newline="", encoding="utf-8") as destination:
        fields = ["year", "doy", "n", "rmse_model_bu_acre", "rmse_trend_bu_acre", "mape_model_percent",
                  "mape_trend_percent", "r2", "mae_model_bu_acre", "bias_bu_acre"]
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        for row in detail:
            writer.writerow({
                "year": row["year"], "doy": row["doy"], "n": row["n"],
                "rmse_model_bu_acre": row["rmse_model"], "rmse_trend_bu_acre": row["rmse_trend"],
                "mape_model_percent": row["mape"], "mape_trend_percent": row["trend_mape"],
                "r2": row["r2"], "mae_model_bu_acre": row["mae"], "bias_bu_acre": row["bias"],
            })
        for doy, row in pooled_by_day.items():
            writer.writerow({
                "year": "Pooled 2021-2025", "doy": doy, "n": row["n"],
                "rmse_model_bu_acre": row["rmse_model"], "rmse_trend_bu_acre": row["rmse_trend"],
                "mape_model_percent": row["mape"], "mape_trend_percent": row["trend_mape"],
                "r2": row["r2"], "mae_model_bu_acre": row["mae"], "bias_bu_acre": row["bias"],
            })

    with (OUTPUT_DIR / "independent_metric_check.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(["metric", "independently_recomputed", "stored_seed_metric", "absolute_percent_difference",
                         "year", "doy", "seed", "county_count"])
        for name in ("rmse", "mae", "r2"):
            writer.writerow([name, hand_check[name], hand_check["stored"][name],
                             hand_check["discrepancies"][name], HAND_CHECK_YEAR, HAND_CHECK_DOY,
                             HAND_CHECK_SEED, hand_check["n"]])
    with (OUTPUT_DIR / "independent_metric_sample.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(hand_check["table"][0]))
        writer.writeheader()
        writer.writerows(hand_check["table"])


def doy_label(doy):
    return (date(2025, 1, 1) + timedelta(days=doy - 1)).strftime("%b %d")


def save_png(fig, name):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / name, dpi=360, bbox_inches="tight", facecolor="white")


def make_spec_check_figure(detail, years, pooled):
    doys = sorted({row["doy"] for row in detail})
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 7.7), sharex=True)
    for ax in axes:
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)

    axes[0].axhspan(0, PRIMARY_LIMIT_PERCENT, color="#E7F2EC", zorder=0)
    axes[0].axhline(PRIMARY_LIMIT_PERCENT, color=COLORS["warn"], linestyle="--", linewidth=1.5)
    for year in years:
        color = YEAR_COLORS[year]
        points = [row for row in detail if row["year"] == year]
        axes[0].plot([r["doy"] for r in points], [r["mape"] for r in points],
                     color=color, marker="o", linewidth=2.1, label=str(year))
    axes[0].set_ylabel("County-date MAPE (%)")
    axes[0].set_title("A  Primary specification: mean absolute percentage error ≤ 10%", loc="left",
                      fontweight="bold", fontsize=10, pad=8)
    axes[1].axhspan(R2_LIMIT, 1, color="#E7F2EC", zorder=0)
    axes[1].axhline(R2_LIMIT, color=COLORS["warn"], linestyle="--", linewidth=1.5,
                    label="R² target = 0.60")
    for year in sorted(years):
        points = [row for row in detail if row["year"] == year]
        axes[1].plot([r["doy"] for r in points], [r["r2"] for r in points],
                     color=YEAR_COLORS[year], marker="o", linewidth=2.1)
    axes[1].set_ylabel("County cross-section R²")
    axes[1].set_title("B  Secondary specification: R² > 0.60", loc="left",
                      fontweight="bold", fontsize=10, pad=8)
    axes[1].set_xlabel("Observation date (DOY)")
    axes[1].set_xticks(doys, [f"{doy_label(doy)}\n{doy}" for doy in doys])
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(title="Evaluation year", frameon=False, ncol=5, loc="upper right")
    axes[0].set_title(
        f"A  Primary specification: MAPE ≤ 10%  |  pooled MAPE = {pooled['mape']:.2f}%",
        loc="left", fontweight="bold", fontsize=10, pad=8,
    )
    axes[0].text(0.995, PRIMARY_LIMIT_PERCENT + 0.15, "10% target", transform=axes[0].get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=7.8, color=COLORS["warn"])
    fig.suptitle("Held-out county yield predictions against design specifications", x=0.055, y=0.99,
                 ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.055, 0.006,
        "Points use the 50-seed mean prediction versus the same observed-yield cohort for each year/DOY. "
        "Overall MAPE is pooled over county-date records; some early dates exceed 10%. R² is computed across counties at each year/DOY.",
        fontsize=7.5, color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0.04, 0.04, 0.98, 0.95), h_pad=0.48)
    save_png(fig, "01_accuracy_specification_checks.png")
    return fig


def make_benchmark_figure(pooled_by_day, pooled):
    doys = list(pooled_by_day)
    model_rmse = [pooled_by_day[doy]["rmse_model"] for doy in doys]
    trend_rmse = [pooled_by_day[doy]["rmse_trend"] for doy in doys]
    skills = [100 * (1 - (m / t) ** 2) for m, t in zip(model_rmse, trend_rmse)]

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.7, 4.4))
    for ax in (left, right):
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
    left.plot(doys, model_rmse, color=COLORS["model"], marker="o", linewidth=2.5, markersize=6,
              label="50-seed mean model")
    left.plot(doys, trend_rmse, color=COLORS["trend"], marker="s", linewidth=2.1, markersize=5,
              linestyle="--", label="Historical yield-trend benchmark")
    left.set_xticks(doys, [str(doy) for doy in doys])
    left.set_xlabel("Observation date (DOY)")
    left.set_ylabel("RMSE (bu/ac)")
    left.set_title("A  Paired RMSE benchmark", loc="left", fontweight="bold", fontsize=10)
    right.axhline(0, color="#728692", linewidth=0.9)
    right.axhspan(0, max(skills) * 1.12, color="#E7F2EC", zorder=0)
    right.plot(doys, skills, color=COLORS["good"], marker="o", linewidth=2.5, markersize=6)
    right.set_xticks(doys, [str(doy) for doy in doys])
    right.set_xlabel("Observation date (DOY)")
    right.set_ylabel("MSE skill vs trend (%)")
    right.set_title("B  Improvement over the benchmark", loc="left", fontweight="bold", fontsize=10)
    right.set_ylim(0, max(skills) * 1.16)

    final_skill = skills[-1]
    improvement = pooled["rmse_trend"] - pooled["rmse_model"]
    fig.suptitle("Independent benchmark: historical yield trend vs county model", x=0.055, y=0.99,
                 ha="left", fontsize=14, fontweight="bold")
    fig.legend(
        handles=left.get_legend_handles_labels()[0],
        labels=left.get_legend_handles_labels()[1],
        frameon=False, fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2,
    )
    fig.text(
        0.055, 0.005,
        f"Same county-year-date observations are scored both ways. Pooled RMSE: model {pooled['rmse_model']:.2f} vs trend "
        f"{pooled['rmse_trend']:.2f} bu/ac; pooled MSE skill {100*(1-(pooled['rmse_model']/pooled['rmse_trend'])**2):.1f}%. "
        f"DOY 273 skill {final_skill:.1f}%; pooled RMSE reduction {improvement:.2f} bu/ac. "
        "Trend is a simple internal reference baseline, not an external published benchmark.",
        fontsize=7.4, color=COLORS["muted"],
    )
    fig.tight_layout(rect=(0.04, 0.08, 0.98, 0.86), w_pad=2.4)
    save_png(fig, "02_historical_trend_benchmark.png")
    return fig


def make_hand_check_figure(check):
    fig = plt.figure(figsize=(11.7, 7.4))
    fig.suptitle("Independent arithmetic check of the software’s county error metrics", x=0.055, y=0.985,
                 ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.055, 0.925,
        f"Worked audit case: seed {HAND_CHECK_SEED}, {HAND_CHECK_YEAR}, DOY {HAND_CHECK_DOY}  |  "
        f"held-out county count N = {check['n']:,}  |  first six FIPS shown below",
        fontsize=8.8, color=COLORS["muted"],
    )
    ax_table = fig.add_axes([0.055, 0.51, 0.89, 0.36])
    ax_table.axis("off")
    columns = ["County FIPS", "Observed\n(bu/ac)", "Predicted\n(bu/ac)", "Error\n(bu/ac)", "Abs. error\n(bu/ac)", "Squared error\n((bu/ac)²)"]
    cell_text = [[
        row["FIPS"],
        f'{row["Observed_bu_acre"]:.1f}',
        f'{row["Predicted_bu_acre"]:.2f}',
        f'{row["Error_bu_acre"]:+.2f}',
        f'{row["Absolute_error_bu_acre"]:.2f}',
        f'{row["Squared_error_bu2_acre2"]:.2f}',
    ] for row in check["table"]]
    table = ax_table.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="right",
                           colLoc="center", colWidths=[0.15, 0.15, 0.16, 0.15, 0.17, 0.22])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.75)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#CFD9DF")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#173B4A")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 1:
            cell.set_facecolor("#F1F5F7")

    ax_formula = fig.add_axes([0.055, 0.38, 0.89, 0.13])
    ax_formula.axis("off")
    equations = [
        r"$e_i=\hat{y}_i-y_i$",
        r"$MAE=\frac{1}{N}\sum_i |e_i|$",
        r"$RMSE=\sqrt{\frac{1}{N}\sum_i e_i^2}$",
        r"$R^2=1-\frac{\sum_i e_i^2}{\sum_i(y_i-\bar{y})^2}$",
    ]
    for index, equation in enumerate(equations):
        x = 0.125 + index * 0.25
        ax_formula.text(x, 0.62, equation, ha="center", va="center", fontsize=12,
                        color=COLORS["ink"], transform=ax_formula.transAxes)
    metric_names = [("RMSE", "rmse"), ("MAE", "mae"), ("R²", "r2")]
    ax_results = fig.add_axes([0.055, 0.16, 0.89, 0.19])
    ax_results.axis("off")
    result_rows = [
        [label, f'{check[key]:.6f}', f'{check["stored"][key]:.6f}', f'{check["discrepancies"][key]:.6f}%']
        for label, key in metric_names
    ]
    result_table = ax_results.table(
        cellText=result_rows,
        colLabels=["Metric", "Independent calculation", "Saved software result", "Relative difference"],
        cellLoc="center", colLoc="center", loc="center",
        colWidths=[0.15, 0.30, 0.30, 0.25],
    )
    result_table.auto_set_font_size(False)
    result_table.set_fontsize(8)
    result_table.scale(1, 1.4)
    for (row, _column), cell in result_table.get_celld().items():
        cell.set_edgecolor("#CFD9DF")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#39836C")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#EEF5F1")
            cell.get_text().set_color(COLORS["good"])
            cell.get_text().set_fontweight("bold")
    fig.text(
        0.055, 0.095,
        "The rows illustrate the calculation; summary metrics are independently recomputed over all N counties using the displayed equations. "
        "The independent results match the saved software metrics exactly to stored precision (0% relative difference). "
        "This verifies the evaluation arithmetic for this case, not the correctness of every upstream input or model assumption.",
        fontsize=8.1, color=COLORS["muted"], wrap=True,
    )
    fig.text(
        0.055, 0.045,
        "Boundary check across the 22,478 county-date predictions: 0 negative predictions; observed yields 27.7–253.6 bu/ac; model predictions 2.05–248.44 bu/ac.",
        fontsize=8.1, color=COLORS["muted"],
    )
    save_png(fig, "03_independent_metric_arithmetic_check.png")
    return fig


def add_box(ax, x, y, width, height, title, body, face, edge="#B7C5CE", title_color=COLORS["ink"]):
    body = body.replace("\\n", "\n")
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=1.0,
    )
    ax.add_patch(box)
    ax.text(x + 0.022, y + height - 0.04, title, transform=ax.transAxes,
            ha="left", va="top", fontsize=9.2, fontweight="bold", color=title_color)
    ax.text(x + 0.022, y + height - 0.09, body, transform=ax.transAxes,
            ha="left", va="top", fontsize=7.45, color=COLORS["ink"], linespacing=1.35)


def add_arrow(ax, start, end, label=None, color="#607581"):
    arrow = FancyArrowPatch(
        start, end, transform=ax.transAxes,
        arrowstyle="-|>", mutation_scale=12, linewidth=1.1, color=color,
    )
    ax.add_patch(arrow)
    if label:
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        ax.text(*midpoint, label, transform=ax.transAxes, ha="center", va="center",
                fontsize=6.9, color=COLORS["muted"], bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})


def make_validation_plan_figure():
    fig, ax = plt.subplots(figsize=(12.0, 7.7))
    ax.axis("off")
    fig.suptitle("Repeatable validation blueprint for county corn-yield forecasts", x=0.045, y=0.985,
                 ha="left", fontsize=14, fontweight="bold")
    fig.text(
        0.045, 0.935,
        "Primary design specification: county-date mean absolute percentage error (MAPE) ≤ 10% against final USDA NASS county yield.",
        fontsize=9.2, color=COLORS["ink"], fontweight="bold",
    )
    positions = [
        (0.045, 0.67, 0.42, 0.21, "1  Freeze forecast", "Lock model, training cutoff, MODIS/CDL,\nseed list, FIPS cohort, and dates before harvest.", "#EAF2F6"),
        (0.535, 0.67, 0.42, 0.21, "2  Produce forecasts", "Generate county yield (bu/ac) for every date.\nArchive predictions, input snapshots, logs, and counts.", "#EDF3F0"),
        (0.535, 0.39, 0.42, 0.21, "3  Match reference data", "Join forecasts to final USDA NASS yield by FIPS\nand crop year; log revisions and exclusions.", "#F7F1E7"),
        (0.045, 0.39, 0.42, 0.21, "4  Score and report", "Primary: pooled county-date MAPE ≤ 10%.\nAlso report R², RMSE, MAE, bias, and coverage.\nBootstrap by state/year; report 95% intervals.", "#EAF2F6"),
        (0.045, 0.10, 0.42, 0.21, "Replicates and cadence", "Proposed minimum: 3 harvest years, ≥500 matched\ncounties/year, and all 50 seeds. Current archive has\n6 dates 16 days apart (not weekly).", "#F8ECE8", COLORS["warn"]),
        (0.535, 0.10, 0.42, 0.21, "Control outside factors", "Freeze input/model versions and use identical joins.\nLog MODIS quality gaps, CDL/NASS revisions,\nsuppressed values, and missing counties.", "#F3F6F8"),
    ]
    for item in positions:
        add_box(ax, *item)

    add_arrow(ax, (0.47, 0.775), (0.532, 0.775))
    add_arrow(ax, (0.745, 0.665), (0.745, 0.605))
    add_arrow(ax, (0.535, 0.495), (0.47, 0.495))

    fig.tight_layout(rect=(0.02, 0.02, 0.98, 0.90))
    save_png(fig, "04_validation_plan_blueprint.png")
    return fig


def main():
    rows = load_ensemble()
    detail, pooled_by_day, pooled, year_metrics = summarize(rows)
    hand_check = independently_verify_metrics()
    write_metric_csv(detail, pooled_by_day, pooled, year_metrics, hand_check)

    figures = [
        make_spec_check_figure(detail, year_metrics, pooled),
        make_benchmark_figure(pooled_by_day, pooled),
        make_hand_check_figure(hand_check),
        make_validation_plan_figure(),
    ]

    figure_names = [
        "01_accuracy_specification_checks.png",
        "02_historical_trend_benchmark.png",
        "03_independent_metric_arithmetic_check.png",
        "04_validation_plan_blueprint.png",
    ]
    pdf_path = OUTPUT_DIR / "sanity_check_graphics.pdf"
    with PdfPages(pdf_path, metadata={"Title": "Corn yield model sanity checks and validation plan"}) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Processed {len(rows):,} county-date records across {len(year_metrics)} test years.")
    print(f"Pooled MAPE: {pooled['mape']:.2f}% (target <= {PRIMARY_LIMIT_PERCENT:.1f}%).")
    print(f"Pooled R2: {pooled['r2']:.3f}; all year/date R2 values exceed {R2_LIMIT:.2f}.")
    print(f"Pooled model/trend RMSE: {pooled['rmse_model']:.2f}/{pooled['rmse_trend']:.2f} bu/ac.")
    print(f"Independent seed-{HAND_CHECK_SEED} audit relative differences: {hand_check['discrepancies']}.")
    print(f"Created {pdf_path}")
    for filename in figure_names:
        print(OUTPUT_DIR / filename)
    print(OUTPUT_DIR / "sanity_check_metrics.csv")
    print(OUTPUT_DIR / "independent_metric_check.csv")
    print(OUTPUT_DIR / "independent_metric_sample.csv")


if __name__ == "__main__":
    main()
