#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_figures.py

Reproduces the summary figures and headline table for the dual-alternative
corn-yield comparison from the model's Cell-18 output CSVs.

Inputs (written by CY2_COMPACT_PRISM_CDL_ICDL_2022_2025_DUAL_PATCHED_v2.py):
    cell18_2022_2025_PRIMARY_AlternativeA_vs_B.csv
    cell18_2022_2025_Davidson_MacKinnon_JTest.csv

Outputs (written to OUTPUT_DIR):
    fig_gap_rmse_improvement.png    RMSE reduction from in-season, by year
    fig_jtest_verdict_grid.png      J-test verdict heatmap
    headline_summary_table.csv      per-year A-vs-B summary

Usage:
    python make_figures.py
    python make_figures.py --results-dir "<path>" --output-dir "<path>"
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

# ------------------------------------------------------------------
# Defaults — edit to your machine, or pass --results-dir / --output-dir
# ------------------------------------------------------------------
DEFAULT_RESULTS_DIR = (
    r"C:\Users\Patron\Downloads\Author's ICDL\11_MODEL_RESULTS"
    r"\cell18_2022_2025_dual_alternative_results"
)

YEARS = [2022, 2023, 2024, 2025]
DOYS = [161, 177, 193, 209, 225, 241, 257, 273]
YEAR_COLORS = {2022: "#1f77b4", 2023: "#2ca02c", 2024: "#ff7f0e", 2025: "#d62728"}
EARLY_DOY = 177  # early-season checkpoint for the headline table


def gap_figure(prim, ncty, out_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for y in YEARS:
        d = prim[prim.Test_Year == y].sort_values("DOY")
        improv = -d["Delta_RMSE_B_minus_A"].values  # A_RMSE - B_RMSE; + = in-season better
        ax.plot(d["DOY"], improv, marker="o", color=YEAR_COLORS[y],
                label=f"{y}  (n={ncty.get(y, '?')})", linewidth=2)
    ax.axhline(0, color="0.4", lw=1, ls="--")
    ax.set_xlabel("Day of Year (checkpoint)")
    ax.set_ylabel("RMSE reduction from in-season ICDL  (bu/ac)\n"
                  "\u2190 previous-CDL better   |   in-season better \u2192")
    ax.set_title("Yield-forecast skill gained from in-season crop mask, by year")
    ax.set_xticks(DOYS)
    ax.grid(alpha=0.3)
    ax.legend(title="Test year", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def jtest_grid(jt, ncty, out_path):
    def cat(row):
        rp = row["Reject_Previous_CDL"]   # in-season adds info beyond previous
        ri = row["Reject_InSeason_ICDL"]  # previous adds info beyond in-season
        if rp and not ri:
            return 3
        if rp and ri:
            return 2
        if ri and not rp:
            return 1
        return 0
    jt = jt.copy()
    jt["cat"] = jt.apply(cat, axis=1)
    grid = jt.pivot(index="Test_Year", columns="DOY", values="cat").reindex(index=YEARS, columns=DOYS)
    pval = jt.pivot(index="Test_Year", columns="DOY", values="p_val_ICDL").reindex(index=YEARS, columns=DOYS)

    cmap = ListedColormap(["#dcdcdc", "#f4a582", "#fddbc7", "#4393c3"])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.imshow(grid.values, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto")
    ax.set_xticks(range(len(DOYS))); ax.set_xticklabels(DOYS)
    ax.set_yticks(range(len(YEARS)))
    ax.set_yticklabels([f"{y}\n(n={ncty.get(y, '?')})" for y in YEARS])
    ax.set_xlabel("Day of Year (checkpoint)")
    ax.set_title("Davidson\u2013MacKinnon J-test verdicts (cluster-robust by state)\n"
                 "annotated with p-value that in-season ICDL adds information")
    for i, y in enumerate(YEARS):
        for j, dd in enumerate(DOYS):
            p = pval.loc[y, dd]
            if pd.isna(p):
                continue
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            ax.text(j, i, f"{p:.3f}\n{star}", ha="center", va="center", fontsize=8)
    legend = [Patch(facecolor="#4393c3", label="In-season adds info; previous redundant"),
              Patch(facecolor="#fddbc7", label="Both carry unique info"),
              Patch(facecolor="#f4a582", label="Previous adds info; in-season redundant"),
              Patch(facecolor="#dcdcdc", label="Neither distinguishable")]
    ax.legend(handles=legend, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def headline_table(prim, jt, ncty, out_path):
    rows = []
    for y in YEARS:
        d = prim[prim.Test_Year == y]
        jy = jt[jt.Test_Year == y]
        e = d[d.DOY == EARLY_DOY]
        if e.empty:
            continue
        e = e.iloc[0]
        rows.append({
            "Year": y, "Counties": ncty.get(y, np.nan),
            "Early A R2": round(e.Alternative_A_R2, 3),
            "Early B R2": round(e.Alternative_B_R2, 3),
            "Early dR2 (B-A)": round(e.Delta_R2_B_minus_A, 3),
            "Early dRMSE (B-A)": round(e.Delta_RMSE_B_minus_A, 2),
            "Season mean dR2 (B-A)": round(d.Delta_R2_B_minus_A.mean(), 3),
            "In-season adds info (of 8)": int(jy.Reject_Previous_CDL.sum()),
            "Previous adds info (of 8)": int(jy.Reject_InSeason_ICDL.sum()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--output-dir", default=None,
                    help="Where to write figures (default: results-dir).")
    args = ap.parse_args()
    rdir = args.results_dir
    odir = args.output_dir or rdir
    os.makedirs(odir, exist_ok=True)

    prim = pd.read_csv(os.path.join(rdir, "cell18_2022_2025_PRIMARY_AlternativeA_vs_B.csv"))
    jt = pd.read_csv(os.path.join(rdir, "cell18_2022_2025_Davidson_MacKinnon_JTest.csv"))

    # county count per year = N_Obs from the J-test (common county set, constant within a year)
    ncol = next((c for c in ["N_Obs", "n_obs", "N", "N_Counties"] if c in jt.columns), None)
    if ncol:
        ncty = jt.groupby("Test_Year")[ncol].max().astype(int).to_dict()
    else:
        ncty = {}

    gap_figure(prim, ncty, os.path.join(odir, "fig_gap_rmse_improvement.png"))
    jtest_grid(jt, ncty, os.path.join(odir, "fig_jtest_verdict_grid.png"))
    headline_table(prim, jt, ncty, os.path.join(odir, "headline_summary_table.csv"))
    print(f"\nWrote figures + headline table to: {odir}")


if __name__ == "__main__":
    main()
