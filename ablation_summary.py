#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ablation_summary.py  --  collate the feature-ablation runs into one comparison,
broken down by crop-mask (CDL) scenario.

Reads a parent folder with one subfolder per ablation config, each containing a
V5 model_comparison_metrics.csv:
    ablation/none/model_comparison_metrics.csv
    ablation/weather/...
    ablation/soil/...
    ablation/anomaly/...
    ablation/veg_only/...

The config is taken from the 'ablation' column if present, else from the folder
name. Each run evaluated every mask scenario, so we report metrics per
(ablation x operating scenario) and the RMSE change when each feature group is
removed (feature-group contribution).

Operating scenarios (canonical training->testing pairs):
    A  Prev-year CDL      PREVIOUS_CDL  -> PREVIOUS_CDL
    B  Published ICDL     SAME_YEAR_CDL -> STUDY
    C  Our ICDL           SAME_YEAR_CDL -> CLEAN_REBUILT
    Oracle  Same-year CDL SAME_YEAR_CDL -> FINAL_CDL

Outputs (to --outdir):
    ablation_summary.csv            mean RMSE / skill per (ablation, scenario) + dRMSE vs full
    fig_ablation_rmse_by_mask.png   grouped bars: RMSE by scenario, one bar per config
    fig_feature_group_contribution.png   dRMSE (removed - full) per group, per scenario

Usage:
    python ablation_summary.py --ablation-dir ./ablation --outdir ./ablation/figs
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAL = {"none": "#444444", "weather": "#2E6FB7", "soil": "#8C6D31",
       "anomaly": "#E07B39", "veg_only": "#3BA99C"}
CONFIG_ORDER = ["none", "weather", "soil", "anomaly", "veg_only"]
CONFIG_LABEL = {"none": "Full model", "weather": "− Weather", "soil": "− Soil",
                "anomaly": "− Anomalies", "veg_only": "Vegetation only"}
# canonical (training_mask, testing_mask) -> scenario label
SCENARIOS = [
    (("PREVIOUS_CDL", "PREVIOUS_CDL"),  "A: Prev-year CDL"),
    (("SAME_YEAR_CDL", "STUDY"),        "B: Published ICDL"),
    (("SAME_YEAR_CDL", "CLEAN_REBUILT"),"C: Our ICDL"),
    (("SAME_YEAR_CDL", "FINAL_CDL"),    "Oracle: Same-year CDL"),
]
SCEN_ORDER = [lab for _, lab in SCENARIOS]
PAIR2LAB = {pair: lab for pair, lab in SCENARIOS}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
                     "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.grid": True, "grid.color": "#DADADA", "grid.linewidth": 0.7})


def load(ablation_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(ablation_dir, "*", "model_comparison_metrics.csv"))):
        cfg_folder = os.path.basename(os.path.dirname(f))
        d = pd.read_csv(f)
        if "ablation" not in d.columns:
            d["ablation"] = cfg_folder
        rows.append(d)
    if not rows:
        raise SystemExit(f"No model_comparison_metrics.csv found under {ablation_dir}/*/")
    m = pd.concat(rows, ignore_index=True)
    # keep only the canonical operating pairs, label them
    m["scenario"] = list(zip(m["training_mask"], m["testing_mask"]))
    m["scenario"] = m["scenario"].map(PAIR2LAB)
    m = m[m["scenario"].notna()].copy()
    return m


def summarize(m, metric="rmse"):
    g = (m.groupby(["ablation", "scenario"])[[metric, "mse_skill_vs_trend"]]
         .mean().reset_index()
         .rename(columns={metric: f"{metric}_mean", "mse_skill_vs_trend": "skill_mean"}))
    # dRMSE vs the full model (ablation == none)
    full = g[g.ablation == "none"].set_index("scenario")[f"{metric}_mean"]
    g["d_vs_full"] = g.apply(lambda r: r[f"{metric}_mean"] - full.get(r.scenario, np.nan), axis=1)
    # order
    g["ablation"] = pd.Categorical(g.ablation, [c for c in CONFIG_ORDER if c in g.ablation.unique()], ordered=True)
    g["scenario"] = pd.Categorical(g.scenario, [s for s in SCEN_ORDER if s in g.scenario.unique()], ordered=True)
    return g.sort_values(["scenario", "ablation"])


def fig_rmse_by_mask(g, outdir, metric="rmse"):
    scen = [s for s in SCEN_ORDER if s in g.scenario.unique()]
    cfgs = [c for c in CONFIG_ORDER if c in g.ablation.unique()]
    x = np.arange(len(scen)); w = 0.8 / len(cfgs)
    fig, ax = plt.subplots(figsize=(max(9, len(scen) * 1.8), 5.6))
    for k, cfg in enumerate(cfgs):
        sub = g[g.ablation == cfg].set_index("scenario").reindex(scen)
        ax.bar(x + (k - (len(cfgs) - 1) / 2) * w, sub[f"{metric}_mean"], w,
               color=PAL.get(cfg, "#888"), label=CONFIG_LABEL.get(cfg, cfg))
    ax.set_xticks(x); ax.set_xticklabels(scen, fontsize=9)
    ax.set_ylabel(f"mean {metric.upper()} (bu/ac) over all year × DOY")
    ax.set_title("Feature ablation — forecast error by crop-mask scenario")
    ax.legend(ncol=len(cfgs), fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="x", alpha=0)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_ablation_rmse_by_mask.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig_ablation_rmse_by_mask.png")


def fig_contribution(g, outdir, metric="rmse"):
    # how much RMSE rises when each group is removed, per scenario (exclude 'none')
    scen = [s for s in SCEN_ORDER if s in g.scenario.unique()]
    cfgs = [c for c in CONFIG_ORDER if c != "none" and c in g.ablation.unique()]
    x = np.arange(len(cfgs)); w = 0.8 / max(len(scen), 1)
    fig, ax = plt.subplots(figsize=(max(8, len(cfgs) * 1.7), 5.4))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(scen)))
    for k, (s, col) in enumerate(zip(scen, cmap)):
        sub = g[g.scenario == s].set_index("ablation").reindex(cfgs)
        ax.bar(x + (k - (len(scen) - 1) / 2) * w, sub["d_vs_full"], w, color=col, label=s)
    ax.axhline(0, color="#444", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([CONFIG_LABEL.get(c, c) for c in cfgs])
    ax.set_ylabel(f"Δ {metric.upper()} vs full model  (removed − full, bu/ac)")
    ax.set_title("Feature-group contribution — RMSE increase when a group is removed\n"
                 "(taller bar = the group carried more of the forecast skill)")
    ax.legend(fontsize=8.5)
    ax.grid(axis="x", alpha=0)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig_feature_group_contribution.png"), bbox_inches="tight")
    plt.close(fig)
    print("  [ok] fig_feature_group_contribution.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ablation-dir", required=True, help="parent folder with one subfolder per config")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--metric", default="rmse")
    args = ap.parse_args()
    outdir = args.outdir or os.path.join(args.ablation_dir, "figs")
    os.makedirs(outdir, exist_ok=True)

    m = load(args.ablation_dir)
    g = summarize(m, args.metric)
    g.to_csv(os.path.join(outdir, "ablation_summary.csv"), index=False)

    fig_rmse_by_mask(g, outdir, args.metric)
    fig_contribution(g, outdir, args.metric)

    # console table
    piv = g.pivot(index="ablation", columns="scenario", values=f"{args.metric}_mean")
    print(f"\nMean {args.metric.upper()} by ablation × scenario:")
    print(piv.to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nSaved ablation_summary.csv + 2 figures to {outdir}")


if __name__ == "__main__":
    main()
