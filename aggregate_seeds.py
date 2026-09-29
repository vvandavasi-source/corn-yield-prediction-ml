#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
aggregate_seeds.py

Multi-seed stability analysis, following Feng et al. (2020), Remote Sensing
12(12):2028 (Tables 5-6): run the model many times under different random
seeds, report MEAN +/- SD of each metric, and use a PAIRED test to compare
two scenarios seed-by-seed.

This answers a DIFFERENT question than run_stat_tests.py:
  - run_stat_tests.py  : is the gap real given the sample of COUNTIES?
                         (bootstrap / DM, model held fixed)
  - aggregate_seeds.py : is the gap real given the randomness of TRAINING?
                         (paired t-test across SEEDS, counties held fixed)
A difference that survives both is robust.

Input: a parent folder containing one subfolder per seed, each with a
V5 `model_comparison_metrics.csv`, e.g.
    v5_seeds/seed_1/model_comparison_metrics.csv
    v5_seeds/seed_2/model_comparison_metrics.csv
    ...
Each file has columns:
    training_mask, testing_mask, year, doy, r2, rmse, mae,
    anomaly_r2, mse_skill_vs_trend, ...

A "scenario" is the (training_mask -> testing_mask) pair. Defaults:
    A = PREVIOUS_CDL  -> PREVIOUS_CDL      (operational previous-year CDL)
    B = SAME_YEAR_CDL -> STUDY             (in-season published ICDL)

Outputs (to --output-dir):
    seed_summary_mean_sd.csv      Table-5 style: mean +/- SD per scenario/metric
    paired_tests_<A>_vs_<B>.csv   Table-6 style: paired t-test + Wilcoxon per year/DOY
    fig_seed_meanSD_<metric>.png  bar chart of mean with SD error bars, per scenario

Usage:
    python aggregate_seeds.py --seeds-dir .\v5_seeds --output-dir .\seed_out
    python aggregate_seeds.py --seeds-dir .\v5_seeds \
        --a-train PREVIOUS_CDL --a-test PREVIOUS_CDL \
        --b-train SAME_YEAR_CDL --b-test STUDY --metric rmse
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["r2", "rmse", "mae", "anomaly_r2", "mse_skill_vs_trend"]


def load_seeds(seeds_dir):
    """Read every seed's metrics file; tag each with its seed folder name."""
    files = sorted(glob.glob(os.path.join(seeds_dir, "*", "model_comparison_metrics.csv")))
    if not files:
        # also allow flat files like metrics_seed1.csv
        files = sorted(glob.glob(os.path.join(seeds_dir, "*model_comparison_metrics*.csv")))
    if not files:
        raise SystemExit(f"No model_comparison_metrics.csv found under {seeds_dir}")
    frames = []
    for i, f in enumerate(files):
        d = pd.read_csv(f)
        seed_tag = os.path.basename(os.path.dirname(f)) or f"file{i}"
        d["seed"] = seed_tag
        frames.append(d)
    allm = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(files)} seed files, {allm.seed.nunique()} distinct seeds.")
    return allm, len(files)


def scenario_slice(df, train, test):
    return df[(df.training_mask == train) & (df.testing_mask == test)].copy()


def mean_sd_table(df):
    """Table-5 style: per (scenario, year, doy) mean +/- SD across seeds, plus a
    per-scenario overall row."""
    df = df.copy()
    df["scenario"] = df.training_mask + " -> " + df.testing_mask
    rows = []
    for (scen, yr, doy), g in df.groupby(["scenario", "year", "doy"]):
        rec = {"scenario": scen, "year": yr, "doy": doy, "n_seeds": g.seed.nunique()}
        for m in METRICS:
            if m in g:
                rec[f"{m}_mean"] = g[m].mean()
                rec[f"{m}_sd"] = g[m].std(ddof=1)
        rows.append(rec)
    per_cell = pd.DataFrame(rows).sort_values(["scenario", "year", "doy"])

    # overall per-scenario (average over all year x DOY, then across seeds)
    overall = []
    for scen, g in df.groupby("scenario"):
        rec = {"scenario": scen, "year": "ALL", "doy": "ALL", "n_seeds": g.seed.nunique()}
        # average each seed's mean-over-cells, then summarize across seeds
        per_seed = g.groupby("seed")[METRICS].mean()
        for m in METRICS:
            rec[f"{m}_mean"] = per_seed[m].mean()
            rec[f"{m}_sd"] = per_seed[m].std(ddof=1)
        overall.append(rec)
    return pd.concat([per_cell, pd.DataFrame(overall)], ignore_index=True)


def paired_tests(df, A, B, metric):
    """Table-6 style: for each (year, doy), pair scenario A and B seed-by-seed and
    run paired t-test + Wilcoxon on the chosen metric. Also a pooled row."""
    a_tr, a_te = A
    b_tr, b_te = B
    a = scenario_slice(df, a_tr, a_te)[["seed", "year", "doy", metric]].rename(columns={metric: "A"})
    b = scenario_slice(df, b_tr, b_te)[["seed", "year", "doy", metric]].rename(columns={metric: "B"})
    merged = a.merge(b, on=["seed", "year", "doy"])
    if merged.empty:
        raise SystemExit("No paired rows: check the four scenario names against the file.")

    rows = []

    def run_pair(sub, year, doy):
        sub = sub.dropna(subset=["A", "B"])
        if len(sub) < 3:
            return None
        diff = sub.A.to_numpy() - sub.B.to_numpy()
        mean_a, mean_b = sub.A.mean(), sub.B.mean()
        sd_a, sd_b = sub.A.std(ddof=1), sub.B.std(ddof=1)
        # paired t-test
        t_stat, t_p = stats.ttest_rel(sub.A, sub.B)
        # Wilcoxon signed-rank (non-parametric backup); guard zero-diff
        try:
            w_stat, w_p = stats.wilcoxon(sub.A, sub.B)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        # for error metrics (rmse/mae) lower is better -> B better if mean_b<mean_a
        lower_better = metric in ("rmse", "mae")
        if lower_better:
            better = "B" if mean_b < mean_a else "A"
        else:
            better = "B" if mean_b > mean_a else "A"
        sig = (t_p < 0.05)
        return dict(year=year, doy=doy, n_seeds=len(sub), metric=metric,
                    A_mean=mean_a, A_sd=sd_a, B_mean=mean_b, B_sd=sd_b,
                    mean_diff_A_minus_B=diff.mean(), t_stat=t_stat, t_p=t_p,
                    wilcoxon_stat=w_stat, wilcoxon_p=w_p,
                    better=better, significant=bool(sig))

    for (yr, doy), sub in merged.groupby(["year", "doy"]):
        r = run_pair(sub, yr, doy)
        if r:
            rows.append(r)
    # pooled across all year x DOY (each seed's mean over cells -> one paired value per seed)
    pooled = merged.groupby("seed")[["A", "B"]].mean().reset_index()
    r = run_pair(pooled.assign(year="ALL", doy="ALL"), "ALL", "ALL")
    if r:
        rows.append(r)
    return pd.DataFrame(rows)


def plot_mean_sd(summary, A, B, metric, path):
    """Bar chart: mean metric with SD error bars, A vs B, per year x DOY (overall row too)."""
    a_lab = f"{A[0]} -> {A[1]}"
    b_lab = f"{B[0]} -> {B[1]}"
    s = summary[summary.scenario.isin([a_lab, b_lab])].copy()
    s = s[(s.year != "ALL")]
    s["cell"] = s.year.astype(str) + "\nD" + s.doy.astype(str)
    cells = sorted(s.cell.unique())
    x = np.arange(len(cells)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(cells) * 0.5), 6))
    for lab, off, col in [(a_lab, -w / 2, "#2463a5"), (b_lab, w / 2, "#803ea0")]:
        sub = s[s.scenario == lab].set_index("cell").reindex(cells)
        ax.bar(x + off, sub[f"{metric}_mean"], w, yerr=sub[f"{metric}_sd"], capsize=3,
               label=lab, color=col, alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(cells, fontsize=7)
    ax.set_ylabel(f"{metric}  (mean +/- SD over seeds)")
    ax.set_title(f"Seed-stability: {metric} by year x DOY  ({a_lab}  vs  {b_lab})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-dir", required=True, help="parent folder of per-seed result subfolders")
    ap.add_argument("--output-dir", default="./seed_out")
    ap.add_argument("--a-train", default="PREVIOUS_CDL")
    ap.add_argument("--a-test", default="PREVIOUS_CDL")
    ap.add_argument("--b-train", default="SAME_YEAR_CDL")
    ap.add_argument("--b-test", default="STUDY")
    ap.add_argument("--metric", default="rmse", choices=METRICS,
                    help="metric for the paired test & bar chart (default rmse)")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df, n_files = load_seeds(args.seeds_dir)
    A = (args.a_train, args.a_test)
    B = (args.b_train, args.b_test)

    # available scenarios sanity print
    scen = (df.training_mask + " -> " + df.testing_mask).unique()
    print("Scenarios present:", sorted(scen))
    if n_files < 3:
        print("WARNING: fewer than 3 seed files found - paired tests need several seeds "
              "(the paper used 50; ~20 recommended).")

    # Table 5
    summary = mean_sd_table(df)
    summary.to_csv(os.path.join(args.output_dir, "seed_summary_mean_sd.csv"), index=False)

    # Table 6
    tag = f"{A[1]}_vs_{B[1]}"
    pt = paired_tests(df, A, B, args.metric)
    pt.to_csv(os.path.join(args.output_dir, f"paired_tests_{tag}.csv"), index=False)

    # figure
    plot_mean_sd(summary, A, B, args.metric,
                 os.path.join(args.output_dir, f"fig_seed_meanSD_{args.metric}_{tag}.png"))

    # console summary
    ov = pt[pt.year == "ALL"]
    print("\n=== PAIRED TEST (pooled over all year x DOY), metric =", args.metric, "===")
    if not ov.empty:
        r = ov.iloc[0]
        print(f"  {A[0]}->{A[1]}: {r.A_mean:.3f} +/- {r.A_sd:.3f}")
        print(f"  {B[0]}->{B[1]}: {r.B_mean:.3f} +/- {r.B_sd:.3f}")
        print(f"  paired t p = {r.t_p:.4g} | Wilcoxon p = {r.wilcoxon_p:.4g} "
              f"| better = {r.better} | seeds = {int(r.n_seeds)}")
    n_sig = int((pt[pt.year != 'ALL'].t_p < 0.05).sum())
    n_cells = int((pt.year != 'ALL').sum())
    print(f"\nPer year x DOY: paired-t p<0.05 in {n_sig}/{n_cells} cells.")
    print(f"\nSaved to {args.output_dir}:")
    print("  seed_summary_mean_sd.csv   (Table-5 style: mean +/- SD)")
    print(f"  paired_tests_{tag}.csv     (Table-6 style: paired t + Wilcoxon)")
    print(f"  fig_seed_meanSD_{args.metric}_{tag}.png")


if __name__ == "__main__":
    main()
