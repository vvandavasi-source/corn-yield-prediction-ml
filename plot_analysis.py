#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_analysis.py  --  one plotting module for the corn-yield crop-mask study.

It turns the CSV outputs of three things into a gallery of publication-style
figures, and it serves BOTH course deliverables:

  * FMEA (Part #1)       -> robustness / correctness evidence: significance,
                            calibration, seed stability, county coverage.
  * Modeling assignment  -> OAT sensitivity plots, pairwise comparison matrix,
                            and a weighted decision-matrix template.

Input files it knows how to read (all optional -- it draws whatever it finds):

  --seed-summary   seed_summary_mean_sd.csv        (from aggregate_seeds.py)
  --paired-tests   paired_tests_<A>_vs_<B>.csv     (from aggregate_seeds.py)
  --stat-tests     stat_tests_<A>_vs_<B>.csv       (from run_stat_tests.py)
  --seed-metrics   a dir of per-seed model_comparison_metrics.csv
                   (same layout aggregate_seeds.py reads: <dir>/*/...csv)
  --sensitivity    sensitivity_oat.csv   (you build this; schema below)
  --pairwise       pairwise.csv          (optional; sensible default used if absent)
  --decision       decision.csv          (optional; template default used if absent)

sensitivity_oat.csv schema (One-At-a-Time sweep -- one row per run):
    param,level_pct,rmse
    n_trees,-25,21.8
    n_trees,-10,20.9
    n_trees,0,20.4
    ...
  'level_pct' is the % change applied to that input (e.g. -25,-10,0,10,25);
  level_pct == 0 is the baseline run. 'rmse' is any output metric you logged;
  rename the column with --sens-metric if you logged something else.

Quick start -- SEE EVERYTHING before your real runs finish:
    python plot_analysis.py --demo --outdir figs_demo
This fabricates small, realistic input CSVs, runs the whole pipeline, and
writes every figure plus figures_index.md describing each one.

Real use:
    python plot_analysis.py \
        --seed-summary seed_out/seed_summary_mean_sd.csv \
        --paired-tests seed_out/paired_tests_PREVIOUS_CDL_vs_STUDY.csv \
        --stat-tests   stat_out/stat_tests_PREVIOUS_CDL_vs_STUDY.csv \
        --seed-metrics v5_seeds \
        --sensitivity  sensitivity_oat.csv \
        --a-label "Previous-season CDL (A)" \
        --b-label "In-season ICDL / STUDY (B)" \
        --outdir figs
"""

import os
import glob
import argparse
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------
# house style  (colourblind-safe blue/orange pair; consistent everywhere)
# ----------------------------------------------------------------------
PAL = {
    "A":      "#2E6FB7",   # Alternative A  -- previous-season CDL
    "B":      "#E07B39",   # Alternative B  -- in-season ICDL / STUDY
    "oracle": "#3BA99C",   # same-year / final CDL (oracle)
    "other":  "#7D7D7D",
    "trend":  "#9AA0A6",
    "sig":    "#2E8B57",   # significant
    "nsig":   "#B9321A",   # not significant
    "grid":   "#D9D9D9",
    "ink":    "#222222",
}
METRICS = ["r2", "rmse", "mae", "anomaly_r2", "mse_skill_vs_trend"]
METRIC_LABEL = {
    "r2": "R$^2$", "rmse": "RMSE (bu/ac)", "mae": "MAE (bu/ac)",
    "anomaly_r2": "Anomaly R$^2$", "mse_skill_vs_trend": "Skill vs trend",
}
LOWER_BETTER = {"rmse", "mae"}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": PAL["grid"], "grid.linewidth": 0.7,
    "legend.fontsize": 9, "legend.frameon": False,
    "xtick.color": PAL["ink"], "ytick.color": PAL["ink"], "text.color": PAL["ink"],
    "axes.labelcolor": PAL["ink"], "axes.titlecolor": PAL["ink"],
    "figure.facecolor": "white", "axes.facecolor": "white",
})

MANIFEST = []   # (filename, assignment, title, takeaway) for the index


def save(fig, outdir, name, assignment, title, takeaway):
    path = os.path.join(outdir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    MANIFEST.append((name, assignment, title, takeaway))
    print(f"  [ok] {name}")


def _cells_only(df):
    """Drop pooled 'ALL' rows and return year/doy as ints, sorted."""
    d = df[(df.get("year").astype(str) != "ALL") &
           (df.get("doy").astype(str) != "ALL")].copy()
    d["year"] = d["year"].astype(float).astype(int)
    d["doy"] = d["doy"].astype(float).astype(int)
    return d.sort_values(["year", "doy"])


def _cell_label(r):
    return f"{int(r.year)}\nD{int(r.doy)}"


def _scen_color(scenario):
    s = scenario.upper()
    if "STUDY" in s or "ICDL" in s:
        return PAL["B"]
    if "PREVIOUS" in s:
        return PAL["A"]
    if "SAME_YEAR" in s or "FINAL" in s:
        return PAL["oracle"]
    return PAL["other"]


# ======================================================================
#  BLOCK 1 -- model results  (seed_summary_mean_sd.csv)
# ======================================================================
def fig_scenario_ranking(summary, outdir, metric="rmse"):
    ov = summary[summary.year.astype(str) == "ALL"].copy()
    if ov.empty:
        print("  [skip] ranking: no overall ('ALL') rows"); return
    col_m, col_s = f"{metric}_mean", f"{metric}_sd"
    if col_m not in ov:
        print(f"  [skip] ranking: no {col_m}"); return
    ov = ov.sort_values(col_m, ascending=(metric in LOWER_BETTER))
    y = np.arange(len(ov))
    fig, ax = plt.subplots(figsize=(8.2, 0.7 * len(ov) + 1.8))
    ax.barh(y, ov[col_m], xerr=ov[col_s], color=[_scen_color(s) for s in ov.scenario],
            height=0.62, capsize=4, error_kw=dict(ecolor="#444", lw=1.2))
    for yi, (m, s) in enumerate(zip(ov[col_m], ov[col_s])):
        ax.text(m, yi, f"  {m:.2f}±{s:.2f}", va="center", ha="left", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(ov.scenario, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"{METRIC_LABEL.get(metric, metric)}  (mean ± SD over seeds)")
    best = "lower" if metric in LOWER_BETTER else "higher"
    ax.set_title(f"Scenario ranking by {METRIC_LABEL.get(metric, metric)}  ({best} is better)")
    ax.grid(axis="y", alpha=0)
    save(fig, outdir, f"01_scenario_ranking_{metric}.png", "FMEA + Modeling",
         f"Overall scenario ranking ({metric})",
         "Headline comparison of every mask scenario, averaged over all year×DOY cells and seeds; "
         "error bars are seed SD.")


def fig_metric_grid(summary, outdir):
    ov = summary[summary.year.astype(str) == "ALL"].copy()
    present = [m for m in METRICS if f"{m}_mean" in ov]
    if ov.empty or not present:
        print("  [skip] metric grid"); return
    n = len(present)
    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.6))
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, present):
        s = ov.sort_values(f"{m}_mean", ascending=(m in LOWER_BETTER))
        x = np.arange(len(s))
        ax.bar(x, s[f"{m}_mean"], yerr=s[f"{m}_sd"], capsize=3,
               color=[_scen_color(v) for v in s.scenario])
        ax.set_xticks(x)
        ax.set_xticklabels([t.split("->")[-1].strip() for t in s.scenario],
                           rotation=40, ha="right", fontsize=7)
        ax.set_title(METRIC_LABEL.get(m, m), fontsize=10)
        ax.grid(axis="x", alpha=0)
    fig.suptitle("All evaluation metrics by scenario (mean ± SD over seeds)",
                 fontweight="bold", y=1.03)
    save(fig, outdir, "02_metric_grid_overall.png", "FMEA + Modeling",
         "All metrics by scenario",
         "Small-multiples view of every metric at once, so no single metric drives the story.")


def fig_metric_by_cell(summary, outdir, metric, a_sub, b_sub):
    d = _cells_only(summary)
    col = f"{metric}_mean"; sd = f"{metric}_sd"
    if col not in d:
        print(f"  [skip] {metric} by cell"); return
    a = d[d.scenario.str.upper().str.contains(a_sub.upper())]
    b = d[d.scenario.str.upper().str.contains(b_sub.upper())]
    if a.empty or b.empty:
        print(f"  [skip] {metric} by cell: A/B scenario not found"); return
    cells = sorted(set(a.apply(lambda r: (r.year, r.doy), axis=1)) |
                   set(b.apply(lambda r: (r.year, r.doy), axis=1)))
    labels = [f"{y}\nD{doy}" for (y, doy) in cells]
    x = np.arange(len(cells)); w = 0.4

    def series(df):
        idx = {(r.year, r.doy): (r[col], r[sd]) for _, r in df.iterrows()}
        mu = [idx.get(c, (np.nan, np.nan))[0] for c in cells]
        er = [idx.get(c, (np.nan, np.nan))[1] for c in cells]
        return np.array(mu), np.array(er)

    amu, aer = series(a); bmu, ber = series(b)
    fig, ax = plt.subplots(figsize=(max(9, len(cells) * 0.52), 5.2))
    ax.bar(x - w / 2, amu, w, yerr=aer, capsize=2.5, color=PAL["A"], label="A")
    ax.bar(x + w / 2, bmu, w, yerr=ber, capsize=2.5, color=PAL["B"], label="B")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(f"{METRIC_LABEL.get(metric, metric)}  (mean ± SD)")
    ax.set_title(f"{METRIC_LABEL.get(metric, metric)} by test year × checkpoint (DOY): A vs B")
    ax.legend(); ax.grid(axis="x", alpha=0)
    save(fig, outdir, f"03_{metric}_by_year_doy.png", "FMEA + Modeling",
         f"{metric} by year×DOY (A vs B)",
         "How the two alternatives track across the season and across test years; "
         "error bars show seed stability at each checkpoint.")


def fig_metric_vs_doy_lines(summary, outdir, metric, a_sub, b_sub):
    d = _cells_only(summary)
    col = f"{metric}_mean"
    if col not in d:
        print(f"  [skip] {metric} lines"); return
    fig, ax = plt.subplots(figsize=(9, 5.4))
    years = sorted(d.year.unique())
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(years)))
    for yr, c in zip(years, cmap):
        for sub, ls, lab in [(a_sub, "-", "A"), (b_sub, "--", "B")]:
            g = d[(d.year == yr) & d.scenario.str.upper().str.contains(sub.upper())].sort_values("doy")
            if g.empty:
                continue
            ax.plot(g.doy, g[col], ls=ls, marker="o", ms=4, color=c,
                    label=f"{yr} {lab}")
    ax.set_xlabel("Day of Year (checkpoint)")
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    ax.set_title(f"{METRIC_LABEL.get(metric, metric)} trajectory through the season\n"
                 f"(solid = A, dashed = B; colour = test year)")
    ax.legend(ncol=2, fontsize=7.5)
    save(fig, outdir, f"05_{metric}_vs_doy_lines.png", "Modeling",
         f"{metric} season trajectory",
         "In-season learning curve: shows whether the in-season mask (B) pulls ahead "
         "as the season progresses.")


# ======================================================================
#  BLOCK 2 -- seed stability  (raw per-seed metrics) + OAT sensitivity
# ======================================================================
def load_seed_metrics(seed_dir):
    files = sorted(glob.glob(os.path.join(seed_dir, "*", "model_comparison_metrics.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(seed_dir, "*model_comparison_metrics*.csv")))
    if not files:
        return None
    frames = []
    for f in files:
        d = pd.read_csv(f)
        d["seed"] = os.path.basename(os.path.dirname(f)) or os.path.basename(f)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def fig_seed_box(raw, outdir, metric, a_sub, b_sub):
    if raw is None or metric not in raw:
        print(f"  [skip] seed box ({metric}): no raw per-seed metrics"); return
    raw = raw.copy()
    raw["scenario"] = raw.training_mask + " -> " + raw.testing_mask
    a = raw[raw.scenario.str.upper().str.contains(a_sub.upper())]
    b = raw[raw.scenario.str.upper().str.contains(b_sub.upper())]
    if a.empty or b.empty:
        print(f"  [skip] seed box ({metric})"); return
    # one value per seed = mean over cells
    av = a.groupby("seed")[metric].mean().to_numpy()
    bv = b.groupby("seed")[metric].mean().to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    parts = ax.violinplot([av, bv], showmeans=False, showmedians=True, widths=0.8)
    for pc, col in zip(parts["bodies"], [PAL["A"], PAL["B"]]):
        pc.set_facecolor(col); pc.set_alpha(0.35)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[key].set_color("#444")
    ax.scatter(np.ones_like(av), av, color=PAL["A"], s=22, zorder=3, alpha=0.8)
    ax.scatter(np.full_like(bv, 2), bv, color=PAL["B"], s=22, zorder=3, alpha=0.8)
    ax.set_xticks([1, 2]); ax.set_xticklabels(["A", "B"])
    ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    cva = 100 * av.std(ddof=1) / av.mean() if av.mean() else np.nan
    cvb = 100 * bv.std(ddof=1) / bv.mean() if bv.mean() else np.nan
    ax.set_title(f"Seed-to-seed spread of {METRIC_LABEL.get(metric, metric)}\n"
                 f"CV: A={cva:.1f}%,  B={cvb:.1f}%  (each dot = one seed)")
    ax.grid(axis="x", alpha=0)
    save(fig, outdir, f"06_seed_spread_{metric}.png", "Modeling (OAT: seed) + FMEA",
         f"Seed spread of {metric}",
         "Random-seed sensitivity: a tight distribution means the result is not an "
         "artefact of initialization. CV printed in title.")


def fig_seed_cv_heat(summary, outdir):
    ov = summary[summary.year.astype(str) == "ALL"].copy()
    present = [m for m in METRICS if f"{m}_mean" in ov and f"{m}_sd" in ov]
    if ov.empty or not present:
        print("  [skip] seed CV heatmap"); return
    mat, rows = [], []
    for _, r in ov.iterrows():
        rows.append(r.scenario)
        mat.append([100 * r[f"{m}_sd"] / abs(r[f"{m}_mean"]) if r[f"{m}_mean"] else np.nan
                    for m in present])
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(1.4 * len(present) + 2.5, 0.6 * len(rows) + 2))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=max(5, np.nanmax(mat)))
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([METRIC_LABEL.get(m, m) for m in present], rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8)
    ax.set_title("Seed instability: coefficient of variation (%)\nSD/|mean| across seeds — lower is more stable")
    fig.colorbar(im, ax=ax, label="CV (%)", fraction=0.046, pad=0.04)
    save(fig, outdir, "07_seed_cv_heatmap.png", "Modeling (OAT: seed) + FMEA",
         "Seed CV heatmap",
         "One-glance stability map: how much each metric wobbles with the random seed, per scenario.")


def load_sensitivity(path, metric_col):
    d = pd.read_csv(path)
    if metric_col not in d.columns:
        raise SystemExit(f"--sens-metric '{metric_col}' not in {path}: {list(d.columns)}")
    d = d.rename(columns={metric_col: "out"})
    base = d[d.level_pct == 0].set_index("param")["out"]
    d["pct_out"] = d.apply(
        lambda r: 100 * (r.out - base.get(r.param, np.nan)) / base.get(r.param, np.nan)
        if base.get(r.param, 0) else np.nan, axis=1)
    return d, base


def fig_oat_multiline(sens, outdir, metric_col):
    fig, ax = plt.subplots(figsize=(8, 5.4))
    params = list(sens.param.unique())
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    for p, c in zip(params, cmap):
        g = sens[sens.param == p].sort_values("level_pct")
        ax.plot(g.level_pct, g.pct_out, marker="o", color=c, label=p, lw=2)
    ax.axhline(0, color="#999", lw=1); ax.axvline(0, color="#999", lw=1)
    ax.set_xlabel("% change in input (One-At-a-Time)")
    ax.set_ylabel(f"% change in output ({metric_col})")
    ax.set_title("OAT sensitivity: % change in input vs % change in output")
    ax.legend(title="parameter")
    save(fig, outdir, "08_oat_multiline.png", "Modeling",
         "OAT sensitivity (multi-line)",
         "Required Modeling plot: steeper line = more sensitive parameter. Flat lines are robust inputs.")


def fig_oat_tornado(sens, outdir, metric_col):
    # sensitivity coefficient = max |%out / %in| over the non-zero levels
    rows = []
    for p, g in sens.groupby("param"):
        gg = g[g.level_pct != 0]
        if gg.empty:
            continue
        coef = (gg.pct_out / gg.level_pct).abs().max()
        rows.append((p, coef))
    if not rows:
        print("  [skip] tornado"); return
    rows.sort(key=lambda t: t[1])
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
    y = np.arange(len(rows))
    colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(rows)))
    fig, ax = plt.subplots(figsize=(7.4, 0.6 * len(rows) + 2))
    ax.barh(y, vals, color=colors, height=0.6)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f"  {v:.2f}", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Sensitivity coefficient  |%Δoutput / %Δinput|")
    ax.set_title("Tornado: parameters ranked by sensitivity coefficient")
    ax.grid(axis="y", alpha=0)
    save(fig, outdir, "09_oat_tornado.png", "Modeling",
         "Sensitivity tornado",
         "Ranks inputs by the required sensitivity coefficient; the top bar is the parameter your "
         "mitigation plan should target.")


def fig_oat_spider(sens, outdir):
    rows = []
    for p, g in sens.groupby("param"):
        gg = g[g.level_pct != 0]
        if gg.empty:
            continue
        rows.append((p, (gg.pct_out / gg.level_pct).abs().max()))
    if len(rows) < 3:
        print("  [skip] spider (need >=3 params)"); return
    labels = [r[0] for r in rows]; vals = [r[1] for r in rows]
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals2 = vals + vals[:1]; ang2 = ang + ang[:1]
    fig, ax = plt.subplots(figsize=(6.4, 6.4), subplot_kw=dict(polar=True))
    ax.plot(ang2, vals2, color=PAL["B"], lw=2)
    ax.fill(ang2, vals2, color=PAL["B"], alpha=0.25)
    ax.set_xticks(ang); ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Sensitivity spider: |%Δoutput / %Δinput| per parameter", y=1.08)
    ax.grid(True, alpha=0.4)
    save(fig, outdir, "10_oat_spider.png", "Modeling",
         "Sensitivity spider",
         "Alternative to the multi-line plot the rubric asks for; area reaches out along the most "
         "sensitive inputs.")


# ======================================================================
#  BLOCK 3 -- statistical significance  (stat_tests_*.csv)
# ======================================================================
def fig_rmse_gap_ci(res, outdir, a_lab, b_lab):
    d = _cells_only(res)
    if "RMSE_gap_A_minus_B" not in d:
        print("  [skip] rmse gap CI"); return
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for yr, g in d.groupby("year"):
        g = g.sort_values("doy")
        ax.errorbar(g.doy, g.RMSE_gap_A_minus_B,
                    yerr=[g.RMSE_gap_A_minus_B - g.RMSE_gap_lo,
                          g.RMSE_gap_hi - g.RMSE_gap_A_minus_B],
                    marker="o", capsize=3, lw=1.6, label=str(int(yr)))
    # shade the "B better" half
    ax.axhline(0, color="#444", ls="--", lw=1)
    ymax = np.nanmax(np.abs(d[["RMSE_gap_lo", "RMSE_gap_hi"]].to_numpy())) * 1.1
    ax.axhspan(0, ymax, color=PAL["B"], alpha=0.05)
    ax.axhspan(-ymax, 0, color=PAL["A"], alpha=0.05)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel("Day of Year (checkpoint)")
    ax.set_ylabel(f"RMSE(A) − RMSE(B)  (bu/ac)")
    ax.set_title(f"RMSE gap with state-clustered bootstrap 95% CI\n"
                 f"above 0 → {b_lab} better;  below 0 → {a_lab} better")
    ax.legend(title="Test year")
    save(fig, outdir, "11_rmse_gap_ci.png", "FMEA + Modeling",
         "RMSE gap with 95% CI",
         "The core significance figure: a CI that clears the dashed line is a statistically "
         "distinguishable improvement on real counties.")


def _p_heatmap(res, outdir, col, name, title, assignment, takeaway):
    d = _cells_only(res)
    if col not in d:
        print(f"  [skip] {name}"); return
    piv = d.pivot(index="year", columns="doy", values=col)
    fig, ax = plt.subplots(figsize=(1.1 * piv.shape[1] + 2.5, 0.8 * piv.shape[0] + 2))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=0, vmax=0.2, aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([f"D{c}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([int(y) for y in piv.index])
    ax.set_xlabel("Day of Year")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                star = "*" if v < 0.05 else ""
                ax.text(j, i, f"{v:.3f}{star}", ha="center", va="center", fontsize=8,
                        color="white" if v < 0.05 else "#222")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="p-value (green = p<0.05)", fraction=0.046, pad=0.04)
    save(fig, outdir, name, assignment, title, takeaway)


def fig_sig_counts(res, outdir):
    d = _cells_only(res)
    n = len(d)
    if n == 0:
        print("  [skip] sig counts"); return
    bars = []
    if "RMSE_gap_sig" in d:
        bars.append(("Bootstrap\nRMSE CI≠0", int(d.RMSE_gap_sig.sum())))
    if "DM_HLN_sq_p" in d:
        bars.append(("DM (sq)\np<0.05", int((d.DM_HLN_sq_p < 0.05).sum())))
    if "DM_HLN_sq_p_holm" in d:
        bars.append(("DM (sq)\nHolm", int((d.DM_HLN_sq_p_holm < 0.05).sum())))
    if "Wilcoxon_p" in d:
        bars.append(("Wilcoxon\np<0.05", int((d.Wilcoxon_p < 0.05).sum())))
    if "Wilcoxon_p_holm" in d:
        bars.append(("Wilcoxon\nHolm", int((d.Wilcoxon_p_holm < 0.05).sum())))
    labels = [b[0] for b in bars]; vals = [b[1] for b in bars]
    x = np.arange(len(bars))
    fig, ax = plt.subplots(figsize=(1.3 * len(bars) + 2, 5))
    ax.bar(x, vals, color=PAL["sig"], width=0.6)
    ax.bar(x, [n - v for v in vals], bottom=vals, color=PAL["grid"], width=0.6)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.1, f"{v}/{n}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("year × DOY cells")
    ax.set_title("Significant cells by test (green) — including Holm-corrected columns")
    ax.grid(axis="x", alpha=0)
    save(fig, outdir, "14_significance_counts.png", "FMEA",
         "Significance tally",
         "How many of the checkpoints show a real A-vs-B difference under each test, before and "
         "after multiple-comparison correction.")


def fig_skill_ci(res, outdir, a_lab, b_lab):
    d = _cells_only(res)
    if "Skill_A" not in d:
        print("  [skip] skill CI"); return
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(max(9, len(d) * 0.5), 5.4))
    ax.errorbar(x - 0.1, d.Skill_A, yerr=[d.Skill_A - d.Skill_A_lo, d.Skill_A_hi - d.Skill_A],
                marker="s", ls="none", capsize=2.5, color=PAL["A"], label=a_lab)
    ax.errorbar(x + 0.1, d.Skill_B, yerr=[d.Skill_B - d.Skill_B_lo, d.Skill_B_hi - d.Skill_B],
                marker="D", ls="none", capsize=2.5, color=PAL["B"], label=b_lab)
    ax.axhline(0, color="#444", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([_cell_label(r) for _, r in d.iterrows()], fontsize=7)
    ax.set_ylabel("Skill vs trend  (1 − MSE/MSE$_{trend}$)")
    ax.set_title("Forecast skill vs the county-trend baseline, with clustered bootstrap CI")
    ax.legend()
    save(fig, outdir, "15_skill_ci.png", "FMEA + Modeling",
         "Skill vs trend (A vs B)",
         "Shows both alternatives beat the naive trend baseline (skill>0) and by how much — the "
         "right yardstick since the model predicts anomalies.")


def fig_bias(res, outdir, a_lab, b_lab):
    d = _cells_only(res)
    if "Bias_A" not in d:
        print("  [skip] bias"); return
    x = np.arange(len(d)); w = 0.4
    fig, ax = plt.subplots(figsize=(max(9, len(d) * 0.5), 5))
    ax.bar(x - w / 2, d.Bias_A, w, color=PAL["A"], label=a_lab)
    ax.bar(x + w / 2, d.Bias_B, w, color=PAL["B"], label=b_lab)
    ax.axhline(0, color="#444", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([_cell_label(r) for _, r in d.iterrows()], fontsize=7)
    ax.set_ylabel("Mean bias  (predicted − actual, bu/ac)")
    ax.set_title("Mincer–Zarnowitz calibration: mean forecast bias by checkpoint")
    ax.legend()
    save(fig, outdir, "16_bias_calibration.png", "FMEA",
         "Calibration / mean bias",
         "A well-calibrated forecast sits near zero bias; large systematic over/under-prediction is "
         "a design flag.")


def fig_gap_forest(res, outdir, a_lab, b_lab):
    d = _cells_only(res).copy()
    if "RMSE_gap_A_minus_B" not in d:
        print("  [skip] forest"); return
    d = d.sort_values("RMSE_gap_A_minus_B")
    y = np.arange(len(d))
    colors = [PAL["sig"] if s else PAL["nsig"] for s in d.get("RMSE_gap_sig", [False] * len(d))]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(d) + 2))
    ax.errorbar(d.RMSE_gap_A_minus_B, y,
                xerr=[d.RMSE_gap_A_minus_B - d.RMSE_gap_lo, d.RMSE_gap_hi - d.RMSE_gap_A_minus_B],
                fmt="o", ecolor="#888", capsize=2.5, ms=0)
    ax.scatter(d.RMSE_gap_A_minus_B, y, c=colors, zorder=3, s=36)
    ax.axvline(0, color="#444", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels([_cell_label(r) for _, r in d.iterrows()], fontsize=7)
    ax.set_xlabel("RMSE(A) − RMSE(B)  (bu/ac)")
    ax.set_title(f"Every checkpoint ranked by RMSE gap (A vs B)\n"
                 f"green = CI excludes 0 (significant), red = not")
    ax.legend(handles=[Patch(color=PAL["sig"], label="significant"),
                       Patch(color=PAL["nsig"], label="n.s.")], loc="lower right")
    save(fig, outdir, "17_rmse_gap_forest.png", "FMEA",
         "RMSE gap forest plot",
         "Every year×DOY cell on one axis, sorted; makes the share of significant improvements "
         "obvious at a glance.")


def fig_county_counts(res, outdir):
    d = _cells_only(res)
    if "N" not in d:
        print("  [skip] county counts"); return
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(max(9, len(d) * 0.5), 4.6))
    ax.bar(x, d.N, color=PAL["oracle"], width=0.6)
    for xi, (nn, ns) in enumerate(zip(d.N, d.get("N_states", [np.nan] * len(d)))):
        lab = f"{int(nn)}" + (f"\n{int(ns)} st" if np.isfinite(ns) else "")
        ax.text(xi, nn, lab, ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([_cell_label(r) for _, r in d.iterrows()], fontsize=7)
    ax.set_ylabel("counties compared")
    ax.set_title("Sample size per checkpoint: matched counties (and states) entering each test")
    ax.grid(axis="x", alpha=0)
    save(fig, outdir, "18_county_counts.png", "FMEA + Modeling",
         "Counties per checkpoint",
         "Documents the evidence base behind every test — answers 'how many counties did you "
         "actually measure?'")


# ======================================================================
#  BLOCK 4 -- paired-seed tests  (paired_tests_*.csv)
# ======================================================================
def fig_paired_diff(pt, outdir, a_lab, b_lab):
    d = _cells_only(pt)
    if "mean_diff_A_minus_B" not in d:
        print("  [skip] paired diff"); return
    x = np.arange(len(d))
    sig = d.get("significant", pd.Series([False] * len(d))).to_numpy()
    colors = [PAL["sig"] if s else PAL["nsig"] for s in sig]
    err = d.get("difference_sd", pd.Series([np.nan] * len(d))).to_numpy()
    fig, ax = plt.subplots(figsize=(max(9, len(d) * 0.5), 5))
    ax.bar(x, d.mean_diff_A_minus_B,
           yerr=err if np.isfinite(err).any() else None,
           capsize=2.5, color=colors, width=0.6)
    ax.axhline(0, color="#444", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([_cell_label(r) for _, r in d.iterrows()], fontsize=7)
    ax.set_ylabel(f"mean {d.metric.iloc[0] if 'metric' in d else ''} difference (A − B) over seeds")
    ax.set_title("Paired across-seed difference per checkpoint\ngreen = paired t-test p<0.05")
    ax.legend(handles=[Patch(color=PAL["sig"], label="p<0.05"),
                       Patch(color=PAL["nsig"], label="n.s.")])
    save(fig, outdir, "19_paired_seed_diff.png", "FMEA",
         "Across-seed paired difference",
         "The seed-level companion to the county-level test: a difference robust to training "
         "randomness, not just to the county sample.")


def fig_paired_p(pt, outdir):
    d = _cells_only(pt)
    cols = [c for c in ["t_p", "t_p_holm", "wilcoxon_p"] if c in d]
    if not cols:
        print("  [skip] paired p"); return
    x = np.arange(len(d)); w = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(max(9, len(d) * 0.55), 5))
    for k, c in enumerate(cols):
        ax.bar(x + (k - (len(cols) - 1) / 2) * w, d[c], w, label=c)
    ax.axhline(0.05, color=PAL["nsig"], ls="--", lw=1.2, label="α = 0.05")
    ax.set_xticks(x); ax.set_xticklabels([_cell_label(r) for _, r in d.iterrows()], fontsize=7)
    ax.set_ylabel("p-value"); ax.set_title("Paired-seed test p-values by checkpoint")
    ax.legend(fontsize=8)
    save(fig, outdir, "20_paired_seed_pvalues.png", "FMEA",
         "Paired-seed p-values",
         "p-values from the across-seed paired t-test (raw + Holm) and Wilcoxon, against the 0.05 line.")


# ======================================================================
#  BLOCK 5 -- decision framework  (Modeling assignment)
# ======================================================================
DEFAULT_CRITERIA = [
    "In-season timeliness",
    "Forecast skill vs trend",
    "RMSE accuracy",
    "Spatial coverage (counties)",
    "Statistical robustness",
    "Computational cost",
    "Data availability / latency",
]
# default pairwise (Saaty-style, row preferred over col; reciprocal filled in).
# Only the upper triangle matters; 1=equal, 3=moderately more important, etc.
DEFAULT_PAIRWISE_UPPER = {
    (0, 1): 1, (0, 2): 1, (0, 3): 3, (0, 4): 2, (0, 5): 5, (0, 6): 3,
    (1, 2): 1, (1, 3): 3, (1, 4): 2, (1, 5): 5, (1, 6): 3,
    (2, 3): 3, (2, 4): 2, (2, 5): 5, (2, 6): 3,
    (3, 4): 1 / 2, (3, 5): 3, (3, 6): 1,
    (4, 5): 4, (4, 6): 2,
    (5, 6): 1 / 3,
}


def build_pairwise(path, criteria):
    n = len(criteria)
    M = np.ones((n, n))
    if path and os.path.exists(path):
        raw = pd.read_csv(path, index_col=0)
        M = raw.to_numpy(dtype=float)
        criteria = list(raw.index)
    else:
        for (i, j), v in DEFAULT_PAIRWISE_UPPER.items():
            M[i, j] = v; M[j, i] = 1 / v
    # AHP weights = normalized row geometric mean
    gm = np.prod(M, axis=1) ** (1 / M.shape[0])
    w = gm / gm.sum()
    # consistency ratio (Saaty)
    lam = (M @ w / w).mean()
    n = M.shape[0]
    ci = (lam - n) / (n - 1) if n > 1 else 0
    RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41}.get(n, 1.45)
    cr = ci / RI if RI else 0
    return M, np.array(w), criteria, cr


def fig_pairwise(path, outdir, criteria):
    M, w, crit, cr = build_pairwise(path, criteria)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 0.6 * len(crit) + 2.5),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    im = ax1.imshow(M, cmap="Blues", vmin=0, vmax=np.nanmax(M))
    ax1.set_xticks(range(len(crit))); ax1.set_yticks(range(len(crit)))
    ax1.set_xticklabels(range(1, len(crit) + 1)); ax1.set_yticklabels(crit, fontsize=8)
    for i in range(len(crit)):
        for j in range(len(crit)):
            ax1.text(j, i, f"{M[i, j]:.2g}", ha="center", va="center", fontsize=7.5,
                     color="white" if M[i, j] > np.nanmax(M) * 0.6 else "#222")
    ax1.set_title(f"Pairwise comparison matrix (AHP)\nconsistency ratio CR = {cr:.3f}"
                  + ("  ✓" if cr < 0.10 else "  (aim < 0.10)"))
    ax1.set_xlabel("criterion # (column)")
    order = np.argsort(w)
    ax2.barh(np.arange(len(crit)), w[order] * 100, color=PAL["A"])
    for k, idx in enumerate(order):
        ax2.text(w[idx] * 100, k, f"  {w[idx]*100:.1f}%", va="center", fontsize=8.5)
    ax2.set_yticks(np.arange(len(crit)))
    ax2.set_yticklabels([crit[i] for i in order], fontsize=8)
    ax2.set_xlabel("derived weight (%)")
    ax2.set_title("Criteria weights")
    ax2.grid(axis="y", alpha=0)
    save(fig, outdir, "21_pairwise_matrix.png", "Modeling",
         "Pairwise comparison + weights",
         "Required Modeling deliverable: weights derived mathematically (AHP row-geometric-mean) with "
         "a consistency check, not assigned arbitrarily.")
    return crit, w, cr


def fig_decision_matrix(path, outdir, criteria, weights, a_lab, b_lab):
    n = len(criteria)
    if path and os.path.exists(path):
        d = pd.read_csv(path)
        crit = d["criterion"].tolist()
        sa = d["score_A"].to_numpy(dtype=float)
        sb = d["score_B"].to_numpy(dtype=float)
        if "weight" in d:
            weights = d["weight"].to_numpy(dtype=float)
    else:
        crit = criteria
        # TEMPLATE illustrative scores on a 1-5 scale (higher = better performance).
        # Replace with your final numbers once both alternatives finish running.
        sa = np.array([2, 3, 3, 4, 3, 4, 4][:n], dtype=float)  # previous-season CDL (A)
        sb = np.array([5, 4, 4, 4, 4, 3, 3][:n], dtype=float)  # in-season ICDL / STUDY (B)
    weights = np.array(weights, dtype=float)
    wa = weights * sa; wb = weights * sb
    tot_a, tot_b = wa.sum(), wb.sum()
    y = np.arange(n); h = 0.38
    fig, (ax, axt) = plt.subplots(1, 2, figsize=(13, 0.62 * n + 2.6),
                                  gridspec_kw={"width_ratios": [2.3, 1]})
    ax.barh(y - h / 2, wa, h, color=PAL["A"], label=a_lab)
    ax.barh(y + h / 2, wb, h, color=PAL["B"], label=b_lab)
    for yi, (a, b, rsa, rsb) in enumerate(zip(wa, wb, sa, sb)):
        ax.text(a, yi - h / 2, f" {rsa:.0f}", va="center", fontsize=8, color=PAL["A"])
        ax.text(b, yi + h / 2, f" {rsb:.0f}", va="center", fontsize=8, color=PAL["B"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}\n(w={w*100:.0f}%)" for c, w in zip(crit, weights)], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("weighted score  (weight × score)")
    ax.set_title("Weighted decision matrix — per-criterion contribution\n"
                 "(small number = raw 1–5 score; TEMPLATE scores, replace with finals)")
    ax.legend(loc="lower right"); ax.grid(axis="y", alpha=0)
    # totals panel
    axt.bar([0, 1], [tot_a, tot_b], color=[PAL["A"], PAL["B"]], width=0.6)
    for xi, t in zip([0, 1], [tot_a, tot_b]):
        axt.text(xi, t, f"{t:.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axt.set_xticks([0, 1]); axt.set_xticklabels(["A", "B"])
    axt.set_ylabel("total weighted score")
    axt.set_title("Total")
    axt.grid(axis="x", alpha=0)
    winner = "B" if tot_b > tot_a else "A"
    axt.text(0.5, max(tot_a, tot_b) * 0.5, f"lead: {winner}", ha="center",
             fontsize=10, color="#444")
    save(fig, outdir, "22_decision_matrix.png", "Modeling",
         "Weighted decision matrix",
         "Required Modeling deliverable: pairwise weights × criterion scores → total. Ships as a "
         "TEMPLATE; drop in final scores after both runs complete.")


# ======================================================================
#  index + demo data
# ======================================================================
def write_index(outdir, a_lab, b_lab):
    lines = ["# Figure index — corn-yield crop-mask study",
             "",
             f"**A** = {a_lab}  **B** = {b_lab}",
             "",
             "Generated by `plot_analysis.py`. Each figure lists the assignment it supports "
             "and the one-line takeaway to use in the caption or talk track.",
             "",
             "| # | file | assignment | what it shows |",
             "|---|------|-----------|---------------|"]
    for name, assignment, title, takeaway in MANIFEST:
        lines.append(f"| {name.split('_')[0]} | `{name}` | {assignment} | **{title}.** {takeaway} |")
    with open(os.path.join(outdir, "figures_index.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [ok] figures_index.md  ({len(MANIFEST)} figures)")


def make_demo_inputs(demo_dir):
    """Fabricate small, realistic CSVs so the whole pipeline can run with no real data."""
    os.makedirs(demo_dir, exist_ok=True)
    rng = np.random.default_rng(7)
    years = [2022, 2023, 2024, 2025]
    doys = [193, 209, 225, 241, 257, 273]
    scen = {
        "PREVIOUS_CDL -> PREVIOUS_CDL": dict(base=22.0, slope=-0.004, sk=0.28),
        "SAME_YEAR_CDL -> STUDY":       dict(base=21.2, slope=-0.006, sk=0.34),
        "SAME_YEAR_CDL -> SAME_YEAR_CDL": dict(base=20.4, slope=-0.006, sk=0.40),
    }
    seeds = [f"seed_{i}" for i in range(1, 13)]

    # --- raw per-seed metrics ---
    seed_root = os.path.join(demo_dir, "v5_seeds")
    rows_summary = []
    for sd in seeds:
        os.makedirs(os.path.join(seed_root, sd), exist_ok=True)
        rr = []
        for scn, p in scen.items():
            tr, te = [x.strip() for x in scn.split("->")]
            for yr in years:
                for doy in doys:
                    rmse = p["base"] + p["slope"] * (doy - 193) * 30 + rng.normal(0, 0.4)
                    r2 = np.clip(0.55 + 0.0008 * (doy - 193) + rng.normal(0, 0.02), 0, 0.95)
                    rr.append(dict(training_mask=tr, testing_mask=te, year=yr, doy=doy,
                                   r2=r2, rmse=rmse, mae=rmse * 0.78,
                                   anomaly_r2=np.clip(r2 - 0.12, 0, 1),
                                   mse_skill_vs_trend=np.clip(p["sk"] + 0.0006 * (doy - 193) + rng.normal(0, 0.02), -0.2, 0.8)))
        pd.DataFrame(rr).to_csv(os.path.join(seed_root, sd, "model_comparison_metrics.csv"), index=False)

    # --- seed_summary_mean_sd.csv ---
    allm = load_seed_metrics(seed_root)
    allm["scenario"] = allm.training_mask + " -> " + allm.testing_mask
    rows = []
    for (scn, yr, doy), g in allm.groupby(["scenario", "year", "doy"]):
        rec = dict(scenario=scn, year=yr, doy=doy, n_seeds=g.seed.nunique())
        for m in METRICS:
            rec[f"{m}_mean"] = g[m].mean(); rec[f"{m}_sd"] = g[m].std(ddof=1)
        rows.append(rec)
    for scn, g in allm.groupby("scenario"):
        per = g.groupby("seed")[METRICS].mean()
        rec = dict(scenario=scn, year="ALL", doy="ALL", n_seeds=g.seed.nunique())
        for m in METRICS:
            rec[f"{m}_mean"] = per[m].mean(); rec[f"{m}_sd"] = per[m].std(ddof=1)
        rows.append(rec)
    seed_summary = os.path.join(demo_dir, "seed_summary_mean_sd.csv")
    pd.DataFrame(rows).to_csv(seed_summary, index=False)

    # --- paired_tests (A=previous, B=study) ---
    from scipy import stats as _st
    A = allm[allm.scenario == "PREVIOUS_CDL -> PREVIOUS_CDL"]
    B = allm[allm.scenario == "SAME_YEAR_CDL -> STUDY"]
    pr = []
    for (yr, doy) in [(y, d) for y in years for d in doys]:
        av = A[(A.year == yr) & (A.doy == doy)].groupby("seed").rmse.mean()
        bv = B[(B.year == yr) & (B.doy == doy)].groupby("seed").rmse.mean()
        j = pd.concat([av, bv], axis=1, keys=["A", "B"]).dropna()
        t, p = _st.ttest_rel(j.A, j.B)
        try:
            w, wp = _st.wilcoxon(j.A, j.B)
        except ValueError:
            w, wp = np.nan, np.nan
        diff = (j.A - j.B)
        pr.append(dict(year=yr, doy=doy, n_seeds=len(j), metric="rmse",
                       A_mean=j.A.mean(), A_sd=j.A.std(ddof=1),
                       B_mean=j.B.mean(), B_sd=j.B.std(ddof=1),
                       mean_diff_A_minus_B=diff.mean(), difference_sd=diff.std(ddof=1),
                       t_stat=t, t_p=p, wilcoxon_stat=w, wilcoxon_p=wp,
                       better="B" if j.B.mean() < j.A.mean() else "A",
                       significant=bool(p < 0.05)))
    ptdf = pd.DataFrame(pr)
    # holm
    pv = ptdf.t_p.to_numpy(); order = np.argsort(pv)
    adj = np.minimum.accumulate((pv[order] * (len(pv) - np.arange(len(pv))))[::-1])[::-1]
    hp = np.empty_like(pv); hp[order] = np.clip(adj, 0, 1)
    ptdf["t_p_holm"] = hp
    paired = os.path.join(demo_dir, "paired_tests_PREVIOUS_CDL_vs_STUDY.csv")
    ptdf.to_csv(paired, index=False)

    # --- stat_tests (county-level) ---
    sr = []
    for yr in years:
        for doy in doys:
            n = rng.integers(980, 1056)
            gap = 0.2 + 0.004 * (doy - 193) + rng.normal(0, 0.15)   # B better late
            half = 0.25 + 0.03 * rng.random()
            lo, hi = gap - half, gap + half
            dmp = max(0.002, 0.5 - 0.0025 * (doy - 193) + rng.normal(0, 0.05))
            wp = max(0.002, dmp + rng.normal(0, 0.03))
            ska = 0.28 + 0.0006 * (doy - 193); skb = ska + 0.05
            sr.append(dict(year=yr, doy=doy, N=int(n), N_states=12,
                           RMSE_A=22 - 0.004 * (doy - 193) * 20,
                           RMSE_B=21.2 - 0.006 * (doy - 193) * 20,
                           RMSE_gap_A_minus_B=gap, RMSE_gap_lo=lo, RMSE_gap_hi=hi,
                           RMSE_gap_sig=bool(lo > 0 or hi < 0),
                           MAE_gap_A_minus_B=gap * 0.8, MAE_gap_lo=lo * 0.8, MAE_gap_hi=hi * 0.8,
                           Wilcoxon_stat=rng.integers(1e5, 3e5), Wilcoxon_p=wp,
                           DM_HLN_sq=gap / (half / 2), DM_HLN_sq_p=dmp,
                           DM_HLN_abs=gap / (half / 2), DM_HLN_abs_p=dmp * 1.1,
                           Skill_A=ska, Skill_A_lo=ska - 0.05, Skill_A_hi=ska + 0.05,
                           Skill_B=skb, Skill_B_lo=skb - 0.05, Skill_B_hi=skb + 0.05,
                           Bias_A=rng.normal(0, 1.5), Bias_B=rng.normal(0, 1.2),
                           Better="B" if lo > 0 else "indistinguishable"))
    stat = pd.DataFrame(sr)
    pv = stat.DM_HLN_sq_p.to_numpy(); order = np.argsort(pv)
    adj = np.minimum.accumulate((pv[order] * (len(pv) - np.arange(len(pv))))[::-1])[::-1]
    hp = np.empty_like(pv); hp[order] = np.clip(adj, 0, 1)
    stat["DM_HLN_sq_p_holm"] = hp
    pv = stat.Wilcoxon_p.to_numpy(); order = np.argsort(pv)
    adj = np.minimum.accumulate((pv[order] * (len(pv) - np.arange(len(pv))))[::-1])[::-1]
    hp = np.empty_like(pv); hp[order] = np.clip(adj, 0, 1)
    stat["Wilcoxon_p_holm"] = hp
    stat_tests = os.path.join(demo_dir, "stat_tests_PREVIOUS_CDL_vs_STUDY.csv")
    stat.to_csv(stat_tests, index=False)

    # --- OAT sensitivity sweep ---
    so = []
    sens_def = {"n_trees": 0.35, "learning_rate": 0.9, "feature_noise": 0.6,
                "train_test_split": 0.25}
    for p, sens in sens_def.items():
        base = 20.4
        for lv in [-25, -10, 0, 10, 25]:
            so.append(dict(param=p, level_pct=lv,
                           rmse=base * (1 + sens * (lv / 100) ** 1 * (1 if "noise" not in p else 1))
                           + (0 if lv else 0)))
    sens_csv = os.path.join(demo_dir, "sensitivity_oat.csv")
    pd.DataFrame(so).to_csv(sens_csv, index=False)

    return dict(seed_summary=seed_summary, paired=paired, stat_tests=stat_tests,
                seed_metrics=seed_root, sensitivity=sens_csv)


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed-summary")
    ap.add_argument("--paired-tests")
    ap.add_argument("--stat-tests")
    ap.add_argument("--seed-metrics", help="dir of per-seed model_comparison_metrics.csv")
    ap.add_argument("--sensitivity")
    ap.add_argument("--sens-metric", default="rmse", help="output column in sensitivity_oat.csv")
    ap.add_argument("--pairwise")
    ap.add_argument("--decision")
    ap.add_argument("--a-scenario", default="PREVIOUS", help="substring that identifies scenario A")
    ap.add_argument("--b-scenario", default="STUDY", help="substring that identifies scenario B")
    ap.add_argument("--a-label", default="Previous-season CDL (A)")
    ap.add_argument("--b-label", default="In-season ICDL / STUDY (B)")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--demo", action="store_true", help="fabricate inputs and render everything")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.demo:
        print("DEMO mode: fabricating inputs...")
        demo = make_demo_inputs(os.path.join(args.outdir, "_demo_inputs"))
        args.seed_summary = args.seed_summary or demo["seed_summary"]
        args.paired_tests = args.paired_tests or demo["paired"]
        args.stat_tests = args.stat_tests or demo["stat_tests"]
        args.seed_metrics = args.seed_metrics or demo["seed_metrics"]
        args.sensitivity = args.sensitivity or demo["sensitivity"]

    print(f"\nWriting figures to {args.outdir}/ ...")

    # ---- model results ----
    if args.seed_summary and os.path.exists(args.seed_summary):
        summary = pd.read_csv(args.seed_summary)
        fig_scenario_ranking(summary, args.outdir, "rmse")
        fig_scenario_ranking(summary, args.outdir, "mse_skill_vs_trend")
        fig_metric_grid(summary, args.outdir)
        fig_metric_by_cell(summary, args.outdir, "rmse", args.a_scenario, args.b_scenario)
        fig_metric_by_cell(summary, args.outdir, "mse_skill_vs_trend", args.a_scenario, args.b_scenario)
        fig_metric_vs_doy_lines(summary, args.outdir, "rmse", args.a_scenario, args.b_scenario)
        fig_seed_cv_heat(summary, args.outdir)
    else:
        summary = None
        print("  [skip] model-results block: no --seed-summary")

    # ---- raw per-seed stability ----
    raw = load_seed_metrics(args.seed_metrics) if args.seed_metrics else None
    if raw is not None:
        fig_seed_box(raw, args.outdir, "rmse", args.a_scenario, args.b_scenario)

    # ---- OAT sensitivity ----
    if args.sensitivity and os.path.exists(args.sensitivity):
        sens, _ = load_sensitivity(args.sensitivity, args.sens_metric)
        fig_oat_multiline(sens, args.outdir, args.sens_metric)
        fig_oat_tornado(sens, args.outdir, args.sens_metric)
        fig_oat_spider(sens, args.outdir)
    else:
        print("  [skip] OAT block: no --sensitivity")

    # ---- statistical significance ----
    if args.stat_tests and os.path.exists(args.stat_tests):
        res = pd.read_csv(args.stat_tests)
        fig_rmse_gap_ci(res, args.outdir, args.a_label, args.b_label)
        _p_heatmap(res, args.outdir, "DM_HLN_sq_p", "12_dm_pvalue_heatmap.png",
                   "HLN–Diebold-Mariano p-value (squared loss)", "FMEA",
                   "Per-checkpoint significance of the error difference; green cells are significant.")
        _p_heatmap(res, args.outdir, "DM_HLN_sq_p_holm", "13_dm_pvalue_holm.png",
                   "DM p-value after Holm correction", "FMEA",
                   "The same test after correcting for testing many checkpoints at once — the honest bar.")
        fig_sig_counts(res, args.outdir)
        fig_skill_ci(res, args.outdir, args.a_label, args.b_label)
        fig_bias(res, args.outdir, args.a_label, args.b_label)
        fig_gap_forest(res, args.outdir, args.a_label, args.b_label)
        fig_county_counts(res, args.outdir)
    else:
        print("  [skip] stat-tests block: no --stat-tests")

    # ---- paired-seed tests ----
    if args.paired_tests and os.path.exists(args.paired_tests):
        pt = pd.read_csv(args.paired_tests)
        fig_paired_diff(pt, args.outdir, args.a_label, args.b_label)
        fig_paired_p(pt, args.outdir)
    else:
        print("  [skip] paired-tests block: no --paired-tests")

    # ---- decision framework ----
    crit, w, cr = fig_pairwise(args.pairwise, args.outdir, DEFAULT_CRITERIA)
    fig_decision_matrix(args.decision, args.outdir, crit, w, args.a_label, args.b_label)

    write_index(args.outdir, args.a_label, args.b_label)
    print(f"\nDone. {len(MANIFEST)} figures + figures_index.md in {args.outdir}/")


if __name__ == "__main__":
    main()
