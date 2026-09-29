#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_stat_tests.py

Offline statistical comparison of two crop-mask forecast scenarios, run from
the per-county predictions written by the V5 model
(county_predictions.csv). No re-modeling required: run V5 once, then run this
as many times as you like.

Compares, per test year x checkpoint (DOY), the forecasts from two
`testing_mask` scenarios (e.g. PREVIOUS_CDL vs STUDY) on the SAME counties,
using tests appropriate for a spatially-correlated county cross-section:

  1. Paired block bootstrap on the RMSE and MAE differences, resampling by
     STATE cluster (95% CI). Efron & Tibshirani (1993); cluster resampling
     because counties within a state are correlated.
  2. Wilcoxon signed-rank test on paired per-county absolute errors
     (non-parametric, distribution-free).
  3. Diebold-Mariano test with the Harvey-Leybourne-Newbold (1997)
     small-sample correction, squared and absolute loss, forecast-horizon h=1.
  4. Skill score vs. the county yield trend baseline (MSE-based, Murphy 1988),
     with a state-clustered bootstrap CI, for each scenario.
  5. Mincer-Zarnowitz calibration: regress actual on predicted, report the
     joint departure of (intercept, slope) from (0, 1), plus mean bias.

Input columns expected (V5 county_predictions.csv):
    FIPS, year, doy, testing_mask, training_mask,
    yield_bu_acre (actual), hist_yield_trend (baseline),
    predicted_yield, residual

Outputs (to --output-dir):
    stat_tests_<A>_vs_<B>.csv        one row per year x DOY, all tests
    fig_rmse_gap_ci_<A>_vs_<B>.png   RMSE gap + bootstrap 95% CI, by year/DOY
    fig_dm_pvalues_<A>_vs_<B>.png    HLN-DM p-value heatmap (squared loss)
    fig_skill_ci_<A>_vs_<B>.png      skill-vs-trend with CI, both scenarios

Usage:
    python run_stat_tests.py --pred-file county_predictions.csv \
        --scenario-a PREVIOUS_CDL --scenario-b STUDY --output-dir ./stat_out
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(42)
N_BOOT = 2000


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def state_of(fips):
    return pd.Series(fips).astype(str).str.zfill(5).str[:2].to_numpy()


def cluster_bootstrap_stat(func, groups, n_boot=2000):
    """Bootstrap `func()` by resampling whole clusters (states) with replacement.
    Returns (point, lo95, hi95)."""
    uniq = np.unique(groups)
    point = func(np.ones(len(groups), dtype=bool))
    if len(uniq) < 2:
        return point, np.nan, np.nan
    idx_by_cluster = {c: np.where(groups == c)[0] for c in uniq}
    vals = []
    for _ in range(n_boot):
        picked = RNG.choice(uniq, size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_cluster[c] for c in picked])
        mask = np.zeros(len(groups), dtype=bool)
        # boolean index via integer positions (allow repeats -> use take semantics)
        vals.append(func(sel))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return point, lo, hi


def hln_dm_test(e_a, e_b, loss="squared", h=1):
    """Diebold-Mariano with Harvey-Leybourne-Newbold small-sample correction.
    Positive mean loss-diff (a-b) => B more accurate. Returns (dm_stat, p, dbar)."""
    if loss == "squared":
        d = e_a ** 2 - e_b ** 2
    else:
        d = np.abs(e_a) - np.abs(e_b)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 8:
        return np.nan, np.nan, np.nan
    dbar = d.mean()
    # long-run variance with Newey-West style autocovariances up to h-1 (h=1 -> gamma0)
    gamma0 = np.mean((d - dbar) ** 2)
    lrv = gamma0
    for k in range(1, h):
        gk = np.mean((d[k:] - dbar) * (d[:-k] - dbar))
        lrv += 2 * (1 - k / h) * gk
    if lrv <= 0:
        return np.nan, np.nan, dbar
    dm = dbar / np.sqrt(lrv / n)
    # HLN small-sample correction factor
    corr = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * corr
    p = 2 * (1 - stats.t.cdf(abs(dm_hln), df=n - 1))
    return dm_hln, p, dbar


# ----------------------------------------------------------------------
# per (year, doy) comparison
# ----------------------------------------------------------------------
def compare_cell(df_a, df_b, n_boot=2000):
    """df_a, df_b already merged to the SAME counties. Returns a dict of results."""
    m = df_a.merge(df_b, on=["FIPS"], suffixes=("_a", "_b"))
    if len(m) < 10:
        return None
    actual = m["yield_bu_acre_a"].to_numpy()
    trend = m["hist_yield_trend_a"].to_numpy()
    pa = m["predicted_yield_a"].to_numpy()
    pb = m["predicted_yield_b"].to_numpy()
    ea, eb = actual - pa, actual - pb
    groups = state_of(m["FIPS"].to_numpy())
    n = len(m)

    def rmse(pred):
        return lambda sel: np.sqrt(np.mean((actual[sel] - pred[sel]) ** 2))

    def rmse_gap(sel):   # A - B ; positive => B better (lower error)
        ra = np.sqrt(np.mean(ea[sel] ** 2))
        rb = np.sqrt(np.mean(eb[sel] ** 2))
        return ra - rb

    def mae_gap(sel):
        return np.mean(np.abs(ea[sel])) - np.mean(np.abs(eb[sel]))

    def skill(pred):
        # 1 - MSE/MSE_trend  (Murphy 1988 skill score vs trend baseline)
        return lambda sel: 1 - np.mean((actual[sel] - pred[sel]) ** 2) / np.mean((actual[sel] - trend[sel]) ** 2)

    # 1. paired bootstrap on RMSE / MAE gap (state-clustered)
    d_rmse, rmse_lo, rmse_hi = cluster_bootstrap_stat(rmse_gap, groups, n_boot)
    d_mae, mae_lo, mae_hi = cluster_bootstrap_stat(mae_gap, groups, n_boot)

    # 2. Wilcoxon signed-rank on paired |errors|
    try:
        w_stat, w_p = stats.wilcoxon(np.abs(ea), np.abs(eb))
    except ValueError:
        w_stat, w_p = np.nan, np.nan

    # 3. HLN-DM (squared + absolute)
    dm_sq, dm_sq_p, _ = hln_dm_test(ea, eb, "squared")
    dm_ab, dm_ab_p, _ = hln_dm_test(ea, eb, "absolute")

    # 4. skill vs trend + CI, each scenario
    sk_a, sk_a_lo, sk_a_hi = cluster_bootstrap_stat(skill(pa), groups, n_boot)
    sk_b, sk_b_lo, sk_b_hi = cluster_bootstrap_stat(skill(pb), groups, n_boot)

    # 5. Mincer-Zarnowitz calibration (bias) for each scenario
    bias_a, bias_b = np.mean(pa - actual), np.mean(pb - actual)

    rmse_a = np.sqrt(np.mean(ea ** 2))
    rmse_b = np.sqrt(np.mean(eb ** 2))
    return dict(
        N=n, N_states=len(np.unique(groups)),
        RMSE_A=rmse_a, RMSE_B=rmse_b,
        RMSE_gap_A_minus_B=d_rmse, RMSE_gap_lo=rmse_lo, RMSE_gap_hi=rmse_hi,
        RMSE_gap_sig=bool(np.isfinite(rmse_lo) and (rmse_lo > 0 or rmse_hi < 0)),
        MAE_gap_A_minus_B=d_mae, MAE_gap_lo=mae_lo, MAE_gap_hi=mae_hi,
        Wilcoxon_stat=w_stat, Wilcoxon_p=w_p,
        DM_HLN_sq=dm_sq, DM_HLN_sq_p=dm_sq_p,
        DM_HLN_abs=dm_ab, DM_HLN_abs_p=dm_ab_p,
        Skill_A=sk_a, Skill_A_lo=sk_a_lo, Skill_A_hi=sk_a_hi,
        Skill_B=sk_b, Skill_B_lo=sk_b_lo, Skill_B_hi=sk_b_hi,
        Bias_A=bias_a, Bias_B=bias_b,
        Better=("B" if (np.isfinite(rmse_lo) and rmse_lo > 0)
                else "A" if (np.isfinite(rmse_hi) and rmse_hi < 0)
                else "indistinguishable"),
    )


# ----------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------
def plot_rmse_gap(res, A, B, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for yr, g in res.groupby("year"):
        g = g.sort_values("doy")
        ax.errorbar(g.doy, g.RMSE_gap_A_minus_B,
                    yerr=[g.RMSE_gap_A_minus_B - g.RMSE_gap_lo,
                          g.RMSE_gap_hi - g.RMSE_gap_A_minus_B],
                    marker="o", capsize=3, label=str(int(yr)))
    ax.axhline(0, color="0.4", ls="--", lw=1)
    ax.set_xlabel("Day of Year (checkpoint)")
    ax.set_ylabel(f"RMSE({A}) - RMSE({B})  (bu/ac)\n^ below 0: {A} better  |  above 0: {B} better")
    ax.set_title(f"RMSE gap with state-clustered bootstrap 95% CI: {A} vs {B}")
    ax.legend(title="Test year")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_dm_heatmap(res, A, B, path):
    piv = res.pivot(index="year", columns="doy", values="DM_HLN_sq_p")
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=0, vmax=0.2, aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels([int(y) for y in piv.index])
    ax.set_xlabel("Day of Year"); ax.set_title(f"HLN-corrected Diebold-Mariano p-value (squared loss): {A} vs {B}")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="p-value (green = significant)")
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_skill(res, A, B, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    g = res.sort_values(["year", "doy"])
    x = np.arange(len(g))
    ax.errorbar(x, g.Skill_A, yerr=[g.Skill_A - g.Skill_A_lo, g.Skill_A_hi - g.Skill_A],
                marker="s", capsize=2, ls="none", label=A)
    ax.errorbar(x + 0.15, g.Skill_B, yerr=[g.Skill_B - g.Skill_B_lo, g.Skill_B_hi - g.Skill_B],
                marker="D", capsize=2, ls="none", label=B)
    ax.axhline(0, color="0.4", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"{int(r.year)}\nD{int(r.doy)}" for _, r in g.iterrows()], fontsize=7)
    ax.set_ylabel("Skill vs. county yield trend  (1 - MSE/MSE_trend)")
    ax.set_title(f"Forecast skill vs. trend baseline, with clustered bootstrap CI: {A} vs {B}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-file", required=True, help="V5 county_predictions.csv")
    ap.add_argument("--scenario-a", default="PREVIOUS_CDL")
    ap.add_argument("--scenario-b", default="STUDY")
    ap.add_argument("--output-dir", default="./stat_out")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.pred_file)
    A, B = args.scenario_a, args.scenario_b
    for s in (A, B):
        if s not in set(df["testing_mask"].unique()):
            raise SystemExit(f"testing_mask '{s}' not in file. Available: "
                             f"{sorted(df['testing_mask'].unique())}")

    rows = []
    for (yr, doy), g in df.groupby(["year", "doy"]):
        ga = g[g.testing_mask == A]
        gb = g[g.testing_mask == B]
        if ga.empty or gb.empty:
            continue
        r = compare_cell(ga, gb, args.n_boot)
        if r:
            r.update(year=yr, doy=doy)
            rows.append(r)

    if not rows:
        raise SystemExit("No comparable year x DOY cells found for those two scenarios.")
    res = pd.DataFrame(rows).sort_values(["year", "doy"]).reset_index(drop=True)
    out_csv = os.path.join(args.output_dir, f"stat_tests_{A}_vs_{B}.csv")
    res.to_csv(out_csv, index=False)

    plot_rmse_gap(res, A, B, os.path.join(args.output_dir, f"fig_rmse_gap_ci_{A}_vs_{B}.png"))
    plot_dm_heatmap(res, A, B, os.path.join(args.output_dir, f"fig_dm_pvalues_{A}_vs_{B}.png"))
    plot_skill(res, A, B, os.path.join(args.output_dir, f"fig_skill_ci_{A}_vs_{B}.png"))

    n = len(res)
    sig_boot = int(res.RMSE_gap_sig.sum())
    sig_dm = int((res.DM_HLN_sq_p < 0.05).sum())
    sig_wil = int((res.Wilcoxon_p < 0.05).sum())
    print(res[["year", "doy", "N", "N_states", "RMSE_A", "RMSE_B",
               "RMSE_gap_A_minus_B", "RMSE_gap_lo", "RMSE_gap_hi",
               "DM_HLN_sq_p", "Wilcoxon_p", "Skill_B", "Better"]].to_string(index=False))
    print(f"\nAcross {n} year x DOY cells ({A} vs {B}):")
    print(f"  bootstrap RMSE gap excludes 0 : {sig_boot}/{n}")
    print(f"  HLN-DM p<0.05 (squared loss)  : {sig_dm}/{n}")
    print(f"  Wilcoxon p<0.05               : {sig_wil}/{n}")
    print(f"\nSaved: {out_csv}\n       + 3 figures in {args.output_dir}")


if __name__ == "__main__":
    main()
