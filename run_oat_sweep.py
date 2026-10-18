#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_oat_sweep.py  --  drive the V5 model across One-At-a-Time (OAT) levels and
write the exact `sensitivity_oat.csv` that plot_analysis.py consumes, plus a
per-seed ensemble directory that feeds the seed-stability plots and
aggregate_seeds.py.

It holds every other input fixed and perturbs ONE parameter at a time, as the
Modeling rubric requires (no simultaneous variation). The parameters it sweeps
are real V5 CLI args:

    --trees                  n_estimators     -> continuous %  sweep
    --small-cohort-threshold structural knob  -> continuous %  sweep (optional)
    --seed                   random init      -> discrete ensemble (its
                             "sensitivity" is the output spread; see 06/07 plots)

NOTE on feature perturbation: V5's --vegetation-scale is a *unit-conversion*
factor with a hard range check (scaled NDVI must stay in [-1, 1], etc.), so it
is NOT used here as a +/-% feature knob -- it would raise on already-scaled data.
If you want a genuine feature-perturbation line, perturb the input CSVs into a
separate --input-dir with clipping, and add it with --extra-input-sweep (see -h).

Each run writes to its own --output-dir so nothing is clobbered and V5 never
reuses a stale model_comparison_metrics.csv.

Typical use on the model machine (Windows, env icdl-authors):

    python run_oat_sweep.py ^
        --model corn_yield_model_CY2_V5_MASK_COMPARISON.py ^
        --input-dir "C:\\BSE509 - Clean Data\\model_inputs" ^
        --work-dir  "C:\\BSE509_Model_V5\\work_v5" ^
        --out-root  ".\\oat_runs" ^
        --baseline-trees 500 --baseline-seed 42 ^
        --sweep trees ^
        --levels -25 -10 10 25 ^
        --seeds 42 7 13 ^
        --scenario STUDY ^
        --test-years 2022 2023 2024 2025 ^
        --make-plots --plotter plot_analysis.py

Minimal (rubric's long-run allowance: 2 params x 2 increments):
    python run_oat_sweep.py --model ... --input-dir ... --work-dir ... \
        --sweep trees small_cohort_threshold --levels -25 25 --seeds 42 7

Outputs under --out-root:
    sensitivity_oat.csv        param,level_pct,param_value,<all metrics>,n_cells,scenario
    run_manifest.csv           one row per model run (cmd, status, score)
    baseline/                  the level_pct=0 reference run
    sweeps/trees_<v>/ ...      continuous-parameter runs
    seeds/seed_<s>/ ...        seed ensemble (point plot_analysis --seed-metrics here)
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd

ALL_METRICS = ["rmse", "mae", "r2", "anomaly_r2", "mse_skill_vs_trend"]
# which CLI flag each sweepable continuous parameter maps to, and its default
PARAM_FLAG = {"trees": "--trees", "small_cohort_threshold": "--small-cohort-threshold"}
PARAM_CAST = {"trees": int, "small_cohort_threshold": int}


def build_cmd(a, output_dir, trees, seed, small_cohort):
    cmd = [a.python, a.model,
           "--input-dir", a.input_dir, "--work-dir", a.work_dir,
           "--output-dir", output_dir,
           "--trees", str(int(trees)), "--seed", str(int(seed))]
    if small_cohort is not None:
        cmd += ["--small-cohort-threshold", str(int(small_cohort))]
    if a.test_years:
        cmd += ["--test-years", *map(str, a.test_years)]
    if a.doys:
        cmd += ["--doys", *map(str, a.doys)]
    if a.extra_args:
        cmd += a.extra_args.split()
    return cmd


def run_model(a, output_dir, trees, seed, small_cohort, label):
    cmd = build_cmd(a, output_dir, trees, seed, small_cohort)
    print(f"\n[{label}] trees={trees} seed={seed} sct={small_cohort}")
    print("   $ " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    if a.dry_run:
        return 0, "dry-run"
    os.makedirs(output_dir, exist_ok=True)
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=a.timeout if a.timeout > 0 else None)
    except subprocess.TimeoutExpired:
        return 1, "TIMEOUT"
    dt = time.time() - t0
    if r.returncode != 0:
        tail = (r.stdout or "")[-1500:] + "\n--- stderr ---\n" + (r.stderr or "")[-1500:]
        sys.stderr.write(tail + "\n")
        return r.returncode, f"FAILED ({dt:.0f}s)"
    return 0, f"ok ({dt:.0f}s)"


def score_run(output_dir, scenario, training):
    """Mean of every metric over the chosen scenario's year x DOY cells."""
    p = os.path.join(output_dir, "model_comparison_metrics.csv")
    if not os.path.exists(p):
        return {m: np.nan for m in ALL_METRICS}, 0
    d = pd.read_csv(p)
    m = d[d["testing_mask"].astype(str).str.upper().str.contains(scenario.upper())]
    if training:
        m = m[m["training_mask"].astype(str).str.upper().str.contains(training.upper())]
    if m.empty:
        return {mt: np.nan for mt in ALL_METRICS}, 0
    out = {}
    for mt in ALL_METRICS:
        out[mt] = float(pd.to_numeric(m[mt], errors="coerce").mean()) if mt in m else np.nan
    return out, int(len(m))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to corn_yield_model_CY2_V5_MASK_COMPARISON.py")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--out-root", default="./oat_runs")
    ap.add_argument("--python", default=sys.executable, help="interpreter to launch V5 with")
    ap.add_argument("--baseline-trees", type=int, default=500)
    ap.add_argument("--baseline-seed", type=int, default=42)
    ap.add_argument("--baseline-small-cohort", type=int, default=30)
    ap.add_argument("--sweep", nargs="*", default=["trees"],
                    choices=list(PARAM_FLAG.keys()),
                    help="continuous parameters to %%-sweep (default: trees)")
    ap.add_argument("--levels", nargs="*", type=float, default=[-25, -10, 10, 25],
                    help="%% changes applied to each swept parameter (0 is added automatically)")
    ap.add_argument("--seeds", nargs="*", type=int, default=[],
                    help="seed ensemble; e.g. 42 7 13. Empty = skip seed runs.")
    ap.add_argument("--scenario", default="STUDY", help="testing_mask substring to score")
    ap.add_argument("--training", default="", help="optional training_mask substring filter")
    ap.add_argument("--metric", default="rmse", choices=ALL_METRICS,
                    help="metric reported as the headline output (all metrics are logged anyway)")
    ap.add_argument("--test-years", nargs="*", type=int, default=[])
    ap.add_argument("--doys", nargs="*", type=int, default=[])
    ap.add_argument("--extra-args", default="", help="extra flags passed verbatim to V5")
    ap.add_argument("--timeout", type=int, default=0, help="per-run timeout seconds (0 = none)")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--make-plots", action="store_true", help="call the plotter at the end")
    ap.add_argument("--plotter", default="plot_analysis.py")
    a = ap.parse_args()

    os.makedirs(a.out_root, exist_ok=True)
    manifest, sens_rows = [], []

    def log(param, level_pct, value, seed, outdir, status):
        metrics, ncell = score_run(outdir, a.scenario, a.training) if "ok" in status or status == "dry-run" else ({m: np.nan for m in ALL_METRICS}, 0)
        row = dict(param=param, level_pct=level_pct, param_value=value, seed=seed,
                   output_dir=outdir, status=status, n_cells=ncell, scenario=a.scenario)
        row.update(metrics)
        manifest.append(row)
        return metrics, ncell

    # ---- 1. baseline (level_pct = 0 for every continuous param) ----
    base_dir = os.path.join(a.out_root, "baseline")
    rc, st = run_model(a, base_dir, a.baseline_trees, a.baseline_seed,
                       a.baseline_small_cohort, "baseline")
    base_metrics, base_n = log("baseline", 0.0, a.baseline_trees, a.baseline_seed, base_dir, st)
    base_val = {"trees": a.baseline_trees, "small_cohort_threshold": a.baseline_small_cohort}
    # one level_pct=0 anchor row per swept param (same baseline score)
    for p in a.sweep:
        sens_rows.append(dict(param=p, level_pct=0.0, param_value=base_val[p],
                              n_cells=base_n, scenario=a.scenario, **base_metrics))

    # ---- 2. continuous %-sweeps (one variable at a time) ----
    for p in a.sweep:
        cast = PARAM_CAST[p]
        for lv in a.levels:
            if lv == 0:
                continue
            val = cast(round(base_val[p] * (1 + lv / 100.0)))
            outdir = os.path.join(a.out_root, "sweeps", f"{p}_{val}")
            trees = val if p == "trees" else a.baseline_trees
            sct = val if p == "small_cohort_threshold" else a.baseline_small_cohort
            rc, st = run_model(a, outdir, trees, a.baseline_seed, sct, f"{p} {lv:+g}%")
            metrics, ncell = log(p, float(lv), val, a.baseline_seed, outdir, st)
            sens_rows.append(dict(param=p, level_pct=float(lv), param_value=val,
                                  n_cells=ncell, scenario=a.scenario, **metrics))

    # ---- 3. seed ensemble (feeds 06/07 stability plots + aggregate_seeds) ----
    seed_scores = []
    for s in a.seeds:
        outdir = os.path.join(a.out_root, "seeds", f"seed_{s}")
        rc, st = run_model(a, outdir, a.baseline_trees, s, a.baseline_small_cohort, f"seed {s}")
        metrics, ncell = log("seed", np.nan, a.baseline_trees, s, outdir, st)
        if np.isfinite(metrics.get(a.metric, np.nan)):
            seed_scores.append(metrics[a.metric])

    # ---- write sensitivity_oat.csv (schema plot_analysis.py expects) ----
    cols = ["param", "level_pct", "param_value"] + ALL_METRICS + ["n_cells", "scenario"]
    sens = pd.DataFrame(sens_rows)[cols].sort_values(["param", "level_pct"])
    sens_path = os.path.join(a.out_root, "sensitivity_oat.csv")
    sens.to_csv(sens_path, index=False)

    man = pd.DataFrame(manifest)
    man.to_csv(os.path.join(a.out_root, "run_manifest.csv"), index=False)

    # ---- console summary ----
    print("\n" + "=" * 64)
    print(f"sensitivity_oat.csv written: {sens_path}")
    print(sens[["param", "level_pct", "param_value", a.metric, "n_cells"]].to_string(index=False))
    # sensitivity coefficients for the headline metric
    print(f"\nSensitivity coefficient |%d{a.metric} / %d input|  (max over levels):")
    for p in a.sweep:
        g = sens[(sens.param == p) & (sens.level_pct != 0)]
        base = sens[(sens.param == p) & (sens.level_pct == 0)][a.metric]
        if g.empty or base.empty or not base.iloc[0]:
            continue
        b = base.iloc[0]
        coef = ((g[a.metric] - b) / b * 100 / g.level_pct).abs().max()
        print(f"   {p:24s} {coef:.3f}")
    if a.seeds and seed_scores:
        sc = np.array(seed_scores)
        cv = 100 * sc.std(ddof=1) / sc.mean() if sc.mean() else np.nan
        print(f"\nSeed ensemble ({len(sc)} seeds): {a.metric} = {sc.mean():.3f} "
              f"+/- {sc.std(ddof=1):.3f}  (CV {cv:.2f}%)  "
              f"range [{sc.min():.3f}, {sc.max():.3f}]")
        print(f"   -> seed stability plots: plot_analysis.py --seed-metrics {os.path.join(a.out_root, 'seeds')}")

    # ---- optional: render the plots now ----
    if a.make_plots and not a.dry_run:
        seeds_dir = os.path.join(a.out_root, "seeds")
        cmd = [a.python, a.plotter, "--sensitivity", sens_path,
               "--sens-metric", a.metric, "--outdir", os.path.join(a.out_root, "figs")]
        if os.path.isdir(seeds_dir) and a.seeds:
            cmd += ["--seed-metrics", seeds_dir]
        print("\n[plots] $ " + " ".join(cmd))
        subprocess.run(cmd)


if __name__ == "__main__":
    main()
