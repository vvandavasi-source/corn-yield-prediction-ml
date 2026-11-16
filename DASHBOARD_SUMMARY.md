# County Yield Dashboards: Work Summary

## Purpose

Two interactive dashboards were built to explore county-level corn-yield
predictions from the project's `v5_out` results. The dashboards complement one
another: the full dashboard summarizes the available seed runs, while the
single-seed dashboard lets a user inspect an individual model run without
averaging it with other seeds.

## Dashboards

### Full ensemble dashboard

Open [the county yield dashboard](v5_out/summary/county_yield_dashboard.html).
Its generator is
[generate_county_yield_dashboard.py](generate_county_yield_dashboard.py).

The dashboard combines predictions across seed files and provides filters for
testing scenario, state, county, and year. A separate date selector controls
the national maps; changing the state filter does not restrict those maps.
Available views include:

- **National predicted-yield map:** county predictions averaged across
  available seeds, colored white-to-green.
- **National signed-error map:** prediction minus observed yield, colored to
  distinguish underprediction, near-zero error, and overprediction.
- **National 95% confidence-interval map:** shows mean predicted yield bounds
  from a two-sided Student-t interval across seed predictions. Hover includes
  lower and upper bounds and interval width. Green marks smaller widths, white
  the middle of the displayed range, and red larger widths.
- **National coefficient-of-variation map:** shows seed-to-seed standard
  deviation relative to absolute mean predicted yield as a percentage. Green
  marks lower CV and red higher CV; CV is unavailable when mean prediction is
  zero.
- **County seasonal yield chart:** observed yield, ensemble prediction,
  historical trend, and a band showing plus/minus one seed standard deviation.
- **County error by date:** signed and absolute prediction error.
- **State error distribution:** county-level signed errors summarized by date.
- **State accuracy by date:** county-averaged MAE and RMSE.
- **Top model features:** the 15 highest mean feature-gain shares for the
  selected year and map date. Mean gain share is averaged across available
  same-year CDL seed models; error bars show standard deviation across seeds.

  The confidence-interval and CV maps are arranged together in a second row
  beneath the yield and signed-error maps.

The feature-importance chart is independent of the selected testing scenario:
it summarizes the same-year CDL training models for that year/date. Importance
is descriptive, not causal.

### Single-seed dashboard

Open [the seed 1 dashboard](v5_out/summary/single_seed_yield_dashboard_seed_01.html).
Its generator is
[generate_single_seed_dashboard.py](generate_single_seed_dashboard.py).

This dashboard has filters for testing scenario, year, state, county, and
observation date. Its maps and charts use only the selected seed. It includes
national predicted-yield and signed-error maps, a selected-county seasonal
yield chart and error chart, state error distributions and accuracy trends,
and feature gain shares for the selected year/date. The feature chart uses
that seed's fitted model rather than aggregating across seeds.

The selected state/year summary reports county-date record and county counts,
RMSE, MAE, MAPE, R², bias, trend RMSE, and skill against the historical trend.
Negative predictions are retained as model diagnostics rather than clipped.

## Data and interpretation

Both generators use county prediction and feature-importance CSVs under
`v5_out/seeds/`. The full dashboard averages predictions across available
seeds and calculates the sample standard deviation of those predictions. This
spread is not a confidence interval. The single-seed dashboard reports no
cross-seed averaging.

The full-dashboard 95% interval uses the available seeds as replicates and
calculates mean ± t × sample SD / √n. It is an interval for the ensemble mean,
not a prediction interval for a future county yield. The CV is 100 × sample
SD / |mean prediction|; it is undefined for a zero mean.

Both dashboards use same-year CDL training results and support the available
previous-CDL, final-CDL, study, and clean-rebuilt testing masks. The final-CDL
scenario uses hindsight inputs and should be treated as a benchmark, not a
like-for-like operational forecast. The county error convention is
**predicted minus observed**: positive values indicate overprediction and
negative values indicate underprediction. Missing counties or dates are not
imputed.

County display names are supplemented from the root NASS corn dataset when
available. Plotly is loaded from a public CDN, and county boundaries are
retrieved from public GeoJSON, so an internet connection is needed to render
the interactive maps.

## Regenerating the dashboards

From the main BSE 508 folder, run:

```powershell
python .\generate_county_yield_dashboard.py
python .\generate_single_seed_dashboard.py --seed 1
```

Replace `1` with a seed number from 1 to 50 to generate another standalone
single-seed dashboard in `v5_out/summary/`.

## Related documentation

See [GRAPHIC_GUIDE.md](GRAPHIC_GUIDE.md) for explanations of the dashboards
and other project graphics.
