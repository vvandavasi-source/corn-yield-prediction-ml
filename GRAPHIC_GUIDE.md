# Corn-yield graphics guide

This guide explains why each generated graphic is useful and how to interpret
it. The current graphics and the dashboard are regenerated from
[`v5_out`](v5_out/), not the earlier `v5_50_seed_all_in_one` archive.

All model-result figures use **same-year CDL training**. “Previous CDL,”
“final CDL,” “study,” and “clean rebuilt” identify test-mask scenarios, not
training protocols. Results use 50 seed runs. A seed standard deviation
describes run-to-run spread; it is not a confidence interval unless a figure
explicitly defines an interval.

The current archive has six observation checkpoints, 16 days apart. The
previous-CDL, final-CDL, and study test masks cover 2021–2025; the clean-rebuilt
mask covers 2022–2025. Comparisons across masks should account for this
different year/date coverage and, where applicable, different county cohorts.

## Shared color language

| Test-mask scenario | Color | Hex |
|---|---|---|
| Previous CDL | Blue | `#176B87` |
| Final CDL (hindsight) | Slate | `#778391` |
| Clean rebuilt | Green | `#39826C` |
| Published study | Purple | `#8664A8` |

Date/year graphics use a consistent year palette: blue (`#176B87`) for 2021,
gold (`#D28A28`) for 2022, green (`#39826C`) for 2023, purple (`#8466A5`) for
2024, and muted red (`#B45C48`) for 2025. The county map uses a continuous
**white-to-green** yield scale from `#FFFFFF` to `#28634E`, without blue.
Lighter counties have lower predicted yields; darker green counties have
higher predicted yields.

## Interactive county dashboard

Open the [ensemble county yield dashboard](v5_out/summary/county_yield_dashboard.html).
It includes filters for training/testing scenario, year, state, and county.
The map has a separate observation-date selector and stays a national overview
when the state filter for the charts changes.
The uncertainty and variability maps occupy a second row beneath the yield and
signed-error maps.

| Graphic | Why it is useful | How to interpret it |
|---|---|---|
| **Predicted yield by county (map)** | Reveals geographic yield patterns and spatially unusual counties that are difficult to spot in a table. | Color encodes the 50-seed mean prediction. Hover shows county/state/FIPS, predicted and observed yield, signed and absolute error, historical trend, seed SD, and seed count. Seed SD is not a confidence interval. |
| **Signed error by county (map)** | Locates spatial clusters of over- or under-prediction alongside the yield map. | Error is prediction minus observed yield. Red indicates underprediction, green overprediction, and white near-zero error. The symmetric color range is fixed for the selected scenario/year across dates. |
| **95% confidence interval for mean predicted yield (map)** | Shows geographic differences in the precision of the ensemble mean at the selected date. | County bounds are the mean prediction ± a two-sided Student-t critical value times seed SD divided by √n. Hover gives lower/upper yield bounds and interval width. Green marks narrower intervals, white the middle of the selected-date width range, and red wider intervals. This is a confidence interval for the mean seed prediction, not a prediction interval for a future observation. |
| **Coefficient of variation in predicted yield (map)** | Compares relative seed-to-seed variability across counties while accounting for differences in mean yield. | CV is 100 × seed SD ÷ absolute mean predicted yield. Green marks smaller CV and red larger CV. It is unavailable if the mean prediction is exactly zero. |
| **County yield through the season** | Shows how estimates evolve across observation dates and compares the selected county with reported yield and its historical trend. | The series distinguish observed yield, ensemble prediction, and historical trend. The shaded band is ±1 seed SD. |
| **County prediction error by observation date** | Reveals over- or under-prediction and whether error changes through the growing season. | Signed error is prediction minus observed yield: positive means over-prediction, negative means under-prediction. Absolute error gives magnitude without direction. |
| **County error distribution by observation date** | Shows the center, spread, and asymmetry of county errors instead of reducing them to a single average. | Each box summarizes county-level signed errors in the selected state; individual counties are not plotted as points. |
| **State accuracy by observation date** | Compares aggregate accuracy through the season using two complementary metrics. | MAE and RMSE are in bushels per acre; lower is better. RMSE penalizes large errors more heavily. Hover includes county counts. |
| **Top model features by year/date** | Highlights which input variables contributed most to the ensemble of seed-trained models at the selected prediction date. | Bars show mean gain share among same-year CDL models; error bars show standard deviation across seeds. This is model-specific descriptive importance, not causal evidence, and does not change with the selected test-mask scenario. |

The dashboard embeds 85,012 county-date-scenario rows and the 15 highest
mean-gain features for each available year/date from `v5_out`. It retrieves
county boundaries from Plotly's public GeoJSON when opened, so an internet
connection is required to render the map.

### Single-seed dashboard

The [single-seed dashboard for seed 1](v5_out/summary/single_seed_yield_dashboard_seed_01.html)
shows one run without averaging predictions across seeds. It includes national
yield and signed-error maps, a selected-county seasonal series, county error
distribution, state accuracy by date, top feature gain shares for the selected
year/date, and state/year summary RMSE, MAE, MAPE, R², bias, and skill against
the historical trend. The statistics are specific to the selected seed,
scenario, year, and state; the maps remain national.

Generate the same view for any seed from 1 through 50 with
`python generate_single_seed_dashboard.py --seed 17`. The generated HTML is
named `single_seed_yield_dashboard_seed_17.html` and is written to
`v5_out/summary/`. Re-run with another seed number to create a different
standalone dashboard.

## Extra-credit model comparisons

See the [seasonal comparison](v5_out/figs/extra_credit/extra_credit_seasonal_comparison.png),
[late-season heatmap](v5_out/figs/extra_credit/extra_credit_late_season_heatmap.png),
[technical appendix](v5_out/summary/extra_credit_technical_appendix.docx), and
[scenario summary CSV](v5_out/summary/extra_credit_scenario_summary.csv).

| Graphic | Why it is useful | How to interpret it |
|---|---|---|
| **Seasonal accuracy across year/date evaluations** | Shows how error and anomaly-based fit evolve during the season, and whether that pattern is consistent across four masks. | RMSE and anomaly R² are in separate panels. Lines summarize available year/date groups; shaded bands show ±1 SD across groups, not uncertainty in the mean. Clean-rebuilt has no 2021 results, so its coverage differs. |
| **Late-season RMSE heatmap** | Makes late-season differences among test masks and years easy to scan. | Each cell gives RMSE at the latest available date for that year/scenario; lower is better. Blank cells mean no result is available. It is a coverage-aware comparison, not a fully controlled ranking. |

The clean-rebuilt and published-study masks share 24 year/date groups. Their
mean clean-rebuilt-minus-study RMSE difference is approximately
`+0.07 ± 0.23 bu/ac` across those groups; this is descriptive and the SD across
groups is not a confidence interval.

## Paired-seed test graphics

The [paired-seed figures and summary](v5_out/figs/paired_seed_tests/) compare
clean-rebuilt and published-study RMSE over 50 paired runs for 24 year/date
groups (2022–2025).

| Graphic | Why it is useful | How to interpret it |
|---|---|---|
| **Paired-seed RMSE difference by date** | Shows the size and direction of the difference across seeds, dates, and years. | The difference is clean rebuilt minus study. Negative values favor clean rebuilt; positive values favor study. Whiskers are approximate 95% intervals computed from paired-seed SD and count. |
| **Holm-adjusted paired t-test heatmap** | Shows which year/date differences remain statistically significant after correcting for multiple comparisons. | Cells show Holm-adjusted p-values; green marks values below 0.05. Nonsignificance does not establish equivalence. |
| **Paired Wilcoxon signed-rank heatmap** | Provides a nonparametric paired-seed check alongside the t-test. | Cells show unadjusted Wilcoxon p-values; they are not corrected for multiple comparisons. Read them with the effect-size plot. |

The new source provides paired-seed tests, not Diebold–Mariano or J-test
results. The updated figures therefore make no DM or J-test claim.

## Sanity-check and validation graphics

See the [sanity-check graphics and audit tables](v5_out/figs/sanity_check_graphics/).
They address accuracy specifications, a historical-trend baseline, metric
arithmetic, and repeatable validation design.

| Graphic | Why it is useful | How to interpret it |
|---|---|---|
| **Accuracy specification checks** | Directly tests the stated primary MAPE and secondary R² requirements. | Year/date points are compared with the marked thresholds. Passing those checks does not prove generalization beyond the available evaluation data. |
| **Historical-trend benchmark** | Tests whether the model adds value beyond a simple historical-yield trend. | RMSE panels compare model and trend on matched records; positive skill means model RMSE is lower than trend RMSE. |
| **Independent metric arithmetic check** | Makes a concrete metric calculation auditable and compares it with the saved per-seed result. | The sample, formulas, and independently recomputed metrics are shown. This verifies arithmetic for the checked case, not every upstream input or model assumption. |
| **Validation-plan blueprint** | Converts the sanity-check requirements into a repeatable process for future runs. | This is a process diagram, not a result figure; it specifies checks and reporting steps. |

For the current study-mask ensemble, pooled county-date MAPE is 8.55%, pooled
R² is 0.821, and model/trend RMSE is 16.20/22.21 bu/ac. The archive contains
one below-zero ensemble prediction at FIPS 18013, 2021 DOY 225 (−0.145 bu/ac);
it is retained rather than clipped and called out in the boundary-check figure.
The same record has a historical trend of −157.33 bu/ac, so inspect it before
interpreting the trend-baseline comparison.

## Final-poster graphics

See the [poster-ready graphics](v5_out/figs/final_poster_graphics/), including
the four-page PDF and key-metrics CSV.

| Graphic | Why it is useful | How to interpret it |
|---|---|---|
| **Model workflow** | Quickly explains how crop-health observations lead to county-yield estimates. | Follow the stages from inputs through modeling to prediction; it is conceptual, not a data-flow audit. |
| **County yield validation** | Compares predicted and reported yields and makes bias and large residuals visible. | Points closer to the one-to-one line indicate better agreement. For 2025 DOY 273 (published-study mask), N=604, R²=0.848, MAPE=5.64%, and RMSE=12.16 bu/ac. These metrics describe that cohort/date only. |
| **Seasonal accuracy** | Shows how error and fit evolve as growing-season observations accrue. | Read the metrics alongside the displayed dates. Checkpoints are 16 days apart, so the plot does not support a weekly-forecast claim. |
| **Model inputs and importance** | Explains what data inform estimates and which features the model uses most at 2025 DOY 273. | Feature importance is descriptive of model use, not evidence that a feature causally changes yield. |

## Regeneration

Run these scripts from the workspace root; generated data-bound materials read
from `v5_out`:

- [`generate_county_yield_dashboard.py`](generate_county_yield_dashboard.py)
- [`generate_single_seed_dashboard.py`](generate_single_seed_dashboard.py)
- [`generate_extra_credit_appendix.py`](generate_extra_credit_appendix.py)
- [`generate_cell18_dm_graphics.py`](generate_cell18_dm_graphics.py)
- [`generate_sanity_check_graphics.py`](generate_sanity_check_graphics.py)
- [`generate_final_poster_graphics.py`](generate_final_poster_graphics.py)
