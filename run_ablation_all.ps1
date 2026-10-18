# run_ablation_all.ps1
# End-to-end feature-ablation study: runs all 5 feature configs, then builds the
# summary table + 6 figures. Edit the paths below, then:  ./run_ablation_all.ps1
# Re-running skips configs already complete.

$base    = "C:\Users\Patron\Downloads\BSE509_Model_V5"
$model   = "corn_yield_model_V5_ABLATION.py"
$in      = "$base\v5_data"
$abldir  = "$base\ablation"
$work    = "$abldir\work"
$configs = @("none", "weather", "soil", "anomaly", "veg_only")

# --- SCOPE ---------------------------------------------------------------
# Fast subset (~25-35 min total). For the full all-years run (~1 hr), set
# $scope = @()  to use the model's defaults (all 5 years, all 6 DOYs).
$scope = @("--test-years", "2022", "2024", "--doys", "225", "257")
# -------------------------------------------------------------------------

foreach ($cfg in $configs) {
  $out = Join-Path $abldir $cfg
  if (Test-Path (Join-Path $out "model_comparison_metrics.csv")) {
    Write-Host "ablation '$cfg' already complete - skipping"; continue
  }
  Write-Host "`n===== ABLATION: $cfg ====="
  python $model --input-dir $in --work-dir $work --output-dir $out `
    --seed 42 --trees 500 --ablate $cfg @scope
  if ($LASTEXITCODE -ne 0) { Write-Host "ablation '$cfg' FAILED (exit $LASTEXITCODE) - stopping"; exit 1 }
}

Write-Host "`n===== SUMMARY + PLOTS ====="
python ablation_summary.py --ablation-dir $abldir --outdir "$abldir\figs"

Write-Host "`nDone. Results in:"
Write-Host "  $abldir\figs\ablation_summary.csv"
Write-Host "  $abldir\figs\  (6 figures + ablation_figures_index.md)"
