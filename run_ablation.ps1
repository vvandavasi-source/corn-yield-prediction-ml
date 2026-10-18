# run_ablation.ps1
# Feature-ablation study: runs the ablation model under 5 feature configurations,
# single seed, all test years. Each run evaluates every mask scenario, so you get
# the "broken down by CDL" view automatically. Shared --work-dir reuses staged inputs.
#
# Edit the paths below, then:  ./run_ablation.ps1
# Re-running skips configs already complete.

$base    = "C:\Users\Patron\Downloads\BSE509_Model_V5"
$model   = "corn_yield_model_V5_ABLATION.py"
$in      = "$base\v5_data"
$work    = "$base\ablation\work"
$configs = @("none", "weather", "soil", "anomaly", "veg_only")

foreach ($cfg in $configs) {
  $out = "$base\ablation\$cfg"
  if (Test-Path (Join-Path $out "model_comparison_metrics.csv")) {
    Write-Host "ablation '$cfg' already complete - skipping"; continue
  }
  Write-Host "===== ABLATION: $cfg ====="
  python $model --input-dir $in --work-dir $work --output-dir $out `
    --seed 42 --trees 500 --ablate $cfg
  if ($LASTEXITCODE -ne 0) { Write-Host "ablation '$cfg' FAILED (exit $LASTEXITCODE) - stopping"; break }
}
Write-Host "`nDone. Summarize with:  python ablation_summary.py --ablation-dir `"$base\ablation`" --outdir `"$base\ablation\figs`""
