"""Generate the model-comparison appendix and publication-quality figures."""

import csv
import statistics
import zipfile
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SUMMARY_DIR = ROOT / "v5_50_seed_all_in_one" / "v5_50_seed_all_in_one" / "summary"
SOURCE = SUMMARY_DIR / "seed_summary_mean_sd.csv"
FIGURE_DIR = SUMMARY_DIR / "figures"
OUTPUT_DOCX = SUMMARY_DIR / "extra_credit_technical_appendix.docx"
OUTPUT_CSV = SUMMARY_DIR / "extra_credit_scenario_summary.csv"
FIGURE_SEASONAL = FIGURE_DIR / "extra_credit_seasonal_comparison.png"
FIGURE_HEATMAP = FIGURE_DIR / "extra_credit_late_season_heatmap.png"

SCENARIO_ORDER = [
    "Previous CDL training / previous CDL testing",
    "Same-year CDL training / previous CDL testing",
    "Final CDL hindsight benchmark",
    "Our ICDLs",
    "Published ICDLs",
]
SCENARIO_SHORT = {
    SCENARIO_ORDER[0]: "Previous CDL train / previous CDL test",
    SCENARIO_ORDER[1]: "Same-year train / previous CDL test",
    SCENARIO_ORDER[2]: "Final CDL train/test (hindsight)",
    SCENARIO_ORDER[3]: "Our ICDLs (clean rebuilt)",
    SCENARIO_ORDER[4]: "Published ICDLs (study)",
}
COLORS = {
    SCENARIO_ORDER[0]: "#176b87",
    SCENARIO_ORDER[1]: "#d4842b",
    SCENARIO_ORDER[2]: "#778391",
    SCENARIO_ORDER[3]: "#39826c",
    SCENARIO_ORDER[4]: "#8664a8",
}
TEST_COHORT_LABELS = {
    "PREVIOUS_CDL": "Previous CDL",
    "FINAL_CDL": "Final CDL (hindsight)",
    "STUDY": "Study",
    "CLEAN_REBUILT": "Clean rebuilt",
}


def read_metrics():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Required model summary not found: {SOURCE}")
    with SOURCE.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"No model comparison rows found in {SOURCE}")
    for row in rows:
        for column in ("year", "doy", "rmse_mean", "rmse_std", "mae_mean", "r2_mean", "anomaly_r2_mean"):
            if row[column] == "":
                raise ValueError(f"Missing {column} in model summary row: {row}")
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["training_mask"], row["testing_mask"], row["scenario"])
        groups[key].append(row)

    summaries = []
    for (training, testing, scenario), items in groups.items():
        metrics = {}
        for metric in ("rmse_mean", "mae_mean", "r2_mean", "anomaly_r2_mean"):
            values = [float(row[metric]) for row in items]
            metrics[metric] = (statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0)
        summaries.append({
            "training": training,
            "testing": testing,
            "scenario": scenario,
            "periods": len(items),
            "years": len({int(row["year"]) for row in items}),
            "metrics": metrics,
        })
    summaries.sort(key=lambda item: SCENARIO_ORDER.index(item["scenario"]) if item["scenario"] in SCENARIO_ORDER else 99)
    return summaries


def write_summary_csv(summaries, rows):
    baseline_key = ("PREVIOUS_CDL", "PREVIOUS_CDL")
    same_year_key = ("SAME_YEAR_CDL", "PREVIOUS_CDL")
    baseline = {
        (int(row["year"]), int(row["doy"])): float(row["rmse_mean"])
        for row in rows
        if (row["training_mask"], row["testing_mask"]) == baseline_key
    }
    alternative = {
        (int(row["year"]), int(row["doy"])): float(row["rmse_mean"])
        for row in rows
        if (row["training_mask"], row["testing_mask"]) == same_year_key
    }
    common = sorted(set(baseline) & set(alternative))
    if not common:
        raise ValueError("No matched evaluation periods found for the previous-CDL test comparison.")
    paired_deltas = [alternative[key] - baseline[key] for key in common]
    delta_mean = statistics.mean(paired_deltas)
    delta_sd = statistics.stdev(paired_deltas) if len(paired_deltas) > 1 else 0.0

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow([
            "scenario", "training_mask", "testing_mask", "evaluation_periods_year_doy",
            "years", "rmse_mean_bu_acre", "rmse_sd_across_periods_bu_acre",
            "mae_mean_bu_acre", "mae_sd_across_periods_bu_acre", "r2_mean",
            "r2_sd_across_periods", "anomaly_r2_mean", "anomaly_r2_sd_across_periods",
            "paired_rmse_delta_vs_previous_cdl_baseline_bu_acre", "paired_periods",
        ])
        for item in summaries:
            metrics = item["metrics"]
            is_matched_alternative = (item["training"], item["testing"]) == same_year_key
            is_baseline = (item["training"], item["testing"]) == baseline_key
            delta = 0.0 if is_baseline else delta_mean if is_matched_alternative else ""
            paired_count = len(common) if is_matched_alternative or is_baseline else ""
            writer.writerow([
                item["scenario"], item["training"], item["testing"], item["periods"],
                item["years"], *[value for metric in ("rmse_mean", "mae_mean", "r2_mean", "anomaly_r2_mean")
                                 for value in metrics[metric]], delta, paired_count,
            ])
    return delta_mean, delta_sd, len(common)


def make_seasonal_figure(rows):
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row["scenario"], int(row["doy"]))].append(row)
    doys = sorted({int(row["doy"]) for row in rows})
    if not doys:
        raise ValueError("No observation dates found in model summary.")

    fig, (ax_rmse, ax_r2) = plt.subplots(2, 1, figsize=(11.4, 7.4), sharex=True)
    fig.patch.set_facecolor("#ffffff")
    for ax in (ax_rmse, ax_r2):
        ax.set_facecolor("#fbfcfd")
        ax.grid(axis="y", color="#dfe7ec", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#aab8c2")
        ax.spines["bottom"].set_color("#aab8c2")

    for scenario in SCENARIO_ORDER:
        available = []
        for doy in doys:
            group = by_group.get((scenario, doy), [])
            if group:
                available.append((
                    doy,
                    statistics.mean(float(row["rmse_mean"]) for row in group),
                    statistics.stdev(float(row["rmse_mean"]) for row in group) if len(group) > 1 else 0.0,
                    statistics.mean(float(row["anomaly_r2_mean"]) for row in group),
                    statistics.stdev(float(row["anomaly_r2_mean"]) for row in group) if len(group) > 1 else 0.0,
                ))
        if not available:
            continue
        x = [item[0] for item in available]
        rmse = [item[1] for item in available]
        rmse_sd = [item[2] for item in available]
        anomaly_r2 = [item[3] for item in available]
        anomaly_sd = [item[4] for item in available]
        color = COLORS.get(scenario, "#444444")
        linestyle = "--" if scenario == "Final CDL hindsight benchmark" else "-"
        label = SCENARIO_SHORT.get(scenario, scenario)
        ax_rmse.plot(x, rmse, color=color, linewidth=2.2, marker="o", markersize=5, linestyle=linestyle, label=label)
        ax_rmse.fill_between(x, np.array(rmse) - rmse_sd, np.array(rmse) + rmse_sd, color=color, alpha=0.11, linewidth=0)
        ax_r2.plot(x, anomaly_r2, color=color, linewidth=2.2, marker="o", markersize=5, linestyle=linestyle, label=label)
        ax_r2.fill_between(x, np.array(anomaly_r2) - anomaly_sd, np.array(anomaly_r2) + anomaly_sd, color=color, alpha=0.11, linewidth=0)

    ax_rmse.set_ylabel("RMSE (bu ac$^{-1}$)", fontsize=10, fontweight="bold")
    ax_r2.set_ylabel("Anomaly $R^2$ (unitless)", fontsize=10, fontweight="bold")
    ax_r2.set_xlabel("Observation date (day of year)", fontsize=10, fontweight="bold")
    ax_rmse.set_title("Seasonal accuracy across available year–date evaluations", loc="left", fontsize=13, fontweight="bold", pad=10)
    ax_rmse.legend(ncol=2, frameon=False, fontsize=8.2, loc="upper right")
    ax_r2.axhline(0, color="#778391", linewidth=0.9, linestyle=":")
    import datetime
    labels = [(datetime.date(2025, 1, 1) + datetime.timedelta(days=doy - 1)).strftime("%b %d") for doy in doys]
    ax_r2.set_xticks(doys, labels)
    ax_rmse.tick_params(axis="both", labelsize=8.5)
    ax_r2.tick_params(axis="both", labelsize=8.5)
    fig.text(
        0.01, 0.005,
        "Lines are means of the reported 50-seed scenario means across available years; shaded bands show ±1 SD across years, not a confidence interval. Test cohorts and year coverage vary by scenario.",
        fontsize=7.3, color="#536574",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=0.45)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_SEASONAL, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_heatmap(rows):
    years = sorted({int(row["year"]) for row in rows})
    scenarios = [scenario for scenario in SCENARIO_ORDER if any(row["scenario"] == scenario for row in rows)]
    last_doy = max(int(row["doy"]) for row in rows)
    values = np.full((len(scenarios), len(years)), np.nan)
    for row in rows:
        if int(row["doy"]) != last_doy:
            continue
        values[scenarios.index(row["scenario"]), years.index(int(row["year"]))] = float(row["rmse_mean"])
    if np.isnan(values).all():
        raise ValueError(f"No late-season (DOY {last_doy}) RMSE values available.")

    fig, ax = plt.subplots(figsize=(11.4, 4.05))
    fig.patch.set_facecolor("#ffffff")
    masked = np.ma.masked_invalid(values)
    image = ax.imshow(masked, aspect="auto", cmap="YlGnBu", interpolation="nearest")
    image.cmap.set_bad("#eef1f3")
    ax.set_xticks(np.arange(len(years)), [str(year) for year in years])
    ax.set_yticks(np.arange(len(scenarios)), [SCENARIO_SHORT.get(s, s) for s in scenarios])
    ax.tick_params(axis="both", length=0, labelsize=8.5)
    ax.set_title(f"Late-season RMSE by scenario and year (DOY {last_doy})", loc="left", fontsize=13, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(scenarios)):
        for j in range(len(years)):
            value = values[i, j]
            label = f"{value:.1f}" if np.isfinite(value) else "—"
            color = "white" if np.isfinite(value) and value > np.nanmedian(values) else "#233746"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, fontweight="bold", color=color)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.025)
    colorbar.set_label("RMSE (bu ac$^{-1}$)", fontsize=9, fontweight="bold")
    colorbar.ax.tick_params(labelsize=8)
    fig.text(
        0.01, 0.005,
        "Lower RMSE is better. Blank cells indicate no reported evaluation for that scenario/year. Test cohorts differ; treat this as a coverage-aware diagnostic, not a controlled ranking.",
        fontsize=7.3, color="#536574",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIGURE_HEATMAP, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def xml_run(text, bold=False, italic=False, size=19, color=None):
    properties = []
    if bold:
        properties.append("<w:b/>")
    if italic:
        properties.append("<w:i/>")
    if color:
        properties.append(f'<w:color w:val="{color}"/>')
    properties.append(f'<w:sz w:val="{size}"/>')
    properties.append(f'<w:szCs w:val="{size}"/>')
    safe = escape(str(text))
    return f'<w:r><w:rPr>{"".join(properties)}</w:rPr><w:t xml:space="preserve">{safe}</w:t></w:r>'


def xml_paragraph(text="", style=None, bold=False, italic=False, size=19, color=None, after=100, before=0, keep=False):
    ppr = [f'<w:spacing w:before="{before}" w:after="{after}" w:line="260" w:lineRule="auto"/>']
    if style:
        ppr.append(f'<w:pStyle w:val="{style}"/>')
    if keep:
        ppr.append("<w:keepNext/>")
    runs = xml_run(text, bold=bold, italic=italic, size=size, color=color) if text else ""
    return f'<w:p><w:pPr>{"".join(ppr)}</w:pPr>{runs}</w:p>'


def xml_cell(text, width, header=False):
    fill = '<w:shd w:fill="173B4A"/>' if header else ""
    color = "FFFFFF" if header else "243746"
    p = xml_paragraph(text, bold=header, size=16 if header else 15, color=color, after=30)
    return (
        f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{fill}'
        '<w:tcMar><w:top w:w="60" w:type="dxa"/><w:start w:w="75" w:type="dxa"/>'
        '<w:bottom w:w="60" w:type="dxa"/><w:end w:w="75" w:type="dxa"/></w:tcMar></w:tcPr>'
        f'{p}</w:tc>'
    )


def xml_table(summaries, delta_mean, delta_sd, n_paired):
    headers = [
        "Model / evaluation scenario",
        "Test cohort",
        "RMSE (bu ac\u207b\u00b9)",
        "MAE (bu ac\u207b\u00b9)",
        "R\u00b2 (unitless)",
        "Year \u00d7 DOY periods (n)",
    ]
    widths = [3000, 1700, 1500, 1500, 1250, 1202]
    borders = (
        '<w:tblBorders><w:top w:val="single" w:sz="8" w:color="173B4A"/>'
        '<w:left w:val="single" w:sz="4" w:color="B8C7D0"/>'
        '<w:bottom w:val="single" w:sz="8" w:color="173B4A"/>'
        '<w:right w:val="single" w:sz="4" w:color="B8C7D0"/>'
        '<w:insideH w:val="single" w:sz="4" w:color="D6E0E6"/>'
        '<w:insideV w:val="single" w:sz="4" w:color="D6E0E6"/></w:tblBorders>'
    )
    rows_xml = []
    head_cells = "".join(xml_cell(value, width, header=True) for value, width in zip(headers, widths))
    rows_xml.append(f'<w:tr><w:trPr><w:tblHeader/></w:trPr>{head_cells}</w:tr>')
    for index, item in enumerate(summaries):
        testing = TEST_COHORT_LABELS.get(item["testing"], item["testing"])
        scenario = SCENARIO_SHORT.get(item["scenario"], item["scenario"])
        metrics = item["metrics"]
        rmse = metrics["rmse_mean"]
        mae = metrics["mae_mean"]
        r2 = metrics["r2_mean"]
        cells = [
            scenario,
            testing,
            f'{rmse[0]:.2f} \u00b1 {rmse[1]:.2f}',
            f'{mae[0]:.2f} \u00b1 {mae[1]:.2f}',
            f'{r2[0]:.3f} \u00b1 {r2[1]:.3f}',
            f'{item["periods"]} ({item["years"]} y)',
        ]
        row_cells = "".join(xml_cell(value, width) for value, width in zip(cells, widths))
        rows_xml.append(f'<w:tr>{row_cells}</w:tr>')
    grid = "".join(f'<w:gridCol w:w="{width}"/>' for width in widths)
    table = (
        '<w:tbl><w:tblPr><w:tblW w:w="10152" w:type="dxa"/>'
        '<w:tblLayout w:type="fixed"/>' + borders + '</w:tblPr>'
        f'<w:tblGrid>{grid}</w:tblGrid>{"".join(rows_xml)}</w:tbl>'
    )
    note = (
        "Values are mean \u00b1 sample SD across available year \u00d7 DOY groups; each underlying group reports 50 seeds. "
        "The two previous-CDL test rows share 30 matched evaluation periods. Their matched RMSE difference "
        f"(same-year training minus previous-CDL training) is {delta_mean:+.2f} \u00b1 {delta_sd:.2f} bu ac\u207b\u00b9 "
        f"across {n_paired} paired groups. Other rows have different test cohorts and/or incomplete year coverage."
    )
    return table + xml_paragraph(note, italic=True, size=15, color="526575", after=130)


def image_paragraph(image_name, rel_id, number, width_inches=6.9, max_height=3.4):
    from PIL import Image

    image_path = FIGURE_DIR / image_name
    with Image.open(image_path) as image:
        width_px, height_px = image.size
    width_inches = min(width_inches, max_height * width_px / height_px)
    height_inches = width_inches * height_px / width_px
    cx = int(width_inches * 914400)
    cy = int(height_inches * 914400)
    return (
        f'<w:p><w:pPr><w:jc w:val="center"/><w:spacing w:before="20" w:after="20"/></w:pPr><w:r><w:drawing>'
        f'<wp:inline distT="0" distB="0" distL="0" distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
        f'<wp:docPr id="{number}" name="{escape(image_name)}"/>'
        '<wp:cNvGraphicFramePr><a:graphicFrameLocks noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        '<a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="Figure"/>'
        '<pic:cNvPicPr><a:picLocks noChangeAspect="1"/></pic:cNvPicPr></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rel_id}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
        '</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
    )


def make_docx(summaries, delta_mean, delta_sd, n_paired):
    paragraphs = [
        xml_paragraph("CORN YIELD MODELING", bold=True, size=16, color="176B87", after=50),
        xml_paragraph("Scenario comparison and seasonal accuracy", style="Title", size=38, color="173B4A", after=70, keep=True),
        xml_paragraph(
            "Technical appendix | 50-seed ensemble | County evaluations, 2021\u20132025 | Yield and error in bushels per acre",
            size=16, color="526575", after=180,
        ),
        xml_paragraph("Engineered summary", style="Heading1", size=24, color="173B4A", after=70, keep=True),
        xml_paragraph(
            "The table compares the available model scenarios using their reported 50-seed mean metrics. "
            "RMSE, MAE, and R\u00b2 are averaged over each scenario\u2019s available year-by-observation-date groups; "
            "the stated SD describes variation across those groups, not uncertainty in a single county prediction.",
            size=18, after=100,
        ),
        xml_paragraph(
            "Table 1. Summary of county-yield model performance by training/testing scenario. Errors are in bu ac\u207b\u00b9; R\u00b2 is dimensionless.",
            bold=True, size=16, color="173B4A", after=50, keep=True,
        ),
        xml_table(summaries, delta_mean, delta_sd, n_paired),
        xml_paragraph("Technical interpretation", style="Heading1", size=24, color="173B4A", before=120, after=70, keep=True),
        xml_paragraph(
            "Across the available scenarios, average RMSE declines from 18.52 bu ac\u207b\u00b9 at DOY 193 "
            "to 14.84 bu ac\u207b\u00b9 at DOY 273, while anomaly R\u00b2 generally rises as the season progresses. "
            "This pattern is consistent with later-season model inputs containing additional crop-development "
            "information: canopy and vegetation signals have had more time to reflect biomass accumulation and "
            "yield formation. The scores alone do not establish that mechanism causally, and the observed "
            "improvement should be interpreted as a seasonal association.",
            size=18, after=90,
        ),
        xml_paragraph(
            "The fairest operational comparison is between the two rows tested on the same previous-CDL cohort. "
            "Across 30 matched year-by-date groups, replacing previous-CDL training with same-year-CDL training "
            "increases mean RMSE by "
            f"{delta_mean:.2f} bu ac\u207b\u00b9 (paired-group SD {delta_sd:.2f}); however, the difference varies "
            "across evaluation periods, so the aggregate alone does not warrant a significance claim. The final-CDL "
            "benchmark has lower average error but uses hindsight inputs; it is not an operational forecast. "
            "The ICDL rows cover fewer years and different test cohorts, so their averages are descriptive rather "
            "than controlled head-to-head rankings.",
            size=18, after=90,
        ),
        xml_paragraph(
            "For deployment, retain the previous-CDL model as the reference candidate for like-for-like previous-CDL "
            "testing, and report the same-year and ICDL variants as separate sensitivity cases until their test "
            "cohorts are aligned. The seasonal curves can guide update timing: later-season estimates are more "
            "accurate on average in these outputs, while early-season estimates remain useful when lead time matters. "
            "Use the year-by-scenario heatmap to monitor whether that pattern is stable across years rather than "
            "selecting a model from the pooled mean alone.",
            size=18, after=80,
        ),
        '<w:p><w:r><w:br w:type="page"/></w:r></w:p>',
        xml_paragraph("Figures and decision diagnostics", style="Heading1", size=27, color="173B4A", after=90, keep=True),
        image_paragraph(FIGURE_SEASONAL.name, "rId2", 1, max_height=3.65),
        xml_paragraph(
            "Figure 1. Seasonal county-accuracy profile. Lines show means across available years; shaded bands show "
            "\u00b11 SD across years. Evaluation support varies by scenario, and the final-CDL line is a hindsight benchmark.",
            style="Caption", size=15, color="526575", after=100,
        ),
        image_paragraph(FIGURE_HEATMAP.name, "rId3", 2, max_height=2.75),
        xml_paragraph(
            "Figure 2. Late-season RMSE by scenario and year at the latest evaluated observation date (DOY 273). "
            "Blank cells indicate unavailable evaluations; differences in test cohorts limit direct ranking.",
            style="Caption", size=15, color="526575", after=100,
        ),
        xml_paragraph(
            "Source: v5_50_seed_all_in_one/summary/seed_summary_mean_sd.csv. The seed-level metrics are summarized "
            "by scenario, year, and DOY; reported seed counts are 50 per group.",
            italic=True, size=14, color="526575", after=0,
        ),
    ]
    document_body = "".join(paragraphs)
    document_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
 <w:body>{document_body}
  <w:sectPr><w:pgSz w:w="12240" w:h="15840"/><w:pgMar w:top="900" w:right="1020" w:bottom="900" w:left="1020" w:header="420" w:footer="420" w:gutter="0"/></w:sectPr>
 </w:body>
</w:document>'''
    styles_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
 <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:lang w:val="en-US"/></w:rPr></w:rPrDefault>
 <w:pPrDefault><w:pPr><w:spacing w:after="100" w:line="260" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults>
 <w:style w:type="paragraph" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:sz w:val="19"/></w:rPr></w:style>
 <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:after="100"/></w:pPr><w:rPr><w:rFonts w:ascii="Aptos Display" w:hAnsi="Aptos Display"/><w:b/><w:sz w:val="38"/><w:color w:val="173B4A"/></w:rPr></w:style>
 <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:qFormat/><w:pPr><w:keepNext/><w:spacing w:before="180" w:after="70"/></w:pPr><w:rPr><w:b/><w:sz w:val="24"/><w:color w:val="173B4A"/></w:rPr></w:style>
 <w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="caption"/><w:basedOn w:val="Normal"/><w:rPr><w:i/><w:sz w:val="15"/><w:color w:val="526575"/></w:rPr></w:style>
</w:styles>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
 <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
 <Default Extension="xml" ContentType="application/xml"/>
 <Default Extension="png" ContentType="image/png"/>
 <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
 <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
 <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>'''
    document_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
 <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/seasonal.png"/>
 <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/heatmap.png"/>
</Relationships>'''
    core_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
 <dc:title>County Corn Yield Scenario Comparison</dc:title>
 <dc:subject>Modeling track technical appendix</dc:subject>
 <dc:creator>Data Visualization Appendix</dc:creator>
</cp:coreProperties>'''
    OUTPUT_DOCX.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_DOCX, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("[Content_Types].xml", content_types)
        package.writestr("_rels/.rels", root_rels)
        package.writestr("docProps/core.xml", core_xml)
        package.writestr("word/document.xml", document_xml)
        package.writestr("word/styles.xml", styles_xml)
        package.writestr("word/_rels/document.xml.rels", document_rels)
        package.write(FIGURE_SEASONAL, "word/media/seasonal.png")
        package.write(FIGURE_HEATMAP, "word/media/heatmap.png")


def main():
    rows = read_metrics()
    summaries = summarize(rows)
    delta_mean, delta_sd, n_paired = write_summary_csv(summaries, rows)
    make_seasonal_figure(rows)
    make_heatmap(rows)
    make_docx(summaries, delta_mean, delta_sd, n_paired)
    print(f"Created {OUTPUT_DOCX}")
    print(f"Created {FIGURE_SEASONAL} and {FIGURE_HEATMAP}")
    print(f"Created {OUTPUT_CSV}")
    print(
        f"Matched previous-CDL test comparison: {n_paired} year-by-date groups; "
        f"mean RMSE change {delta_mean:+.2f} ± {delta_sd:.2f} bu/acre."
    )


if __name__ == "__main__":
    main()
