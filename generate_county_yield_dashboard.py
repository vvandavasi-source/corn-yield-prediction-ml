"""Build the interactive dashboard from v5_out; run with python generate_county_yield_dashboard.py."""

import csv
import html
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "v5_out"
SEEDS_DIR = DATA_DIR / "seeds"
NASS_FILE = ROOT / "NASS_corn_2008_2026-02-05.csv"
OUTPUT_FILE = DATA_DIR / "summary" / "county_yield_dashboard.html"
COUNTY_GEOJSON_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

SCENARIOS = {
    ("SAME_YEAR_CDL", "PREVIOUS_CDL"): "Same-year CDL training / previous CDL testing",
    ("SAME_YEAR_CDL", "FINAL_CDL"): "Same-year CDL training / final CDL testing (hindsight)",
    ("SAME_YEAR_CDL", "STUDY"): "Same-year CDL training / study testing",
    ("SAME_YEAR_CDL", "CLEAN_REBUILT"): "Same-year CDL training / clean rebuilt testing",
}

STATE_NAMES = {
    "01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas",
    "06": "California", "08": "Colorado", "09": "Connecticut", "10": "Delaware",
    "11": "District of Columbia", "12": "Florida", "13": "Georgia", "15": "Hawaii",
    "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa",
    "20": "Kansas", "21": "Kentucky", "22": "Louisiana", "23": "Maine",
    "24": "Maryland", "25": "Massachusetts", "26": "Michigan", "27": "Minnesota",
    "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska",
    "32": "Nevada", "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico",
    "36": "New York", "37": "North Carolina", "38": "North Dakota", "39": "Ohio",
    "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
    "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas",
    "49": "Utah", "50": "Vermont", "51": "Virginia", "53": "Washington",
    "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming",
}


def load_county_names():
    names = {}
    if not NASS_FILE.is_file():
        return names
    with NASS_FILE.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if row.get("Crop type", "").strip().lower() != "corn":
                continue
            code = row.get("FIPS", "").strip().zfill(5)
            location = row.get("Location", "").strip()
            if len(code) == 5 and code.isdigit() and location:
                names.setdefault(code, location.split(",")[-1].strip().title())
    return names


def read_seed_predictions():
    seed_files = sorted(SEEDS_DIR.glob("seed_*/county_predictions.csv"))
    if not seed_files:
        raise FileNotFoundError(f"No seed prediction CSV files found in {SEEDS_DIR}")

    aggregates = {}
    for seed_file in seed_files:
        with seed_file.open(newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                scenario_key = (row["training_mask"], row["testing_mask"])
                scenario = SCENARIOS.get(scenario_key)
                if scenario is None:
                    continue

                fips = row["FIPS"].strip().zfill(5)
                year = int(row["year"])
                doy = int(row["doy"])
                actual = float(row["yield_bu_acre"])
                trend = float(row["hist_yield_trend"])
                prediction = float(row["predicted_yield"])
                key = (scenario, year, doy, fips)

                item = aggregates.get(key)
                if item is None:
                    aggregates[key] = [1, prediction, 0.0, actual, trend]
                    continue

                count, mean, m2, first_actual, first_trend = item
                if not math.isclose(actual, first_actual, rel_tol=0, abs_tol=0.001):
                    raise ValueError(f"Inconsistent observed yield for {key} in {seed_file}")
                if not math.isclose(trend, first_trend, rel_tol=0, abs_tol=0.001):
                    raise ValueError(f"Inconsistent historical trend for {key} in {seed_file}")
                count += 1
                delta = prediction - mean
                mean += delta / count
                m2 += delta * (prediction - mean)
                aggregates[key] = [count, mean, m2, first_actual, first_trend]

    county_names = load_county_names()
    records = []
    for (scenario, year, doy, fips), (count, mean, m2, actual, trend) in sorted(aggregates.items()):
        seed_sd = math.sqrt(m2 / (count - 1)) if count > 1 else 0.0
        state_code = fips[:2]
        county = county_names.get(fips, f"County {fips}")
        records.append({
            "s": scenario,
            "st": STATE_NAMES.get(state_code, f"State {state_code}"),
            "sc": state_code,
            "f": fips,
            "c": county,
            "y": year,
            "d": doy,
            "a": round(actual, 3),
            "t": round(trend, 3),
            "p": round(mean, 3),
            "sd": round(seed_sd, 3),
            "n": count,
        })
    return records


def write_dashboard(records):
    payload = json.dumps(records, separators=(",", ":"), ensure_ascii=False).replace("<", "\\u003c")
    scenarios = sorted(
        {row["s"] for row in records},
        key=lambda scenario: (0 if "study testing" in scenario.lower() else 1, scenario),
    )
    years = sorted({row["y"] for row in records})
    states = sorted({(row["sc"], row["st"]) for row in records}, key=lambda item: item[1])
    if not records:
        raise ValueError("No county predictions were available for the configured scenarios.")

    scenario_options = "\n".join(
        f'<option value="{html.escape(scenario, quote=True)}">{html.escape(scenario)}</option>'
        for scenario in scenarios
    )
    year_options = "\n".join(
        f'<option value="{year}"{" selected" if year == max(years) else ""}>{year}</option>'
        for year in years
    )
    map_day_options = "\n".join(
        f'<option value="{doy}">{doy}</option>'
        for doy in sorted({row["d"] for row in records})
    )
    state_options = "\n".join(
        f'<option value="{code}">{html.escape(name)}</option>' for code, name in states
    )

    page = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>County Corn Yield Explorer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root { color-scheme: light; --ink:#172b3a; --muted:#617486; --line:#dbe4eb; --blue:#176b87; --gold:#d28a28; --green:#39826c; }
    * { box-sizing:border-box; }
    body { margin:0; background:#f3f6f8; color:var(--ink); font:15px/1.45 "Segoe UI",Arial,sans-serif; }
    header { padding:28px max(24px,calc((100vw - 1440px)/2)); background:#102d3a; color:white; }
    header h1 { margin:0 0 6px; font-size:27px; letter-spacing:.2px; }
    header p { margin:0; max-width:1000px; color:#c9d9df; }
    main { max-width:1440px; margin:22px auto; padding:0 20px 36px; }
    .controls { display:grid; grid-template-columns:2fr 1fr 1.4fr .8fr; gap:12px; padding:16px; background:white; border:1px solid var(--line); border-radius:10px; box-shadow:0 2px 8px #18394b0b; }
    label { display:block; color:var(--muted); font-size:12px; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }
    select { width:100%; margin-top:6px; padding:10px 12px; border:1px solid #c8d4dc; border-radius:6px; color:var(--ink); background:white; font:inherit; }
    .kpis { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin:14px 0; }
    .kpi { background:white; padding:14px 16px; border:1px solid var(--line); border-radius:9px; }
    .kpi span { display:block; color:var(--muted); font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
    .kpi strong { display:block; margin-top:5px; font-size:22px; font-weight:650; }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
    .panel { min-width:0; background:white; border:1px solid var(--line); border-radius:10px; padding:12px 12px 4px; box-shadow:0 2px 8px #18394b0b; }
    .panel h2 { margin:4px 10px 0; font-size:16px; }
    .panel p { margin:2px 10px 0; color:var(--muted); font-size:12px; }
    .chart { height:390px; }
    .wide { grid-column:1 / -1; }
    .map-controls { display:flex; align-items:center; gap:10px; margin:8px 10px 0; }
    .map-controls label { min-width:150px; }
    .map-controls select { max-width:180px; }
    .map-summary { margin:8px 10px 0; color:var(--ink); font-size:13px; font-weight:600; }
    .map-status { min-height:18px; margin:2px 10px 0; color:#8a3d32; font-size:12px; }
    #countyMap { height:610px; }
    .note { margin:14px 2px 0; color:var(--muted); font-size:12px; }
    .warning { display:none; margin:12px 0; padding:12px; border:1px solid #f1c27d; border-radius:6px; background:#fff8e9; color:#664b1c; }
    @media (max-width:850px) { .controls { grid-template-columns:1fr 1fr; } .grid { grid-template-columns:1fr; } .wide { grid-column:auto; } .kpis { grid-template-columns:1fr 1fr; } }
    @media (max-width:520px) { .controls { grid-template-columns:1fr; } header { padding:22px 20px; } .chart { height:340px; } }
  </style>
</head>
<body>
  <header>
    <h1>County Corn Yield Explorer</h1>
    <p>Explore county predictions from the same-year CDL training protocol across the available states, growing-season observation dates, years, and test-mask scenarios. Yield and error are in bushels per acre.</p>
  </header>
  <main>
    <section class="controls" aria-label="Dashboard filters">
      <label>Training / testing scenario<select id="scenario">__SCENARIOS__</select></label>
      <label>State<select id="state">__STATES__</select></label>
      <label>County<select id="county"></select></label>
      <label>Year<select id="year">__YEARS__</select></label>
    </section>
    <div id="warning" class="warning"></div>
    <section class="kpis" aria-label="Selected county summary">
      <div class="kpi"><span>Selected county</span><strong id="countyKpi">—</strong></div>
      <div class="kpi"><span>Observed yield average</span><strong id="yieldKpi">—</strong></div>
      <div class="kpi"><span>County MAE · selected year</span><strong id="maeKpi">—</strong></div>
      <div class="kpi"><span>Counties in state · selected year</span><strong id="countKpi">—</strong></div>
    </section>
    <section class="grid">
      <article class="panel wide">
        <h2>Predicted yield by county</h2>
        <p>National overview for the selected training/testing scenario and year. Hover a county for predicted and observed yield, error, historical trend, and seed spread. The map is independent of the state filter used by the charts below.</p>
        <div class="map-controls">
          <label>Map observation date<select id="mapDoy">__MAP_DAYS__</select></label>
        </div>
        <p id="mapSummary" class="map-summary"></p>
        <div id="mapStatus" class="map-status" role="status" aria-live="polite"></div>
        <div id="countyMap" class="chart"></div>
      </article>
      <article class="panel">
        <h2>County yield through the season</h2>
        <p>Observed yield, 50-seed mean prediction, historical trend, and ±1 seed standard deviation.</p>
        <div id="yieldChart" class="chart"></div>
      </article>
      <article class="panel">
        <h2>County prediction error by observation date</h2>
        <p>Signed error is prediction − observed yield; absolute error shows error magnitude.</p>
        <div id="countyErrorChart" class="chart"></div>
      </article>
      <article class="panel">
        <h2>County error distribution by observation date</h2>
        <p>Each box summarizes prediction − observed yield across counties in the selected state.</p>
        <div id="errorBoxChart" class="chart"></div>
      </article>
      <article class="panel">
        <h2>State accuracy by observation date</h2>
        <p>County-averaged MAE and RMSE; county count is shown in hover details.</p>
        <div id="accuracyChart" class="chart"></div>
      </article>
    </section>
    <p class="note">The seed spread is the sample standard deviation of predictions across available model seeds, not a confidence interval. State metrics give each available county equal weight. Missing counties/dates are not imputed. The final-CDL scenario uses hindsight inputs and is a benchmark, not a like-for-like operational forecast.</p>
  </main>
  <script>
    const rows = __DATA__;
    const scenarioSelect = document.getElementById("scenario");
    const stateSelect = document.getElementById("state");
    const countySelect = document.getElementById("county");
    const yearSelect = document.getElementById("year");
    const mapDoySelect = document.getElementById("mapDoy");
    const stateNames = Object.fromEntries([...document.querySelectorAll("#state option")].map(o => [o.value, o.textContent]));
    const palette = { observed:"#176b87", predicted:"#39826c", trend:"#778391", error:"#b9503d", abs:"#d28a28", mae:"#176b87", rmse:"#39826c" };
    let renderedMapKey = "";
    const commonLayout = {
      autosize:true, paper_bgcolor:"#fff", plot_bgcolor:"#fff", font:{family:"Segoe UI, Arial, sans-serif",color:"#304657",size:12},
      margin:{l:58,r:22,t:20,b:56}, hovermode:"x unified",
      xaxis:{title:"Day of year (DOY)",gridcolor:"#edf1f4",zerolinecolor:"#dbe4eb"},
      yaxis:{gridcolor:"#edf1f4",zerolinecolor:"#dbe4eb"},
      legend:{orientation:"h",y:1.12,x:0}
    };
    const plotConfig = {responsive:true,displaylogo:false};
    const round = value => Number.isFinite(value) ? value.toFixed(1) : "—";
    const byDoy = (a,b) => a.d-b.d;
    function filteredRows() {
      return rows.filter(r => r.s === scenarioSelect.value && r.sc === stateSelect.value && r.y === Number(yearSelect.value));
    }
    function updateYears() {
      const available = [...new Set(rows.filter(r => r.s === scenarioSelect.value).map(r => r.y))].sort((a,b) => a-b);
      const previous = Number(yearSelect.value);
      yearSelect.replaceChildren(...available.map(year => {
        const option = document.createElement("option");
        option.value = year;
        option.textContent = year;
        return option;
      }));
      yearSelect.value = available.includes(previous) ? String(previous) : String(available[available.length - 1] || "");
    }
    function updateStates() {
      const available = new Map(
        rows.filter(r => r.s === scenarioSelect.value && r.y === Number(yearSelect.value))
          .map(r => [r.sc, r.st])
      );
      const previous = stateSelect.value;
      stateSelect.replaceChildren(...[...available].sort((a,b) => a[1].localeCompare(b[1])).map(([code,name]) => {
        const option = document.createElement("option");
        option.value = code;
        option.textContent = name;
        return option;
      }));
      stateSelect.value = available.has(previous) ? previous : stateSelect.options[0]?.value || "";
    }
    function updateCounties() {
      const stateRows = rows.filter(r => r.s === scenarioSelect.value && r.sc === stateSelect.value && r.y === Number(yearSelect.value));
      const counties = new Map();
      for (const r of stateRows) counties.set(r.f, r.c);
      const previous = countySelect.value;
      countySelect.replaceChildren();
      [...counties].sort((a,b) => a[1].localeCompare(b[1]) || a[0].localeCompare(b[0])).forEach(([fips,name]) => {
        const option = document.createElement("option");
        option.value = fips;
        option.textContent = `${name} (${fips})`;
        countySelect.append(option);
      });
      if (counties.has(previous)) countySelect.value = previous;
    }
    function updateMapDates() {
      const available = [...new Set(
        rows.filter(r => r.s === scenarioSelect.value && r.y === Number(yearSelect.value)).map(r => r.d)
      )].sort((a,b) => a-b);
      const previous = Number(mapDoySelect.value);
      mapDoySelect.replaceChildren(...available.map(doy => {
        const option = document.createElement("option");
        option.value = doy;
        option.textContent = doy;
        return option;
      }));
      mapDoySelect.value = available.includes(previous) ? String(previous) : String(available[available.length - 1] || "");
      return available;
    }
    function renderCountyMap() {
      const year = Number(yearSelect.value);
      const doy = Number(mapDoySelect.value);
      const mapRows = rows.filter(r =>
        r.s === scenarioSelect.value && r.y === year && r.d === doy
      );
      const status = document.getElementById("mapStatus");
      document.getElementById("mapSummary").textContent =
        `${scenarioSelect.value} · ${year} · DOY ${doy} · ${mapRows.length.toLocaleString()} counties across ${new Set(mapRows.map(r => r.sc)).size} states. Color scale is fixed across dates for this scenario and year.`;
      if (!mapRows.length) {
        renderedMapKey = "";
        status.textContent = "No county predictions are available for this map selection.";
        Plotly.purge("countyMap");
        return;
      }
      const mapKey = `${scenarioSelect.value}|${year}|${doy}`;
      if (mapKey === renderedMapKey) return;
      status.textContent = "";
      const rangeRows = rows.filter(r => r.s === scenarioSelect.value && r.y === year);
      const values = rangeRows.map(r => r.p);
      const customdata = mapRows.map(r => [r.c, r.st, r.f, r.a, r.p-r.a, Math.abs(r.p-r.a), r.t, r.sd, r.n]);
      Plotly.react("countyMap", [{
        type:"choropleth",
        geojson:__COUNTY_GEOJSON_URL__,
        featureidkey:"id",
        locations:mapRows.map(r => r.f),
        z:mapRows.map(r => r.p),
        zmin:Math.min(...values),
        zmax:Math.max(...values),
        colorscale:[[0,"#FFFFFF"],[0.2,"#E9F3EF"],[0.5,"#AFCFC0"],[0.8,"#67A287"],[1,"#28634E"]],
        marker:{line:{color:"#ffffff",width:0.25}},
        colorbar:{title:{text:"Yield<br>(bu/ac)"},thickness:18},
        customdata,
        hovertemplate:
          "<b>%{customdata[0]}, %{customdata[1]}</b><br>" +
          "FIPS: %{customdata[2]}<br>" +
          `Year ${year} · DOY ${doy}<br>` +
          "Predicted (50-seed mean): %{z:.1f} bu/ac<br>" +
          "Observed yield: %{customdata[3]:.1f} bu/ac<br>" +
          "Signed error: %{customdata[4]:.1f} bu/ac<br>" +
          "Absolute error: %{customdata[5]:.1f} bu/ac<br>" +
          "Historical trend: %{customdata[6]:.1f} bu/ac<br>" +
          "Seed SD (not a confidence interval): %{customdata[7]:.2f} bu/ac<br>" +
          "Seeds: %{customdata[8]}<extra></extra>"
      }],{
        autosize:true,paper_bgcolor:"#fff",plot_bgcolor:"#fff",
        font:{family:"Segoe UI, Arial, sans-serif",color:"#304657",size:12},
        margin:{l:0,r:0,t:8,b:0},
        geo:{
          scope:"usa",projection:{type:"albers usa"},fitbounds:"locations",
          showland:true,landcolor:"#f5f7f8",showlakes:true,lakecolor:"#fff",
          showcountries:false,showsubunits:true,subunitcolor:"#b9c7ce"
        }
      },plotConfig).then(() => {
        renderedMapKey = mapKey;
      }).catch(error => {
        renderedMapKey = "";
        console.error("County yield map failed to render.",error);
        status.textContent = "Map could not be loaded. Check your internet connection and reload; county boundary data is loaded from Plotly's public GeoJSON.";
      });
    }
    function dateLabels(doys) {
      const year = Number(yearSelect.value);
      return doys.map(doy => {
        const date = new Date(year, 0, doy);
        return `${date.toLocaleString("en-US",{month:"short",day:"numeric"})} · ${doy}`;
      });
    }
    function setWarning(message) {
      const warning = document.getElementById("warning");
      warning.textContent = message || "";
      warning.style.display = message ? "block" : "none";
    }
    function render() {
      updateYears();
      updateStates();
      updateCounties();
      updateMapDates();
      renderCountyMap();
      const selected = filteredRows();
      const countyRows = selected.filter(r => r.f === countySelect.value).sort(byDoy);
      const dates = [...new Set(selected.map(r => r.d))].sort((a,b) => a-b);
      const labels = dateLabels(dates);
      const stateName = stateNames[stateSelect.value] || `State ${stateSelect.value}`;
      const countyName = countyRows[0] ? countyRows[0].c : countySelect.options[0]?.textContent || "No county data";
      setWarning(selected.length ? "" : "No prediction records are available for these filters. Try a different scenario, state, or year.");

      const meanActual = countyRows.length ? countyRows.reduce((sum,r) => sum+r.a,0)/countyRows.length : NaN;
      const countyMae = countyRows.length ? countyRows.reduce((sum,r) => sum+Math.abs(r.p-r.a),0)/countyRows.length : NaN;
      document.getElementById("countyKpi").textContent = countyRows.length ? `${countyName} (${countySelect.value})` : "No data";
      document.getElementById("yieldKpi").textContent = Number.isFinite(meanActual) ? `${round(meanActual)} bu/ac` : "—";
      document.getElementById("maeKpi").textContent = Number.isFinite(countyMae) ? `${round(countyMae)} bu/ac` : "—";
      document.getElementById("countKpi").textContent = new Set(selected.map(r => r.f)).size.toLocaleString();

      const lower = countyRows.map(r => r.p-r.sd);
      const upper = countyRows.map(r => r.p+r.sd);
      const yieldTraces = [
        {x:countyRows.map(r=>r.d),y:upper,mode:"lines",line:{width:0},showlegend:false,hoverinfo:"skip"},
        {x:countyRows.map(r=>r.d),y:lower,mode:"lines",line:{width:0},fill:"tonexty",fillcolor:"rgba(57,130,108,.17)",name:"±1 seed SD",hoverinfo:"skip"},
        {x:countyRows.map(r=>r.d),y:countyRows.map(r=>r.t),mode:"lines+markers",name:"Historical trend",line:{color:palette.trend,dash:"dot"},marker:{size:6}},
        {x:countyRows.map(r=>r.d),y:countyRows.map(r=>r.p),mode:"lines+markers",name:"50-seed mean prediction",line:{color:palette.predicted,width:2.5},marker:{size:7},customdata:countyRows.map(r=>[r.sd,r.n]),hovertemplate:"DOY %{x}<br>Predicted %{y:.1f} bu/ac<br>Seed SD %{customdata[0]:.2f}<br>Seeds n=%{customdata[1]}<extra></extra>"},
        {x:countyRows.map(r=>r.d),y:countyRows.map(r=>r.a),mode:"lines+markers",name:"Observed yield",line:{color:palette.observed,width:2.5},marker:{size:7}}
      ];
      Plotly.react("yieldChart",yieldTraces,{
        ...commonLayout,title:{text:`${countyName} County, ${stateName} · ${yearSelect.value}`,font:{size:14}},
        xaxis:{...commonLayout.xaxis,tickmode:"array",tickvals:dates,ticktext:labels},
        yaxis:{...commonLayout.yaxis,title:"Yield (bu/acre)"}
      },plotConfig);

      const errors = countyRows.map(r=>r.p-r.a);
      Plotly.react("countyErrorChart",[
        {x:countyRows.map(r=>r.d),y:errors,type:"bar",name:"Signed error",marker:{color:palette.error},customdata:countyRows.map(r=>r.a),hovertemplate:"DOY %{x}<br>Signed error %{y:.1f} bu/ac<br>Observed %{customdata:.1f} bu/ac<extra></extra>"},
        {x:countyRows.map(r=>r.d),y:errors.map(Math.abs),mode:"lines+markers",name:"Absolute error",line:{color:palette.abs,width:2.5},marker:{size:8}}
      ],{
        ...commonLayout,barmode:"overlay",xaxis:{...commonLayout.xaxis,tickmode:"array",tickvals:dates,ticktext:labels},
        yaxis:{...commonLayout.yaxis,title:"Error (bu/acre)",zeroline:true,zerolinecolor:"#738391",zerolinewidth:1}
      },plotConfig);

      const boxTraces = dates.map((d,index) => {
        const dayRows = selected.filter(r=>r.d===d);
        return {y:dayRows.map(r=>r.p-r.a),name:labels[index],type:"box",boxpoints:false,marker:{color:palette.observed},line:{color:palette.observed},hovertemplate:"%{y:.1f} bu/ac<extra></extra>"};
      });
      Plotly.react("errorBoxChart",boxTraces,{
        ...commonLayout,showlegend:false,hovermode:"closest",
        xaxis:{...commonLayout.xaxis,title:"Observation date",tickangle:-18},
        yaxis:{...commonLayout.yaxis,title:"Prediction − observed (bu/acre)",zeroline:true,zerolinecolor:"#738391",zerolinewidth:1}
      },plotConfig);

      const stats = dates.map(d => {
        const dayRows = selected.filter(r=>r.d===d);
        const errorsForDay = dayRows.map(r=>r.p-r.a);
        const mae = errorsForDay.length ? errorsForDay.reduce((sum,e)=>sum+Math.abs(e),0)/errorsForDay.length : NaN;
        const rmse = errorsForDay.length ? Math.sqrt(errorsForDay.reduce((sum,e)=>sum+e*e,0)/errorsForDay.length) : NaN;
        return {d,mae,rmse,count:dayRows.length};
      });
      Plotly.react("accuracyChart",[
        {x:stats.map((_,i)=>labels[i]),y:stats.map(s=>s.mae),mode:"lines+markers",name:"MAE",line:{color:palette.mae,width:2.5},marker:{size:8},customdata:stats.map(s=>[s.count,s.rmse]),hovertemplate:"%{x}<br>MAE %{y:.1f} bu/ac<br>RMSE %{customdata[1]:.1f} bu/ac<br>Counties n=%{customdata[0]}<extra></extra>"},
        {x:stats.map((_,i)=>labels[i]),y:stats.map(s=>s.rmse),mode:"lines+markers",name:"RMSE",line:{color:palette.rmse,width:2.5},marker:{size:8},customdata:stats.map(s=>[s.count,s.mae]),hovertemplate:"%{x}<br>RMSE %{y:.1f} bu/ac<br>MAE %{customdata[1]:.1f} bu/ac<br>Counties n=%{customdata[0]}<extra></extra>"}
      ],{
        ...commonLayout,hovermode:"x unified",xaxis:{...commonLayout.xaxis,title:"Observation date",tickangle:-18},
        yaxis:{...commonLayout.yaxis,title:"Error (bu/acre)",rangemode:"tozero"}
      },plotConfig);
    }
    for (const control of [scenarioSelect,stateSelect,countySelect,yearSelect,mapDoySelect]) control.addEventListener("change",render);
    render();
  </script>
</body>
</html>
"""
    page = page.replace("__SCENARIOS__", scenario_options)
    page = page.replace("__STATES__", state_options)
    page = page.replace("__YEARS__", year_options)
    page = page.replace("__MAP_DAYS__", map_day_options)
    page = page.replace("__COUNTY_GEOJSON_URL__", json.dumps(COUNTY_GEOJSON_URL))
    page = page.replace("__DATA__", payload)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(page, encoding="utf-8")


def main():
    records = read_seed_predictions()
    write_dashboard(records)
    state_count = len({record["sc"] for record in records})
    county_count = len({record["f"] for record in records})
    print(f"Wrote {OUTPUT_FILE}")
    print(f"Embedded {len(records):,} county-date-scenario records for {county_count:,} counties across {state_count} states.")


if __name__ == "__main__":
    main()
