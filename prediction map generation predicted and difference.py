import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Load and prepare data
df = pd.read_csv('ensemble_county_predictions.csv')

# Format FIPS to 5-digit strings
df['FIPS'] = df['FIPS'].astype(str).str.zfill(5)

# Calculate difference: (Predicted - Actual)
df['yield_diff'] = df['predicted_yield'] - df['yield_bu_acre']

# Aggregate duplicate mask configurations per County/DOY/Year
df_agg = df.groupby(['year', 'doy', 'FIPS'], as_index=False).agg({
    'predicted_yield': 'mean',
    'yield_bu_acre': 'mean',
    'yield_diff': 'mean'
})

# Correct GeoJSON link for US Counties
geojson_url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

output_dir = "predicted_and_difference_interactive_maps_by_year"
os.makedirs(output_dir, exist_ok=True)

# 2. Build one interactive HTML file per year
for year in sorted(df_agg['year'].unique()):
    df_year = df_agg[df_agg['year'] == year]
    doys = sorted(df_year['doy'].unique())
    
    # Color scale limits
    pred_min, pred_max = df_year['predicted_yield'].min(), df_year['predicted_yield'].max()
    diff_limit = max(abs(df_year['yield_diff'].min()), abs(df_year['yield_diff'].max()))

    # Create 1x2 subplot layout
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Predicted Yield (bu/acre)", "Yield Difference (Predicted - Actual)"),
        specs=[[{"type": "choropleth"}, {"type": "choropleth"}]]
    )

    num_doys = len(doys)

    # Add choropleth traces for each DOY
    for i, doy in enumerate(doys):
        df_doy = df_year[df_year['doy'] == doy]
        
        # Left Map: Predicted Yield
        fig.add_trace(
            go.Choropleth(
                geojson=geojson_url,
                locations=df_doy['FIPS'],
                z=df_doy['predicted_yield'],
                colorscale='YlGn',
                zmin=pred_min,
                zmax=pred_max,
                colorbar=dict(title='bu/acre', x=0.45, len=0.8),
                name=f'Predicted (DOY {doy})',
                visible=(i == 0)  # Only show first DOY initially
            ),
            row=1, col=1
        )
        
        # Right Map: Yield Difference
        fig.add_trace(
            go.Choropleth(
                geojson=geojson_url,
                locations=df_doy['FIPS'],
                z=df_doy['yield_diff'],
                colorscale='RdBu_r',  # Red = Overpredicted, Blue = Underpredicted
                zmin=-diff_limit,
                zmax=diff_limit,
                colorbar=dict(title='Diff (bu/acre)', x=1.0, len=0.8),
                name=f'Diff (DOY {doy})',
                visible=(i == 0)  # Only show first DOY initially
            ),
            row=1, col=2
        )

    # 3. Create DOY Slider Control
    steps = []
    for i, doy in enumerate(doys):
        # Create visibility boolean vector (2 traces per DOY)
        visible_vec = [False] * (2 * num_doys)
        visible_vec[2 * i] = True
        visible_vec[2 * i + 1] = True
        
        step = dict(
            method="update",
            args=[{"visible": visible_vec},
                  {"title": f"Year {year} — Day of Year (DOY): {doy}"}],
            label=str(doy)
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Day of Year (DOY): "},
        pad={"t": 50},
        steps=steps
    )]

    # Configure USA Map Layout
    fig.update_layout(
        sliders=sliders,
        title=dict(text=f"Year {year} — Day of Year (DOY): {doys[0]}", x=0.5, font=dict(size=18)),
        geo=dict(scope="usa", projection_type="albers usa", showlakes=True, lakecolor="rgb(255, 255, 255)"),
        geo2=dict(scope="usa", projection_type="albers usa", showlakes=True, lakecolor="rgb(255, 255, 255)"),
        height=650,
        margin=dict(l=10, r=10, t=80, b=40)
    )

    # Save to a single HTML file per year
    output_path = os.path.join(output_dir, f"yield_map_{year}.html")
    fig.write_html(output_path)
    print(f"Saved: {output_path}")