
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle
import requests
import rasterio
from scipy import ndimage
from pyproj import Transformer

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Carbon Offset Engine",
    page_icon="🌱",
    layout="wide"
)

# ── Load data and models ──────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("data/official_processed.csv")

@st.cache_resource
def load_models():
    with open("models/habitat_recovery_model.pkl", "rb") as f:
        ml_model = pickle.load(f)
    with open("models/model_metrics.pkl", "rb") as f:
        model_metrics = pickle.load(f)
    return ml_model, model_metrics

official     = load_data()
ml_model, model_metrics = load_models()

# ── Transformers ──────────────────────────────────────────────
transformer_to_latlon = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
transformer_to_osgb   = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)

# ── LCM configuration ─────────────────────────────────────────
LCM_FILES = {
    1990: "land_1990/data/LCM.tif",
    2000: "land_2000/data/LCM2000.tif",
    2007: "land_2007/data/LCM2007.tif",
    2015: "land_2015/data/LCM2015.tif",
    2019: "land_2019/data/LCM.tif",
    2023: "land_2023/data/LCM.tif",
}

FRESHWATER_CLASSES = {
    1990: [14], 2000: [131], 2007: [14],
    2015: [14], 2019: [14],  2023: [14],
}

SALTWATER_CLASSES = {
    1990: [13], 2000: [151, 161], 2007: [13],
    2015: [13], 2019: [13],       2023: [13],
}

INTACT_CLASSES_BY_YEAR = {
    1990: [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    2000: [11, 21, 41, 42, 43, 51, 52, 61, 71, 81, 91, 101, 121, 131, 151, 161, 181, 191, 221],
    2007: [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    2015: [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    2019: [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    2023: [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
}

ECOSYSTEM_VALUE_PER_HA_YR = 7500

HABITAT_QUALITY_FACTOR = {
    "Solar":   0.3,
    "Wind":    0.7,
    "Hydro":   0.5,
    "Nuclear": 0.2,
    "Biomass": 0.1,
}

# ── Helper functions ──────────────────────────────────────────
def get_lcm_for_year(closure_year):
    available = sorted(LCM_FILES.keys())
    return LCM_FILES[min(available, key=lambda x: abs(x - closure_year))]

def get_lcm_year(closure_year):
    available = sorted(LCM_FILES.keys())
    return min(available, key=lambda x: abs(x - closure_year))

def distance_to_intact_habitat_lcm(lat, lon, lcm_file, lcm_year, search_radius_pixels=400):
    if pd.isna(lat) or pd.isna(lon):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    intact_classes   = INTACT_CLASSES_BY_YEAR.get(lcm_year, INTACT_CLASSES_BY_YEAR[2023])
    freshwater_class = FRESHWATER_CLASSES.get(lcm_year, [14])
    saltwater_class  = SALTWATER_CLASSES.get(lcm_year, [13])
    try:
        easting, northing = transformer_to_osgb.transform(lon, lat)
        with rasterio.open(lcm_file) as src:
            row, col  = src.index(easting, northing)
            row_min   = max(0, row - search_radius_pixels)
            row_max   = min(src.height, row + search_radius_pixels)
            col_min   = max(0, col - search_radius_pixels)
            col_max   = min(src.width, col + search_radius_pixels)
            window    = rasterio.windows.Window(col_min, row_min, col_max - col_min, row_max - row_min)
            data      = src.read(1, window=window)
            plant_row = row - row_min
            plant_col = col - col_min
            intact_mask  = np.isin(data, intact_classes)
            intact_rows, intact_cols = np.where(intact_mask)
            dist_any = round(np.sqrt((intact_rows - plant_row)**2 + (intact_cols - plant_col)**2).min() * 25 / 1000, 3) if len(intact_rows) > 0 else np.nan
            water_mask = np.isin(data, freshwater_class)
            water_rows, water_cols = np.where(water_mask)
            dist_water = round(np.sqrt((water_rows - plant_row)**2 + (water_cols - plant_col)**2).min() * 25 / 1000, 3) if len(water_rows) > 0 else np.nan
            salt_mask  = np.isin(data, saltwater_class)
            salt_rows, salt_cols = np.where(salt_mask)
            dist_salt  = round(np.sqrt((salt_rows - plant_row)**2 + (salt_cols - plant_col)**2).min() * 25 / 1000, 3) if len(salt_rows) > 0 else np.nan
            labelled, num_patches = ndimage.label(intact_mask)
            patch_sizes = ndimage.sum(intact_mask, labelled, range(1, num_patches + 1))
            def dist_to_min_patch(min_pixels):
                mask2 = np.zeros_like(intact_mask, dtype=bool)
                for pid, size in enumerate(patch_sizes, 1):
                    if size >= min_pixels:
                        mask2[labelled == pid] = True
                rows2, cols2 = np.where(mask2)
                return round(np.sqrt((rows2 - plant_row)**2 + (cols2 - plant_col)**2).min() * 25 / 1000, 3) if len(rows2) > 0 else np.nan
            return dist_any, dist_to_min_patch(16), dist_to_min_patch(80), dist_water, dist_salt
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan, np.nan

def get_annual_rainfall(lat, lon, closure_year):
    if pd.isna(lat) or pd.isna(lon):
        return 800.0
    try:
        response = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":   lat,
                "longitude":  lon,
                "start_date": f"{max(1940, closure_year - 10)}-01-01",
                "end_date":   f"{closure_year}-12-31",
                "daily":      "precipitation_sum",
                "timezone":   "Europe/London",
            }
        )
        data = response.json()
        if "daily" not in data:
            return 800.0
        daily = [p if p is not None else 0 for p in data["daily"]["precipitation_sum"]]
        return round(sum(daily) / (len(daily) / 365.25), 1)
    except:
        return 800.0

def get_unique_species(lat, lon, year_from, year_to, radius_deg=0.045):
    try:
        response = requests.get(
            "https://api.gbif.org/v1/occurrence/search",
            params={
                "decimalLatitude":    f"{lat-radius_deg},{lat+radius_deg}",
                "decimalLongitude":   f"{lon-radius_deg},{lon+radius_deg}",
                "year":               f"{year_from},{year_to}",
                "country":            "GB",
                "hasCoordinate":      "true",
                "hasGeospatialIssue": "false",
                "limit":              0,
                "facet":              "speciesKey",
                "facetLimit":         100000,
            }
        )
        data   = response.json()
        facets = data.get("facets", [])
        return len(facets[0].get("counts", [])) if facets else 0
    except:
        return None

def predict_species_recovery(plant_row, opening_year, closure_year):
    easting  = plant_row["X-Coordinate"]
    northing = plant_row["Y-Coordinate"]
    if pd.isna(easting) or pd.isna(northing):
        return None
    lon, lat            = transformer_to_latlon.transform(easting, northing)
    years_since_closure = 2026 - closure_year
    years_operational   = closure_year - opening_year
    site_ha             = plant_row["site_ha"] if pd.notna(plant_row["site_ha"]) else 20
    rainfall            = get_annual_rainfall(lat, lon, closure_year)
    lcm_file            = get_lcm_for_year(closure_year)
    lcm_year            = get_lcm_year(closure_year)
    dist_any, dist_1ha, dist_5ha, dist_water, dist_salt = distance_to_intact_habitat_lcm(lat, lon, lcm_file, lcm_year)
    species_before      = get_unique_species(lat, lon, closure_year - 3, closure_year)

    MIN_YEARS_SINCE_CLOSURE = 3
    if years_since_closure < MIN_YEARS_SINCE_CLOSURE:
        return {
            "species_5yr":         species_before or 150,
            "species_10yr":        species_before or 150,
            "species_before":      species_before or 150,
            "lat": lat, "lon": lon,
            "rainfall":            rainfall,
            "dist_1ha":            dist_1ha,
            "years_operational":   years_operational,
            "prediction_reliable": False,
        }

    input_data = pd.DataFrame([{
        "Years Operational":           years_operational,
        "Site Area (ha)":              site_ha,
        "annual_rainfall_mm":          rainfall,
        "species_before":              species_before if species_before else 150,
        "dist_any_intact_closure_km":  dist_any  if not np.isnan(dist_any)  else 2.0,
        "dist_1ha_patch_closure_km":   dist_1ha  if not np.isnan(dist_1ha)  else 2.0,
        "dist_5ha_patch_closure_km":   dist_5ha  if not np.isnan(dist_5ha)  else 4.0,
        "lat": lat, "lon": lon,
        "years_since_closure":         years_since_closure,
    }])
    prediction = ml_model.predict(input_data)[0]
    return {
        "species_5yr":         max(species_before or 150, prediction[0]),
        "species_10yr":        max(species_before or 150, prediction[1]),
        "species_before":      species_before if species_before else 150,
        "lat": lat, "lon": lon,
        "rainfall":            rainfall,
        "dist_1ha":            dist_1ha,
        "years_operational":   years_operational,
        "prediction_reliable": True,
    }

def calculate_species_roi(prediction, plant_row):
    if prediction is None:
        return None
    species_5yr    = prediction["species_5yr"]
    species_10yr   = prediction["species_10yr"]
    species_before = prediction["species_before"]
    site_ha        = plant_row["site_ha"] if pd.notna(plant_row["site_ha"]) else 10
    if species_before == 0:
        species_before = 1
    uplift_5yr  = max(0, (species_5yr  - species_before) / species_before)
    uplift_10yr = max(0, (species_10yr - species_before) / species_before)
    annual_5yr  = ECOSYSTEM_VALUE_PER_HA_YR * uplift_5yr  * site_ha
    annual_10yr = ECOSYSTEM_VALUE_PER_HA_YR * uplift_10yr * site_ha
    cumulative  = []
    running     = 0
    for year in range(1, 21):
        running += annual_5yr if year <= 5 else annual_10yr
        cumulative.append(running)
    return {
        "species_gained_5yr":   max(0, species_5yr  - species_before),
        "species_gained_10yr":  max(0, species_10yr - species_before),
        "uplift_5yr":           uplift_5yr,
        "uplift_10yr":          uplift_10yr,
        "annual_value_5yr":     annual_5yr,
        "annual_value_10yr":    annual_10yr,
        "cumulative_20yr":      cumulative[-1],
        "cumulative":           cumulative,
    }

def renewable_land_biodiversity_impact(plant_row, renewable_type):
    land_change = plant_row.get(f"{renewable_type}_net_land_ha", 0)
    land_change = land_change if pd.notna(land_change) else 0
    quality     = HABITAT_QUALITY_FACTOR.get(renewable_type, 0.5)
    if land_change < 0:
        return -(abs(land_change) * ECOSYSTEM_VALUE_PER_HA_YR * (1 - quality))
    return land_change * ECOSYSTEM_VALUE_PER_HA_YR * quality

def get_roi_for_renewable(plant_row, renewable_type):
    def safe(key):
        val = plant_row.get(f"{renewable_type}_{key}", 0)
        return val if pd.notna(val) else 0
    co2_value   = safe("co2_saving_kgCO2e_per_year") * 0.06423
    water_value = safe("water_saving_litres_per_year") * 0.000001
    nox_value   = safe("NOX_saving_kg") * 10.193
    sox_value   = safe("SOX_saving_kg") * 26.193
    ch4_value   = safe("CH4_saving_kg") * 1.79844
    land_change = plant_row.get(f"{renewable_type}_net_land_ha", 0)
    land_change = land_change if pd.notna(land_change) else 0
    land_value  = abs(land_change) * ECOSYSTEM_VALUE_PER_HA_YR
    return {
        "co2_value":       co2_value,
        "water_value":     water_value,
        "nox_value":       nox_value,
        "sox_value":       sox_value,
        "ch4_value":       ch4_value,
        "land_value":      land_value,
        "existing_annual": co2_value + water_value + nox_value + sox_value + ch4_value,
    }

# ── Streamlit UI ──────────────────────────────────────────────
st.title("🌱 Carbon Offset Engine")
st.markdown("### Environmental ROI of decommissioning UK fossil fuel power plants")
st.markdown("---")

# ── Sidebar inputs ────────────────────────────────────────────
st.sidebar.header("Plant Selection")

plant_names  = sorted(official["Site Name"].dropna().unique().tolist())
selected_plant = st.sidebar.selectbox("Select a power plant", plant_names)

opening_year = st.sidebar.number_input(
    "Year plant opened", min_value=1900, max_value=2030, value=1970
)
closure_year = st.sidebar.number_input(
    "Year plant closed / will close", min_value=1900, max_value=2030, value=2025
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Model performance:**\n"
    f"- Species 5yr R²={model_metrics['species_5yr']['r2']:.2f}\n"
    f"- Species 10yr R²={model_metrics['species_10yr']['r2']:.2f}\n"
    "- Based on 89 historical UK sites"
)

# ── Run calculations ──────────────────────────────────────────
if st.sidebar.button("Calculate ROI", type="primary"):

    plant_row  = official[official["Site Name"] == selected_plant].iloc[0]
    renewables = ["Solar", "Wind", "Hydro", "Nuclear", "Biomass"]

    with st.spinner("Calculating species recovery (this may take a minute)..."):
        prediction  = predict_species_recovery(plant_row, opening_year, closure_year)
        species_roi = calculate_species_roi(prediction, plant_row)
        species_annual = species_roi["annual_value_10yr"] if species_roi else 0

    if prediction and not prediction.get("prediction_reliable", True):
        st.warning("⚠️ Plant closed too recently for reliable species prediction. Showing baseline species count.")

    # baseline
    baseline_roi        = get_roi_for_renewable(plant_row, "Solar")
    land_freed          = plant_row["site_ha"] if pd.notna(plant_row["site_ha"]) else 0
    land_value_baseline = land_freed * ECOSYSTEM_VALUE_PER_HA_YR
    baseline_annual     = (
        baseline_roi["existing_annual"] +
        land_value_baseline +
        species_annual
    )

    # all renewables
    results = {}
    for renewable in renewables:
        roi        = get_roi_for_renewable(plant_row, renewable)
        bio_impact = renewable_land_biodiversity_impact(plant_row, renewable)
        total      = roi["existing_annual"] + roi["land_value"] + species_annual + bio_impact
        results[renewable] = {**roi, "bio_impact": bio_impact, "species_value": species_annual, "total_annual": total}

    # ── Plant info ────────────────────────────────────────────
    st.header(f"📍 {selected_plant}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Capacity",        f"{plant_row[' InstalledCapacity (MW)']:.0f} MW")
    col2.metric("Fuel Type",       plant_row["Primary Fuel"])
    col3.metric("Years Operational", f"{closure_year - opening_year} years")
    col4.metric("Site Area",       f"{plant_row['site_ha']:.1f} ha" if pd.notna(plant_row["site_ha"]) else "Unknown")

    st.markdown("---")

    # ── Summary table ─────────────────────────────────────────
    st.subheader("📊 Annual Environmental Value by Renewable Type")

    table_data = {
        "":               ["CO2 savings", "Water savings", "NOX savings", "SOX savings", "CH4 savings", "Land value", "Species recovery", "Renewable bio impact", "**NET ANNUAL**"],
        "Close Only":     [
            f"£{baseline_roi['co2_value']/1e6:.2f}M",
            f"£{baseline_roi['water_value']/1e6:.2f}M",
            f"£{baseline_roi['nox_value']/1e6:.2f}M",
            f"£{baseline_roi['sox_value']/1e6:.2f}M",
            f"£{baseline_roi['ch4_value']/1e6:.2f}M",
            f"£{land_value_baseline/1e6:.2f}M",
            f"£{species_annual/1e6:.2f}M",
            "£0.00M",
            f"**£{baseline_annual/1e6:.2f}M**",
        ],
    }
    for r in renewables:
        table_data[r] = [
            f"£{results[r]['co2_value']/1e6:.2f}M",
            f"£{results[r]['water_value']/1e6:.2f}M",
            f"£{results[r]['nox_value']/1e6:.2f}M",
            f"£{results[r]['sox_value']/1e6:.2f}M",
            f"£{results[r]['ch4_value']/1e6:.2f}M",
            f"£{results[r]['land_value']/1e6:.2f}M",
            f"£{results[r]['species_value']/1e6:.2f}M",
            f"£{results[r]['bio_impact']/1e6:.2f}M",
            f"**£{results[r]['total_annual']/1e6:.2f}M**",
        ]

    st.dataframe(pd.DataFrame(table_data).set_index(""), use_container_width=True)

    st.markdown("---")

    # ── Chart 1: net annual value ─────────────────────────────
    st.subheader("📈 Net Annual Environmental Value")

    x_labels  = ["Close
Only"] + renewables
    x_pos     = range(len(x_labels))
    bar_width = 0.6
    net_vals  = [baseline_annual] + [results[r]["total_annual"] for r in renewables]
    net_m     = [v / 1e6 for v in net_vals]
    colours   = ["#95a5a6"] + [
        "#2ecc71" if v >= baseline_annual else "#e74c3c"
        for v in [results[r]["total_annual"] for r in renewables]
    ]

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(x_pos, net_m, bar_width, color=colours, alpha=0.9)
    for bar, val in zip(bars, net_m):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(net_m) * 0.01,
            f"£{val:.1f}M", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, fontsize=10)
    ax1.set_ylabel("Net Annual Value (£ millions)")
    ax1.set_title(f"Net Annual Environmental Value — {selected_plant}\n(green = better than close only)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.1f}M"))
    ax1.axhline(baseline_annual / 1e6, color="#95a5a6", linewidth=1.5, linestyle="--", label="Close only baseline")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig1)

    st.markdown("---")

    # ── Chart 2: 20 year cumulative ───────────────────────────
    st.subheader("📈 20 Year Cumulative Environmental ROI")

    years = list(range(1, 21))
    renewable_colours = {
        "Solar": "#f1c40f", "Wind": "#2ecc71", "Hydro": "#3498db",
        "Nuclear": "#9b59b6", "Biomass": "#e67e22",
    }

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    baseline_cumulative = [baseline_annual * year / 1e6 for year in years]
    ax2.plot(years, baseline_cumulative, color="#95a5a6", linewidth=2,
             linestyle="--", label="Close Only", zorder=5)
    ax2.text(20.2, baseline_cumulative[-1], f"£{baseline_cumulative[-1]:.0f}M",
             va="center", fontsize=9, fontweight="bold", color="#95a5a6")

    if prediction and species_roi:
        sp_before   = prediction["species_before"]
        site_ha_val = plant_row["site_ha"] if pd.notna(plant_row["site_ha"]) else 10
        if sp_before == 0: sp_before = 1
        uplift_5yr  = max(0, (prediction["species_5yr"]  - sp_before) / sp_before)
        uplift_10yr = max(0, (prediction["species_10yr"] - sp_before) / sp_before)
        ann_sp_5yr  = ECOSYSTEM_VALUE_PER_HA_YR * uplift_5yr  * site_ha_val
        ann_sp_10yr = ECOSYSTEM_VALUE_PER_HA_YR * uplift_10yr * site_ha_val
    else:
        ann_sp_5yr = ann_sp_10yr = 0

    for renewable in renewables:
        r = results[renewable]
        non_species = r["co2_value"] + r["water_value"] + r["nox_value"] + r["sox_value"] + r["ch4_value"] + r["land_value"] + r["bio_impact"]
        cumulative_line = []
        running = 0
        for year in years:
            running += non_species + (ann_sp_5yr if year <= 5 else ann_sp_10yr)
            cumulative_line.append(running / 1e6)
        ax2.plot(years, cumulative_line, color=renewable_colours[renewable],
                 linewidth=2, label=renewable)
        ax2.text(20.2, cumulative_line[-1], f"£{cumulative_line[-1]:.0f}M",
                 va="center", fontsize=9, fontweight="bold", color=renewable_colours[renewable])

    ax2.set_xlabel("Year", fontsize=11)
    ax2.set_ylabel("Cumulative Environmental Value (£ millions)", fontsize=11)
    ax2.set_title(f"20 Year Cumulative Net Environmental Value — {selected_plant}")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.0f}M"))
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(alpha=0.3)
    ax2.set_xlim(1, 22)
    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("---")

    # ── Chart 3: species recovery ─────────────────────────────
    st.subheader("🌿 Ecological Recovery")

    if prediction:
        sp_before  = prediction["species_before"]
        sp_5yr     = prediction["species_5yr"]
        sp_10yr    = prediction["species_10yr"]
        site_ha_v  = plant_row["site_ha"] if pd.notna(plant_row["site_ha"]) else 10
        if sp_before == 0: sp_before = 1
        uplift_5yr  = max(0, (sp_5yr  - sp_before) / sp_before)
        uplift_10yr = max(0, (sp_10yr - sp_before) / sp_before)
        ann_5       = ECOSYSTEM_VALUE_PER_HA_YR * uplift_5yr  * site_ha_v
        ann_10      = ECOSYSTEM_VALUE_PER_HA_YR * uplift_10yr * site_ha_v

        fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 6))
        fig3.suptitle(f"Ecological Recovery — {selected_plant}", fontsize=14, fontweight="bold")

        # species trajectory
        trajectory = []
        for year in years:
            if year <= 5:
                trajectory.append(sp_before + (sp_5yr - sp_before) * year / 5)
            elif year <= 10:
                trajectory.append(sp_5yr + (sp_10yr - sp_5yr) * (year - 5) / 5)
            else:
                trajectory.append(sp_10yr)

        ax3.plot(years, trajectory, color="#2ecc71", linewidth=2.5)
        ax3.axhline(sp_before, color="#e74c3c", linewidth=1.5, linestyle="--",
                    label=f"Baseline ({sp_before:.0f} species)")
        ax3.scatter([5, 10], [sp_5yr, sp_10yr], color="#2ecc71", s=80, zorder=5)
        ax3.annotate(f"{sp_5yr:.0f} species", xy=(5, sp_5yr), xytext=(5.3, sp_5yr),
                     fontsize=9, fontweight="bold", color="#2ecc71")
        ax3.annotate(f"{sp_10yr:.0f} species", xy=(10, sp_10yr), xytext=(10.3, sp_10yr),
                     fontsize=9, fontweight="bold", color="#2ecc71")
        ax3.fill_between(years, sp_before, trajectory, alpha=0.15, color="#2ecc71", label="Species gained")
        ax3.set_xlabel("Years since closure", fontsize=11)
        ax3.set_ylabel("Predicted species count", fontsize=11)
        ax3.set_title("Predicted Species Recovery Trajectory")
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
        ax3.set_xlim(1, 20)
        r2_5  = model_metrics["species_5yr"]["r2"]
        r2_10 = model_metrics["species_10yr"]["r2"]
        ax3.text(0.02, 0.02,
                 f"⚠️ Model R²: yr5={r2_5:.2f}, yr10={r2_10:.2f}\nBased on 89 historical UK power station sites",
                 transform=ax3.transAxes, fontsize=8, color="grey", va="bottom")

        # cumulative species value
        sp_cumulative = []
        running = 0
        for year in years:
            running += ann_5 if year <= 5 else ann_10
            sp_cumulative.append(running / 1e3)

        ax4.plot(years, sp_cumulative, color="#f1c40f", linewidth=2.5)
        ax4.axvline(5, color="#2ecc71", linewidth=1.5, linestyle="--",
                    alpha=0.7, label="Year 5 — recovery accelerates")
        ax4.fill_between(years, 0, sp_cumulative, alpha=0.15, color="#f1c40f")
        ax4.annotate(f"£{sp_cumulative[4]:,.0f}k", xy=(5, sp_cumulative[4]),
                     xytext=(5.3, sp_cumulative[4]), fontsize=9, fontweight="bold", color="#f1c40f")
        ax4.annotate(f"£{sp_cumulative[-1]:,.0f}k", xy=(20, sp_cumulative[-1]),
                     xytext=(17, sp_cumulative[-1] * 0.95), fontsize=9, fontweight="bold", color="#f1c40f")
        ax4.set_xlabel("Years since closure", fontsize=11)
        ax4.set_ylabel("Cumulative Species Recovery Value (£ thousands)", fontsize=11)
        ax4.set_title("Cumulative Ecosystem Service Value\nfrom Species Recovery")
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        ax4.set_xlim(1, 20)
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}k"))

        plt.tight_layout()
        st.pyplot(fig3)

        # species metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline Species",    f"{sp_before:.0f}")
        col2.metric("Predicted Year 5",    f"{sp_5yr:.0f}")
        col3.metric("Predicted Year 10",   f"{sp_10yr:.0f}")

        if not prediction.get("prediction_reliable", True):
            st.info("ℹ️ Species predictions unreliable — plant closed too recently. Showing baseline as estimate.")
        else:
            st.caption(f"⚠️ Species predictions based on 89 historical UK power station sites. R²={r2_10:.2f}. Use as directional indicator only.")

    st.markdown("---")
    st.caption("Sources: ONS Natural Capital Accounts 2023 | GBIF Biodiversity Database | CEH Land Cover Map | Open-Meteo Climate API | DUKES 2024")

else:
    st.info("👈 Select a plant and click **Calculate ROI** to begin")
    st.markdown("""
    ### How it works
    1. Select a fossil fuel power plant from the dropdown
    2. Enter the opening and closing years
    3. The engine calculates the full environmental ROI of decommissioning

    ### What it measures
    - **CO2 savings** — avoided greenhouse gas emissions
    - **Water savings** — reduced water consumption  
    - **Air quality** — NOX, SOX and CH4 reduction
    - **Land value** — ecosystem services from freed land
    - **Species recovery** — ML-predicted biodiversity improvement
    - **Renewable impact** — net biodiversity effect of replacement energy

    ### Data sources
    - DUKES 2024 — UK power plant registry
    - GBIF — Global Biodiversity Information Facility
    - CEH Land Cover Map — UK habitat data
    - Open-Meteo — Historical climate data
    - ONS Natural Capital Accounts 2023
    """)
