import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pickle

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

@st.cache_data
def load_structured():
    return pd.read_csv("data/structured_official.csv")

@st.cache_data
def load_habitat_data():
    return pd.read_csv("data/plant_habitat_distances.csv")

@st.cache_data
def load_species_data():
    return pd.read_csv("data/plant_species_rainfall.csv")

@st.cache_resource
def load_models():
    with open("models/habitat_recovery_model.pkl", "rb") as f:
        ml_model = pickle.load(f)
    with open("models/model_metrics.pkl", "rb") as f:
        model_metrics = pickle.load(f)
    return ml_model, model_metrics

official       = load_data()
structured     = load_structured()
habitat_data   = load_habitat_data()
species_data   = load_species_data()
ml_model, model_metrics = load_models()

# ── Constants ─────────────────────────────────────────────────
ECOSYSTEM_VALUE_PER_HA_YR = 7500
CURRENT_YEAR = 2026

HABITAT_QUALITY_FACTOR = {
    "Solar":   0.3,
    "Wind":    0.7,
    "Hydro":   0.5,
    "Nuclear": 0.2,
    "Biomass": 0.1,
}

BUILD_CARBON_KG_PER_MW = {
    "Solar":   670000,
    "Wind":    80616,
    "Hydro":   2500000,
    "Nuclear": 1025000,
    "Biomass": 500000,
}

RENEWABLE_COLOURS = {
    "Solar":   "#f39c12",
    "Wind":    "#2ecc71",
    "Hydro":   "#3498db",
    "Nuclear": "#9b59b6",
    "Biomass": "#e74c3c",
}

RENEWABLES = ["Solar", "Wind", "Hydro", "Nuclear", "Biomass"]

# ── Helper functions ──────────────────────────────────────────
def predict_species_recovery(plant_row, opening_year, closure_year):
    name = plant_row["Site Name"]
    hab  = habitat_data[habitat_data["Site Name"] == name]
    spe  = species_data[species_data["Site Name"] == name]
    if hab.empty or spe.empty:
        return None
    hab = hab.iloc[0]
    spe = spe.iloc[0]
    lat               = hab["lat"]
    lon               = hab["lon"]
    years_operational = closure_year - opening_year
    site_ha           = plant_row["site_ha"] if pd.notna(plant_row["site_ha"]) else 20
    species_before    = spe["species_before"]  if pd.notna(spe["species_before"])  else 150
    rainfall          = spe["annual_rainfall"]  if pd.notna(spe["annual_rainfall"]) else 800
    dist_any          = hab["dist_any_intact_km"] if pd.notna(hab["dist_any_intact_km"]) else None
    dist_1ha          = hab["dist_1ha_patch_km"]  if pd.notna(hab["dist_1ha_patch_km"])  else None
    dist_5ha          = hab["dist_5ha_patch_km"]  if pd.notna(hab["dist_5ha_patch_km"])  else None

    # ── check habitat data is valid ───────────────────────────
    # NaN = site outside LCM coverage (islands etc)
    # 0.0 = coordinates landed on wrong pixel
    invalid_dist = (
        dist_any is None or dist_1ha is None or dist_5ha is None or
        (dist_any == 0.0 and dist_1ha == 0.0 and dist_5ha == 0.0)
    )

    if invalid_dist:
        return {
            "species_5yr":          species_before,
            "species_10yr":         species_before,
            "species_before":       species_before,
            "prediction_year_5yr":  CURRENT_YEAR + 5,
            "prediction_year_10yr": CURRENT_YEAR + 10,
            "not_yet_closed":       closure_year > CURRENT_YEAR,
            "lat":                  lat,
            "lon":                  lon,
            "rainfall":             rainfall,
            "dist_1ha":             dist_1ha,
            "years_operational":    years_operational,
            "prediction_reliable":  False,
            "reason":               "Habitat distance data is unreliable for this location. "
                                    "The site may be on an island, coastal area, or the coordinates "
                                    "may fall outside the CEH Land Cover Map coverage area.",
        }

    # predict from whichever is later — now or closure date
    # ensures every plant gets a genuine 5 and 10 year recovery window
    effective_start  = max(CURRENT_YEAR, closure_year)
    years_since_5yr  = (effective_start + 5)  - closure_year
    years_since_10yr = (effective_start + 10) - closure_year

    def predict_at(years_since):
        input_data = pd.DataFrame([{
            "Years Operational":           years_operational,
            "Site Area (ha)":              site_ha,
            "annual_rainfall_mm":          rainfall,
            "species_before":              species_before,
            "dist_any_intact_closure_km":  dist_any,
            "dist_1ha_patch_closure_km":   dist_1ha,
            "dist_5ha_patch_closure_km":   dist_5ha,
            "lat":                         lat,
            "lon":                         lon,
            "years_since_closure":         years_since,
        }])
        return ml_model.predict(input_data)[0]

    pred_5yr  = predict_at(years_since_5yr)
    pred_10yr = predict_at(years_since_10yr)

    return {
        "species_5yr":          max(species_before, pred_5yr[0]),
        "species_10yr":         max(species_before, pred_10yr[1]),
        "species_before":       species_before,
        "prediction_year_5yr":  effective_start + 5,
        "prediction_year_10yr": effective_start + 10,
        "not_yet_closed":       closure_year > CURRENT_YEAR,
        "lat":                  lat,
        "lon":                  lon,
        "rainfall":             rainfall,
        "dist_1ha":             dist_1ha,
        "years_operational":    years_operational,
        "prediction_reliable":  True,
        "reason":               None,
    }

def calculate_species_roi(prediction, plant_row):
    if prediction is None:
        return None
    species_5yr    = prediction["species_5yr"]
    species_10yr   = prediction["species_10yr"]
    species_before = prediction["species_before"]
    site_ha        = plant_row["site_ha"] if pd.notna(plant_row["site_ha"]) else 10
    if species_before == 0: species_before = 1
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
    co2_value   = safe("co2_saving_kgCO2e_per_year")  * 0.06423
    water_value = safe("water_saving_litres_per_year") * 0.000001
    nox_value   = safe("NOX_saving_kg")                * 10.193
    sox_value   = safe("SOX_saving_kg")                * 26.193
    ch4_value   = safe("CH4_saving_kg")                * 1.79844
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
plant_names    = sorted(official["Site Name"].dropna().unique().tolist())
selected_plant = st.sidebar.selectbox("Select a power plant", plant_names)
opening_year   = st.sidebar.number_input("Year plant opened",              min_value=1900, max_value=2050, value=1970)
closure_year   = st.sidebar.number_input("Year plant closed / will close", min_value=1900, max_value=2050, value=2025)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**ML Model performance:**\n"
    f"- Species 5yr R²={model_metrics['species_5yr']['r2']:.2f}\n"
    f"- Species 10yr R²={model_metrics['species_10yr']['r2']:.2f}\n"
    f"- Based on 89 historical UK sites"
)

# ── Run calculations ──────────────────────────────────────────
if st.sidebar.button("Calculate ROI", type="primary"):

    plant_row  = official[official["Site Name"] == selected_plant].iloc[0]
    struct_row = structured[structured["Site Name"] == selected_plant]

    if struct_row.empty:
        st.error(f"No data found for {selected_plant}")
        st.stop()

    struct_row  = struct_row.iloc[0]
    capacity_mw = struct_row["Capacity MW"]

    with st.spinner("Calculating environmental ROI..."):
        prediction     = predict_species_recovery(plant_row, opening_year, closure_year)
        species_roi    = calculate_species_roi(prediction, plant_row)
        species_annual = species_roi["annual_value_10yr"] if species_roi else 0

        baseline_roi        = get_roi_for_renewable(plant_row, "Solar")
        land_freed          = plant_row["site_ha"] if pd.notna(plant_row["site_ha"]) else 0
        land_value_baseline = land_freed * ECOSYSTEM_VALUE_PER_HA_YR
        baseline_annual     = baseline_roi["existing_annual"] + land_value_baseline + species_annual

        results = {}
        for renewable in RENEWABLES:
            roi        = get_roi_for_renewable(plant_row, renewable)
            bio_impact = renewable_land_biodiversity_impact(plant_row, renewable)
            total      = roi["existing_annual"] + roi["land_value"] + species_annual + bio_impact
            results[renewable] = {**roi, "bio_impact": bio_impact,
                                  "species_value": species_annual, "total_annual": total}

    # ── Plant info ────────────────────────────────────────────
    st.header(f"📍 {selected_plant}")

    if prediction and prediction.get("not_yet_closed"):
        st.info(
            f"ℹ️ {selected_plant} is scheduled to close in {closure_year}. "
            f"Species predictions show recovery by "
            f"{prediction['prediction_year_5yr']} and {prediction['prediction_year_10yr']}."
        )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Capacity",          f"{capacity_mw:.0f} MW")
    col2.metric("Fuel Type",         struct_row["Primary Fuel"])
    col3.metric("Years Operational", f"{closure_year - opening_year} years")
    col4.metric("Site Area",         f"{plant_row['site_ha']:.1f} ha" if pd.notna(plant_row["site_ha"]) else "Unknown")
    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    # SECTION 1 — INDIVIDUAL METRIC CHARTS
    # ═══════════════════════════════════════════════════════════
    st.header("📊 Individual Environmental Metrics")

    years = list(range(1, 21))

    # ── CO2 chart ─────────────────────────────────────────────
    st.subheader("CO2 Emissions")
    co2_labels = ["Total CO2\n(current)", "Solar\nsaving", "Wind\nsaving",
                  "Hydro\nsaving", "Nuclear\nsaving", "Biomass\nsaving"]
    co2_values = [
        struct_row["CO2_Total CO2 per year"],
        struct_row["CO2_Solar saving"],
        struct_row["CO2_Wind saving"],
        struct_row["CO2_Hydro saving"],
        struct_row["CO2_Nuclear saving"],
        struct_row["CO2_Biomass saving"],
    ]
    co2_colors = ["#e74c3c"] + ["#2ecc71"] * 5

    fig_co2, ax_co2 = plt.subplots(figsize=(10, 5))
    ax_co2.bar(co2_labels, co2_values, color=co2_colors, edgecolor="white")
    ax_co2.set_title(f"CO2 Savings — {selected_plant}")
    ax_co2.set_ylabel("kgCO2e per year")
    ax_co2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax_co2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_co2)

    # ── Water chart ───────────────────────────────────────────
    st.subheader("Water Consumption")
    water_labels = ["Total\n(current)", "Solar\nsaving", "Wind\nsaving",
                    "Hydro\nsaving", "Nuclear\nsaving", "Biomass\nsaving"]
    water_values = [
        struct_row["Water_Consumption litres/yr"],
        struct_row["Water_Solar saving"],
        struct_row["Water_Wind saving"],
        struct_row["Water_Hydro saving"],
        struct_row["Water_Nuclear saving"],
        struct_row["Water_Biomass saving"],
    ]
    water_colors = ["#e74c3c"] + ["#9b59b6"] * 5

    fig_water, ax_water = plt.subplots(figsize=(10, 5))
    ax_water.bar(water_labels, water_values, color=water_colors, edgecolor="white")
    ax_water.set_title(f"Water Savings — {selected_plant}")
    ax_water.set_ylabel("Litres per year")
    ax_water.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e9:.2f}B"))
    ax_water.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_water)

    # ── Emissions charts ──────────────────────────────────────
    st.subheader("Air Quality Emissions (NOX, SOX, CH4)")
    fig_em, axes_em = plt.subplots(1, 3, figsize=(15, 5))
    emissions_config = {
        "NOX": {
            "total": "Emissions_NOX kg",
            "savings": [f"Emissions_{r} NOX saving" for r in RENEWABLES],
        },
        "SOX": {
            "total": "Emissions_SOX kg",
            "savings": [f"Emissions_{r} SOX saving" for r in RENEWABLES],
        },
        "CH4": {
            "total": "Emissions_CH4 kg",
            "savings": [f"Emissions_{r} CH4 saving" for r in RENEWABLES],
        },
    }
    em_labels = ["Current"] + RENEWABLES
    em_colors = ["#e74c3c"] + ["#3498db"] * 5

    for ax_em, (em_type, cfg) in zip(axes_em, emissions_config.items()):
        em_values = [struct_row[cfg["total"]]] + [
            struct_row[col] if col in struct_row.index and pd.notna(struct_row[col]) else 0
            for col in cfg["savings"]
        ]
        ax_em.bar(em_labels, em_values, color=em_colors, edgecolor="white")
        ax_em.set_title(f"{em_type} — {selected_plant}")
        ax_em.set_ylabel("kg per year")
        ax_em.tick_params(axis="x", rotation=30)
        ax_em.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig_em)

    # ── Land use chart ────────────────────────────────────────
    st.subheader("Land Use Change")
    land_labels = ["Site\n(current)", "Solar\nnet land", "Wind\nnet land",
                   "Hydro\nnet land", "Nuclear\nnet land", "Biomass\nnet land"]
    land_values = [
        struct_row["Land_Site ha"],
        struct_row["Land_Solar net land"],
        struct_row["Land_Wind net land"],
        struct_row["Land_Hydro net land"],
        struct_row["Land_Nuclear net land"],
        struct_row["Land_Biomass net land"],
    ]
    land_colors = ["#e74c3c"] + [
        "#2ecc71" if v >= 0 else "#f39c12"
        for v in land_values[1:]
    ]

    fig_land, ax_land = plt.subplots(figsize=(10, 5))
    ax_land.bar(land_labels, land_values, color=land_colors, edgecolor="white")
    ax_land.set_title(f"Land Use Change — {selected_plant}")
    ax_land.set_ylabel("Hectares")
    ax_land.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_land.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_land)

    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    # SECTION 2 — CO2 PAYBACK TIMELINE
    # ═══════════════════════════════════════════════════════════
    st.header("⏱️ CO2 Carbon Payback Timeline")

    annual_co2_savings = {r: struct_row[f"CO2_{r} saving"] for r in RENEWABLES}
    build_carbon_kg    = {r: capacity_mw * BUILD_CARBON_KG_PER_MW[r] for r in RENEWABLES}

    co2_timelines = {}
    for renewable in RENEWABLES:
        running_total = -build_carbon_kg[renewable]
        cumulative    = []
        for year in years:
            running_total += annual_co2_savings[renewable]
            cumulative.append(running_total)
        co2_timelines[renewable] = cumulative

    fig_payback, ax_payback = plt.subplots(figsize=(12, 6))
    for renewable in RENEWABLES:
        ax_payback.plot(
            years, co2_timelines[renewable],
            label=renewable, color=RENEWABLE_COLOURS[renewable], linewidth=2
        )
    ax_payback.axhline(y=0, color="black", linestyle="--", linewidth=0.8, label="Break even")
    ax_payback.set_title(f"20 Year CO2 Carbon Payback — {selected_plant}")
    ax_payback.set_xlabel("Year")
    ax_payback.set_ylabel("Cumulative CO2 saved (kg)")
    ax_payback.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e9:.2f}B kg"))
    ax_payback.legend()
    ax_payback.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_payback)

    payback_data = {}
    for renewable in RENEWABLES:
        payback = next(
            (y for y, v in zip(years, co2_timelines[renewable]) if v > 0), None
        )
        payback_data[renewable] = f"Year {payback}" if payback else "Not within 20 years"

    st.dataframe(
        pd.DataFrame(payback_data, index=["Carbon Payback Year"]),
        use_container_width=True
    )
    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    # SECTION 3 — NET ENVIRONMENTAL ROI
    # ═══════════════════════════════════════════════════════════
    st.header("💰 Net Environmental ROI")

    st.subheader("Annual Environmental Value by Renewable Type")
    table_data = {
        "": ["CO2 savings", "Water savings", "NOX savings", "SOX savings",
             "CH4 savings", "Land value", "Species recovery",
             "Renewable bio impact", "NET ANNUAL"],
        "Close Only": [
            f"£{baseline_roi['co2_value']/1e6:.2f}M",
            f"£{baseline_roi['water_value']/1e6:.2f}M",
            f"£{baseline_roi['nox_value']/1e6:.2f}M",
            f"£{baseline_roi['sox_value']/1e6:.2f}M",
            f"£{baseline_roi['ch4_value']/1e6:.2f}M",
            f"£{land_value_baseline/1e6:.2f}M",
            f"£{species_annual/1e6:.2f}M",
            "£0.00M",
            f"£{baseline_annual/1e6:.2f}M",
        ],
    }
    for r in RENEWABLES:
        table_data[r] = [
            f"£{results[r]['co2_value']/1e6:.2f}M",
            f"£{results[r]['water_value']/1e6:.2f}M",
            f"£{results[r]['nox_value']/1e6:.2f}M",
            f"£{results[r]['sox_value']/1e6:.2f}M",
            f"£{results[r]['ch4_value']/1e6:.2f}M",
            f"£{results[r]['land_value']/1e6:.2f}M",
            f"£{results[r]['species_value']/1e6:.2f}M",
            f"£{results[r]['bio_impact']/1e6:.2f}M",
            f"£{results[r]['total_annual']/1e6:.2f}M",
        ]
    st.dataframe(pd.DataFrame(table_data).set_index(""), use_container_width=True)

    st.subheader("Net Annual Environmental Value")
    x_labels  = ["Close\nOnly"] + RENEWABLES
    x_pos     = range(len(x_labels))
    bar_width = 0.6
    net_vals  = [baseline_annual] + [results[r]["total_annual"] for r in RENEWABLES]
    net_m     = [v / 1e6 for v in net_vals]
    colours   = ["#95a5a6"] + [
        "#2ecc71" if v >= baseline_annual else "#e74c3c"
        for v in [results[r]["total_annual"] for r in RENEWABLES]
    ]

    fig_net, ax_net = plt.subplots(figsize=(10, 6))
    bars = ax_net.bar(x_pos, net_m, bar_width, color=colours, alpha=0.9)
    for bar, val in zip(bars, net_m):
        ax_net.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(net_m) * 0.01,
            f"£{val:.1f}M", ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    ax_net.set_xticks(x_pos)
    ax_net.set_xticklabels(x_labels, fontsize=10)
    ax_net.set_ylabel("Net Annual Value (£ millions)")
    ax_net.set_title(f"Net Annual Environmental Value — {selected_plant}\n(green = better than close only)")
    ax_net.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.1f}M"))
    ax_net.axhline(baseline_annual / 1e6, color="#95a5a6", linewidth=1.5,
                   linestyle="--", label="Close only baseline")
    ax_net.legend(fontsize=9)
    ax_net.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_net)

    st.subheader("20 Year Cumulative Environmental ROI")

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

    fig_cum, ax_cum = plt.subplots(figsize=(12, 7))
    baseline_cumulative = [baseline_annual * year / 1e6 for year in years]
    ax_cum.plot(years, baseline_cumulative, color="#95a5a6", linewidth=2,
                linestyle="--", label="Close Only", zorder=5)
    ax_cum.text(20.2, baseline_cumulative[-1], f"£{baseline_cumulative[-1]:.0f}M",
                va="center", fontsize=9, fontweight="bold", color="#95a5a6")

    for renewable in RENEWABLES:
        r           = results[renewable]
        non_species = (r["co2_value"] + r["water_value"] + r["nox_value"] +
                       r["sox_value"] + r["ch4_value"] + r["land_value"] + r["bio_impact"])
        cumulative_line = []
        running = 0
        for year in years:
            running += non_species + (ann_sp_5yr if year <= 5 else ann_sp_10yr)
            cumulative_line.append(running / 1e6)
        ax_cum.plot(years, cumulative_line, color=RENEWABLE_COLOURS[renewable],
                    linewidth=2, label=renewable)
        ax_cum.text(20.2, cumulative_line[-1], f"£{cumulative_line[-1]:.0f}M",
                    va="center", fontsize=9, fontweight="bold",
                    color=RENEWABLE_COLOURS[renewable])

    ax_cum.set_xlabel("Year", fontsize=11)
    ax_cum.set_ylabel("Cumulative Environmental Value (£ millions)", fontsize=11)
    ax_cum.set_title(f"20 Year Cumulative Net Environmental Value — {selected_plant}")
    ax_cum.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:.0f}M"))
    ax_cum.legend(fontsize=10, loc="upper left")
    ax_cum.grid(alpha=0.3)
    ax_cum.set_xlim(1, 22)
    plt.tight_layout()
    st.pyplot(fig_cum)
    st.markdown("---")

    # ═══════════════════════════════════════════════════════════
    # SECTION 4 — ECOLOGICAL RECOVERY
    # ═══════════════════════════════════════════════════════════
    st.header("🌿 Ecological Recovery")

    if prediction and not prediction.get("prediction_reliable", True):
        st.warning(f"⚠️ {prediction.get('reason', 'Species prediction not available for this plant.')}")

    elif prediction and prediction.get("prediction_reliable", True):
        sp_before  = prediction["species_before"]
        sp_5yr     = prediction["species_5yr"]
        sp_10yr    = prediction["species_10yr"]
        yr5_label  = prediction["prediction_year_5yr"]
        yr10_label = prediction["prediction_year_10yr"]
        site_ha_v  = plant_row["site_ha"] if pd.notna(plant_row["site_ha"]) else 10
        if sp_before == 0: sp_before = 1
        uplift_5yr  = max(0, (sp_5yr  - sp_before) / sp_before)
        uplift_10yr = max(0, (sp_10yr - sp_before) / sp_before)
        ann_5       = ECOSYSTEM_VALUE_PER_HA_YR * uplift_5yr  * site_ha_v
        ann_10      = ECOSYSTEM_VALUE_PER_HA_YR * uplift_10yr * site_ha_v

        fig_eco, (ax_sp, ax_val) = plt.subplots(1, 2, figsize=(14, 6))
        fig_eco.suptitle(f"Ecological Recovery — {selected_plant}",
                         fontsize=14, fontweight="bold")

        trajectory = []
        for year in years:
            if year <= 5:
                trajectory.append(sp_before + (sp_5yr - sp_before) * year / 5)
            elif year <= 10:
                trajectory.append(sp_5yr + (sp_10yr - sp_5yr) * (year - 5) / 5)
            else:
                trajectory.append(sp_10yr)

        calendar_years = [CURRENT_YEAR + y for y in years]

        ax_sp.plot(calendar_years, trajectory, color="#2ecc71", linewidth=2.5)
        ax_sp.axhline(sp_before, color="#e74c3c", linewidth=1.5, linestyle="--",
                      label=f"Baseline ({sp_before:.0f} species)")
        ax_sp.scatter([yr5_label, yr10_label], [sp_5yr, sp_10yr],
                      color="#2ecc71", s=80, zorder=5)
        ax_sp.annotate(f"{sp_5yr:.0f} species ({yr5_label})",
                       xy=(yr5_label, sp_5yr), xytext=(yr5_label + 0.3, sp_5yr),
                       fontsize=9, fontweight="bold", color="#2ecc71")
        ax_sp.annotate(f"{sp_10yr:.0f} species ({yr10_label})",
                       xy=(yr10_label, sp_10yr), xytext=(yr10_label + 0.3, sp_10yr),
                       fontsize=9, fontweight="bold", color="#2ecc71")
        ax_sp.fill_between(calendar_years, sp_before, trajectory, alpha=0.15,
                           color="#2ecc71", label="Species gained")
        ax_sp.set_xlabel("Year", fontsize=11)
        ax_sp.set_ylabel("Predicted species count", fontsize=11)
        ax_sp.set_title("Predicted Species Recovery Trajectory")
        ax_sp.legend(fontsize=9)
        ax_sp.grid(alpha=0.3)
        r2_5  = model_metrics["species_5yr"]["r2"]
        r2_10 = model_metrics["species_10yr"]["r2"]
        ax_sp.text(0.02, 0.02,
                   f"⚠️ R²: yr5={r2_5:.2f}, yr10={r2_10:.2f}\nBased on 89 historical sites",
                   transform=ax_sp.transAxes, fontsize=8, color="grey", va="bottom")

        sp_cumulative = []
        running = 0
        for year in years:
            running += ann_5 if year <= 5 else ann_10
            sp_cumulative.append(running / 1e3)

        ax_val.plot(calendar_years, sp_cumulative, color="#f1c40f", linewidth=2.5)
        ax_val.axvline(yr5_label, color="#2ecc71", linewidth=1.5, linestyle="--",
                       alpha=0.7, label=f"{yr5_label} — recovery accelerates")
        ax_val.fill_between(calendar_years, 0, sp_cumulative, alpha=0.15, color="#f1c40f")
        ax_val.annotate(f"£{sp_cumulative[4]:,.0f}k",
                        xy=(yr5_label, sp_cumulative[4]),
                        xytext=(yr5_label + 0.3, sp_cumulative[4]),
                        fontsize=9, fontweight="bold", color="#f1c40f")
        ax_val.annotate(f"£{sp_cumulative[-1]:,.0f}k",
                        xy=(calendar_years[-1], sp_cumulative[-1]),
                        xytext=(calendar_years[-3], sp_cumulative[-1] * 0.95),
                        fontsize=9, fontweight="bold", color="#f1c40f")
        ax_val.set_xlabel("Year", fontsize=11)
        ax_val.set_ylabel("Cumulative Species Value (£ thousands)", fontsize=11)
        ax_val.set_title("Cumulative Ecosystem Service Value\nfrom Species Recovery")
        ax_val.legend(fontsize=9)
        ax_val.grid(alpha=0.3)
        ax_val.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}k"))
        plt.tight_layout()
        st.pyplot(fig_eco)

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline Species",        f"{sp_before:.0f}")
        col2.metric(f"Predicted {yr5_label}",  f"{sp_5yr:.0f}")
        col3.metric(f"Predicted {yr10_label}", f"{sp_10yr:.0f}")

        st.caption(
            f"⚠️ Species predictions based on 89 historical UK power station sites. "
            f"R²={r2_10:.2f}. Use as directional indicator only."
        )
    else:
        st.info("No species prediction available for this plant.")

    st.markdown("---")
    st.caption(
        "Sources: ONS Natural Capital Accounts 2023 | GBIF Biodiversity Database | "
        "CEH Land Cover Map | Open-Meteo Climate API | DUKES 2024"
    )

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
    - **Land use** — ecosystem services from freed land
    - **CO2 payback** — how long until the renewable offsets its build carbon
    - **Species recovery** — ML-predicted biodiversity improvement
    - **Net ROI** — full comparison across all renewable types

    ### Data sources
    - DUKES 2024 — UK power plant registry
    - GBIF — Global Biodiversity Information Facility
    - CEH Land Cover Map — UK habitat data
    - Open-Meteo — Historical climate data
    - ONS Natural Capital Accounts 2023
    """)
