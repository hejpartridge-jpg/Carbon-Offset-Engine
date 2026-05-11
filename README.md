# Carbon Offset Engine 🌱

A Streamlit app that calculates the full environmental ROI of decommissioning UK fossil fuel power plants and replacing them with renewable energy.

## What it does

- Calculates CO2, water, NOX, SOX and CH4 savings
- Predicts species recovery using a machine learning model trained on 89 historical UK power station sites
- Compares net environmental value across Solar, Wind, Hydro, Nuclear and Biomass replacements
- Shows 20 year cumulative environmental value

## Data sources

- DUKES 2024 — UK power plant registry
- GBIF — Global Biodiversity Information Facility  
- CEH Land Cover Map — UK habitat data
- Open-Meteo — Historical climate data
- ONS Natural Capital Accounts 2023

## Model performance

- Species 5yr R²=0.45
- Species 10yr R²=0.58
- Trained on 89 historical UK power station sites

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer

Species predictions are based on 89 historical sites and carry significant uncertainty. Use as directional indicator only.
