# PROJECT HEIMDALL: AI-Driven Hydro-Meteorological Forecasting & Agricultural Drought Early Warning System

**Author:** Engr. Carl Joseph Torres, ECE, ECT 
**Live Dashboard:** [Project HEIMDALL Vercel Deployment](https://project-heimdall-nasa-spaceapps-psi.vercel.app/)

---

## 📌 Project Overview
Project HEIMDALL is an advanced AI-driven early warning dashboard designed for Southeast Asia and the Philippines. It fuses multi-source satellite remote sensing with deep learning time-series forecasting to predict root-zone soil moisture depletion and identify agricultural drought and thermal stress hotspots before they impact food security.

## 🛰️ Data Sources & Acquisition
* **NASA FLDAS (NOAHMP001):** Area-averaged hydro-meteorological time-series data (Precipitation flux, Surface air temperature, Root-zone soil moisture $0–10\text{ cm}$) extracted via **NASA Giovanni** (2023–2026).
* **NASA MODIS (MOD11A2 v061):** 1km-resolution daytime and nighttime Land Surface Temperature (LST) statistics extracted via **NASA AppEEARS** using a custom Philippine boundary polygon.

1. **NASA Giovanni (FLDAS NOAHMP001):**
   * **Utilization:** Used as the primary training and inference feed for our Long Short-Term Memory (LSTM) deep learning model to evaluate baseline hydro-meteorological coupling (Pearson $r = 0.8161$) and execute 30-day forward drought predictions.

2. **NASA AppEEARS (MODIS MOD11A2 v061):**
   * **Utilization:** Used as a high-resolution thermal baseline to power our Philippine agricultural risk warning map, combining LST anomalies with LSTM soil moisture forecasts to pinpoint localized thermal stress hotspots in Central Luzon, Cagayan Valley, Western Visayas, and Bukidnon.

## 🧠 Machine Learning Core
* **Model Architecture:** Long Short-Term Memory (LSTM) neural network (`project-heimdall-climate-SEAV2.keras`).
* **Performance Metrics:** 
  * $R^2 \text{ Score} = 0.9140$
  * $\text{RMSE} = 0.0217 \text{ m}^3/\text{m}^3$
  * $\text{MAE} = 0.0183 \text{ m}^3/\text{m}^3$
* **Statistical Validation:** Evaluated using Pearson correlation ($r = 0.8161, p < 0.001$), OLS regression, and Mann-Kendall trend tests, confirming periodic seasonal cycles rather than long-term linear drift.

## 🚀 Features
1. **Interactive Time-Series Chart:** Visualizes historical observations alongside a 30-day forward LSTM forecast with 95% confidence intervals.
2. **Super El Niño Stress Simulator:** Interactive sliders allowing users to simulate rainfall deficits and temperature anomalies in real-time.
3. **Philippine Agricultural Hotspot Map:** Leaflet map rendering dynamic risk zones (Central Luzon, Cagayan Valley, Western Visayas, Bukidnon) combining MODIS LST and LSTM soil moisture outputs.
4. **Adjustable Forecast Table:** Fully editable data grid enabling manual scenario overrides.
5. **Empirical Evidence Section:** Scroll-down validation suite containing statistical heatmaps and OLS trend plots.

## 🛠️ Local Setup & Execution
1. Clone the repository and ensure all FLDAS CSVs, MODIS stats, and model files (`.keras`, `.save`) are in the root directory.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the FastAPI Backend:
   ```bash
   uvicorn app:app --reload --port 8000
   ```
4. Open index.html in your browser (or serve it via a live server). The dashboard will automatically connect to the live FastAPI backend and render real-time LSTM predictions and Philippine MODIS thermal hotspots.
