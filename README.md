OEM Demand Forecasting — Python Application
A complete Streamlit-based demand forecasting platform for OEM demand planning and export requirements.
---
Project structure
```
demand_forecasting/
├── app.py                        ← Streamlit main application
├── requirements.txt
├── data/
│   └── loader.py                 ← Data loading + feature engineering
├── models/
│   ├── xgboost_model.py          ← XGBoost regressor
│   ├── lightgbm_model.py         ← LightGBM regressor
│   ├── rf_gwo_model.py           ← Random Forest + Grey Wolf Optimizer
│   ├── sarima_model.py           ← SARIMA / SARIMAX
│   ├── prophet_model.py          ← Facebook Prophet (with India holidays)
│   └── lstm_model.py             ← LSTM (TensorFlow/Keras)
├── utils/
│   ├── metrics.py                ← MAPE, MAE, RMSE, R², accuracy
│   └── tableau_export.py         ← CSV + Excel export for Tableau
└── exports/                      ← Downloaded Tableau files land here
```
---
Setup
1. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
> **Note:** Prophet requires additional system packages on some platforms:
> ```bash
> pip install pystan==2.19.1.1
> pip install prophet
> ```
> **Note:** TensorFlow (for LSTM) requires Python 3.8–3.11.
3. Run the app
```bash
streamlit run app.py
```
The app opens at `http://localhost:8501`
---
How to use
Step 1 — Data
Use the built-in demo data (72 months of synthetic Indian OEM demand with monsoon, festive, fuel price signals), or
Upload your own CSV / Excel with at minimum `date` and `demand_units` columns.

Step 2 — Model selection (sidebar)
Choose from:
Model	Best for
XGBoost	Feature-rich, seasonal data
LightGBM	Large dealer network datasets
RF + GWO	Hyperparameter-optimized RF
SARIMA	Classic time-series baseline
Prophet	Holiday/festive decomposition
LSTM	Long-range sequential patterns
Enable Compare all models to run a full benchmark.

Step 3 — Run forecast
Click ▶ Run Forecast in the sidebar.

Step 4 — Analyse results
Forecast tab: actual vs predicted chart, feature importance, GWO convergence, Prophet decomposition
Performance tab: MAPE, MAE, RMSE, R², accuracy; model comparison table
Demand Trends tab: seasonal pattern, YoY growth, correlation heatmap
What-if tab: 5-driver scenario sliders, tornado chart, scenario saving

Step 5 — Export to Tableau
Go to Tableau Export tab and download:
`oem_forecast_tableau.csv` — flat file for Tableau Server/Online
`oem_forecast_tableau.xlsx` — multi-sheet workbook (Forecast_Results, Model_Metrics, Raw_Data, Scenario_Analysis)
In Tableau Desktop: Data → Connect → Microsoft Excel → select the `.xlsx` file.

---

Extending the app
Add a new model
Create `models/your_model.py` with a `run_your_model(train, test, feature_cols, **kwargs) -> dict` function.
The return dict must contain: `predicted`, `fitted`, `actual_test`, `actual_train`, `params`.
Add it to `MODEL_OPTIONS` in `app.py` and add a branch in `run_model()`.
Add external data sources
Add columns to your CSV (e.g., `google_trends_index`, `imd_rainfall_mm`) and they will automatically be picked up as features by `get_feature_columns()` and used by all tree-based models.

---

Tableau dashboard guide
After connecting the Excel export, build these sheets:
Forecast line chart — `date` dimension, `actual` + `predicted` measures, color by `model`
Seasonal heatmap — rows: `year`, columns: `month`, color: `demand_units`
Model accuracy bar — `Model` dimension, `Accuracy (%)` + `MAPE (%)` measures
What-if table — connect `Scenario_Analysis` sheet, parameter action to filter rows
Error distribution — `error_pct` histogram, filtered to test period
Combine all sheets into a Dashboard with filter actions linking `date` across views.
