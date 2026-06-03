"""
OEM Demand Forecasting Application
====================================
Streamlit app for selecting, training, and evaluating demand forecasting models.
Outputs Tableau-ready exports.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from data.loader import generate_demo_data, load_user_data, engineer_features, get_feature_columns, train_test_split_ts
from utils.metrics import compute_metrics, build_forecast_df, compare_models
from utils.tableau_export import export_to_csv, export_to_excel, build_scenario_export

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OEM Demand Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; font-size: 14px; }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
    .model-badge {
        display: inline-block; padding: 2px 10px;
        border-radius: 6px; font-size: 12px; font-weight: 500; margin-left: 6px;
    }
    .badge-best { background: #EAF3DE; color: #3B6D11; }
    .badge-fast { background: #E6F1FB; color: #185FA5; }
    .badge-ensemble { background: #EEEDFE; color: #534AB7; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    st.subheader("📂 Data source")
    data_source = st.radio("Choose data", ["Use demo data", "Upload CSV / Excel"], label_visibility="collapsed")
    uploaded_file = None
    if data_source == "Upload CSV / Excel":
        uploaded_file = st.file_uploader("Upload file", type=["csv", "xlsx"])
        st.caption("Required columns: `date`, `demand_units`")

    st.divider()
    st.subheader("🔬 Model selection")

    MODEL_OPTIONS = {
        "XGBoost (Gradient Boosting)": "xgboost",
        "LightGBM (Fast GBDT)": "lightgbm",
        "RF + GWO (Optimized RF)": "rf_gwo",
        "SARIMA / SARIMAX": "sarima",
        "Facebook Prophet": "prophet",
        "LSTM (Deep Learning)": "lstm",
    }

    selected_model_label = st.selectbox(
        "Primary model",
        list(MODEL_OPTIONS.keys()),
        index=0,
    )
    selected_model = MODEL_OPTIONS[selected_model_label]

    run_comparison = st.checkbox("Compare all models", value=False)
    if run_comparison:
        st.caption("⚠️ Runs all 6 models — may take 2–5 min")

    st.divider()
    st.subheader("📅 Forecast settings")
    test_months = st.slider("Test period (months)", 6, 24, 12)
    forecast_horizon = st.selectbox("Future horizon", ["3 months", "6 months", "12 months"], index=1)

    st.divider()
    st.subheader("🔧 Model hyperparameters")

    with st.expander("XGBoost / LightGBM"):
        n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
        max_depth = st.slider("max_depth", 2, 15, 5)
        learning_rate = st.select_slider("learning_rate", [0.01, 0.03, 0.05, 0.1, 0.15, 0.2], value=0.05)

    with st.expander("RF + GWO"):
        use_gwo = st.checkbox("Enable GWO optimization", value=True)
        n_wolves = st.slider("Wolves (GWO population)", 4, 20, 8)
        gwo_iterations = st.slider("GWO iterations", 5, 50, 15)

    with st.expander("SARIMA"):
        sarima_p = st.slider("p (AR order)", 0, 3, 1)
        sarima_d = st.slider("d (differencing)", 0, 2, 1)
        sarima_q = st.slider("q (MA order)", 0, 3, 1)

    with st.expander("LSTM"):
        lstm_window = st.slider("Lookback window (months)", 6, 24, 12)
        lstm_units = st.select_slider("LSTM units", [32, 64, 128, 256], value=64)
        lstm_epochs = st.slider("Max epochs", 30, 200, 80)

    st.divider()
    run_btn = st.button("▶ Run Forecast", type="primary", use_container_width=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(source: str, file=None) -> pd.DataFrame:
    if source == "demo":
        df = generate_demo_data(n_months=72)
    else:
        df = load_user_data(file)
    return engineer_features(df)


# ── Run a single model ────────────────────────────────────────────────────────
def run_model(model_id: str, train: pd.DataFrame, test: pd.DataFrame, feat_cols: list) -> dict:
    if model_id == "xgboost":
        from models.xgboost_model import run_xgboost
        return run_xgboost(train, test, feat_cols,
                           n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    elif model_id == "lightgbm":
        from models.lightgbm_model import run_lightgbm
        return run_lightgbm(train, test, feat_cols,
                            n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    elif model_id == "rf_gwo":
        from models.rf_gwo_model import run_rf_gwo
        progress_bar = st.progress(0, text="GWO optimizing RF hyperparameters...")
        def gwo_cb(t, total, score, params):
            progress_bar.progress(t / total, text=f"GWO iteration {t}/{total} | Best MAPE: {score:.2f}%")
        result = run_rf_gwo(train, test, feat_cols,
                            n_wolves=n_wolves, max_iter=gwo_iterations,
                            use_gwo=use_gwo, progress_callback=gwo_cb if use_gwo else None)
        progress_bar.empty()
        return result
    elif model_id == "sarima":
        from models.sarima_model import run_sarima
        exog = [c for c in ["fuel_price_inr", "monsoon_index", "interest_rate_pct"] if c in train.columns]
        return run_sarima(train, test,
                          order=(sarima_p, sarima_d, sarima_q),
                          seasonal_order=(1, 1, 1, 12),
                          exog_cols=exog if exog else None)
    elif model_id == "prophet":
        from models.prophet_model import run_prophet
        regs = [c for c in ["fuel_price_inr", "monsoon_index"] if c in train.columns]
        return run_prophet(train, test, add_regressors=regs if regs else None)
    elif model_id == "lstm":
        from models.lstm_model import run_lstm
        return run_lstm(train, test, feat_cols,
                        window=lstm_window, units=lstm_units, epochs=lstm_epochs)
    else:
        raise ValueError(f"Unknown model: {model_id}")


# ── Plot helpers ──────────────────────────────────────────────────────────────
COLORS = {
    "actual": "#2C2C2A",
    "predicted": "#378ADD",
    "train_fit": "#1D9E75",
    "error": "#D85A30",
    "festive": "rgba(186,117,23,0.12)",
    "monsoon": "rgba(29,158,117,0.10)",
}


def plot_forecast(train, test, result, model_name: str):
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.72, 0.28],
        subplot_titles=("Actual vs Forecast", "Forecast Error (%)"),
        vertical_spacing=0.08,
    )

    # Train fit
    fig.add_trace(go.Scatter(
        x=train["date"], y=result["fitted"],
        mode="lines", name="In-sample fit",
        line=dict(color=COLORS["train_fit"], width=1.5, dash="dot"),
    ), row=1, col=1)

    # Actuals
    fig.add_trace(go.Scatter(
        x=test["date"], y=result["actual_test"],
        mode="lines+markers", name="Actual demand",
        line=dict(color=COLORS["actual"], width=2),
        marker=dict(size=5),
    ), row=1, col=1)

    # Predicted
    fig.add_trace(go.Scatter(
        x=test["date"], y=result["predicted"],
        mode="lines+markers", name=f"{model_name} forecast",
        line=dict(color=COLORS["predicted"], width=2),
        marker=dict(size=5, symbol="diamond"),
    ), row=1, col=1)

    # Shade festive months
    for _, row in test.iterrows():
        if row["month"] in [9, 10, 11]:
            fig.add_vrect(
                x0=row["date"], x1=row["date"] + pd.DateOffset(months=1),
                fillcolor=COLORS["festive"], opacity=1, line_width=0,
                annotation_text="Festive" if row["month"] == 10 else "",
                annotation_font_size=9, row=1, col=1,
            )
        elif row["month"] in [6, 7, 8]:
            fig.add_vrect(
                x0=row["date"], x1=row["date"] + pd.DateOffset(months=1),
                fillcolor=COLORS["monsoon"], opacity=1, line_width=0,
                annotation_text="Monsoon" if row["month"] == 7 else "",
                annotation_font_size=9, row=1, col=1,
            )

    # Error %
    error_pct = ((result["actual_test"] - result["predicted"]) / (result["actual_test"] + 1e-9)) * 100
    colors_err = ["#E24B4A" if e > 0 else "#1D9E75" for e in error_pct]

    fig.add_trace(go.Bar(
        x=test["date"], y=error_pct,
        name="Error %",
        marker_color=colors_err,
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#888", row=2, col=1)

    fig.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=12),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", gridwidth=0.5)
    return fig


def plot_feature_importance(importances: dict, top_n: int = 12):
    items = list(importances.items())[:top_n]
    labels = [i[0].replace("_", " ") for i in items]
    values = [i[1] for i in items]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=labels[::-1],
        orientation="h",
        marker_color="#378ADD",
    ))
    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Importance", yaxis_title="",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=12),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=False)
    return fig


def plot_model_comparison(comp_df: pd.DataFrame):
    fig = px.bar(
        comp_df.sort_values("MAPE (%)"),
        x="MAPE (%)", y="Model",
        orientation="h",
        color="Accuracy (%)",
        color_continuous_scale=["#E24B4A", "#EF9F27", "#1D9E75"],
        text="Accuracy (%)",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        coloraxis_showscale=False,
        font=dict(size=12),
    )
    return fig


def plot_seasonal_decomposition(df: pd.DataFrame):
    monthly = df.groupby("month")["demand_units"].mean().reset_index()
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly["month_name"] = monthly["month"].apply(lambda m: month_names[m - 1])

    colors = []
    for m in monthly["month"]:
        if m in [9, 10, 11]: colors.append("#BA7517")
        elif m in [6, 7, 8]: colors.append("#1D9E75")
        elif m in [2, 3]:    colors.append("#7F77DD")
        else:                colors.append("#888780")

    fig = go.Figure(go.Bar(
        x=monthly["month_name"], y=monthly["demand_units"],
        marker_color=colors,
        text=monthly["demand_units"].apply(lambda v: f"{v/1000:.0f}K"),
        textposition="outside",
    ))
    fig.add_annotation(
        x=9.5, y=monthly["demand_units"].max() * 1.05,
        text="Festive surge", showarrow=False,
        font=dict(color="#BA7517", size=11),
    )
    fig.add_annotation(
        x=6.5, y=monthly["demand_units"].max() * 0.88,
        text="Monsoon dip", showarrow=False,
        font=dict(color="#1D9E75", size=11),
    )
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Avg units", xaxis_title="",
        showlegend=False, font=dict(size=12),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def plot_gwo_convergence(history: list):
    fig = go.Figure(go.Scatter(
        x=list(range(len(history))), y=history,
        mode="lines+markers",
        line=dict(color="#7F77DD", width=2),
        marker=dict(size=5),
        fill="tozeroy", fillcolor="rgba(127,119,221,0.08)",
    ))
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="GWO iteration", yaxis_title="Best MAPE (%)",
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(size=12),
    )
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


# ── What-if scenario engine ───────────────────────────────────────────────────
def compute_scenario(base_demand: float, monsoon: float, fuel: float, rate: float, festive: float, export_shock: float) -> float:
    d = base_demand
    d += d * (monsoon * -0.003)
    d += d * (fuel   * -0.004)
    d += d * (rate   * -0.0002)
    d += d * (festive *  0.005)
    d += d * (export_shock * 0.004)
    return max(0, d)


# ── Main layout ───────────────────────────────────────────────────────────────
st.title("📊 OEM Demand Forecasting Platform")
st.caption("Predictive demand planning · Seasonal intelligence · Tableau-ready export")

# Load data
with st.spinner("Loading data..."):
    try:
        if data_source == "Upload CSV / Excel" and uploaded_file is not None:
            df = load_data("upload", uploaded_file)
        else:
            df = load_data("demo")
    except Exception as e:
        st.error(f"Data load error: {e}")
        st.stop()

feat_cols = get_feature_columns(df)
train_df, test_df = train_test_split_ts(df, test_months=test_months)

# KPI header
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total data points", f"{len(df):,}", f"{len(df) // 12} years")
col2.metric("Training months", f"{len(train_df)}", f"{len(train_df) / 12:.1f} yr window")
col3.metric("Test months", f"{len(test_df)}", "hold-out period")
col4.metric("Feature columns", f"{len(feat_cols)}", "engineered")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 Forecast", "📈 Performance", "🌦 Demand Trends", "⚡ What-if Analysis", "📤 Tableau Export"
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — FORECAST
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    if run_btn or "last_result" in st.session_state:
        if run_btn:
            with st.spinner(f"Training {selected_model_label}..."):
                try:
                    result = run_model(selected_model, train_df, test_df, feat_cols)
                    st.session_state["last_result"] = result
                    st.session_state["last_model"] = selected_model_label
                    st.session_state["last_model_id"] = selected_model
                    metrics = compute_metrics(result["actual_test"], result["predicted"])
                    st.session_state["last_metrics"] = metrics
                    forecast_df_out = build_forecast_df(
                        test_df["date"], result["actual_test"], result["predicted"], selected_model_label
                    )
                    st.session_state["forecast_df_out"] = forecast_df_out
                except Exception as e:
                    st.error(f"Model error: {e}")
                    import traceback; st.code(traceback.format_exc())
                    st.stop()

        result = st.session_state["last_result"]
        model_name = st.session_state["last_model"]
        metrics = st.session_state["last_metrics"]

        # Metric strip
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{metrics['Accuracy (%)']:.1f}%")
        m2.metric("MAPE", f"{metrics['MAPE (%)']:.2f}%")
        m3.metric("MAE", f"{metrics['MAE']:,.0f}")
        m4.metric("RMSE", f"{metrics['RMSE']:,.0f}")
        m5.metric("R²", f"{metrics['R² Score']:.4f}")

        st.plotly_chart(plot_forecast(train_df, test_df, result, model_name), use_container_width=True)

        # Feature importance (tree models)
        if "feature_importances" in result:
            st.subheader("Feature importance")
            st.plotly_chart(plot_feature_importance(result["feature_importances"]), use_container_width=True)

        # GWO convergence
        if result.get("gwo_results") and result["gwo_results"].get("convergence"):
            st.subheader("GWO convergence (hyperparameter optimization)")
            gcol1, gcol2 = st.columns([2, 1])
            with gcol1:
                st.plotly_chart(plot_gwo_convergence(result["gwo_results"]["convergence"]), use_container_width=True)
            with gcol2:
                st.markdown("**Best hyperparameters found**")
                bp = result["gwo_results"]["best_params"]
                for k, v in bp.items():
                    st.code(f"{k}: {v}")

        # Prophet components
        if "components" in result:
            st.subheader("Prophet decomposition — trend & seasonality")
            comp = result["forecast_df"]
            fig_comp = make_subplots(rows=1, cols=2, subplot_titles=("Trend", "Yearly seasonality"))
            fig_comp.add_trace(go.Scatter(x=comp["ds"], y=comp["trend"], mode="lines", line=dict(color="#378ADD")), row=1, col=1)
            if "yearly" in comp.columns:
                fig_comp.add_trace(go.Scatter(x=comp["ds"], y=comp["yearly"], mode="lines", line=dict(color="#1D9E75")), row=1, col=2)
            fig_comp.update_layout(height=300, showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_comp, use_container_width=True)

    else:
        st.info("Configure settings in the sidebar and click **▶ Run Forecast** to begin.")
        st.image("https://via.placeholder.com/900x300?text=Configure+model+%E2%86%92+Run+Forecast",
                 use_column_width=True) if False else None


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE / MODEL COMPARISON
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    if run_comparison and run_btn:
        all_results = {}
        all_metrics = {}

        progress = st.progress(0)
        models_to_run = list(MODEL_OPTIONS.items())

        for idx, (label, mid) in enumerate(models_to_run):
            progress.progress((idx) / len(models_to_run), text=f"Running {label}...")
            try:
                r = run_model(mid, train_df, test_df, feat_cols)
                m = compute_metrics(r["actual_test"], r["predicted"])
                all_results[label] = r
                all_metrics[label] = m
            except Exception as e:
                st.warning(f"{label} failed: {e}")
        progress.empty()

        st.session_state["all_metrics"] = all_metrics
        comp_df = compare_models(all_metrics)
        st.session_state["comp_df"] = comp_df

    if "comp_df" in st.session_state:
        comp_df = st.session_state["comp_df"]
        st.subheader("Model comparison")
        st.plotly_chart(plot_model_comparison(comp_df), use_container_width=True)

        st.dataframe(
            comp_df.style.background_gradient(subset=["Accuracy (%)"], cmap="Greens")
                         .background_gradient(subset=["MAPE (%)"], cmap="Reds_r")
                         .format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}"}),
            use_container_width=True,
        )
    elif "last_metrics" in st.session_state:
        st.subheader(f"Metrics — {st.session_state['last_model']}")
        m = st.session_state["last_metrics"]
        mdf = pd.DataFrame([m])
        st.dataframe(mdf, use_container_width=True)
        st.info("Enable **Compare all models** in the sidebar to run a full benchmark.")
    else:
        st.info("Run a forecast first to see performance metrics.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEMAND TRENDS
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Seasonal demand pattern (monthly average)")
    st.plotly_chart(plot_seasonal_decomposition(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Year-on-year growth")
        yearly = df.groupby("year")["demand_units"].sum().reset_index()
        yearly["yoy_pct"] = yearly["demand_units"].pct_change() * 100
        fig_yoy = go.Figure(go.Bar(
            x=yearly["year"], y=yearly["demand_units"],
            marker_color=["#378ADD"] * len(yearly),
            text=yearly["yoy_pct"].apply(lambda v: f"{v:+.1f}%" if pd.notna(v) else ""),
            textposition="outside",
        ))
        fig_yoy.update_layout(height=290, plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(l=0, r=0, t=10, b=0), font=dict(size=12))
        st.plotly_chart(fig_yoy, use_container_width=True)

    with c2:
        st.subheader("Demand vs fuel price correlation")
        if "fuel_price_inr" in df.columns:
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=df["fuel_price_inr"], y=df["demand_units"],
                mode="markers",
                marker=dict(color=df["month"], colorscale="Teal", size=7, opacity=0.7),
                text=df["date"].dt.strftime("%b %Y"),
            ))
            fig_corr.update_layout(
                height=290, xaxis_title="Fuel price (INR/L)", yaxis_title="Demand units",
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=0, r=0, t=10, b=0), font=dict(size=12),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Key demand drivers — correlation heatmap")
    num_cols = [c for c in ["demand_units", "fuel_price_inr", "monsoon_index", "interest_rate_pct",
                             "ev_share_pct", "gdp_growth_pct", "dealer_inventory_index"] if c in df.columns]
    corr = df[num_cols].corr()
    fig_heat = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        text_auto=".2f",
        aspect="auto",
    )
    fig_heat.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), font=dict(size=11))
    st.plotly_chart(fig_heat, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — WHAT-IF ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Scenario planning")

    base = float(df["demand_units"].tail(12).mean())

    sc1, sc2 = st.columns(2)
    with sc1:
        s_monsoon  = st.slider("Monsoon intensity Δ (%)",  -30, 30, 0, help="+ = heavier monsoon → lower demand")
        s_fuel     = st.slider("Fuel price Δ (%)",         -20, 20, 0)
        s_rate     = st.slider("Interest rate Δ (bps)",    -100, 100, 0, step=25)
    with sc2:
        s_festive  = st.slider("Festive spend index Δ (%)", -20, 40, 0)
        s_export   = st.slider("Export order shock (%)",    -30, 30, 0)

    adj_demand = compute_scenario(base, s_monsoon, s_fuel, s_rate / 100, s_festive, s_export)
    delta = ((adj_demand - base) / base) * 100

    r1, r2, r3 = st.columns(3)
    r1.metric("Baseline demand (avg)", f"{base:,.0f} units/mo")
    r2.metric("Adjusted demand", f"{adj_demand:,.0f} units/mo", f"{delta:+.1f}%")
    r3.metric("Annual projection", f"{adj_demand * 12:,.0f} units")

    # Sensitivity tornado chart
    st.subheader("Sensitivity analysis — driver impact")
    drivers = {
        "Monsoon intensity": compute_scenario(base, 20, 0, 0, 0, 0) - base,
        "Fuel price (+20%)": compute_scenario(base, 0, 20, 0, 0, 0) - base,
        "Interest rate (+100bps)": compute_scenario(base, 0, 0, 100/100, 0, 0) - base,
        "Festive spend (+40%)": compute_scenario(base, 0, 0, 0, 40, 0) - base,
        "Export shock (+30%)": compute_scenario(base, 0, 0, 0, 0, 30) - base,
    }
    driver_df = pd.DataFrame({"Driver": list(drivers.keys()), "Impact": list(drivers.values())})
    driver_df = driver_df.sort_values("Impact")
    bar_colors = ["#E24B4A" if v < 0 else "#1D9E75" for v in driver_df["Impact"]]
    fig_tornado = go.Figure(go.Bar(
        x=driver_df["Impact"], y=driver_df["Driver"],
        orientation="h", marker_color=bar_colors,
        text=driver_df["Impact"].apply(lambda v: f"{v:+,.0f}"),
        textposition="outside",
    ))
    fig_tornado.add_vline(x=0, line_width=1.5, line_color="#333")
    fig_tornado.update_layout(
        height=300, xaxis_title="Demand impact (units/month)",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=0, r=0, t=10, b=0), showlegend=False, font=dict(size=12),
    )
    st.plotly_chart(fig_tornado, use_container_width=True)

    # Save scenario
    if st.button("💾 Save this scenario"):
        scenario_record = {
            "scenario": f"Scenario_{len(st.session_state.get('scenarios', [])) + 1}",
            "monsoon_delta": s_monsoon,
            "fuel_delta": s_fuel,
            "rate_delta_bps": s_rate,
            "festive_delta": s_festive,
            "export_delta": s_export,
            "adjusted_demand": round(adj_demand, 0),
            "change_pct": round(delta, 2),
        }
        if "scenarios" not in st.session_state:
            st.session_state["scenarios"] = []
        st.session_state["scenarios"].append(scenario_record)
        st.success("Scenario saved for Tableau export.")

    if "scenarios" in st.session_state and st.session_state["scenarios"]:
        st.subheader("Saved scenarios")
        st.dataframe(pd.DataFrame(st.session_state["scenarios"]), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — TABLEAU EXPORT
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Export to Tableau")
    st.markdown("""
Connect your Tableau Desktop to the exported `.xlsx` file via **Data → Connect → Microsoft Excel**.
Four ready-made sheets are included:

| Sheet | Use in Tableau |
|---|---|
| `Forecast_Results` | Line chart: actual vs predicted; error % by month |
| `Model_Metrics` | Bar chart: MAPE / accuracy comparison across models |
| `Raw_Data` | Correlation scatter, driver analysis |
| `Scenario_Analysis` | What-if parameter vs demand output |
""")

    forecast_ready = "forecast_df_out" in st.session_state
    metrics_ready = "last_metrics" in st.session_state

    if not forecast_ready:
        st.warning("Run at least one forecast to enable export.")
    else:
        forecast_df_out = st.session_state["forecast_df_out"]
        metrics_df_out = pd.DataFrame([st.session_state["last_metrics"]])
        metrics_df_out.insert(0, "Model", st.session_state["last_model"])

        if "comp_df" in st.session_state:
            metrics_df_out = st.session_state["comp_df"]

        scenario_df_out = None
        if "scenarios" in st.session_state and st.session_state["scenarios"]:
            scenario_df_out = build_scenario_export(st.session_state["scenarios"])

        col_a, col_b = st.columns(2)

        with col_a:
            csv_bytes = export_to_csv(forecast_df_out, metrics_df_out, scenario_df_out)
            st.download_button(
                "⬇ Download Forecast CSV",
                data=csv_bytes,
                file_name="oem_forecast_tableau.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_b:
            excel_bytes = export_to_excel(forecast_df_out, metrics_df_out, df, scenario_df_out)
            st.download_button(
                "⬇ Download Excel Workbook (Tableau-ready)",
                data=excel_bytes,
                file_name="oem_forecast_tableau.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        st.subheader("Preview — forecast results")
        st.dataframe(forecast_df_out.head(20), use_container_width=True)

        st.subheader("Tableau dashboard suggestions")
        st.markdown("""
**Sheet 1 — Demand forecast line chart**
- Dimensions: `date` (Month/Year)
- Measures: `actual`, `predicted`
- Color: `model`
- Add calculated field: `Error % = (actual - predicted) / actual * 100`

**Sheet 2 — Model accuracy bar chart**
- Dimension: `Model`
- Measure: `Accuracy (%)`, `MAPE (%)`

**Sheet 3 — Seasonal heatmap**
- Rows: `year`, Columns: `month`
- Color: `demand_units`

**Sheet 4 — What-if scenario table**
- Connect `Scenario_Analysis` sheet
- Parameter actions to switch between scenarios
""")
