"""
XGBoost model for Wheels India OEM Demand Forecasting
======================================================
Adapted for WheelsIndia_72Month_POC_Dataset.csv

Column mapping from original → this dataset:
  Volume              → Actual_Volume
  Month (string)      → Month (int 1-12, already numeric)
  Fiscal_Year / FY    → Fiscal_Year (already present)
  Calendar_Year / CY  → Calendar_Year (already present)
  Wheel Code Number   → WheelCode

Extra columns in this dataset used as direct features:
  Working_Days, Holiday_Count, Festive_Flag, Diwali_Flag,
  Monsoon_Flag, Quarter_End_Flag, Year_End_Flag,
  Industry_Growth_Pct, Segment_Growth_Pct, Monsoon_Index,
  Rainfall_Index, Export_Demand_Index, Fuel_Price_Index,
  Economic_Growth_Index, Market_Sentiment_Index,
  Forecast_Confidence, Market_Risk_Score, Forecast_Gap_Pct,
  Product_Family, Region

All 9 original bug-fixes are preserved:
  BUG 1  Cross-series lag leakage     → groupby(['Customer','WheelCode'])
  BUG 2  Raw target skew              → log1p / expm1
  BUG 3  No Indian holiday features   → build_holiday_features()
  BUG 4  No monsoon intensity         → MONSOON_INTENSITY_INDEX
  BUG 5  No kharif/rabi cycle         → agricultural flags
  BUG 6  No dealer inventory push     → is_qtr_end, is_month_end_push
  BUG 7  No identity features         → LabelEncoder per entity
  BUG 8  COVID FY21 contamination     → is_covid derived from Date
  BUG 9  Zero-volume MAPE → inf       → filter Volume == 0 from eval
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_percentage_error
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Indian Holiday Calendar  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
INDIAN_HOLIDAY_CALENDAR = {
    2018: {'diwali': 11, 'navratri': 10, 'akshaya': 4,  'onam': 8,  'gudi': 3},
    2019: {'diwali': 10, 'navratri': 10, 'akshaya': 5,  'onam': 9,  'gudi': 4},
    2020: {'diwali': 11, 'navratri': 10, 'akshaya': 4,  'onam': 8,  'gudi': 3},
    2021: {'diwali': 11, 'navratri': 10, 'akshaya': 5,  'onam': 8,  'gudi': 4},
    2022: {'diwali': 10, 'navratri': 10, 'akshaya': 5,  'onam': 8,  'gudi': 4},
    2023: {'diwali': 11, 'navratri': 10, 'akshaya': 4,  'onam': 8,  'gudi': 3},
    2024: {'diwali': 11, 'navratri': 10, 'akshaya': 5,  'onam': 9,  'gudi': 4},
    2025: {'diwali': 10, 'navratri': 10, 'akshaya': 4,  'onam': 8,  'gudi': 3},
    2026: {'diwali': 10, 'navratri': 10, 'akshaya': 5,  'onam': 9,  'gudi': 4},
    2027: {'diwali': 11, 'navratri': 10, 'akshaya': 4,  'onam': 8,  'gudi': 3},
}

MONSOON_INTENSITY_INDEX = {
    2018: 0.91,
    2019: 1.10,
    2020: 1.09,
    2021: 0.99,
    2022: 1.08,
    2023: 0.94,
    2024: 1.07,
    2025: 1.05,
    2026: 1.00,
}

# ─────────────────────────────────────────────────────────────────────────────
# Columns to exclude from feature matrix
# Updated for this dataset's column names
# ─────────────────────────────────────────────────────────────────────────────
_EXCLUDE = {
    # Identity / target / index columns — never feed as features
    'Date',
    'Calendar_Year',    # represented by time_idx
    'Fiscal_Year',      # represented by time_idx
    'Month',            # represented by month_sin / month_cos
    'Quarter',          # represented by quarter (derived)
    'Customer',
    'Segment',
    'WheelCode',
    'Product_Family',   # label-encoded separately as product_enc
    'Region',           # label-encoded separately as region_enc
    'Actual_Volume',    # TARGET
    'is_covid',
    # Customer forecast is a leaky feature in a real pipeline
    # (it uses information available at forecast time, not at model-train time).
    # Keep it as an optional external signal — comment out to disable.
    # 'Customer_Forecast',
    # 'Forecast_Gap_Pct',
    # 'Forecast_Confidence',
}


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Data loading
# Adapted: Month is already int; target column is Actual_Volume
# ─────────────────────────────────────────────────────────────────────────────

def load_data(filepath):
    """
    Load WheelsIndia dataset
    """

    if filepath.lower().endswith(".csv"):
        df = pd.read_csv(filepath)

    elif filepath.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(filepath)

    else:
        raise ValueError(
            f"Unsupported file type: {filepath}"
        )

    df.columns = [str(c).strip() for c in df.columns]

    # Rename target
    df = df.rename(
        columns={
            "Actual_Volume": "Volume"
        }
    )

    df["Date"] = pd.to_datetime(df["Date"])

    df["month_num"] = df["Month"].astype(int)

    df["cal_year"] = df["Date"].dt.year

    df = df.sort_values(
        ["Customer", "WheelCode", "Date"]
    ).reset_index(drop=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Indian holiday & seasonality features  (unchanged logic)
# The dataset already has Festive_Flag, Diwali_Flag, Monsoon_Flag — we ADD
# the richer 27-feature set on top of those coarse flags.
# ─────────────────────────────────────────────────────────────────────────────

def build_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 27 holiday, seasonality, and dealer-cycle features.
    The dataset's built-in Festive_Flag / Diwali_Flag / Monsoon_Flag are
    coarse (Oct+Nov = festive, Jun-Sep = monsoon). This function adds:
      • Exact Diwali month per year (Diwali shifts Oct ↔ Nov annually)
      • Pre/post Diwali months, Navratri, festive window
      • Regional festivals: Akshaya Tritiya, Onam, Gudi Padwa, Pongal, Baisakhi
      • Fiscal calendar: FY-end (Feb-Mar), FY-start (Apr), post-budget (Mar-Apr)
      • Monsoon intensity from IMD index with heavy/deficit flags
      • Kharif / rabi agricultural cycle (tractor demand leading indicators)
      • Dealer inventory push: quarter-end, month-end push
    """
    calendar = INDIAN_HOLIDAY_CALENDAR
    monsoon  = MONSOON_INTENSITY_INDEX

    feat_rows = []
    for _, row in df.iterrows():
        yr  = int(row['cal_year'])
        mon = int(row['month_num'])
        h   = calendar.get(yr, calendar[2025])
        mi  = monsoon.get(yr, 1.0)

        dw = h['diwali']

        # ── Diwali family ─────────────────────────────────────────────────
        is_diwali       = int(mon == dw)
        is_pre_diwali   = int(mon == (dw - 1 if dw > 1 else 12))
        is_post_diwali  = int(mon == (dw + 1 if dw < 12 else 1))
        is_navratri     = int(mon == h['navratri'])
        is_prefestive   = int(mon == (dw - 2 if dw > 2 else 12 - (2 - dw)))
        is_fest_window  = int(mon in [
            (dw - 1) % 12 + 1 if dw > 1 else 12,
            dw,
            (dw % 12) + 1,
        ])

        # ── Regional festivals ────────────────────────────────────────────
        is_akshaya      = int(mon == h['akshaya'])
        is_onam         = int(mon == h['onam'])
        is_gudi         = int(mon == h['gudi'])
        is_pongal       = int(mon == 1)
        is_baisakhi     = int(mon == 4)
        is_independence = int(mon == 8)

        # ── Fiscal calendar ───────────────────────────────────────────────
        is_fy_end       = int(mon in [2, 3])
        is_fy_start     = int(mon == 4)
        is_post_budget  = int(mon in [3, 4])

        # ── Monsoon (richer than dataset's Monsoon_Flag) ──────────────────
        is_monsoon_rich = int(mon in [6, 7, 8])
        is_heavy        = int(is_monsoon_rich == 1 and mi > 1.05)
        is_deficit      = int(is_monsoon_rich == 1 and mi < 0.95)
        is_post_monsoon = int(mon == 9)
        monsoon_x_int   = is_monsoon_rich * mi

        # ── Agricultural cycles ───────────────────────────────────────────
        is_kharif_sow   = int(mon in [4, 5, 6])
        is_kharif_harv  = int(mon in [10, 11])
        is_rabi_sow     = int(mon in [10, 11])
        is_rabi_harv    = int(mon in [3, 4])

        # ── Dealer push cycles ────────────────────────────────────────────
        is_qtr_end      = int(mon in [6, 9, 12, 3])
        is_month_push   = int(mon in [3, 9])

        feat_rows.append({
            'is_diwali_month':      is_diwali,
            'is_pre_diwali':        is_pre_diwali,
            'is_post_diwali':       is_post_diwali,
            'is_navratri_month':    is_navratri,
            'is_prefestive_stock':  is_prefestive,
            'is_festive_window':    is_fest_window,
            'is_akshaya_tritiya':   is_akshaya,
            'is_onam':              is_onam,
            'is_gudi_padwa':        is_gudi,
            'is_pongal':            is_pongal,
            'is_baisakhi':          is_baisakhi,
            'is_independence_day':  is_independence,
            'is_fy_end':            is_fy_end,
            'is_fy_start':          is_fy_start,
            'is_post_budget':       is_post_budget,
            'is_monsoon_rich':      is_monsoon_rich,
            'monsoon_intensity':    mi,
            'is_heavy_monsoon':     is_heavy,
            'is_deficit_monsoon':   is_deficit,
            'is_post_monsoon':      is_post_monsoon,
            'monsoon_x_intensity':  monsoon_x_int,
            'is_kharif_sowing':     is_kharif_sow,
            'is_kharif_harvest':    is_kharif_harv,
            'is_rabi_sowing':       is_rabi_sow,
            'is_rabi_harvest':      is_rabi_harv,
            'is_qtr_end':           is_qtr_end,
            'is_month_end_push':    is_month_push,
        })

    holiday_df = pd.DataFrame(feat_rows)
    return pd.concat([df.reset_index(drop=True),
                      holiday_df.reset_index(drop=True)], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Per-series lag & rolling features  (BUG 1 fix preserved)
# Additional dataset columns (indices, flags) are passed through untouched.
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute lag / rolling features strictly within each Customer × WheelCode
    group (BUG 1 fix).  The extra columns from the POC dataset
    (Working_Days, Industry_Growth_Pct, Fuel_Price_Index, etc.) are already
    present in df and will be picked up automatically by feature selection.

    Additions vs original:
      • Product_Family and Region label-encoded (product_enc, region_enc)
      • time_idx anchored to 2020 (dataset starts 2020-01-01)
      • is_covid derived from Date (2020-04-01 → 2021-03-31)
    """
    series_parts = []

    for (customer, sku), grp in df.groupby(['Customer', 'WheelCode']):
        g = grp.sort_values('Date').copy()

        # ── Per-series lag features (BUG 1: within-group only) ────────────
        for lag in [1, 2, 3, 6, 12]:
            g[f'lag_{lag}'] = g['Volume'].shift(lag)

        # ── Rolling statistics ────────────────────────────────────────────
        for w in [3, 6, 12]:
            shifted = g['Volume'].shift(1)
            g[f'roll_mean_{w}'] = shifted.rolling(w).mean()
            g[f'roll_std_{w}']  = shifted.rolling(w).std()

        g['roll_max_3'] = g['Volume'].shift(1).rolling(3).max()
        g['roll_min_3'] = g['Volume'].shift(1).rolling(3).min()

        # ── Year-on-year growth ───────────────────────────────────────────
        g['yoy_growth'] = g['Volume'].pct_change(12).clip(-2, 10) * 100

        # ── 6-month linear trend slope ────────────────────────────────────
        g['trend_6'] = (
            g['Volume'].shift(1)
            .rolling(6)
            .apply(lambda x: np.polyfit(range(6), x, 1)[0], raw=True)
        )

        # ── SKU-level expanding mean / std ────────────────────────────────
        g['sku_hist_mean'] = g['Volume'].expanding().mean().shift(1)
        g['sku_hist_std']  = g['Volume'].expanding().std().shift(1)

        # ── Demand deviation from 6-month rolling mean ────────────────────
        g['demand_deviation'] = (
            g['Volume'].shift(1) - g['Volume'].shift(1).rolling(6).mean()
        )

        series_parts.append(g)

    df_eng = pd.concat(series_parts, ignore_index=True)
    df_eng = df_eng.replace([np.inf, -np.inf], np.nan)

    # ── Calendar features ─────────────────────────────────────────────────
    df_eng['month_sin'] = np.sin(2 * np.pi * df_eng['month_num'] / 12)
    df_eng['month_cos'] = np.cos(2 * np.pi * df_eng['month_num'] / 12)
    df_eng['quarter']   = ((df_eng['month_num'] - 1) // 3 + 1)
    # Anchor to 2020 (dataset start) instead of 2018
    df_eng['time_idx']  = (df_eng['Date'].dt.year - 2020) * 12 + df_eng['month_num']

    # ── COVID disruption flag (BUG 8 fix) ────────────────────────────────
    df_eng['is_covid'] = df_eng['Date'].between(
        '2020-04-01', '2021-03-31'
    ).astype(int)

    # ── Identity encodings (BUG 7 fix) ───────────────────────────────────
    df_eng['customer_enc']  = LabelEncoder().fit_transform(df_eng['Customer'])
    df_eng['segment_enc']   = LabelEncoder().fit_transform(df_eng['Segment'])
    df_eng['sku_enc']        = LabelEncoder().fit_transform(df_eng['WheelCode'])

    # ── New entity encodings (columns specific to this dataset) ──────────
    df_eng['product_enc']   = LabelEncoder().fit_transform(df_eng['Product_Family'])
    df_eng['region_enc']    = LabelEncoder().fit_transform(df_eng['Region'])

    return df_eng.dropna().reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Train / test split  (logic unchanged; column name fix only)
# ─────────────────────────────────────────────────────────────────────────────

def make_train_test(df: pd.DataFrame, test_start: str = '2025-04-01'):
    """
    Temporal split with three data-quality filters on training data.
    Same logic as original; 'Volume' is used (renamed from Actual_Volume
    in load_data so downstream code is identical).

    Dataset spans Jan 2020 – Dec 2025.
    Default test_start = '2025-04-01' → last 9 months as holdout.
    Adjust test_start earlier for a larger test window if needed.
    """
    train = df[
        (df['Date'] < test_start) &
        (df['is_covid'] == 0) &
        (df['Volume'] > 0)
    ].copy()

    test_full = df[df['Date'] >= test_start].copy()
    test_eval = test_full[test_full['Volume'] > 0].copy()

    return train, test_eval, test_full


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — XGBoost model  (unchanged except docstring update)
# ─────────────────────────────────────────────────────────────────────────────

def run_xgboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list = None,
    n_estimators: int           = 1000,
    max_depth: int              = 7,
    learning_rate: float        = 0.03,
    subsample: float            = 0.8,
    colsample_bytree: float     = 0.8,
    min_child_weight: int       = 5,
    reg_alpha: float            = 0.1,
    reg_lambda: float           = 1.0,
    early_stopping_rounds: int  = 50,
    model_save_path: str        = None,
) -> dict:
    """
    Fit XGBoost on log1p(Volume); inverse-transform predictions.
    Feature set includes both the engineered lag/holiday features AND
    the dataset's native columns (indices, flags, Working_Days, etc.)
    that survived the _EXCLUDE filter.
    """
    if feature_cols is None:
        feature_cols = [c for c in train.columns if c not in _EXCLUDE]

    X_train = train[feature_cols]
    X_test  = test[feature_cols]

    # BUG 2 fix: log-transform skewed target
    y_train = np.log1p(train['Volume'].values)
    y_test  = np.log1p(test['Volume'].values)

    model = xgb.XGBRegressor(
        n_estimators           = n_estimators,
        max_depth              = max_depth,
        learning_rate          = learning_rate,
        subsample              = subsample,
        colsample_bytree       = colsample_bytree,
        min_child_weight       = min_child_weight,
        reg_alpha              = reg_alpha,
        reg_lambda             = reg_lambda,
        objective              = 'reg:squarederror',
        random_state           = 42,
        n_jobs                 = -1,
        verbosity              = 0,
        early_stopping_rounds  = early_stopping_rounds,
    )

    model.fit(
        X_train, y_train,
        eval_set = [(X_test, y_test)],
        verbose  = False,
    )

    # Inverse-transform back to volume scale
    predicted = np.maximum(np.expm1(model.predict(X_test)), 0)
    fitted    = np.maximum(np.expm1(model.predict(X_train)), 0)

    # ── Overall metrics ───────────────────────────────────────────────────
    mape     = mean_absolute_percentage_error(test['Volume'].values, predicted) * 100
    accuracy = max(0.0, 100 - mape)

    # ── Per-segment accuracy ──────────────────────────────────────────────
    seg_metrics = {}
    test_out = test.copy()
    test_out['_pred'] = predicted
    for seg, grp in test_out.groupby('Segment'):
        m = mean_absolute_percentage_error(
            grp['Volume'].values, grp['_pred'].values
        ) * 100
        seg_metrics[seg] = {
            'MAPE (%)':     round(m, 2),
            'Accuracy (%)': round(max(0, 100 - m), 2),
            'n_rows':       len(grp),
        }

    # ── Feature importances (gain) ────────────────────────────────────────
    importances = dict(zip(feature_cols, model.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump({
            'model':        model,
            'feature_cols': feature_cols,
            'params': {'log_transform': True, 'covid_excluded': True},
        }, model_save_path)
        print(f"Model saved → {model_save_path}")

    return {
        'predicted':           predicted,
        'fitted':              fitted,
        'actual_test':         test['Volume'].values,
        'actual_train':        train['Volume'].values,
        'model_obj':           model,
        'feature_importances': importances,
        'metrics': {
            'MAPE (%)':       round(mape, 2),
            'Accuracy (%)':   round(accuracy, 2),
            'Best iteration': model.best_iteration,
            'n_features':     len(feature_cols),
        },
        'segment_metrics': seg_metrics,
        'params': {
            'n_estimators':          n_estimators,
            'max_depth':             max_depth,
            'learning_rate':         learning_rate,
            'subsample':             subsample,
            'colsample_bytree':      colsample_bytree,
            'min_child_weight':      min_child_weight,
            'reg_alpha':             reg_alpha,
            'reg_lambda':            reg_lambda,
            'early_stopping_rounds': early_stopping_rounds,
            'log_transform':         True,
            'covid_rows_excluded':   True,
            'zero_rows_excluded':    True,
            'holiday_features':      True,
            'monsoon_intensity':     True,
            'agricultural_cycle':    True,
            'dealer_inventory_proxy':True,
            'product_family_encoded':True,
            'region_encoded':        True,
            'dataset_native_features':True,
            'features':              feature_cols,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Standalone run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    path = "/kaggle/input/datasets/ritikrai09/72-months-data/WheelsIndia_72Month_POC_Dataset.csv"

    print("PATH =", path)
    
    
    df_raw  = load_data(path)
    print(f"     Shape: {df_raw.shape}  |  Date range: {df_raw['Date'].min().date()} → {df_raw['Date'].max().date()}")
    print(f"     Customers: {df_raw['Customer'].nunique()}  |  SKUs: {df_raw['WheelCode'].nunique()}  |  Segments: {df_raw['Segment'].nunique()}")

    print("\n2/4  Adding Indian holiday & seasonality features ...")
    df_hol  = build_holiday_features(df_raw)

    print("3/4  Engineering per-series lag & rolling features ...")
    df_feat = engineer_features(df_hol)
    print(f"     Post-engineering shape (after dropna): {df_feat.shape}")

    print("\n4/4  Training XGBoost ...")
    train_df, test_df, _ = make_train_test(df_feat)
    feat_cols = [c for c in df_feat.columns if c not in _EXCLUDE]

    print(f"\n     Train rows  : {len(train_df):,}")
    print(f"     Test rows   : {len(test_df):,}")
    print(f"     Features    : {len(feat_cols)}")
    print(f"       ├─ lag / rolling        : {sum(1 for f in feat_cols if f.startswith(('lag_','roll_','trend','yoy','sku_hist','demand_dev')))}")
    print(f"       ├─ holiday flags        : {sum(1 for f in feat_cols if f.startswith('is_') and f not in ['is_covid'])}")
    print(f"       ├─ monsoon / agri       : {sum(1 for f in feat_cols if 'monsoon' in f or 'kharif' in f or 'rabi' in f)}")
    print(f"       ├─ dataset native cols  : {sum(1 for f in feat_cols if f in ['Working_Days','Holiday_Count','Festive_Flag','Diwali_Flag','Monsoon_Flag','Quarter_End_Flag','Year_End_Flag','Industry_Growth_Pct','Segment_Growth_Pct','Monsoon_Index','Rainfall_Index','Export_Demand_Index','Fuel_Price_Index','Economic_Growth_Index','Market_Sentiment_Index','Forecast_Confidence','Market_Risk_Score','Forecast_Gap_Pct','Customer_Forecast'])}")
    print(f"       └─ identity / calendar  : {sum(1 for f in feat_cols if f.endswith('_enc') or f in ['month_sin','month_cos','quarter','time_idx'])}")

    result = run_xgboost(train_df, test_df, feat_cols)

    print(f"\n{'─'*52}")
    print(f"  Overall MAPE      : {result['metrics']['MAPE (%)']:.2f} %")
    print(f"  Overall Accuracy  : {result['metrics']['Accuracy (%)']:.2f} %")
    print(f"  Best iteration    : {result['metrics']['Best iteration']}")
    print(f"{'─'*52}")

    print("\nPer-segment accuracy:")
    for seg, m in result['segment_metrics'].items():
        bar = '█' * int(m['Accuracy (%)'] / 5)
        print(f"  {seg:12s}  {m['Accuracy (%)']:5.1f}%  {bar}")

    print("\nTop 15 features by importance (gain):")
    for i, (feat, imp) in enumerate(
            list(result['feature_importances'].items())[:15], start=1):
        print(f"  {i:2d}. {feat:30s}  {imp:.4f}")
