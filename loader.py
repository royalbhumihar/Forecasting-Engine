"""
Data loader and feature engineering for OEM demand forecasting.
Supports CSV upload or synthetic demo data generation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_demo_data(n_months: int = 60, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic synthetic OEM demand data for India.
    Includes: monsoon dip, festive surge, EV trend, regional patterns.
    """
    np.random.seed(seed)
    start = datetime(2019, 1, 1)
    dates = [start + timedelta(days=30 * i) for i in range(n_months)]

    base_demand = 80000

    records = []
    for i, date in enumerate(dates):
        month = date.month
        year = date.year

        # Long-term growth trend
        trend = 1 + (i / n_months) * 0.35

        # Monsoon dip: June–Aug → -18 to -22%
        if month in [6, 7, 8]:
            seasonality = np.random.uniform(0.78, 0.82)
        # Festive surge: Sep–Nov → +25 to +40%
        elif month in [9, 10, 11]:
            seasonality = np.random.uniform(1.25, 1.40)
        # Year-end fleet orders: Feb–Mar → +10 to +18%
        elif month in [2, 3]:
            seasonality = np.random.uniform(1.10, 1.18)
        # Pre-monsoon stocking: Apr–May → +5 to +10%
        elif month in [4, 5]:
            seasonality = np.random.uniform(1.05, 1.10)
        # Dec–Jan: mild dip
        elif month in [12, 1]:
            seasonality = np.random.uniform(0.90, 0.96)
        else:
            seasonality = np.random.uniform(0.95, 1.05)

        # Fuel price impact (synthetic, correlated with oil cycles)
        fuel_price = 90 + 15 * np.sin(i / 12 * np.pi) + np.random.normal(0, 3)
        fuel_impact = 1 - (fuel_price - 90) * 0.002

        # Monsoon intensity index (higher → worse for demand)
        monsoon_index = 100 + 20 * np.sin((month - 6) / 3 * np.pi) * (month in [6, 7, 8, 9]) + np.random.normal(0, 5)
        monsoon_index = max(0, monsoon_index)

        # Interest rate (RBI policy cycle)
        interest_rate = 6.5 + 0.5 * np.sin(i / 18 * np.pi) + np.random.normal(0, 0.1)
        emi_impact = 1 - (interest_rate - 6.5) * 0.02

        # EV substitution effect (gradual from 2022)
        ev_share = max(0, (year - 2021) * 0.03 + np.random.normal(0, 0.01)) if year >= 2022 else 0

        # Noise
        noise = np.random.normal(1.0, 0.03)

        demand = int(base_demand * trend * seasonality * fuel_impact * emi_impact * noise)
        demand = max(30000, demand)

        records.append({
            "date": date,
            "year": year,
            "month": month,
            "demand_units": demand,
            "fuel_price_inr": round(fuel_price, 2),
            "monsoon_index": round(monsoon_index, 1),
            "interest_rate_pct": round(interest_rate, 2),
            "ev_share_pct": round(ev_share * 100, 2),
            "is_festive": 1 if month in [9, 10, 11] else 0,
            "is_monsoon": 1 if month in [6, 7, 8] else 0,
            "is_year_end": 1 if month in [2, 3] else 0,
            "gdp_growth_pct": round(6.5 + np.random.normal(0, 0.5), 2),
            "dealer_inventory_index": round(100 + np.random.normal(0, 10), 1),
        })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_user_data(uploaded_file) -> pd.DataFrame:
    """
    Load user-uploaded CSV/Excel.
    Expected columns: date, demand_units (minimum).
    All other columns used as exogenous features.
    """
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    if "date" not in df.columns:
        raise ValueError("Dataset must contain a 'date' column.")
    if "demand_units" not in df.columns:
        raise ValueError("Dataset must contain a 'demand_units' column.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based and lag features for ML models.
    """
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["quarter"] = df["date"].dt.quarter
    df["time_index"] = np.arange(len(df))

    for lag in [1, 2, 3, 6, 12]:
        df[f"lag_{lag}"] = df["demand_units"].shift(lag)

    for window in [3, 6, 12]:
        df[f"rolling_mean_{window}"] = df["demand_units"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["demand_units"].shift(1).rolling(window).std()

    df["yoy_growth"] = df["demand_units"].pct_change(12) * 100

    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return ML-ready feature columns (exclude date, raw demand)."""
    exclude = {"date", "demand_units", "yoy_growth"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]


def train_test_split_ts(df: pd.DataFrame, test_months: int = 12):
    """Time-series aware train/test split (no shuffle)."""
    split_idx = len(df) - test_months
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test
