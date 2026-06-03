"""
Facebook Prophet model for OEM demand forecasting.
Handles holiday/festive effects, seasonality decomposition.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


def get_india_holidays(years: list) -> pd.DataFrame:
    """Return key Indian automotive demand events as Prophet holidays."""
    holidays = []
    for year in years:
        holidays += [
            {"holiday": "Diwali", "ds": f"{year}-10-24", "lower_window": -10, "upper_window": 5},
            {"holiday": "Navratri", "ds": f"{year}-10-03", "lower_window": -3, "upper_window": 3},
            {"holiday": "Dhanteras", "ds": f"{year}-10-22", "lower_window": -2, "upper_window": 1},
            {"holiday": "Onam", "ds": f"{year}-08-29", "lower_window": -5, "upper_window": 2},
            {"holiday": "Gudi_Padwa", "ds": f"{year}-04-02", "lower_window": -7, "upper_window": 3},
            {"holiday": "FY_End_Push", "ds": f"{year}-03-25", "lower_window": -7, "upper_window": 5},
        ]
    return pd.DataFrame(holidays)


def run_prophet(
    train: pd.DataFrame,
    test: pd.DataFrame,
    add_regressors: list = None,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = False,
    seasonality_mode: str = "multiplicative",
) -> dict:
    """
    Fit Prophet model with India-specific holidays and optional regressors.
    """
    if not PROPHET_AVAILABLE:
        raise ImportError("Prophet not installed. Run: pip install prophet")

    all_years = list(range(train["year"].min(), test["year"].max() + 2))
    holidays = get_india_holidays(all_years)

    prophet_train = train[["date", "demand_units"]].rename(columns={"date": "ds", "demand_units": "y"})
    prophet_test = test[["date", "demand_units"]].rename(columns={"date": "ds", "demand_units": "y"})

    model = Prophet(
        holidays=holidays,
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=0.05,
    )

    if add_regressors:
        for reg in add_regressors:
            if reg in train.columns:
                model.add_regressor(reg)
        prophet_train = prophet_train.join(train[add_regressors].reset_index(drop=True))
        prophet_test = prophet_test.join(test[add_regressors].reset_index(drop=True))

    model.fit(prophet_train)

    future_test = prophet_test[["ds"] + (add_regressors if add_regressors else [])].copy()
    forecast = model.predict(future_test)

    predicted = np.maximum(forecast["yhat"].values, 0)
    y_test = test["demand_units"].values

    # In-sample
    future_train = prophet_train[["ds"] + (add_regressors if add_regressors else [])].copy()
    fitted_df = model.predict(future_train)
    fitted = np.maximum(fitted_df["yhat"].values, 0)

    return {
        "predicted": predicted,
        "fitted": fitted,
        "actual_test": y_test,
        "actual_train": train["demand_units"].values,
        "model_obj": model,
        "forecast_df": forecast,
        "components": forecast[["ds", "trend", "yearly", "holidays"]
                                if "holidays" in forecast.columns
                                else ["ds", "trend", "yearly"]],
        "params": {
            "seasonality_mode": seasonality_mode,
            "yearly_seasonality": yearly_seasonality,
            "regressors": add_regressors,
        },
    }
