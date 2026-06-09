"""
SARIMA / SARIMAX model for OEM demand forecasting.
Uses statsmodels SARIMAX with optional exogenous regressors.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX


def run_sarima(
    train: pd.DataFrame,
    test: pd.DataFrame,
    horizon: int = 12,
    order: tuple = (1, 1, 1),
    seasonal_order: tuple = (1, 1, 1, 12),
    exog_cols: list = None,
) -> dict:
    """
    Fit SARIMAX on training data, forecast over test period.

    Returns:
        dict with keys: predicted, fitted, model_summary, params
    """
    y_train = train["demand_units"].values
    y_test = test["demand_units"].values

    exog_train = train[exog_cols].values if exog_cols else None
    exog_test = test[exog_cols].values if exog_cols else None

    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fit = model.fit(disp=False, maxiter=200)

    n_steps = len(y_test)
    forecast = fit.forecast(steps=n_steps, exog=exog_test)
    forecast = np.maximum(forecast, 0)

    in_sample = fit.fittedvalues
    in_sample = np.maximum(in_sample, 0)

    return {
        "predicted": forecast,
        "fitted": in_sample,
        "actual_test": y_test,
        "actual_train": y_train,
        "model_obj": fit,
        "params": {
            "order": order,
            "seasonal_order": seasonal_order,
            "exog_cols": exog_cols,
            "n_train": len(y_train),
            "n_test": n_steps,
        },
    }
