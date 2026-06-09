"""
XGBoost model for OEM demand forecasting.
Uses lag + calendar + external features as regressors.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import joblib
import os


def run_xgboost(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    model_save_path: str = None,
) -> dict:
    """
    Fit XGBoost regressor on training features, predict test period.
    """
    X_train = train[feature_cols].values
    y_train = train["demand_units"].values
    X_test = test[feature_cols].values
    y_test = test["demand_units"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train_s,
        y_train,
        eval_set=[(X_test_s, y_test)],
        verbose=False,
    )

    predicted = np.maximum(model.predict(X_test_s), 0)
    fitted = np.maximum(model.predict(X_train_s), 0)

    # Feature importances
    importances = dict(zip(feature_cols, model.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    if model_save_path:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump({"model": model, "scaler": scaler, "features": feature_cols}, model_save_path)

    return {
        "predicted": predicted,
        "fitted": fitted,
        "actual_test": y_test,
        "actual_train": y_train,
        "model_obj": model,
        "feature_importances": importances,
        "scaler": scaler,
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "features": feature_cols,
        },
    }
