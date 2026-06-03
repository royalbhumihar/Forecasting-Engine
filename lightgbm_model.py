"""
LightGBM model for OEM demand forecasting.
Fast gradient boosting — well suited for large dealer network datasets.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def run_lightgbm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    subsample: float = 0.8,
) -> dict:
    """
    Fit LightGBM regressor and predict test period.
    """
    X_train = train[feature_cols].values
    y_train = train["demand_units"].values
    X_test = test[feature_cols].values
    y_test = test["demand_units"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=subsample,
        objective="regression",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )

    callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    model.fit(
        X_train_s, y_train,
        eval_set=[(X_test_s, y_test)],
        callbacks=callbacks,
    )

    predicted = np.maximum(model.predict(X_test_s), 0)
    fitted = np.maximum(model.predict(X_train_s), 0)

    importances = dict(zip(feature_cols, model.feature_importances_))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

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
            "num_leaves": num_leaves,
            "subsample": subsample,
            "features": feature_cols,
        },
    }
