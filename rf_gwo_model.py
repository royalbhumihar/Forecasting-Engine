"""
Random Forest + Grey Wolf Optimizer (GWO) hybrid model.
GWO is a nature-inspired metaheuristic that optimizes RF hyperparameters
(n_estimators, max_depth, min_samples_split, max_features).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


class GreyWolfOptimizer:
    """
    Grey Wolf Optimizer for hyperparameter tuning.
    Minimizes MAPE on cross-validation set.
    """

    def __init__(self, n_wolves: int = 8, max_iter: int = 20, seed: int = 42):
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.seed = seed

        # Search bounds: [n_estimators, max_depth, min_samples_split, max_features_idx]
        self.lb = np.array([50,   3,  2, 0])
        self.ub = np.array([400, 15, 10, 2])

    def _decode(self, pos: np.ndarray) -> dict:
        """Map continuous wolf position to RF hyperparameters."""
        pos = np.clip(pos, self.lb, self.ub)
        max_features_options = ["sqrt", "log2", None]
        return {
            "n_estimators": int(round(pos[0])),
            "max_depth": int(round(pos[1])),
            "min_samples_split": int(round(pos[2])),
            "max_features": max_features_options[int(round(pos[3]))],
        }

    def _fitness(self, pos: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate 3-fold CV MAPE for a given hyperparameter position."""
        params = self._decode(pos)
        rf = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            max_features=params["max_features"],
            random_state=42,
            n_jobs=-1,
        )
        scores = cross_val_score(rf, X, y, cv=3, scoring="neg_mean_absolute_percentage_error")
        mape = -scores.mean() * 100
        return mape

    def optimize(self, X: np.ndarray, y: np.ndarray, progress_callback=None) -> dict:
        """Run GWO and return best hyperparameters."""
        np.random.seed(self.seed)
        dim = len(self.lb)

        # Initialize wolf population
        positions = self.lb + np.random.rand(self.n_wolves, dim) * (self.ub - self.lb)
        fitness = np.array([self._fitness(positions[i], X, y) for i in range(self.n_wolves)])

        sorted_idx = np.argsort(fitness)
        alpha_pos, alpha_score = positions[sorted_idx[0]].copy(), fitness[sorted_idx[0]]
        beta_pos, beta_score  = positions[sorted_idx[1]].copy(), fitness[sorted_idx[1]]
        delta_pos, delta_score = positions[sorted_idx[2]].copy(), fitness[sorted_idx[2]]

        history = [alpha_score]

        for t in range(self.max_iter):
            a = 2 - t * (2 / self.max_iter)  # linearly decreases 2→0

            for i in range(self.n_wolves):
                for j in range(dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                    X2 = beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                    X3 = delta_pos[j] - A3 * D_delta

                    positions[i, j] = (X1 + X2 + X3) / 3

                positions[i] = np.clip(positions[i], self.lb, self.ub)

            fitness = np.array([self._fitness(positions[i], X, y) for i in range(self.n_wolves)])
            sorted_idx = np.argsort(fitness)

            if fitness[sorted_idx[0]] < alpha_score:
                alpha_pos, alpha_score = positions[sorted_idx[0]].copy(), fitness[sorted_idx[0]]
            if fitness[sorted_idx[1]] < beta_score:
                beta_pos, beta_score   = positions[sorted_idx[1]].copy(), fitness[sorted_idx[1]]
            if fitness[sorted_idx[2]] < delta_score:
                delta_pos, delta_score = positions[sorted_idx[2]].copy(), fitness[sorted_idx[2]]

            history.append(alpha_score)
            if progress_callback:
                progress_callback(t + 1, self.max_iter, alpha_score, self._decode(alpha_pos))

        best_params = self._decode(alpha_pos)
        return {"best_params": best_params, "best_mape": alpha_score, "convergence": history}


def run_rf_gwo(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list,
    n_wolves: int = 8,
    max_iter: int = 20,
    use_gwo: bool = True,
    progress_callback=None,
) -> dict:
    """
    Fit Random Forest with GWO-optimized hyperparameters.
    If use_gwo=False, uses default RF hyperparameters (faster).
    """
    X_train = train[feature_cols].values
    y_train = train["demand_units"].values
    X_test = test[feature_cols].values
    y_test = test["demand_units"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    gwo_results = None
    if use_gwo:
        gwo = GreyWolfOptimizer(n_wolves=n_wolves, max_iter=max_iter)
        gwo_results = gwo.optimize(X_train_s, y_train, progress_callback=progress_callback)
        best_params = gwo_results["best_params"]
    else:
        best_params = {
            "n_estimators": 200,
            "max_depth": 8,
            "min_samples_split": 4,
            "max_features": "sqrt",
        }

    model = RandomForestRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)

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
        "gwo_results": gwo_results,
        "params": {**best_params, "use_gwo": use_gwo, "features": feature_cols},
    }
