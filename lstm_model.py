"""
LSTM (Long Short-Term Memory) model for OEM demand forecasting.
Uses a sliding window approach with optional multivariate input.
Falls back gracefully if TensorFlow is not installed.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    tf.get_logger().setLevel("ERROR")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


def create_sequences(data: np.ndarray, window: int):
    """Create (X, y) sequences for LSTM from a 2D array (n_steps, n_features)."""
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i - window:i])
        y.append(data[i, 0])  # first column is always demand_units
    return np.array(X), np.array(y)


def run_lstm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list,
    window: int = 12,
    units: int = 64,
    dropout: float = 0.2,
    epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 0.001,
) -> dict:
    """
    Fit a stacked LSTM and forecast the test period.
    Uses all feature_cols as multivariate inputs.
    """
    if not TENSORFLOW_AVAILABLE:
        raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

    from sklearn.preprocessing import MinMaxScaler

    all_cols = ["demand_units"] + [c for c in feature_cols if c != "demand_units"]
    full = pd.concat([train, test], ignore_index=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(full[all_cols].values)

    n_train = len(train)
    n_features = len(all_cols)

    X_all, y_all = create_sequences(scaled, window)
    # Adjust split to account for sequence window
    train_end = n_train - window
    X_train, y_train = X_all[:train_end], y_all[:train_end]
    X_test_seq, y_test_seq = X_all[train_end:], y_all[train_end:]

    model = Sequential([
        Input(shape=(window, n_features)),
        LSTM(units, return_sequences=True),
        Dropout(dropout),
        LSTM(units // 2),
        Dropout(dropout),
        Dense(32, activation="relu"),
        Dense(1),
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="huber")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5),
    ]

    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0,
    )

    # Inverse transform predictions
    def inverse_demand(scaled_vals):
        dummy = np.zeros((len(scaled_vals), n_features))
        dummy[:, 0] = scaled_vals
        return scaler.inverse_transform(dummy)[:, 0]

    pred_scaled = model.predict(X_test_seq, verbose=0).flatten()
    predicted = np.maximum(inverse_demand(pred_scaled), 0)

    fit_scaled = model.predict(X_train, verbose=0).flatten()
    fitted = np.maximum(inverse_demand(fit_scaled), 0)

    y_test_actual = inverse_demand(y_test_seq)
    y_train_actual = inverse_demand(y_train)

    return {
        "predicted": predicted,
        "fitted": fitted,
        "actual_test": y_test_actual,
        "actual_train": y_train_actual,
        "model_obj": model,
        "scaler": scaler,
        "params": {
            "window": window,
            "units": units,
            "dropout": dropout,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "features": all_cols,
        },
    }
