import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

RESULTS_PATH = "results"
PLOTS_PATH = os.path.join(RESULTS_PATH, "plots")
MODEL_WEIGHTS_PATH = os.path.join(RESULTS_PATH, "model_weights")
os.makedirs(PLOTS_PATH, exist_ok=True)
os.makedirs(MODEL_WEIGHTS_PATH, exist_ok=True)

# ------------------------------------------------
# PREDICTION INTERVALS
# ------------------------------------------------
def prediction_intervals(mc_preds, percentile=95):
    """
    Compute prediction intervals from MC Dropout predictions.
    mc_preds: np.array, shape (num_MC_samples, num_timesteps, horizon)
    """
    lower = np.percentile(mc_preds, (100 - percentile) / 2, axis=0)
    upper = np.percentile(mc_preds, 100 - (100 - percentile) / 2, axis=0)
    return lower, upper

# ------------------------------------------------
# METRICS
# ------------------------------------------------
def evaluate_forecast(y_true, y_pred):
    """
    y_true: array-like, shape (num_samples, horizon) or (num_samples, horizon, 1)
    y_pred: array-like, same shape as y_true
    Returns: MAE, RMSE, MAPE
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Flatten last dimension if 3D
    if y_true.ndim == 3 and y_true.shape[2] == 1:
        y_true = y_true[:,:,0]
    if y_pred.ndim == 3 and y_pred.shape[2] == 1:
        y_pred = y_pred[:,:,0]

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

# ------------------------------------------------
# SAVE METRICS
# ------------------------------------------------
def save_metrics(metrics_dict, filename="metrics.csv"):
    os.makedirs(RESULTS_PATH, exist_ok=True)
    df = pd.DataFrame(metrics_dict)
    df.to_csv(os.path.join(RESULTS_PATH, filename), index=False)
    print(f"Metrics saved to {os.path.join(RESULTS_PATH, filename)}")

# ------------------------------------------------
# UNIVERSAL FORECAST PLOT
# ------------------------------------------------
def plot_forecast(y_true, y_pred, mc_preds=None, title="Forecast vs Actual", filename="forecast.png"):
    """
    Generalized forecast plotting function for any horizon.
    y_true: np.array or pd.Series, true values
    y_pred: np.array, shape (num_samples, horizon) or (num_samples, horizon, 1)
    mc_preds: optional np.array, MC dropout samples (num_MC x num_samples x horizon)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred)
    
    # Flatten last dimension if present
    if y_pred.ndim == 3 and y_pred.shape[2] == 1:
        y_pred = y_pred[:,:,0]
    
    horizon = y_pred.shape[1]
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    
    dates = pd.date_range(start=0, periods=min_len)
    
    plt.figure(figsize=(14,6))
    
    # Plot actuals
    plt.plot(dates, y_true, label="Actual", color="blue", linewidth=2)
    
    # Plot first step predictions
    plt.plot(dates, y_pred[:min_len, 0], label=f"Predicted (1st of {horizon})", color="red", linestyle="--", linewidth=2)
    
    # Shaded horizon region
    for i in range(min_len):
        h_len = min(horizon, min_len - i)
        y_pred_slice = y_pred[i,:h_len].flatten()
        y_true_slice = y_true[i:i+h_len].flatten()
        plt.fill_between(dates[i:i+h_len], y_pred_slice, y_true_slice,
                         color="red", alpha=0.1)
    
    # Optional: MC dropout intervals
    if mc_preds is not None:
        lower, upper = prediction_intervals(mc_preds)
        if lower.ndim == 2:
            lower = lower[:,0].flatten()
            upper = upper[:,0].flatten()
        plt.fill_between(dates, lower[:min_len], upper[:min_len],
                         color="orange", alpha=0.2, label="95% PI")
    
    plt.xlabel("Time")
    plt.ylabel("Demand (MW)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_PATH, filename), dpi=150)
    plt.close()
    print(f"Plot saved to {os.path.join(PLOTS_PATH, filename)}")