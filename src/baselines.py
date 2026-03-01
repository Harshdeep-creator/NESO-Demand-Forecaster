import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings("ignore")

DATA_PATH = "data/processed/demand_daily_2019_2026.csv"
RESULTS_PATH = "results"
FORECAST_HORIZON = 7

os.makedirs(RESULTS_PATH, exist_ok=True)


# ------------------------------------------------
# METRICS
# ------------------------------------------------
def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Safe MAPE
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return mae, rmse, mape


# ------------------------------------------------
# CHRONO SPLIT
# ------------------------------------------------
def split_data(df):
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test


# ------------------------------------------------
# NAIVE (Direct Multi-Step)
# ------------------------------------------------
def naive_forecast(train, test):
    """
    Direct multi-step naive forecast.
    Repeats last observed train value.
    """
    last_value = train.values[-1][0]
    predictions = np.repeat(last_value, len(test))
    return predictions


# ------------------------------------------------
# SEASONAL NAIVE (7-day block)
# ------------------------------------------------
def seasonal_naive_forecast(train, test, season_length=7):
    """
    Direct seasonal naive forecast.
    Repeats last seasonal pattern.
    """
    last_season = train.values[-season_length:].flatten()

    repeats = int(np.ceil(len(test) / season_length))
    predictions = np.tile(last_season, repeats)[:len(test)]

    return predictions


# ------------------------------------------------
# MOVING AVERAGE (Direct Multi-Step)
# ------------------------------------------------
def moving_average_forecast(train, test, window=7):
    """
    Direct multi-step moving average forecast.
    """
    avg = np.mean(train.values[-window:])
    predictions = np.repeat(avg, len(test))
    return predictions


# ------------------------------------------------
# SARIMA (Optional – Slow)
# ------------------------------------------------
def sarima_forecast(train, test):
    print("Fitting SARIMA... this may take time")

    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=len(test))

    return forecast.values


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main():

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()

    print("Dataset shape:", df.shape)

    train, val, test = split_data(df)
    print("Train:", len(train), "Test:", len(test))

    y_true = test.values.flatten()

    # ---- NAIVE ----
    print("\nRunning Naive...")
    naive_pred = naive_forecast(train, test)
    naive_metrics = evaluate_forecast(y_true, naive_pred)

    # ---- SEASONAL NAIVE ----
    print("Running Seasonal Naive...")
    seasonal_pred = seasonal_naive_forecast(train, test)
    seasonal_metrics = evaluate_forecast(y_true, seasonal_pred)

    # ---- MOVING AVERAGE ----
    print("Running Moving Average...")
    ma_pred = moving_average_forecast(train, test)
    ma_metrics = evaluate_forecast(y_true, ma_pred)

    print("\nBaseline Results:")
    print("Naive:", naive_metrics)
    print("Seasonal Naive:", seasonal_metrics)
    print("Moving Average:", ma_metrics)

    # Save results
    results = pd.DataFrame({
        "Model": ["Naive", "Seasonal Naive", "Moving Average"],
        "MAE": [naive_metrics[0], seasonal_metrics[0], ma_metrics[0]],
        "RMSE": [naive_metrics[1], seasonal_metrics[1], ma_metrics[1]],
        "MAPE": [naive_metrics[2], seasonal_metrics[2], ma_metrics[2]]
    })

    results.to_csv(os.path.join(RESULTS_PATH, "baseline_results.csv"), index=False)
    print("\nBaseline results saved to results/baseline_results.csv")

    # Optional SARIMA
    """
    print("Running SARIMA...")
    sarima_pred = sarima_forecast(train, test)
    sarima_metrics = evaluate_forecast(y_true, sarima_pred)
    print("SARIMA:", sarima_metrics)
    """


if __name__ == "__main__":
    main()