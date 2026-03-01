import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.data_loader import INPUT_WINDOW, FORECAST_HORIZON
from src.evaluation import evaluate_forecast

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------
# CREATE SEQUENCES
# ------------------------------------------------
def create_sequences(data, input_window, forecast_horizon):
    X, y = [], []

    for i in range(len(data) - input_window - forecast_horizon + 1):
        X.append(data[i:i + input_window])
        y.append(data[i + input_window:i + input_window + forecast_horizon])

    return np.array(X), np.array(y)


# ------------------------------------------------
# GENERIC WALK-FORWARD BACKTEST
# ------------------------------------------------
def walk_forward_backtest(
    df,
    model_class,
    weight_path,
    model_kwargs,
    initial_train_size=1200,
    step_size=14,
    fine_tune_epochs=2
):

    print(f"\nStarting Walk-Forward Backtest for {model_class.__name__}...\n")

    values = df.values
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values)

    all_true = []
    all_pred = []

    start = initial_train_size
    window_count = 0

    # Load base model once
    base_model = model_class(**model_kwargs).to(DEVICE)
    base_model.load_state_dict(
        torch.load(weight_path, map_location=DEVICE)
    )

    while start + FORECAST_HORIZON < len(values_scaled):

        window_count += 1
        train_data = values_scaled[:start]

        X_train, y_train = create_sequences(
            train_data,
            INPUT_WINDOW,
            FORECAST_HORIZON
        )

        X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        y_train = torch.tensor(
            y_train.squeeze(-1),
            dtype=torch.float32
        ).to(DEVICE)

        # Copy base model correctly
        model = model_class(**model_kwargs).to(DEVICE)
        model.load_state_dict(base_model.state_dict())

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        criterion = nn.MSELoss()

        # Light fine-tuning
        for _ in range(fine_tune_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        # Forecast
        model.eval()
        last_window = values_scaled[start - INPUT_WINDOW:start]
        last_window = torch.tensor(
            last_window.reshape(1, INPUT_WINDOW, 1),
            dtype=torch.float32
        ).to(DEVICE)

        with torch.no_grad():
            forecast = model(last_window).cpu().numpy()

        true_future = values_scaled[start:start + FORECAST_HORIZON]

        all_pred.append(forecast.flatten())
        all_true.append(true_future.flatten())

        start += step_size

    print(f"Total Windows Used: {window_count}")

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    all_pred = scaler.inverse_transform(all_pred.reshape(-1, 1)).flatten()
    all_true = scaler.inverse_transform(all_true.reshape(-1, 1)).flatten()

    mae, rmse, mape = evaluate_forecast(all_true, all_pred)

    direction_true = np.sign(np.diff(all_true))
    direction_pred = np.sign(np.diff(all_pred))
    hit_rate = (direction_true == direction_pred).mean()

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Directional_Accuracy": hit_rate
    }