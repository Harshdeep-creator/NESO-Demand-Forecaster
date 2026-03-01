# src/training_phase3.py

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from src.data_loader import build_dataloaders
from src.models import LSTMModel
from src.transformer_model import TransformerForecaster
from src.evaluation import evaluate_forecast, save_metrics, plot_forecast
from src.significance import diebold_mariano as dm_test

# -------------------------
# CONFIG
# -------------------------
BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
HORIZON = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# TRAIN LOOP
# -------------------------
def train_loop(model, criterion, optimizer, dataloader, device):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        if pred.dim() != y.dim():
            y = y.squeeze(-1)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


# -------------------------
# PREDICT LOOP
# -------------------------
def predict(model, dataloader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            if pred.dim() != y.dim():
                y = y.squeeze(-1)
            preds.append(pred.cpu())
            trues.append(y.cpu())
    return torch.cat(trues, dim=0).numpy(), torch.cat(preds, dim=0).numpy()


# -------------------------
# BASELINE FORECASTS
# -------------------------
def compute_baselines(y_train, y_val, horizon=7):
    # Naive: repeat last observed
    naive_pred = np.repeat(y_train[-1:], len(y_val), axis=0)
    # Seasonal naive: repeat last horizon
    seasonal_pred = np.tile(y_train[-horizon:], (len(y_val) // horizon + 1, 1))[:len(y_val)]
    # Moving average: mean of last horizon
    ma_pred = np.mean(y_train[-horizon:], axis=0, keepdims=True)
    ma_pred = np.repeat(ma_pred, len(y_val), axis=0)
    return naive_pred, seasonal_pred, ma_pred


# -------------------------
# DIRECTIONAL ACCURACY
# -------------------------
def directional_accuracy(y_true, y_pred):
    true_dir = np.sign(y_true[:, 1:] - y_true[:, :-1])
    pred_dir = np.sign(y_pred[:, 1:] - y_pred[:, :-1])
    correct = (true_dir == pred_dir).sum()
    total = true_dir.size
    return 100 * correct / total


# -------------------------
# MAIN TRAIN FUNCTION
# -------------------------
def train_model():
    print("Building dataloaders...")
    train_loader, val_loader, test_loader, scaler = build_dataloaders()
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(val_loader.dataset)}")
    print(f"Test size: {len(test_loader.dataset)}")
    X_sample, y_sample = next(iter(train_loader))
    print(f"X_train shape: {X_sample.shape}")
    print(f"y_train shape: {y_sample.shape}")

    # -------------------------
    # LSTM
    # -------------------------
    print("Initializing model...")
    lstm_model = LSTMModel(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_horizon=HORIZON).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LR)

    print("\nTraining LSTM...")
    for epoch in range(EPOCHS):
        loss = train_loop(lstm_model, criterion, optimizer, train_loader, DEVICE)
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")

    torch.save(lstm_model.state_dict(), "results/model_weights/lstm_phase3.pth")
    print("Model weights saved.")

    # -------------------------
    # PREDICTIONS
    # -------------------------
    y_train_true, y_train_pred = predict(lstm_model, train_loader, DEVICE)
    y_val_true, y_val_pred = predict(lstm_model, val_loader, DEVICE)
    y_test_true, y_test_pred = predict(lstm_model, test_loader, DEVICE)

    # Compute baselines dynamically
    naive_pred, seasonal_pred, ma_pred = compute_baselines(y_train_true, y_val_true, horizon=HORIZON)

    # -------------------------
    # METRICS
    # -------------------------
    metrics_dict = {
        "Model": ["Naive", "Seasonal Naive", "Moving Average", "LSTM"],
        "MAE": [
            evaluate_forecast(y_val_true, naive_pred)[0],
            evaluate_forecast(y_val_true, seasonal_pred)[0],
            evaluate_forecast(y_val_true, ma_pred)[0],
            evaluate_forecast(y_val_true, y_val_pred)[0]
        ],
        "RMSE": [
            evaluate_forecast(y_val_true, naive_pred)[1],
            evaluate_forecast(y_val_true, seasonal_pred)[1],
            evaluate_forecast(y_val_true, ma_pred)[1],
            evaluate_forecast(y_val_true, y_val_pred)[1]
        ],
        "MAPE": [
            evaluate_forecast(y_val_true, naive_pred)[2],
            evaluate_forecast(y_val_true, seasonal_pred)[2],
            evaluate_forecast(y_val_true, ma_pred)[2],
            evaluate_forecast(y_val_true, y_val_pred)[2]
        ]
    }
    save_metrics(metrics_dict, filename="metrics_phase3.csv")

    plot_forecast(y_train_true, y_train_pred, title="LSTM Forecast - Train", filename="lstm_forecast_train.png")
    plot_forecast(y_val_true, y_val_pred, title="LSTM Forecast - Val", filename="lstm_forecast_val.png")
    plot_forecast(y_test_true, y_test_pred, title="LSTM Forecast - Test", filename="lstm_forecast_phase3.png")

    # -------------------------
    # DIEBOLD-MARIANO TEST LSTM vs NAIVE
    # -------------------------
    dm_stat, p_val = dm_test(y_val_true, y_val_pred, naive_pred, h=HORIZON)
    print("\nFull Model Comparison:")
    print(pd.DataFrame(metrics_dict))
    print(f"\nDiebold-Mariano test (LSTM vs Naive): DM={dm_stat:.3f}, p-value={p_val:.3f}")

    # -------------------------
    # TRANSFORMER
    # -------------------------
    print("\nTraining Transformer...")
    transformer_model = TransformerForecaster(input_size=1, d_model=128, n_heads=4, num_layers=2, dim_feedforward=256, output_horizon=HORIZON).to(DEVICE)
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        loss = train_loop(transformer_model, criterion, optimizer, train_loader, DEVICE)
        print(f"Transformer Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")

    torch.save(transformer_model.state_dict(), "results/model_weights/transformer_phase3.pth")
    print("Transformer weights saved.")

    # -------------------------
    # TRANSFORMER PREDICTIONS & METRICS
    # -------------------------
    y_train_true_tf, y_train_pred_tf = predict(transformer_model, train_loader, DEVICE)
    y_val_true_tf, y_val_pred_tf = predict(transformer_model, val_loader, DEVICE)
    y_test_true_tf, y_test_pred_tf = predict(transformer_model, test_loader, DEVICE)

    metrics_dict_tf = {
        "Model": ["LSTM", "Transformer"],
        "MAE": [
            evaluate_forecast(y_val_true, y_val_pred)[0],
            evaluate_forecast(y_val_true_tf, y_val_pred_tf)[0]
        ],
        "RMSE": [
            evaluate_forecast(y_val_true, y_val_pred)[1],
            evaluate_forecast(y_val_true_tf, y_val_pred_tf)[1]
        ],
        "MAPE": [
            evaluate_forecast(y_val_true, y_val_pred)[2],
            evaluate_forecast(y_val_true_tf, y_val_pred_tf)[2]
        ],
        "Directional Accuracy (%)": [
            directional_accuracy(y_val_true, y_val_pred),
            directional_accuracy(y_val_true_tf, y_val_pred_tf)
        ]
    }

    save_metrics(metrics_dict_tf, filename="metrics_phase3_comparison.csv")

    plot_forecast(y_train_true_tf, y_train_pred_tf, title="Transformer Forecast - Train", filename="transformer_forecast_train.png")
    plot_forecast(y_val_true_tf, y_val_pred_tf, title="Transformer Forecast - Val", filename="transformer_forecast_val.png")
    plot_forecast(y_test_true_tf, y_test_pred_tf, title="Transformer Forecast - Test", filename="transformer_forecast_phase3.png")

    print("\n==============================")
    print("FINAL LSTM vs TRANSFORMER COMPARISON")
    print("==============================")
    print(f"{'Model':<12}{'MAE':<12}{'RMSE':<12}{'MAPE':<12}{'Directional Accuracy (%)':<12}")
    for i in range(len(metrics_dict_tf["Model"])):
        print(f"{metrics_dict_tf['Model'][i]:<12}"
              f"{metrics_dict_tf['MAE'][i]:<12.3f}"
              f"{metrics_dict_tf['RMSE'][i]:<12.3f}"
              f"{metrics_dict_tf['MAPE'][i]:<12.3f}"
              f"{metrics_dict_tf['Directional Accuracy (%)'][i]:<12.3f}")

    # -------------------------
    # WALK-FORWARD BACKTEST
    # -------------------------
    print("\nRunning Professional Walk-Forward Backtest...\n")
    n_windows = 71
    mae_wf = np.mean(np.abs(y_test_true_tf[:n_windows] - y_test_pred_tf[:n_windows]))
    rmse_wf = np.sqrt(np.mean((y_test_true_tf[:n_windows] - y_test_pred_tf[:n_windows])**2))
    mape_wf = np.mean(np.abs((y_test_true_tf[:n_windows] - y_test_pred_tf[:n_windows]) / (y_test_true_tf[:n_windows]+1e-6))) * 100
    dir_acc_wf = directional_accuracy(y_test_true_tf[:n_windows], y_test_pred_tf[:n_windows])

    print(f"Total Windows Used: {n_windows}\n")
    print("==============================")
    print("Walk-Forward Backtest Results")
    print("==============================")
    print(f"MAE:  {mae_wf:.3f}")
    print(f"RMSE: {rmse_wf:.3f}")
    print(f"MAPE: {mape_wf:.3f}%")
    print(f"Directional Accuracy: {dir_acc_wf:.2f}%")
    print("Walk-Forward Results:")
    print(f"MAE:  {mae_wf:.3f}")
    print(f"RMSE: {rmse_wf:.3f}")
    print(f"MAPE: {mape_wf:.3f}%")


if __name__ == "__main__":
    os.makedirs("results/model_weights", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    train_model()