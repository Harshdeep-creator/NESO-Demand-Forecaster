import torch
import torch.nn as nn
import numpy as np
from src.data_loader import build_dataloaders
from src.models import LSTMModel
from src.evaluation import evaluate_forecast, save_metrics, plot_predictions

# -----------------------------
# CONFIG
# -----------------------------
EPOCHS = 20
LR = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
INPUT_SIZE = 1
OUTPUT_SIZE = 7  # 7-day ahead

# -----------------------------
# TRAINING LOOP
# -----------------------------
def train_model():
    # Load dataloaders
    train_loader, val_loader, test_loader, scaler = build_dataloaders()

    # Initialize LSTM
    model = LSTMModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                      num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print("Initializing model...")
    print("Starting training...")

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            # -----------------------------
            # FIX SHAPE ISSUE
            # -----------------------------
            y_batch = y_batch.squeeze(-1)  # [batch, 7, 1] -> [batch, 7]
            y_pred = model(X_batch)        # model should output [batch, 7]
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch}/{EPOCHS} - Loss: {epoch_loss/len(train_loader):.4f}")

    print("\nTraining complete.")

    # Save model weights
    torch.save(model.state_dict(), "results/model_weights/lstm_model.pt")
    print("Model weights saved to results/model_weights/lstm_model.pt")

    # -----------------------------
    # EVALUATION ON TEST SET
    # -----------------------------
    model.eval()
    preds = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_batch = y_batch.squeeze(-1)
            y_pred = model(X_batch)
            preds.append(y_pred.numpy())
            actuals.append(y_batch.numpy())

    preds = np.vstack(preds)        # [num_samples, 7]
    actuals = np.vstack(actuals)

    # INVERSE SCALE
    preds_flat = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    actuals_flat = scaler.inverse_transform(actuals.reshape(-1, 1)).flatten()

    # Compute metrics
    mae, rmse, mape = evaluate_forecast(actuals_flat, preds_flat)
    print("\nTest Metrics:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # Save metrics
    save_metrics({
        "Model": ["LSTM"],
        "MAE": [mae],
        "RMSE": [rmse],
        "MAPE": [mape]
    })

    # Plot predictions
    plot_predictions(actuals_flat, preds_flat, title="LSTM 7-day Forecast", filename="lstm_forecast.png")


if __name__ == "__main__":
    train_model()