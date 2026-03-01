# app.py
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.models import LSTMModel
from src.transformer_model import TransformerForecaster
from src.data_loader import INPUT_WINDOW, FORECAST_HORIZON
from src.evaluation import evaluate_forecast, plot_forecast  # optional plotting helper

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="NESO Demand Forecast", layout="wide")
st.title("NESO Multi-Step Demand Forecasting")
st.markdown(
    """
Predict 7-day ahead demand using LSTM and Transformer models.
Select the model and provide the last 30 days of data (or use default).  
"""
)

# -------------------------
# Load Data
# -------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/demand_daily_2019_2026.csv")
    return df

data = load_data()

# Sidebar: User input
st.sidebar.header("Input Options")
use_default = st.sidebar.checkbox("Use last 30 days from dataset", value=True)

if use_default:
    df_input = data['demand'].values[-INPUT_WINDOW:]
else:
    uploaded_file = st.sidebar.file_uploader("Upload 30-day CSV", type=["csv"])
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file).values.flatten()
    else:
        st.warning("Please upload a CSV or use default")
        st.stop()

# Ensure correct shape
if len(df_input) != INPUT_WINDOW:
    st.error(f"Input length must be {INPUT_WINDOW} days")
    st.stop()

input_sequence = np.array(df_input).reshape(1, INPUT_WINDOW, 1)
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_sequence.reshape(-1, 1)).reshape(1, INPUT_WINDOW, 1)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(DEVICE)

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_lstm_model():
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, output_horizon=FORECAST_HORIZON).to(DEVICE)
    model.load_state_dict(torch.load("results/model_weights/lstm_phase3.pth", map_location=DEVICE))
    model.eval()
    return model

@st.cache_resource
def load_transformer_model():
    model = TransformerForecaster(input_size=1, d_model=128, n_heads=4, num_layers=2,
                                  dim_feedforward=256, output_horizon=FORECAST_HORIZON).to(DEVICE)
    model.load_state_dict(torch.load("results/model_weights/transformer_phase3.pth", map_location=DEVICE))
    model.eval()
    return model

# Select model
model_choice = st.sidebar.selectbox("Select Model", ["LSTM", "Transformer"])

if model_choice == "LSTM":
    model = load_lstm_model()
else:
    model = load_transformer_model()

# -------------------------
# Forecast
# -------------------------
with torch.no_grad():
    forecast_scaled = model(input_tensor).cpu().numpy().flatten()
forecast = scaler.inverse_transform(forecast_scaled.reshape(-1,1)).flatten()

# Display Results
st.subheader(f"{model_choice} 7-Day Forecast")
forecast_df = pd.DataFrame({
    "Day": [f"Day +{i+1}" for i in range(FORECAST_HORIZON)],
    "Forecast": forecast
})
st.table(forecast_df)

# Optional: plot
st.subheader("Forecast Plot")
st.line_chart(forecast_df.set_index("Day"))

st.markdown("---")
st.markdown("**Note:** Transformer generally outperforms LSTM in accuracy and directional prediction.")
