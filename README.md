NESO Daily Demand Forecasting Project

Predict 7-day ahead daily demand using LSTM and Transformer models, compared against classical baselines. This project demonstrates multi-step forecasting, performance evaluation, and walk-forward backtesting for real-world applicability.

Table of Contents

Project Overview

Features

Implementation Flow

Results

Why Directional Accuracy Matters

Takeaways & Learning Outcomes

Project Structure

Quick Start

Tools & Libraries

Project Overview

Goal: Forecast 7-day ahead daily demand.

Models: LSTM and Transformer for multi-step forecasting.

Baselines: Naive, Seasonal Naive, and Moving Average.

Evaluation Metrics: MAE, RMSE, MAPE, Directional Accuracy, and Diebold-Mariano test.

Validation: Walk-forward backtesting to simulate real-world deployment.

Features
Data Handling

Daily demand data (2019–2026)

Train/Validation/Test split: 70% / 15% / 15%

Input window: 30 days → Output horizon: 7 days

Models

LSTM

2 hidden layers, 64 units each

Input: past 30 days → Output: next 7 days

Optimizer: Adam, Loss: MSE

Transformer

Parameters: d_model=128, n_heads=4, num_layers=2, dim_feedforward=256

Captures long-range dependencies better than LSTM

Baselines

Naive: repeats last observed value

Seasonal Naive: repeats last 7-day pattern

Moving Average: average of last 7 days

Metrics
Metric	Description
MAE	Average absolute error
RMSE	Penalizes large errors
MAPE	Scale-invariant percent error
Directional Accuracy (%)	% of correctly predicted trends
Diebold-Mariano Test	Statistical comparison of forecasts
Visualization

Forecast plots for Train, Validation, and Test sets

Metrics saved as CSV for record-keeping

Implementation Flow

Data Preparation

Load CSV (data/processed/demand_daily_2019_2026.csv)

Scale data using StandardScaler

Convert to sequences for PyTorch: 30-day input → 7-day output

Load into DataLoader

Model Training

Train LSTM and Transformer using MSE loss and Adam optimizer

Monitor epoch-wise loss

Early plotting for convergence inspection

Prediction & Evaluation

Predict on train, val, test sets

Compute baselines

Evaluate metrics: MAE, RMSE, MAPE, Directional Accuracy

Compare models using Diebold-Mariano test

Walk-Forward Backtesting

Sequential rolling windows on test data

Predict next window, roll forward, repeat

Compute overall metrics for robust evaluation

Results
Baselines vs LSTM
Model	MAE	RMSE	MAPE
Naive	0.207	0.263	97%
Seasonal Naive	0.222	0.275	129%
Moving Average	0.183	0.222	114%
LSTM	0.126	0.154	74%

Interpretation: LSTM outperforms all baselines. DM test confirms statistical significance.

LSTM vs Transformer
Model	MAE	RMSE	MAPE	Directional Accuracy (%)
LSTM	0.126	0.154	74.4%	61.9
Transformer	0.099	0.128	59.1%	74.0

Interpretation: Transformer captures complex demand patterns better, improving trend prediction.

Walk-Forward Backtest
Metric	Value
MAE	0.096
RMSE	0.121
MAPE	35.6%
Directional Accuracy	73.9%
Why Directional Accuracy Matters

Even with reasonable absolute errors, predicting trends is key for:

Inventory planning

Stock replenishment

Resource allocation

74% directional accuracy demonstrates practical applicability of the Transformer forecast.

Takeaways & Learning Outcomes

Transformer outperforms LSTM and baselines consistently

Statistical tests validate model superiority

Walk-forward backtesting ensures robustness

Outputs (plots, CSVs, model weights) make project deployment-ready

Learning Outcomes:

Multi-step forecasting with PyTorch

Sequence modeling & attention mechanisms

Time series performance evaluation

Statistical testing for forecast significance

Project Structure
forecasting-project/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploration.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── data_loader.py
│   ├── baselines.py
│   ├── models.py
│   ├── transformer_model.py
│   ├── training.py
│   ├── evaluation.py
│   ├── significance.py
│   ├── training_phase3.py
│   └── backtesting.py
│
├── results/
│   ├── metrics_phase3.csv
│   ├── metrics_phase3_comparison.csv
│   ├── plots/
│   └── model_weights/
│
├── README.md
└── requirements.txt
Quick Start
# Install dependencies
pip install -r requirements.txt

# Run full training and evaluation
python -m src.training_phase3

All outputs (metrics, plots, model weights) are saved in results/.

Tools & Libraries

Python 3.9+

PyTorch

NumPy, Pandas

scikit-learn

Matplotlib / Seaborn

Key Highlights

Multi-step forecasting using LSTM & Transformer

Evaluation via absolute, relative, and directional metrics

Walk-forward backtesting ensures real-world robustness

Fully reproducible and deployment-ready
