NESO Daily Demand Forecasting Project

1. Project Overview

Goal: Predict 7-day ahead daily demand using sequence models and evaluate against classical baselines.

This project implements LSTM and Transformer models for multi-step forecasting and compares them with Naive, Seasonal Naive, and Moving Average methods. Performance is validated using MAE, RMSE, MAPE, Directional Accuracy, and Diebold-Mariano statistical test, followed by walk-forward backtesting to simulate real-world deployment.

2. Project Features
Data Handling

Daily demand data (2019–2026)

Train/Validation/Test split: 70% / 15% / 15%

Input Window: 30 days

Output Horizon: 7 days

Models
LSTM

Input: 30 days of past demand

Output: 7 days ahead

2 hidden layers, 64 units each

Optimizer: Adam, Loss: MSE

Transformer

Input: same 30 days

Output: 7 days ahead

Parameters: d_model=128, n_heads=4, num_layers=2, dim_feedforward=256

Attention mechanism captures long-range dependencies better than LSTM

Baselines

Naive: repeats last observed value

Seasonal Naive: repeats last 7-day seasonal pattern

Moving Average: average of last 7 days repeated

Metrics
Metric	Description
MAE	Average absolute error. Lower is better.
RMSE	Penalizes large errors more. Lower is better.
MAPE	Percent error relative to actual. Useful for scale-invariant comparison.
Directional Accuracy (%)	% of correctly predicted trends. Crucial for operational decisions.
Diebold-Mariano Test	Compares forecast performance statistically.
Visualization & Outputs

Forecast plots for Train, Validation, and Test sets

Metrics saved as CSVs for record-keeping

Walk-forward backtesting simulates real-world sequential forecasting

3. Implementation Flow
Data Preparation

Load CSV (data/processed/demand_daily_2019_2026.csv)

Scale data using StandardScaler

Convert into PyTorch sequences: 30-day input → 7-day output

Load into DataLoader

Model Training

LSTM and Transformer trained with MSE Loss and Adam optimizer

Epoch-wise loss monitoring

Early plotting for convergence inspection

Prediction & Evaluation

Predict on train, val, test sets

Compute baseline predictions

Evaluate metrics: MAE, RMSE, MAPE, Directional Accuracy

Compare models using Diebold-Mariano test

Walk-Forward Backtest

Sequentially split test data into rolling windows

Predict next window, roll forward, repeat

Compute overall metrics to assess robust real-world performance

4. Project Outputs
Baselines vs LSTM Metrics
Model	MAE	RMSE	MAPE
Naive	0.207	0.263	97%
Seasonal Naive	0.222	0.275	129%
Moving Average	0.183	0.222	114%
LSTM	0.126	0.154	74%

Interpretation: LSTM outperforms all baselines. DM test shows statistically significant improvement over Naive.

LSTM vs Transformer Metrics
Model	MAE	RMSE	MAPE	Directional Accuracy (%)
LSTM	0.126	0.154	74.4%	61.9
Transformer	0.099	0.128	59.1%	74.0

Interpretation: Transformer captures complex demand patterns better, reducing error and improving trend prediction.

Walk-Forward Backtest
Metric	Value
MAE	0.096
RMSE	0.121
MAPE	35.6%
Directional Accuracy	73.9%

Interpretation: Model is robust under realistic sequential forecasting conditions.

5. Why Directional Accuracy Matters

Even if absolute errors are reasonable, predicting demand trends is critical for:

Inventory planning

Stock replenishment

Resource allocation

74% directional accuracy demonstrates practical applicability of the Transformer forecast.

6. Takeaways

Transformer consistently outperforms LSTM and baselines.

Statistical tests validate model superiority.

Walk-forward backtesting ensures model robustness.

Professional outputs (plots, CSVs, models) make the project deployment-ready.

Learning Outcomes:

Multi-step forecasting with PyTorch

Sequence modeling & attention mechanisms

Performance evaluation for time series

Statistical testing for forecast significance

7. Project Structure
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
8. Quick Start
# Install dependencies
pip install -r requirements.txt

# Run full training and evaluation
python -m src.training_phase3

Outputs will be saved in results/ as CSV metrics, plots, and model weights.

9. Tools & Libraries

Python 3.9+

PyTorch

NumPy, Pandas

scikit-learn

Matplotlib / Seaborn

10. Key Highlights

Multi-step forecasting with LSTM & Transformer

Evaluation using absolute, relative, and directional metrics

Robustness verified via walk-forward backtesting

Fully reproducible and deployment-ready
