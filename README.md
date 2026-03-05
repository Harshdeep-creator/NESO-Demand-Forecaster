# NESO Daily Demand Forecasting Project



## 1. Project Overview

**Goal:** Predict 7-day ahead daily demand using sequence models and evaluate against classical baselines.

This project implements **LSTM** and **Transformer** models for multi-step forecasting and compares them with **Naive**, **Seasonal Naive**, and **Moving Average** methods. Performance is validated using MAE, RMSE, MAPE, Directional Accuracy, and Diebold-Mariano statistical test, followed by walk-forward backtesting to simulate real-world deployment.

## 1. Dataset

**Source:** National Energy System Operator (NESO) – [neso.energy](https://www.neso.energy/)  
**Time Period:** 2019–2024  
**Type:** Real daily demand data  

**Preprocessing Steps:**  
1. **Load raw data** from multiple files.  
2. **Create datetime index** for proper time series alignment.  
3. **Select relevant columns** needed for forecasting.  
4. **Fix missing timestamps** to ensure continuity.  
5. **Resample to daily frequency** to standardize the data.  
6. **Save processed dataset** for modeling and analysis (`data/processed/demand_daily_2019_2024.csv`).  

These steps ensure the dataset is **clean, consistent, and ready for multi-step forecasting**.

## 3. Project Features

### Data Handling
- Train/Validation/Test split: 70% / 15% / 15%
- Input Window: 90 days
- Output Horizon: 7 days

### Models

#### LSTM
- Input: 90 days of past demand
- Output: 7 days ahead
- 2 hidden layers, 64 units each
- Optimizer: Adam, Loss: MSE

#### Transformer
- Input: same 90 days
- Output: 7 days ahead
- Parameters: `d_model=224`, `n_heads=7`, `num_layers=4`, `dim_feedforward=896`
- Attention mechanism captures long-range dependencies better than LSTM

### Baselines
- **Naive:** repeats last observed value  
- **Seasonal Naive:** repeats last 7-day seasonal pattern  
- **Moving Average:** average of last 7 days repeated  

### Metrics
| Metric | Description |
|--------|-------------|
| MAE | Average absolute error. Lower is better. |
| RMSE | Penalizes large errors more. Lower is better. |
| MAPE | Percent error relative to actual. Useful for scale-invariant comparison. |
| Directional Accuracy (%) | % of correctly predicted trends. Crucial for operational decisions. |
| Diebold-Mariano Test | Compares forecast performance statistically. |

### Visualization & Outputs
- Forecast plots for Train, Validation, and Test sets
- Metrics saved as CSVs for record-keeping
- Walk-forward backtesting simulates real-world sequential forecasting



## 4. Implementation Flow

<details>
<summary>Click to expand Implementation Steps</summary>

### 4.1 Data Preparation
1. Load CSV: `data/processed/demand_daily_2019_2024.csv`
2. Scale data using `StandardScaler`
3. Convert into PyTorch sequences: 90-day input → 7-day output
4. Load into `DataLoader`

### 4.2 Model Training
1. Train **LSTM** and **Transformer** with MSE Loss and Adam optimizer
2. Monitor epoch-wise loss
3. Plot convergence early for inspection

### 4.3 Prediction & Evaluation
1. Predict on train, val, test sets
2. Compute baseline predictions
3. Evaluate metrics: MAE, RMSE, MAPE, Directional Accuracy
4. Compare models using **Diebold-Mariano test**

### 4.4 Walk-Forward Backtest
1. Sequentially split test data into rolling windows
2. Predict next window, roll forward, repeat
3. Compute overall metrics to assess robust real-world performance

</details>



## 5. Project Outputs

### Baselines vs LSTM Metrics
| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Naive | 0.203201   |  0.256921 | 100% |
| Seasonal Naive | 0.22503 | 0.273033 | 131% |
| Moving Average |  0.179111  | 0.218067 | 118% |
| LSTM | 0.098599 | 0.127598 | 62.5% |

**Interpretation:** LSTM outperforms all baselines. DM test shows statistically significant improvement over Naive.

### LSTM vs Transformer Metrics
| Model | MAE | RMSE | MAPE | Directional Accuracy (%) |
|-------|-----|------|------|-------------------------|
| LSTM | 0.099 |  0.128   | 62.506  |72.604 |
| Transformer | 0.069 | 0.095  | 34.207  | 84.478 |

**Interpretation:** Transformer captures complex demand patterns better, reducing error and improving trend prediction.

### Walk-Forward Backtest
| Metric | Value |
|--------|-------|
| MAE | 0.074 |
| RMSE | 0.101 |
| MAPE | 30.52% |
| Directional Accuracy |82.63%|

**Interpretation:** Model is robust under realistic sequential forecasting conditions.



## 6. Why Directional Accuracy Matters
Even if absolute errors are reasonable, predicting demand trends is critical for:
- Inventory planning
- Stock replenishment
- Resource allocation

74% directional accuracy demonstrates practical applicability of the Transformer forecast.


## 7. Takeaways
- Transformer consistently outperforms LSTM and baselines
- Statistical tests validate model superiority
- Walk-forward backtesting ensures model robustness
- Professional outputs (plots, CSVs, models) make the project deployment-ready

**Learning Outcomes:**
- Multi-step forecasting with PyTorch
- Sequence modeling & attention mechanisms
- Performance evaluation for time series
- Statistical testing for forecast significance



## 8. Project Structure
```
forecasting-project/
│
├── data/
│ ├── raw/
│ └── processed/
├── src/
│ ├── preprocessing.py
│ ├── data_loader.py
│ ├── baselines.py
│ ├── models.py
│ ├── transformer_model.py
│ ├── training.py
│ ├── evaluation.py
│ ├── significance.py
│ ├── training_phase3.py
│ └── backtesting.py
│
├── results/
│ ├── metrics_phase3.csv
│ ├── metrics_phase3_comparison.csv
│ ├── plots/
│ └── model_weights/
└── requirements.txt
```


## 9. Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full training and evaluation
python -m src.training_phase3

Outputs will be saved in results/ as CSV metrics, plots, and model weights.



10. Tools & Libraries

This project leverages the following technologies:

- **Python 3.9+** – Core programming language  
- **PyTorch** – Deep learning framework for LSTM & Transformer  
- **NumPy** – Numerical computations  
- **Pandas** – Data handling and manipulation  
- **scikit-learn** – Data preprocessing and evaluation metrics  
- **Matplotlib / Seaborn** – Visualization of forecasts and metrics  

>  Tip: Use a virtual environment (`venv` or `conda`) to manage dependencies.



11. Key Highlights

- Multi-step forecasting with **LSTM** and **Transformer** models  
- Evaluation using **absolute**, **relative**, and **directional** metrics  
- Statistical validation with **Diebold-Mariano test**  
- Robustness verified via **walk-forward backtesting**  
- Professional outputs: plots, CSVs, and model weights  
- Fully **reproducible** and deployment-ready  
