import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------
# CONFIG
# ------------------------------------------------
DATA_PATH = "data/processed/demand_daily_2019_2026.csv"

INPUT_WINDOW = 30
FORECAST_HORIZON = 7
DEFAULT_BATCH_SIZE = 32

# ------------------------------------------------
# CUSTOM DATASET
# ------------------------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()
    return df

# ------------------------------------------------
# CHRONOLOGICAL SPLIT
# ------------------------------------------------
def split_data(df):
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    print("Train size:", len(train))
    print("Val size:", len(val))
    print("Test size:", len(test))

    return train, val, test

# ------------------------------------------------
# SCALING (FIT ONLY ON TRAIN)
# ------------------------------------------------
def scale_data(train, val, test):
    """
    Scale train, val, test using MinMaxScaler.
    Returns scaled NumPy arrays.
    """
    scaler = MinMaxScaler()

    train_np = np.array(train).reshape(-1, 1)
    val_np   = np.array(val).reshape(-1, 1)
    test_np  = np.array(test).reshape(-1, 1)

    train_scaled = scaler.fit_transform(train_np).reshape(-1)
    val_scaled   = scaler.transform(val_np).reshape(-1)
    test_scaled  = scaler.transform(test_np).reshape(-1)

    return train_scaled, val_scaled, test_scaled, scaler

# ------------------------------------------------
# CREATE SLIDING WINDOWS
# ------------------------------------------------
def create_sequences(data):
    X, y = [], []
    for i in range(len(data) - INPUT_WINDOW - FORECAST_HORIZON + 1):
        X.append(data[i:i+INPUT_WINDOW])
        y.append(data[i+INPUT_WINDOW:i+INPUT_WINDOW+FORECAST_HORIZON])
    X = np.array(X).reshape(-1, INPUT_WINDOW, 1)   # [samples, input_window, 1]
    y = np.array(y).reshape(-1, FORECAST_HORIZON)  # [samples, forecast_horizon]
    return X, y

# ------------------------------------------------
# BUILD DATALOADERS
# ------------------------------------------------
def build_dataloaders(batch_size=DEFAULT_BATCH_SIZE):
    df = load_data()
    train, val, test = split_data(df)
    train_scaled, val_scaled, test_scaled, scaler = scale_data(train, val, test)

    X_train, y_train = create_sequences(train_scaled)
    X_val, y_val = create_sequences(val_scaled)
    X_test, y_test = create_sequences(test_scaled)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler

# ------------------------------------------------
# TEST RUN
# ------------------------------------------------
if __name__ == "__main__":
    build_dataloaders()