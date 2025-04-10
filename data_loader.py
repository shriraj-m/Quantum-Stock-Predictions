import yfinance as yf
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler


def load_stock_data(symbol: str, sequence_length, train_split=0.8):
    df = yf.download(symbol, period="1y")
    df = df['Close'].values.reshape(-1,1)

    scaler = MinMaxScaler()
    scaled_df = scaler.fit_transform(df)

    X = []
    y = []

    for i in range(len(scaled_df) - sequence_length):
        X.append(scaled_df[i:i+sequence_length].flatten())
        y.append(scaled_df[i+sequence_length])

    X = np.array(X)
    y = np.array(y).reshape(-1,1)

    # convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # test and train split
    split_index = int(train_split * len(X_tensor))
    X_train, X_test = X_tensor[:split_index], X_tensor[split_index:]
    y_train, y_test = y_tensor[:split_index], y_tensor[split_index:]

    return X_train, X_test, y_train, y_test, scaler
